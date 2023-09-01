import os
import sys
import torch
import einops
from cuda import cudart
from trt_infer import TRTInfer
from transformers import CLIPTokenizer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pytorch.dpm_solver_pp import NoiseScheduleVP


device = 'cuda'
n_samples = 1
seed = 1234
z_shape = (4, 64, 64)
clip_img_dim = 512
clip_text_dim = 768
text_dim = 64  # reduce dimension
sample_steps = 50
scale = 7.
t2i_cfg_mode = "true_uncond"
contexts_low_dim = 64

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

_betas = torch.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, 1000, dtype=torch.float32) ** 2
N = len(_betas)

prompts = ["a dog under the sea"]

# step1
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
batch_encoding = tokenizer(prompts, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
input_ids = batch_encoding["input_ids"].type(torch.int32).contiguous().cuda()
clip = TRTInfer("outputs/clip_float16.trt")
clip_input_data_ptr = clip.inputs[0]["tensor"].data_ptr()
clip_input_data_size = clip.inputs[0]["size"]
cudart.cudaMemcpy(clip_input_data_ptr, input_ids.data_ptr(), clip_input_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
clip.infer()

# step2
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

uvit = TRTInfer("outputs/uvit_float32.trt")
x_data_ptr = uvit.inputs[0]["tensor"].data_ptr()
x_data_size = uvit.inputs[0]["size"]
ts_data_ptr = uvit.inputs[1]["tensor"].data_ptr()
ts_data_size = uvit.inputs[1]["size"]
contexts_data_ptr = uvit.inputs[2]["tensor"].data_ptr()
contexts_data_size = uvit.inputs[2]["size"]
text_N_data_ptr = uvit.inputs[3]["tensor"].data_ptr()
text_N_data_size = uvit.inputs[3]["size"]

def model_fn(x, t):
    text_N = torch.randn_like(clip.outputs[0]["tensor"]).type(torch.float32).cuda()
    ts = t * N
    cudart.cudaMemcpy(x_data_ptr, x.data_ptr(), x_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
    cudart.cudaMemcpy(ts_data_ptr, ts.data_ptr(), ts_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
    cudart.cudaMemcpy(contexts_data_ptr, clip.outputs[0]["tensor"].data_ptr(), contexts_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
    cudart.cudaMemcpy(text_N_data_ptr, text_N.data_ptr(), text_N_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
    uvit.infer()
    noise = uvit.outputs[0]["tensor"]
    alpha_t, sigma_t = noise_schedule.marginal_alpha(t), noise_schedule.marginal_std(t)
    dims = len(x.shape) - 1
    x0 = (x - sigma_t[(...,) + (None,)*dims] * noise) / alpha_t[(...,) + (None,)*dims]
    return x0


def dpm_solver_first_update(x, s, t):
    dims = len(x.shape) - 1
    lambda_s, lambda_t = noise_schedule.marginal_lambda(s), noise_schedule.marginal_lambda(t)
    h = lambda_t - lambda_s
    log_alpha_s, log_alpha_t = noise_schedule.marginal_log_mean_coeff(s), noise_schedule.marginal_log_mean_coeff(t)
    sigma_s, sigma_t = noise_schedule.marginal_std(s), noise_schedule.marginal_std(t)
    alpha_t = torch.exp(log_alpha_t)
    phi_1 = (torch.exp(-h) - 1.) / (-1.)
    noise_s = model_fn(x, s)
    x_t = ((sigma_t / sigma_s)[(...,) + (None,)*dims] * x + (alpha_t * phi_1)[(...,) + (None,)*dims] * noise_s)
    return x_t


def dpm_solver_second_update(x, s, t, r1):
    dims = len(x.shape) - 1
    lambda_s, lambda_t = noise_schedule.marginal_lambda(s), noise_schedule.marginal_lambda(t)
    h = lambda_t - lambda_s
    lambda_s1 = lambda_s + r1 * h
    s1 = noise_schedule.inverse_lambda(lambda_s1)
    log_alpha_s, log_alpha_s1, log_alpha_t = noise_schedule.marginal_log_mean_coeff(s), noise_schedule.marginal_log_mean_coeff(s1), noise_schedule.marginal_log_mean_coeff(t)
    sigma_s, sigma_s1, sigma_t = noise_schedule.marginal_std(s), noise_schedule.marginal_std(s1), noise_schedule.marginal_std(t)
    alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)
    phi_11 = torch.expm1(-r1 * h)
    phi_1 = torch.expm1(-h)
    noise_s = model_fn(x, s)
    x_s1 = ((sigma_s1 / sigma_s)[(...,) + (None,)*dims] * x - (alpha_s1 * phi_11)[(...,) + (None,)*dims] * noise_s)
    noise_s1 = model_fn(x_s1, s1)
    x_t = ((sigma_t / sigma_s)[(...,) + (None,)*dims] * x - (alpha_t * phi_1)[(...,) + (None,)*dims] * noise_s - (0.5 / r1) * (alpha_t * phi_1)[(...,) + (None,)*dims] * (noise_s1 - noise_s))
    return x_t


def dpm_solver_third_update(x, s, t, r1, r2):
    dims = len(x.shape) - 1
    lambda_s, lambda_t = noise_schedule.marginal_lambda(s), noise_schedule.marginal_lambda(t)
    h = lambda_t - lambda_s
    lambda_s1 = lambda_s + r1 * h
    lambda_s2 = lambda_s + r2 * h
    s1 = noise_schedule.inverse_lambda(lambda_s1)
    s2 = noise_schedule.inverse_lambda(lambda_s2)
    log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = noise_schedule.marginal_log_mean_coeff(s), noise_schedule.marginal_log_mean_coeff(s1), noise_schedule.marginal_log_mean_coeff(s2), noise_schedule.marginal_log_mean_coeff(t)
    sigma_s, sigma_s1, sigma_s2, sigma_t = noise_schedule.marginal_std(s), noise_schedule.marginal_std(s1), noise_schedule.marginal_std(s2), noise_schedule.marginal_std(t)
    alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)
    phi_11 = torch.expm1(-r1 * h)
    phi_12 = torch.expm1(-r2 * h)
    phi_1 = torch.expm1(-h)
    phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
    phi_2 = phi_1 / h + 1.
    noise_s = model_fn(x, s)
    x_s1 = ((sigma_s1 / sigma_s)[(...,) + (None,)*dims] * x - (alpha_s1 * phi_11)[(...,) + (None,)*dims] * noise_s)
    noise_s1 = model_fn(x_s1, s1)
    x_s2 = ((sigma_s2 / sigma_s)[(...,) + (None,)*dims] * x - (alpha_s2 * phi_12)[(...,) + (None,)*dims] * noise_s + r2 / r1 * (alpha_s2 * phi_22)[(...,) + (None,)*dims] * (noise_s1 - noise_s))
    noise_s2 = model_fn(x_s2, s2)
    x_t = ((sigma_t / sigma_s)[(...,) + (None,)*dims] * x - (alpha_t * phi_1)[(...,) + (None,)*dims] * noise_s + (1. / r2) * (alpha_t * phi_2)[(...,) + (None,)*dims] * (noise_s2 - noise_s))
    return x_t


z_init = torch.randn(n_samples, *z_shape, device=device)
clip_img_init = torch.randn(n_samples, 1, clip_img_dim, device=device)
z = einops.rearrange(z_init, 'B C H W -> B (C H W)')
clip_img = einops.rearrange(clip_img_init, 'B L D -> B (L D)')
x = torch.concat([z, clip_img], dim=-1)

t_0 = 1. / N
t_T = 1.
device = x.device
order=3
K = sample_steps // 3 + 1
if sample_steps % 3 == 0:
    orders = [3,] * (K - 2) + [2, 1]
elif sample_steps % 3 == 1:
    orders = [3,] * (K - 1) + [1]
else:
    orders = [3,] * (K - 1) + [2]

timesteps = torch.linspace(t_T, t_0, sample_steps + 1).to(device)   # time_uniform
i = 0
for order in orders:
    vec_s = torch.ones((x.shape[0],)).to(device) * timesteps[i]
    vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i + order]
    h = noise_schedule.marginal_lambda(timesteps[i + order]) - noise_schedule.marginal_lambda(timesteps[i])
    r1 = None if order <= 1 else (noise_schedule.marginal_lambda(timesteps[i + 1]) - noise_schedule.marginal_lambda(timesteps[i])) / h
    r2 = None if order <= 2 else (noise_schedule.marginal_lambda(timesteps[i + 2]) - noise_schedule.marginal_lambda(timesteps[i])) / h
    if order == 1:
        x = dpm_solver_first_update(x, vec_s, vec_t)
    elif order == 2:
        x = dpm_solver_second_update(x, vec_s, vec_t, r1)
    elif order == 3:
        x = dpm_solver_third_update(x, vec_s, vec_t, r1, r2)
    else:
        raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))
    i += order

C, H, W = z_shape
z_dim = C * H * W
z, clip_img = x.split([z_dim, clip_img_dim], dim=1)
z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
