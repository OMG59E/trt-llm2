import torch
import utils
import random
import einops
import numpy as np
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image
from libs.caption_decoder import CaptionDecoder
from transformers import CLIPTokenizer, CLIPTextModel
from dpm_solver_pp import NoiseScheduleVP


device = 'cuda'
nnet_path = "models/uvit_v1.pth"
output_path = "out"
prompt = "a dog under the sea"
n_samples = 1
seed = 1234
z_shape = (4, 64, 64)
clip_img_dim = 512
clip_text_dim = 768
text_dim = 64  # reduce dimension
sample_steps = 50
scale = 7.
t2i_cfg_mode = "true_uncond"
  

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    return _betas.numpy()


def split(x):
    C, H, W = z_shape
    z_dim = C * H * W
    z, clip_img = x.split([z_dim, clip_img_dim], dim=1)
    z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
    clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=clip_img_dim)
    return z, clip_img
    
    
def combine(z, clip_img):
    z = einops.rearrange(z, 'B C H W -> B (C H W)')
    clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
    return torch.concat([z, clip_img], dim=-1)
    

def t2i_nnet(x, timesteps, text, nnet, N):  # text is the low dimension version of the text clip embedding
    z, clip_img = split(x)
    t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)
    data_type = torch.zeros_like(t_text, device=device, dtype=torch.int) + 1
    # z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text, data_type=data_type)
    # x_out = combine(z_out, clip_img_out)
    # if scale == 0.:
    #     return x_out
    text_N = torch.randn_like(text)  # 3 other possible choices
    # z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N, data_type=data_type)
    # x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)

    _z = torch.cat([z, z], dim=0)
    _clip_img = torch.cat([clip_img, clip_img], dim=0)
    _text = torch.cat([text, text_N], dim=0)
    _t_img = torch.cat([timesteps, timesteps], dim=0)
    _t_text = torch.cat([t_text, torch.ones_like(timesteps) * N], dim=0)
    _data_type = torch.cat([data_type, data_type], dim=0)
    z_out, clip_img_out, _ = nnet(_z, _clip_img, text=_text, t_img=_t_img, t_text=_t_text, data_type=_data_type)
    x_out = combine(z_out, clip_img_out)
    return x_out[0] + scale * (x_out[0] - x_out[1])
    

def model_fn(x, t, ns, contexts, nnet, N):
    alpha_t, sigma_t = ns.marginal_alpha(t), ns.marginal_std(t)
    noise = t2i_nnet(x, t * N, contexts, nnet, N)
    dims = len(x.shape) - 1
    x0 = (x - sigma_t[(...,) + (None,)*dims] * noise) / alpha_t[(...,) + (None,)*dims]
    return x0
                               
def dpm_solver_first_update(x, s, t, ns, contexts, nnet, N):
    dims = len(x.shape) - 1
    lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
    h = lambda_t - lambda_s
    log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
    sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
    alpha_t = torch.exp(log_alpha_t)
    phi_1 = (torch.exp(-h) - 1.) / (-1.)
    noise_s = model_fn(x, s, ns, contexts, nnet, N)
    x_t = ((sigma_t / sigma_s)[(...,) + (None,)*dims] * x + (alpha_t * phi_1)[(...,) + (None,)*dims] * noise_s)
    return x_t

def dpm_solver_second_update(x, s, t, r1, ns, contexts, nnet, N):
    dims = len(x.shape) - 1
    lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
    h = lambda_t - lambda_s
    lambda_s1 = lambda_s + r1 * h
    s1 = ns.inverse_lambda(lambda_s1)
    log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
    sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
    alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)
    phi_11 = torch.expm1(-r1 * h)
    phi_1 = torch.expm1(-h)
    noise_s = model_fn(x, s, ns, contexts, nnet, N)
    x_s1 = ((sigma_s1 / sigma_s)[(...,) + (None,)*dims] * x - (alpha_s1 * phi_11)[(...,) + (None,)*dims] * noise_s)
    noise_s1 = model_fn(x_s1, s1, ns, contexts, nnet, N)
    x_t = ((sigma_t / sigma_s)[(...,) + (None,)*dims] * x - (alpha_t * phi_1)[(...,) + (None,)*dims] * noise_s - (0.5 / r1) * (alpha_t * phi_1)[(...,) + (None,)*dims] * (noise_s1 - noise_s))
    return x_t


def dpm_solver_third_update(x, s, t, r1, r2, ns, contexts, nnet, N):
    dims = len(x.shape) - 1
    lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
    h = lambda_t - lambda_s
    lambda_s1 = lambda_s + r1 * h
    lambda_s2 = lambda_s + r2 * h
    s1 = ns.inverse_lambda(lambda_s1)
    s2 = ns.inverse_lambda(lambda_s2)
    log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
    sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
    alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)
    phi_11 = torch.expm1(-r1 * h)
    phi_12 = torch.expm1(-r2 * h)
    phi_1 = torch.expm1(-h)
    phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
    phi_2 = phi_1 / h + 1.
    noise_s = model_fn(x, s, ns, contexts, nnet, N)
    x_s1 = ((sigma_s1 / sigma_s)[(...,) + (None,)*dims] * x - (alpha_s1 * phi_11)[(...,) + (None,)*dims] * noise_s)
    noise_s1 = model_fn(x_s1, s1, ns, contexts, nnet, N)
    x_s2 = ((sigma_s2 / sigma_s)[(...,) + (None,)*dims] * x - (alpha_s2 * phi_12)[(...,) + (None,)*dims] * noise_s + r2 / r1 * (alpha_s2 * phi_22)[(...,) + (None,)*dims] * (noise_s1 - noise_s))
    noise_s2 = model_fn(x_s2, s2, ns, contexts, nnet, N)
    x_t = ((sigma_t / sigma_s)[(...,) + (None,)*dims] * x - (alpha_t * phi_1)[(...,) + (None,)*dims] * noise_s + (1. / r2) * (alpha_t * phi_2)[(...,) + (None,)*dims] * (noise_s2 - noise_s))
    return x_t
    
    
def main():    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(seed)
    
    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)
    
    nnet_dict = {
        "name": 'uvit_multi_post_ln_v1',
        "img_size": 64,
        "in_chans": 4,
        "patch_size":2,
        "embed_dim": 1536,
        "depth": 30,
        "num_heads": 24,
        "mlp_ratio": 4,
        "qkv_bias": False,
        "pos_drop_rate": 0.,
        "drop_rate": 0.,
        "attn_drop_rate": 0.,
        "mlp_time_embed": False,
        "text_dim": 64,
        "num_text_tokens": 77,
        "clip_img_dim": 512,
        "use_checkpoint": False}
    
    nnet = utils.get_nnet(**nnet_dict)
    nnet.load_state_dict(torch.load(nnet_path, map_location='cpu'))
    nnet.to(device)
    nnet.eval()
    
    autoencoder = libs.autoencoder.get_model(pretrained_path='models/autoencoder_kl.pth')
    autoencoder.to(device)

    # step1 CLIP
    prompts = [prompt] * n_samples
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    batch_encoding = tokenizer(prompts, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"].to(device)

    transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    transformer = transformer.cuda().eval()
    tokens = batch_encoding["input_ids"].to(device)
    outputs = transformer(input_ids=tokens)
    contexts = outputs.last_hidden_state
    # the low dimensional version of the contexts, which is the input to the nnet
    caption_decoder = CaptionDecoder(device=device, pretrained_path="models/caption_decoder.pth", hidden_dim=64)
    contexts_low_dim = caption_decoder.encode_prefix(contexts)  

    # step2 
    z_init = torch.randn(n_samples, *z_shape, device=device)
    clip_img_init = torch.randn(n_samples, 1, clip_img_dim, device=device)
    z = einops.rearrange(z_init, 'B C H W -> B (C H W)')
    clip_img = einops.rearrange(clip_img_init, 'B L D -> B (L D)')
    x = torch.concat([z, clip_img], dim=-1)
    
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())
                             
    with torch.no_grad():
        with torch.autocast(device_type=device):
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
                vec_s, vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i], torch.ones((x.shape[0],)).to(device) * timesteps[i + order]
                h = noise_schedule.marginal_lambda(timesteps[i + order]) - noise_schedule.marginal_lambda(timesteps[i])
                r1 = None if order <= 1 else (noise_schedule.marginal_lambda(timesteps[i + 1]) - noise_schedule.marginal_lambda(timesteps[i])) / h
                r2 = None if order <= 2 else (noise_schedule.marginal_lambda(timesteps[i + 2]) - noise_schedule.marginal_lambda(timesteps[i])) / h
                if order == 1:
                    x = dpm_solver_first_update(x, vec_s, vec_t, noise_schedule, contexts_low_dim, nnet, N)
                elif order == 2:
                    x = dpm_solver_second_update(x, vec_s, vec_t, r1, noise_schedule, contexts_low_dim, nnet, N)
                elif order == 3:
                    x = dpm_solver_third_update(x, vec_s, vec_t, r1, r2, noise_schedule, contexts_low_dim, nnet, N)
                else:
                    raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))
                i += order
            z, _ = split(x)
        
        # step3 decoder
        samples = 0.5 * (autoencoder.decode(z) + 1.) 
        samples = samples.mul(255.).clamp_(0., 255.).permute((0, 2, 3, 1))[:, :, :, [2, 1, 0]].cpu().numpy().astype(np.uint8)
        import cv2
        cv2.imwrite("sample.jpg", samples[0])
    
if __name__ == "__main__":
    main()