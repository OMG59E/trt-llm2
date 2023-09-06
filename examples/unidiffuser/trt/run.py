import os
import sys
import cv2
import time
import numpy as np
import torch
import einops
from cuda import cudart
from trt_infer import TRTInfer
from transformers import CLIPTokenizer
from plugin import GroupNormLayer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pytorch.dpm_solver_pp import NoiseScheduleVP


class UnidiffuserText2ImgTRT(object):
    def __init__(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip = TRTInfer("outputs/clip_float16.trt")
        self.uvit = TRTInfer("outputs/uvit_float16.trt")
        self.decoder = TRTInfer("outputs/decoder_float16.trt")
        
        self.device = 'cuda'
        self.n_samples = 1
        self.z_shape = (4, 64, 64)
        self.clip_img_dim = 512
        self.sample_steps = 50
        
        self._betas = torch.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, 1000, dtype=torch.float32) ** 2
        self.N = len(self._betas)
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(self._betas.numpy(), device=self.device).float())

        t_0 = 1. / self.N
        t_T = 1.
        K = self.sample_steps // 3 + 1
        if self.sample_steps % 3 == 0:
            self.orders = [3,] * (K - 2) + [2, 1]
        elif self.sample_steps % 3 == 1:
            self.orders = [3,] * (K - 1) + [1]
        else:
            self.orders = [3,] * (K - 1) + [2]
            
        self.timesteps = torch.linspace(t_T, t_0, self.sample_steps + 1).cuda()   # time_uniform

    def model_fn(self, x, t):
        text_N = torch.randn_like(self.clip.outputs[0]["tensor"]).type(torch.float32).contiguous().cuda()
        ts = t * self.N
        alpha_t = self.noise_schedule.marginal_alpha(t).contiguous().cuda()
        sigma_t = self.noise_schedule.marginal_std(t).contiguous().cuda()
        # t_start = time.time()
        x_data_ptr = self.uvit.inputs[0]["tensor"].data_ptr()
        x_data_size = self.uvit.inputs[0]["size"]
        ts_data_ptr = self.uvit.inputs[1]["tensor"].data_ptr()
        ts_data_size = self.uvit.inputs[1]["size"]
        contexts_data_ptr = self.uvit.inputs[2]["tensor"].data_ptr()
        contexts_data_size = self.uvit.inputs[2]["size"]
        text_N_data_ptr = self.uvit.inputs[3]["tensor"].data_ptr()
        text_N_data_size = self.uvit.inputs[3]["size"]
        sigma_data_ptr = self.uvit.inputs[4]["tensor"].data_ptr()
        sigma_data_size = self.uvit.inputs[4]["size"]
        alpha_data_ptr = self.uvit.inputs[5]["tensor"].data_ptr()
        alpha_data_size = self.uvit.inputs[5]["size"]
        cudart.cudaMemcpy(x_data_ptr, x.contiguous().data_ptr(), x_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(ts_data_ptr, ts.contiguous().data_ptr(), ts_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(contexts_data_ptr, self.clip.outputs[0]["tensor"].data_ptr(), contexts_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(text_N_data_ptr, text_N.data_ptr(), text_N_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(sigma_data_ptr, sigma_t.data_ptr(), sigma_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(alpha_data_ptr, alpha_t.data_ptr(), alpha_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        self.uvit.infer()
        # print("uvit: {:.3f}ms".format((time.time() - t_start) * 1000))
        x0 = self.uvit.outputs[0]["tensor"]  # bs, 16896 (4*64*64 + 512)
        return x0
    
    def dpm_solver_first_update(self, x, s, t):
        lambda_s = self.noise_schedule.marginal_lambda(s)
        lambda_t = self.noise_schedule.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s = self.noise_schedule.marginal_log_mean_coeff(s)
        log_alpha_t = self.noise_schedule.marginal_log_mean_coeff(t)
        sigma_s = self.noise_schedule.marginal_std(s)
        sigma_t = self.noise_schedule.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        phi_1 = 1. - torch.exp(-h)
        noise_s = self.model_fn(x, s)
        x_t = (sigma_t / sigma_s) * x + (alpha_t * phi_1) * noise_s
        return x_t

    def dpm_solver_second_update(self, x, s, t, r1):
        lambda_s = self.noise_schedule.marginal_lambda(s)
        lambda_t = self.noise_schedule.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = self.noise_schedule.inverse_lambda(lambda_s1)
        
        log_alpha_s = self.noise_schedule.marginal_log_mean_coeff(s)
        log_alpha_s1 = self.noise_schedule.marginal_log_mean_coeff(s1)
        log_alpha_t = self.noise_schedule.marginal_log_mean_coeff(t)
        
        sigma_s = self.noise_schedule.marginal_std(s) 
        sigma_s1 = self.noise_schedule.marginal_std(s1)
        sigma_t = self.noise_schedule.marginal_std(t)
        
        alpha_s1 = torch.exp(log_alpha_s1)
        alpha_t = torch.exp(log_alpha_t)
        
        phi_11 = torch.expm1(-r1 * h)
        phi_1 = torch.expm1(-h)
        noise_s = self.model_fn(x, s)
        x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * noise_s
        noise_s1 = self.model_fn(x_s1, s1)
        x_t = (sigma_t / sigma_s) * x - (alpha_t * phi_1) * noise_s - (0.5 / r1) * (alpha_t * phi_1) * (noise_s1 - noise_s)
        return x_t

    def dpm_solver_third_update(self, x, s, t, r1, r2):
        lambda_s = self.noise_schedule.marginal_lambda(s)
        lambda_t = self.noise_schedule.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = self.noise_schedule.inverse_lambda(lambda_s1)
        s2 = self.noise_schedule.inverse_lambda(lambda_s2)

        log_alpha_s = self.noise_schedule.marginal_log_mean_coeff(s)
        log_alpha_s1 = self.noise_schedule.marginal_log_mean_coeff(s1)
        log_alpha_s2 = self.noise_schedule.marginal_log_mean_coeff(s2)
        log_alpha_t = self.noise_schedule.marginal_log_mean_coeff(t)
        
        sigma_s = self.noise_schedule.marginal_std(s)
        sigma_s1 = self.noise_schedule.marginal_std(s1)
        sigma_s2 = self.noise_schedule.marginal_std(s2)
        sigma_t = self.noise_schedule.marginal_std(t)

        alpha_s1 = torch.exp(log_alpha_s1)
        alpha_s2 = torch.exp(log_alpha_s2)
        alpha_t = torch.exp(log_alpha_t)

        phi_11 = torch.expm1(-r1 * h)
        phi_12 = torch.expm1(-r2 * h)
        phi_1 = torch.expm1(-h)
        phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
        phi_2 = phi_1 / h + 1.
        noise_s = self.model_fn(x, s)
        x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * noise_s
        noise_s1 = self.model_fn(x_s1, s1)
        x_s2 = (sigma_s2 / sigma_s) * x - (alpha_s2 * phi_12) * noise_s + r2 / r1 * (alpha_s2 * phi_22) * (noise_s1 - noise_s)
        noise_s2 = self.model_fn(x_s2, s2)
        x_t = (sigma_t / sigma_s) * x - (alpha_t * phi_1) * noise_s + (1. / r2) * (alpha_t * phi_2) * (noise_s2 - noise_s)
        return x_t
                     
    def process(self, prompt="a dog under the sea", seed=1234) -> np.ndarray:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # step1
        t_start = time.time()
        batch_encoding = self.tokenizer([prompt], truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        input_ids = batch_encoding["input_ids"].type(torch.int32).contiguous().cuda()
        clip_input_data_ptr = self.clip.inputs[0]["tensor"].data_ptr()
        clip_input_data_size = self.clip.inputs[0]["size"]
        cudart.cudaMemcpy(clip_input_data_ptr, input_ids.data_ptr(), clip_input_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        self.clip.infer()
        print("clip: {:.3f}ms".format((time.time() - t_start) * 1000))

        # step2
        t_start = time.time()
        z_init = torch.randn(self.n_samples, *(self.z_shape), device=self.device)  # 1,4,64,64
        clip_img_init = torch.randn(self.n_samples, 1, self.clip_img_dim, device=self.device)  # 1,1,512
        z = einops.rearrange(z_init, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img_init, 'B L D -> B (L D)')
        x = torch.concat([z, clip_img], dim=-1)   # 1,16896

        i = 0
        for order in self.orders:
            vec_s = torch.ones((x.shape[0],)).cuda() * self.timesteps[i]
            vec_t = torch.ones((x.shape[0],)).cuda() * self.timesteps[i + order]
            h = self.noise_schedule.marginal_lambda(self.timesteps[i + order]) - self.noise_schedule.marginal_lambda(self.timesteps[i])
            r1 = None if order <= 1 else (self.noise_schedule.marginal_lambda(self.timesteps[i + 1]) - self.noise_schedule.marginal_lambda(self.timesteps[i])) / h
            r2 = None if order <= 2 else (self.noise_schedule.marginal_lambda(self.timesteps[i + 2]) - self.noise_schedule.marginal_lambda(self.timesteps[i])) / h
            if order == 1:
                x = self.dpm_solver_first_update(x, vec_s, vec_t)
            elif order == 2:
                x = self.dpm_solver_second_update(x, vec_s, vec_t, r1)
            elif order == 3:
                x = self.dpm_solver_third_update(x, vec_s, vec_t, r1, r2)
            else:
                raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))
            i += order

        C, H, W = self.z_shape
        z_dim = C * H * W
        z, _ = x.split([z_dim, self.clip_img_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        print("all uvit: {:.3f}ms".format((time.time() - t_start) * 1000))

        # decoder
        t_start = time.time()
        z_data_ptr = self.decoder.inputs[0]["tensor"].data_ptr()
        z_data_size = self.decoder.inputs[0]["size"]
        cudart.cudaMemcpy(z_data_ptr, z.contiguous().data_ptr(), z_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        self.decoder.infer()
        print("decoder: {:.3f}ms".format((time.time() - t_start) * 1000))
        return self.decoder.outputs[0]["tensor"].cpu().numpy()
        
        
if __name__ == "__main__":           
    m = UnidiffuserText2ImgTRT()
    for idx in range(1):
        t_start = time.time()
        samples = m.process(prompt="a dog under the sea", seed=29764)
        print(idx, "end2end {:.3f}ms".format((time.time() - t_start) * 1000))
    cv2.imwrite("sample_trt.jpg", samples[0])

