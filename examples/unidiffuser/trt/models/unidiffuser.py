import os
import sys
import torch
import numpy as np
import tensorrt as trt
from tensorrt_llm.functional import Tensor, constant, select, unsqueeze, concat
from tensorrt_llm.layers import Linear
from tensorrt_llm.module import Module 
from .clip import CLIPTextTransformer
from .uvit import UViT, split, combine
from .decoder import Decoder
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pytorch.dpm_solver_pp import NoiseScheduleVP


class CaptionEncoder(Module):
    def __init__(self, hidden_dim, dtype) -> None:
        super().__init__()
        self.encode_prefix = Linear(768, hidden_dim, dtype=dtype)
    
    def forward(self, x):
        return self.encode_prefix(x)
    
    
class Unidiffuser(Module):
    def __init__(self) -> None:
        super().__init__()
        self.caption_encode = CaptionEncoder(64, trt.float32)
        self.clip = CLIPTextTransformer(
                    hidden_size=768, 
                    vocab_size=49408, 
                    max_position_embeddings=77, 
                    num_attention_heads=12, 
                    intermediate_size=3072, 
                    hidden_act="quick_gelu", 
                    num_hidden_layers=12, 
                    output_attentions=False, 
                    output_hidden_states=False, 
                    use_return_dict=True, 
                    np_dtype=np.float32)

        self.nnet = UViT(img_size=64, 
                         in_chans=4, 
                         patch_size=2, 
                         embed_dim=1536, 
                         depth=30, 
                         num_heads=24, 
                         mlp_ratio=4., 
                         qkv_bias=False, 
                         qk_scale=None,
                         mlp_time_embed=False, 
                         text_dim=64, 
                         num_text_tokens=77, 
                         clip_img_dim=512, 
                         np_dtype=np.float32)

        self.decoder = Decoder(
            ch=128, 
            out_ch=3, 
            ch_mult=(1, 2, 4, 4), 
            num_res_blocks=2, 
            embed_dim=4, 
            attn_resolutions=[], 
            resamp_with_conv=True, 
            in_channels=3, 
            resolution=256, 
            z_channels=4, 
            give_pre_end=False, 
            tanh_out=False, 
            use_linear_attn=False, 
            scale_factor=0.18215,
            np_dtype=np.float32)
        
        self.sample_steps = 50
        K = self.sample_steps // 3 + 1
        if self.sample_steps % 3 == 0:
            self.orders = [3,] * (K - 2) + [2, 1]
        elif self.sample_steps % 3 == 1:
            self.orders = [3,] * (K - 1) + [1]
        else:
            self.orders = [3,] * (K - 1) + [2]

        self._betas = torch.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, 1000, dtype=torch.float32) ** 2
        self.N = len(self._betas)
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(self._betas.numpy()).float())

        self.timesteps = torch.linspace(1.0, 0.001, self.sample_steps + 1).type(torch.float32)   # time_uniform
        self.marginal_lambda = self.noise_schedule.marginal_lambda(self.timesteps).type(torch.float32)
        self.marginal_alpha = self.noise_schedule.marginal_alpha(self.timesteps).type(torch.float32)
        self.marginal_std = self.noise_schedule.marginal_std(self.timesteps).type(torch.float32)
        self.marginal_log_mean_coeff = self.noise_schedule.marginal_log_mean_coeff(self.timesteps).type(torch.float32)

        self.inverse_lambda = self.noise_schedule.inverse_lambda(self.marginal_lambda).type(torch.float32)
        self.marginal_log_mean_coeff_inverse = self.noise_schedule.marginal_log_mean_coeff(self.inverse_lambda).type(torch.float32)
        self.marginal_alpha_inverse = self.noise_schedule.marginal_alpha(self.inverse_lambda).type(torch.float32)
        self.marginal_std_inverse = self.noise_schedule.marginal_std(self.inverse_lambda).type(torch.float32)
        
        self.alpha_s1 = torch.exp(self.marginal_log_mean_coeff_inverse).type(torch.float32)
        self.alpha_s2 = torch.exp(self.marginal_log_mean_coeff_inverse).type(torch.float32)
        self.alpha_t = torch.exp(self.marginal_log_mean_coeff).type(torch.float32)
        self.marginal_lambda_exp = torch.exp(self.marginal_lambda).type(torch.float32)

    def model_fn(self, x, timesteps, text, text_N, sigma, alpha):
        bs = x.shape[0]
        z, clip_img = split(x)
        t_text0 = constant(np.zeros(shape=(bs,), dtype=np.int32))
        t_text1 = constant(np.ones(shape=(bs,), dtype=np.int32) * 1000)
        data_type = constant(np.zeros(shape=(bs,), dtype=np.int32) + 1)
        text = concat([text, text_N], dim=0)  # 2*bs, 77, 64
        t_text = concat([t_text0, t_text1], dim=0)  # 2*bs,
        z_out, clip_img_out = self.nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text, data_type=data_type)
        x_out = combine(z_out, clip_img_out)
        x_out0, x_out1 = x_out.split([1, 1], dim=0)
        noise = x_out0 + 7. * (x_out0 - x_out1)
        x_out = (x - sigma * noise) / alpha
        return x_out

    def dpm_solver_first_update(self, z: Tensor, clip_img: Tensor, idx: Tensor, text: Tensor, text_N: Tensor) -> Tensor:
        timesteps = unsqueeze(constant(self.timesteps.detach().cpu().numpy()), axis=0)  # [1, 51]
        marginal_alpha = unsqueeze(constant(self.marginal_alpha.detach().cpu().numpy()), axis=0)  # [1, 51]
        marginal_std = unsqueeze(constant(self.marginal_std.detach().cpu().numpy()), axis=0)  # [1, 51]        
        marginal_lambda_exp = unsqueeze(constant(self.marginal_lambda_exp.detach().cpu().numpy()), axis=0)  # [1, 51]
        _alpha_t = unsqueeze(constant(self.alpha_t.detach().cpu().numpy()), axis=0)  # [1, 51]
        x = combine(z, clip_img)
        alpha_t = select(_alpha_t, dim=1, index=idx + 1)
        phi_1 = 1. - select(marginal_lambda_exp, dim=1, index=idx) / select(marginal_lambda_exp, dim=1, index=idx + 1)
        
        sigma_s = select(marginal_std, dim=1, index=idx)
        sigma_t = select(marginal_std, dim=1, index=idx + 1)

        ts = select(timesteps, dim=1, index=idx) * 1000.
        alpha = select(marginal_alpha, dim=1, index=idx)
        sigma = select(marginal_std, dim=1, index=idx)
        noise_s = self.model_fn(x, ts, text, text_N, sigma, alpha)
        x_t = (sigma_t / sigma_s) * x + (alpha_t * phi_1) * noise_s
        z_out, clip_img_out = split(x_t)
        return z_out, clip_img_out
    
    def dpm_solver_second_update(self, z: Tensor, clip_img: Tensor, idx: Tensor, text: Tensor, text_N0: Tensor, text_N1: Tensor) -> Tensor:
        timesteps = unsqueeze(constant(self.timesteps.detach().cpu().numpy()), axis=0)  # [1, 51]
        marginal_alpha = unsqueeze(constant(self.marginal_alpha.detach().cpu().numpy()), axis=0)  # [1, 51]
        marginal_std = unsqueeze(constant(self.marginal_std.detach().cpu().numpy()), axis=0)  # [1, 51]
        marginal_lambda = unsqueeze(constant(self.marginal_lambda.detach().cpu().numpy()), axis=0)
        marginal_std_inverse = unsqueeze(constant(self.marginal_std_inverse.detach().cpu().numpy()), axis=0)
        marginal_lambda_exp = unsqueeze(constant(self.marginal_lambda_exp.detach().cpu().numpy()), axis=0)
        marginal_alpha_inverse = unsqueeze(constant(self.marginal_alpha_inverse.detach().cpu().numpy()), axis=0)
        marginal_std_inverse = unsqueeze(constant(self.marginal_std_inverse.detach().cpu().numpy()), axis=0)
        _alpha_s1 = unsqueeze(constant(self.alpha_s1.detach().cpu().numpy()), axis=0)
        _alpha_t = unsqueeze(constant(self.alpha_t.detach().cpu().numpy()), axis=0)
        x = combine(z, clip_img)
        r1 = (select(marginal_lambda, dim=1, index=idx + 1) - select(marginal_lambda, dim=1, index=idx)) / (select(marginal_lambda, dim=1, index=idx + 2) - select(marginal_lambda, dim=1, index=idx))
             
        sigma_s = select(marginal_std, dim=1, index=idx)
        sigma_s1 = select(marginal_std_inverse, dim=1, index=idx + 1)
        sigma_t = select(marginal_std, dim=1, index=idx + 2)

        alpha_s1 = select(_alpha_s1, dim=1, index=idx + 1)
        alpha_t = select(_alpha_t, dim=1, index=idx + 2)

        phi_11 = select(marginal_lambda_exp, dim=1, index=idx) / select(marginal_lambda_exp, dim=1, index=idx + 1) - 1.
        phi_1 = select(marginal_lambda_exp, dim=1, index=idx) / select(marginal_lambda_exp, dim=1, index=idx + 2) - 1.

        # 1
        ts = select(timesteps, dim=1, index=idx) * 1000.
        alpha = select(marginal_alpha, dim=1, index=idx)
        sigma = select(marginal_std, dim=1, index=idx)
        noise_s = self.model_fn(x, ts, text, text_N0, sigma, alpha)
        x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * noise_s

        # 2
        ts = select(timesteps, dim=1, index=idx + 1) * 1000.
        alpha = select(marginal_alpha_inverse, dim=1, index=idx + 1)
        sigma = select(marginal_std_inverse, dim=1, index=idx + 1)
        noise_s1 = self.model_fn(x_s1, ts, text, text_N1, sigma, alpha)
        x_t = (sigma_t / sigma_s) * x - (alpha_t * phi_1) * noise_s - (alpha_t * phi_1 * 0.5 / r1) * (noise_s1 - noise_s)
        z_out, clip_img_out = split(x_t)
        return z_out, clip_img_out

    def dpm_solver_third_update(self, z: Tensor, clip_img: Tensor, idx: Tensor, text: Tensor, text_N0: Tensor, text_N1: Tensor, text_N2: Tensor) -> Tensor:
        timesteps = unsqueeze(constant(self.timesteps.detach().cpu().numpy()), axis=0)  # [1, 51]
        marginal_alpha = unsqueeze(constant(self.marginal_alpha.detach().cpu().numpy()), axis=0)  # [1, 51]
        marginal_std = unsqueeze(constant(self.marginal_std.detach().cpu().numpy()), axis=0)  # [1, 51]
        marginal_alpha_inverse = unsqueeze(constant(self.marginal_alpha_inverse.detach().cpu().numpy()), axis=0)
        marginal_std_inverse = unsqueeze(constant(self.marginal_std_inverse.detach().cpu().numpy()), axis=0)
        marginal_lambda = unsqueeze(constant(self.marginal_lambda.detach().cpu().numpy()), axis=0)
        marginal_lambda_exp = unsqueeze(constant(self.marginal_lambda_exp.detach().cpu().numpy()), axis=0)
        _alpha_s1 = unsqueeze(constant(self.alpha_s1.detach().cpu().numpy()), axis=0)
        _alpha_s2 = unsqueeze(constant(self.alpha_s2.detach().cpu().numpy()), axis=0)
        _alpha_t = unsqueeze(constant(self.alpha_t.detach().cpu().numpy()), axis=0)
        x = combine(z, clip_img)
        r1 = (select(marginal_lambda, dim=1, index=idx + 1) - select(marginal_lambda, dim=1, index=idx)) / (select(marginal_lambda, dim=1, index=idx + 3) - select(marginal_lambda, dim=1, index=idx))
        r2 = (select(marginal_lambda, dim=1, index=idx + 2) - select(marginal_lambda, dim=1, index=idx)) / (select(marginal_lambda, dim=1, index=idx + 3) - select(marginal_lambda, dim=1, index=idx))
        
        sigma_s = select(marginal_std, dim=1, index=idx)
        sigma_s1 = select(marginal_std_inverse, dim=1, index=idx + 1)
        sigma_s2 = select(marginal_std_inverse, dim=1, index=idx + 2)
        sigma_t = select(marginal_std, dim=1, index=idx + 3)

        alpha_s1 = select(_alpha_s1, dim=1, index=idx + 1)
        alpha_s2 = select(_alpha_s2, dim=1, index=idx + 2)
        alpha_t = select(_alpha_t, dim=1, index=idx + 3)

        phi_11 = select(marginal_lambda_exp, dim=1, index=idx) / select(marginal_lambda_exp, dim=1, index=idx + 1) - 1.
        phi_12 = select(marginal_lambda_exp, dim=1, index=idx) / select(marginal_lambda_exp, dim=1, index=idx + 2) - 1.
        phi_1 = select(marginal_lambda_exp, dim=1, index=idx) / select(marginal_lambda_exp, dim=1, index=idx + 3) - 1.

        phi_22 = phi_12 / (select(marginal_lambda, dim=1, index=idx + 2) - select(marginal_lambda, dim=1, index=idx)) + 1.
        phi_2 = phi_1 / (select(marginal_lambda, dim=1, index=idx + 3) - select(marginal_lambda, dim=1, index=idx)) + 1.

        # 1
        ts = select(timesteps, dim=1, index=idx) * 1000.
        alpha = select(marginal_alpha, dim=1, index=idx)
        sigma = select(marginal_std, dim=1, index=idx)
        noise_s = self.model_fn(x, ts, text, text_N0, sigma, alpha)
        x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * noise_s

        # 2
        ts = select(timesteps, dim=1, index=idx + 1) * 1000.
        alpha = select(marginal_alpha_inverse, dim=1, index=idx + 1)
        sigma = select(marginal_std_inverse, dim=1, index=idx + 1)
        noise_s1 = self.model_fn(x_s1, ts, text, text_N1, sigma, alpha)
        x_s2 = (sigma_s2 / sigma_s) * x - (alpha_s2 * phi_12) * noise_s + (r2 / r1) * (alpha_s2 * phi_22) * (noise_s1 - noise_s)

        # 3
        ts = select(timesteps, dim=1, index=idx + 2) * 1000.
        alpha = select(marginal_alpha_inverse, dim=1, index=idx + 2)
        sigma = select(marginal_std_inverse, dim=1, index=idx + 2)
        noise_s2 = self.model_fn(x_s2, ts, text, text_N2, sigma, alpha)
        x_t = (sigma_t / sigma_s) * x - (alpha_t * phi_1) * noise_s + (alpha_t * phi_2 / r2) * (noise_s2 - noise_s)
        z_out, clip_img_out = split(x_t)
        return z_out, clip_img_out

    def forward(self, input_ids: Tensor, z: Tensor, clip_img: Tensor, text_N: Tensor) -> Tensor:
        """
        input_ids: [1, 77]
        z: [1, 4, 64, 64]
        clip_img: [1, 1, 512]
        text_N: [50, 1, 77, 64]
        """
        text = self.clip(input_ids)
        text = self.caption_encode(text)
        i = 0
        for order in self.orders:
            idx = constant(np.array(i).reshape([1,]).astype(np.int32))
            if order == 1:
                text_N0 = select(text_N, dim=0, index=i)
                z, clip_img = self.dpm_solver_first_update(z, clip_img, idx, text, text_N0)
            elif order == 2:
                text_N0 = select(text_N, dim=0, index=i)
                text_N1 = select(text_N, dim=0, index=i + 1)
                z, clip_img = self.dpm_solver_second_update(z, clip_img, idx, text, text_N0, text_N1)
            elif order == 3:
                text_N0 = select(text_N, dim=0, index=i)
                text_N1 = select(text_N, dim=0, index=i + 1)
                text_N2 = select(text_N, dim=0, index=i + 2)
                z, clip_img = self.dpm_solver_third_update(z, clip_img, idx, text, text_N0, text_N1, text_N2)
            else:
                raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))
            i += order
        z = self.decoder(z)
        return z
        