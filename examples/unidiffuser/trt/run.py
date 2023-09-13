import os
import sys
import cv2
import time
import tqdm
import numpy as np
import torch
import einops
from cuda import cudart
from trt_infer import TRTInfer
from transformers import CLIPTokenizer
from plugin import GroupNormLayer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pytorch.dpm_solver_pp import NoiseScheduleVP


prompts = [
    ("A dog adventuring under the sea", 29764),
    ("A dog beneath the sea", 38671),
    ("A rabbit floating in the galaxy", 1234),
    ("A rabbit amidst the stars", 356),
    ("A dog exploring the deep sea", 657),
    ("A rabbit in the depths of the ocean", 109),
    ("A rabbit drifting in the galaxy", 12345),
    ("A dog in the Milky Way", 32562),
    ("A dog wandering beneath the ocean", 11879),
    ("A rabbit in the depths of the sea", 22480),
    ("A rabbit adventuring in the cosmos", 115),
    ("A dog in the universe", 120),
    ("A dog drifting in the depths of the ocean", 110),
    ("A rabbit in the deep blue sea", 1187),
    ("A rabbit leaping between the stars", 11678),
    ("A dog among the interstellar spaces", 32443),
    ("A dog playing in the deep blue sea", 7768),
    ("A rabbit on the ocean's seabed", 5672),
    ("A rabbit floating near a cosmic black hole", 9090),
    ("A dog by the side of a cosmic black hole", 3306),
    ("A cat under the sea", 13244),
    ("A penguin floating in the galaxy", 987),
    ("A dolphin exploring the deep sea", 1234),
    ("A koala in the midst of the stars", 355),
    ("A sea turtle drifting in the cosmos", 8796),
    ("A giraffe riding a submarine through an underwater forest", 22345),
    ("An elephant surfing on a comet in the Milky Way", 33467),
    ("A talking parrot singing opera in a coral reef concert hall", 11332),
    ("A kangaroo boxing with asteroids in the asteroid belt", 5634),
    ("A group of squirrels hosting a tea party on the rings of Saturn", 6645),
    ("A polar bear disco dancing at the bottom of the Mariana Trench", 21),
    ("A wizardly octopus casting spells in a cosmic library", 56),
    ("A group of fireflies lighting up a magical underwater cave", 13078),
    ("A time-traveling rhinoceros exploring ancient constellations", 2311),
    ("A cybernetic hummingbird sipping data nectar in cyberspace", 32455),
    ("A quantum panda meditating on the event horizon of a black hole", 78906),
    ("A steam-powered owl delivering messages on a steampunk asteroid", 7865),
    ("A group of intergalactic jellyfish having a floating tea party", 8796),
    ("A cosmic sloth stargazing from a hammock in the Andromeda Galaxy", 8801),
    ("A team of robotic bees pollinating holographic flowers on Mars", 43765),
]


class UnidiffuserText2ImgTRT(object):
    def __init__(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip = TRTInfer("outputs/clip_float16.trt", use_cuda_graph=False)
        self.uvit1 = TRTInfer("outputs/uvit1_float16.trt", use_cuda_graph=False)
        self.uvit2 = TRTInfer("outputs/uvit2_float16.trt", use_cuda_graph=False)
        self.uvit3 = TRTInfer("outputs/uvit3_float16.trt", use_cuda_graph=False)
        self.decoder = TRTInfer("outputs/decoder_float16.trt", use_cuda_graph=False)
        
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
            
        self.timesteps = torch.linspace(t_T, t_0, self.sample_steps + 1).type(torch.float32).cuda()   # time_uniform
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

        self.text_N = None
        
        self.total_clip_ms = 0
        self.total_uvit_x50_ms = 0
        self.total_decoder_ms = 0

    def dpm_solver_first_update(self, x: torch.Tensor, idx: int):
        x_data_ptr = self.uvit1.inputs[0]["tensor"].data_ptr()
        x_data_size = self.uvit1.inputs[0]["size"]
        text_data_ptr = self.uvit1.inputs[2]["tensor"].data_ptr()
        text_data_size = self.uvit1.inputs[2]["size"]
        text_N_data_ptr = self.uvit1.inputs[3]["tensor"].data_ptr()
        text_N_data_size = self.uvit1.inputs[3]["size"]

        self.uvit1.inputs[1]["tensor"][0] = idx
        cudart.cudaMemcpy(x_data_ptr, x.contiguous().data_ptr(), x_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(text_data_ptr, self.clip.outputs[0]["tensor"].data_ptr(), text_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(text_N_data_ptr, self.text_N[idx].contiguous().data_ptr(), text_N_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)        
        self.uvit1.infer()
        return self.uvit1.outputs[0]["tensor"]

    def dpm_solver_second_update(self, x: torch.Tensor, idx: int):
        x_data_ptr = self.uvit2.inputs[0]["tensor"].data_ptr()
        x_data_size = self.uvit2.inputs[0]["size"]
        text_data_ptr = self.uvit2.inputs[2]["tensor"].data_ptr()
        text_data_size = self.uvit2.inputs[2]["size"]
        text_N0_data_ptr = self.uvit2.inputs[3]["tensor"].data_ptr()
        text_N0_data_size = self.uvit2.inputs[3]["size"]
        text_N1_data_ptr = self.uvit2.inputs[4]["tensor"].data_ptr()
        text_N1_data_size = self.uvit2.inputs[4]["size"]

        self.uvit2.inputs[1]["tensor"][0] = idx
        cudart.cudaMemcpy(x_data_ptr, x.data_ptr(), x_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(text_data_ptr, self.clip.outputs[0]["tensor"].data_ptr(), text_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(text_N0_data_ptr, self.text_N[idx].contiguous().data_ptr(), text_N0_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)     
        cudart.cudaMemcpy(text_N1_data_ptr, self.text_N[idx + 1].contiguous().data_ptr(), text_N1_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)    
        self.uvit2.infer()
        return self.uvit2.outputs[0]["tensor"]

    def dpm_solver_third_update(self, x: torch.Tensor, idx: int):
        x_data_ptr = self.uvit3.inputs[0]["tensor"].data_ptr()
        x_data_size = self.uvit3.inputs[0]["size"]
        text_data_ptr = self.uvit3.inputs[2]["tensor"].data_ptr()
        text_data_size = self.uvit3.inputs[2]["size"]
        text_N0_data_ptr = self.uvit3.inputs[3]["tensor"].data_ptr()
        text_N0_data_size = self.uvit3.inputs[3]["size"]
        text_N1_data_ptr = self.uvit3.inputs[4]["tensor"].data_ptr()
        text_N1_data_size = self.uvit3.inputs[4]["size"]
        text_N2_data_ptr = self.uvit3.inputs[5]["tensor"].data_ptr()
        text_N2_data_size = self.uvit3.inputs[5]["size"]

        self.uvit3.inputs[1]["tensor"][0] = idx
        cudart.cudaMemcpy(x_data_ptr, x.contiguous().data_ptr(), x_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(text_data_ptr, self.clip.outputs[0]["tensor"].data_ptr(), text_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(text_N0_data_ptr, self.text_N[idx].data_ptr(), text_N0_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)     
        cudart.cudaMemcpy(text_N1_data_ptr, self.text_N[idx + 1].data_ptr(), text_N1_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)    
        cudart.cudaMemcpy(text_N2_data_ptr, self.text_N[idx + 2].data_ptr(), text_N2_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice) 
        self.uvit3.infer()
        return self.uvit3.outputs[0]["tensor"]
                     
    def process(self, prompt="a dog under the sea", seed=1234, cumulative_time=False) -> np.ndarray:
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
        if cumulative_time:
            self.total_clip_ms += (time.time() - t_start) * 1000
            
        # step2
        t_start = time.time()
        z_init = torch.randn(self.n_samples, *(self.z_shape), device=self.device)  # 1,4,64,64
        clip_img_init = torch.randn(self.n_samples, 1, self.clip_img_dim, device=self.device)  # 1,1,512
        self.text_N = []
        for n in range(self.sample_steps):
            self.text_N.append(torch.randn_like(self.clip.outputs[0]["tensor"]).type(dtype=torch.float32).contiguous().cuda())

        z = einops.rearrange(z_init, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img_init, 'B L D -> B (L D)')
        x = torch.concat([z, clip_img], dim=-1).contiguous().cuda()  # 1,16896

        i = 0
        for order in self.orders:
            idx = i
            if order == 1:
                x = self.dpm_solver_first_update(x, idx)
            elif order == 2:
                x = self.dpm_solver_second_update(x, idx)
            elif order == 3:
                x = self.dpm_solver_third_update(x, idx)
            else:
                raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))
            i += order
        if cumulative_time:
            self.total_uvit_x50_ms += (time.time() - t_start) * 1000

        # decoder
        t_start = time.time()
        z_data_ptr = self.decoder.inputs[0]["tensor"].data_ptr()
        z_data_size = self.decoder.inputs[0]["size"]
        cudart.cudaMemcpy(z_data_ptr, x.data_ptr(), z_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        self.decoder.infer()
        if cumulative_time:
            self.total_decoder_ms += (time.time() - t_start) * 1000
        return self.decoder.outputs[0]["tensor"].cpu().numpy()
        
        
if __name__ == "__main__":           
    m = UnidiffuserText2ImgTRT()
    total_ms = 0
    if not os.path.exists("images"):
        os.makedirs("images")
    
    warmup = 5
    for _ in range(warmup):
        prompt, seed = prompts[0]
        samples = m.process(prompt=prompt, seed=seed)
        
    for idx, v in enumerate(tqdm.tqdm(prompts)):
        prompt, seed = v
        t_start = time.time()
        samples = m.process(prompt=prompt, seed=seed, cumulative_time=True)
        total_ms += (time.time() - t_start) * 1000
        cv2.imwrite("images/{}.jpg".format(str(idx).zfill(4)), samples[0])
    print("clip: {:.3f}ms".format(m.total_clip_ms / len(prompts)))  
    print("uvit: {:.3f}ms, {:.3f}ms".format(m.total_uvit_x50_ms / len(prompts), m.total_uvit_x50_ms / len(prompts) / 50))
    print("decoder: {:.3f}ms".format(m.total_decoder_ms / len(prompts)))
    print("mean end2end: {:.3f}ms".format(total_ms / len(prompts)))
