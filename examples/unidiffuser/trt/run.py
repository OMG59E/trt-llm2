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
        self.total_clip_ms = 0
        self.total_uvit_x50_ms = 0
        self.total_decoder_ms = 0

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
        # print("clip: {:.3f}ms".format((time.time() - t_start) * 1000))

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
        if cumulative_time:
            self.total_uvit_x50_ms += (time.time() - t_start) * 1000
        # print("all uvit: {:.3f}ms".format((time.time() - t_start) * 1000))

        # decoder
        t_start = time.time()
        z_data_ptr = self.decoder.inputs[0]["tensor"].data_ptr()
        z_data_size = self.decoder.inputs[0]["size"]
        cudart.cudaMemcpy(z_data_ptr, x.contiguous().data_ptr(), z_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        self.decoder.infer()
        if cumulative_time:
            self.total_decoder_ms += (time.time() - t_start) * 1000
        # print("decoder: {:.3f}ms".format((time.time() - t_start) * 1000))
        return self.decoder.outputs[0]["tensor"].cpu().numpy()
        
        
if __name__ == "__main__":           
    m = UnidiffuserText2ImgTRT()
    total_ms = 0
    if not os.path.exists("images"):
        os.makedirs("images")
    warmup = 5
    for idx, v in enumerate(tqdm.tqdm(prompts)):
        prompt, seed = v
        # warmup
        if idx < warmup:
            samples = m.process(prompt=prompt, seed=seed)
            continue
        t_start = time.time()
        samples = m.process(prompt=prompt, seed=seed, cumulative_time=True)
        total_ms += (time.time() - t_start) * 1000
        cv2.imwrite("images/{}.jpg".format(str(idx - warmup).zfill(4)), samples[0])
    print("clip: {:.3f}ms".format(m.total_clip_ms / (len(prompts) - warmup)))  
    print("uvit: {:.3f}ms".format(m.total_uvit_x50_ms / (len(prompts) - warmup)))
    print("decoder: {:.3f}ms".format(m.total_decoder_ms / (len(prompts) - warmup)))
    print("end2end {:.3f}ms".format(total_ms / (len(prompts) - warmup)))
    

