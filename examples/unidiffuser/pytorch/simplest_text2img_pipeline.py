import os
import cv2
import time
import tqdm
import einops
import numpy as np
import torch
import libs.autoencoder
import libs.clip
from libs.caption_decoder import CaptionDecoder
from transformers import CLIPTokenizer, CLIPTextModel
from dpm_solver_pp import NoiseScheduleVP
from libs.uvit_multi_post_ln_v1 import UViT


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


class UnidiffuserText2ImgTorch(object):
    def __init__(self) -> None:
        self.device = 'cuda'
        self.n_samples = 1
        self.z_shape = (4, 64, 64)
        self.clip_img_dim = 512
        self.clip_text_dim = 768
        self.text_dim = 64  # reduce dimension
        self.sample_steps = 50
        self.scale = 7.
        self.t2i_cfg_mode = "true_uncond"

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.transformer.cuda().eval()
        self.caption_decoder = CaptionDecoder(device=self.device, pretrained_path="models/caption_decoder.pth", hidden_dim=64)
        
        nnet_dict = {
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
            "use_checkpoint": False
        }

        self.nnet = UViT(**nnet_dict)
        self.nnet.load_state_dict(torch.load("models/uvit_v1.pth", map_location='cpu'))
        self.nnet.to(self.device)
        self.nnet.eval()

        self.autoencoder = libs.autoencoder.get_model(pretrained_path='models/autoencoder_kl.pth')
        self.autoencoder.to(self.device)
        
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

    def split(self, x):
        C, H, W = self.z_shape
        z_dim = C * H * W
        z, clip_img = x.split([z_dim, self.clip_img_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=self.clip_img_dim)
        return z, clip_img
    
    def combine(self, z, clip_img):
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        return torch.concat([z, clip_img], dim=-1)

    def t2i_nnet(self, x, timesteps, text, text_N):  # text is the low dimension version of the text clip embedding
        z, clip_img = self.split(x)
        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=self.device)
        data_type = torch.zeros_like(t_text, device=self.device, dtype=torch.int) + 1
        _z = torch.cat([z, z], dim=0)
        _clip_img = torch.cat([clip_img, clip_img], dim=0)
        _text = torch.cat([text, text_N], dim=0)
        _t_img = torch.cat([timesteps, timesteps], dim=0)
        _t_text = torch.cat([t_text, torch.ones_like(timesteps) * self.N], dim=0)
        _data_type = torch.cat([data_type, data_type], dim=0)
        z_out, clip_img_out, _ = self.nnet(_z, _clip_img, text=_text, t_img=_t_img, t_text=_t_text, data_type=_data_type)
        x_out = self.combine(z_out, clip_img_out)
        return x_out[0] + self.scale * (x_out[0] - x_out[1])

    def model_fn(self, x, t, contexts):
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        text_N = torch.randn_like(contexts)
        noise = self.t2i_nnet(x, t * self.N, contexts, text_N)
        x0 = (x - sigma_t * noise) / alpha_t
        return x0

    def dpm_solver_first_update(self, x, s, t, contexts):
        lambda_s, lambda_t = self.noise_schedule.marginal_lambda(s), self.noise_schedule.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = self.noise_schedule.marginal_log_mean_coeff(s), self.noise_schedule.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = self.noise_schedule.marginal_std(s), self.noise_schedule.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        phi_1 = (torch.exp(-h) - 1.) / (-1.)
        noise_s = self.model_fn(x, s, contexts)
        x_t = (sigma_t / sigma_s) * x + (alpha_t * phi_1) * noise_s
        return x_t

    def dpm_solver_second_update(self, x, s, t, r1, contexts):
        lambda_s, lambda_t = self.noise_schedule.marginal_lambda(s), self.noise_schedule.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = self.noise_schedule.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = self.noise_schedule.marginal_log_mean_coeff(s), self.noise_schedule.marginal_log_mean_coeff(s1), self.noise_schedule.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = self.noise_schedule.marginal_std(s), self.noise_schedule.marginal_std(s1), self.noise_schedule.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)
        phi_11 = torch.expm1(-r1 * h)
        phi_1 = torch.expm1(-h)
        noise_s = self.model_fn(x, s, contexts)
        x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * noise_s
        noise_s1 = self.model_fn(x_s1, s1, contexts)
        x_t = (sigma_t / sigma_s) * x - (alpha_t * phi_1) * noise_s - (0.5 / r1) * (alpha_t * phi_1) * (noise_s1 - noise_s)
        return x_t

    def dpm_solver_third_update(self, x, s, t, r1, r2, contexts):
        lambda_s, lambda_t = self.noise_schedule.marginal_lambda(s), self.noise_schedule.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = self.noise_schedule.inverse_lambda(lambda_s1)
        s2 = self.noise_schedule.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = self.noise_schedule.marginal_log_mean_coeff(s), self.noise_schedule.marginal_log_mean_coeff(s1), self.noise_schedule.marginal_log_mean_coeff(s2), self.noise_schedule.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = self.noise_schedule.marginal_std(s), self.noise_schedule.marginal_std(s1), self.noise_schedule.marginal_std(s2), self.noise_schedule.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)
        phi_11 = torch.expm1(-r1 * h)
        phi_12 = torch.expm1(-r2 * h)
        phi_1 = torch.expm1(-h)
        phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
        phi_2 = phi_1 / h + 1.
        noise_s = self.model_fn(x, s, contexts)
        x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * noise_s
        noise_s1 = self.model_fn(x_s1, s1, contexts)
        x_s2 = (sigma_s2 / sigma_s) * x - (alpha_s2 * phi_12) * noise_s + r2 / r1 * (alpha_s2 * phi_22) * (noise_s1 - noise_s)
        noise_s2 = self.model_fn(x_s2, s2, contexts)
        x_t = (sigma_t / sigma_s) * x - (alpha_t * phi_1) * noise_s + (1. / r2) * (alpha_t * phi_2) * (noise_s2 - noise_s)
        return x_t
        
    def process(self, prompt, seed=1234, cumulative_time=False):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # step1 clip
        t_start = time.time()
        batch_encoding = self.tokenizer([prompt], truncation=True, max_length=77, return_length=True, 
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)
        contexts = outputs.last_hidden_state
        # the low dimensional version of the contexts, which is the input to the nnet
        contexts_low_dim = self.caption_decoder.encode_prefix(contexts)
        if cumulative_time:
            self.total_clip_ms += (time.time() - t_start) * 1000
        # print("clip: {:.3f}ms".format((time.time() - t_start) * 1000))

        # step2 uvit
        t_start = time.time()
        z_init = torch.randn(self.n_samples, *(self.z_shape), device=self.device)
        clip_img_init = torch.randn(self.n_samples, 1, self.clip_img_dim, device=self.device)
        z = einops.rearrange(z_init, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img_init, 'B L D -> B (L D)')
        x = torch.concat([z, clip_img], dim=-1)

        with torch.no_grad():
            with torch.autocast(device_type=self.device):
                i = 0
                for order in self.orders:
                    vec_s, vec_t = torch.ones((x.shape[0],)).to(self.device) * self.timesteps[i], torch.ones((x.shape[0],)).to(self.device) * self.timesteps[i + order]
                    h = self.noise_schedule.marginal_lambda(self.timesteps[i + order]) - self.noise_schedule.marginal_lambda(self.timesteps[i])
                    r1 = None if order <= 1 else (self.noise_schedule.marginal_lambda(self.timesteps[i + 1]) - self.noise_schedule.marginal_lambda(self.timesteps[i])) / h
                    r2 = None if order <= 2 else (self.noise_schedule.marginal_lambda(self.timesteps[i + 2]) - self.noise_schedule.marginal_lambda(self.timesteps[i])) / h
                    if order == 1:
                        x = self.dpm_solver_first_update(x, vec_s, vec_t, contexts_low_dim)
                    elif order == 2:
                        x = self.dpm_solver_second_update(x, vec_s, vec_t, r1, contexts_low_dim)
                    elif order == 3:
                        x = self.dpm_solver_third_update(x, vec_s, vec_t, r1, r2, contexts_low_dim)
                    else:
                        raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))
                    i += order
                z, _ = self.split(x)
            if cumulative_time:
                self.total_uvit_x50_ms += (time.time() - t_start) * 1000
            # print("uvit: {:.3f}ms".format((time.time() - t_start) * 1000))

            # step3 decoder
            t_start = time.time()
            samples = 0.5 * (self.autoencoder.decode(z) + 1.) 
            samples = samples.mul(255.).clamp_(0., 255.).permute((0, 2, 3, 1))[:, :, :, [2, 1, 0]].cpu().numpy().astype(np.uint8)
            if cumulative_time:
                self.total_decoder_ms += (time.time() - t_start) * 1000
            # print("decoder: {:.3f}ms".format((time.time() - t_start) * 1000))
            return samples


if __name__ == "__main__":  
    m = UnidiffuserText2ImgTorch()
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

