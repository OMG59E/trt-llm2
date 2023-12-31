import os
import sys
import cv2
import time
import tqdm
import numpy as np
import torch
from cuda import cudart
from trt_infer import TRTInfer
from transformers import CLIPTokenizer
from plugin import GroupNormLayer


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
        self.tokenizer = CLIPTokenizer.from_pretrained("../pytorch/models/clip-vit-large-patch14")
        self.unidiffuser = TRTInfer("outputs/unidiffuser_float16.trt", use_cuda_graph=True)
        
        self.device = 'cuda'
        self.n_samples = 1
        self.sample_steps = 50
        self.z_shape = (4, 64, 64)
        self.clip_img_dim = 512    
        self.text_N = torch.zeros([self.sample_steps, 1, 77, 64], dtype=torch.float32).cuda()
    
    def process(self, prompt="a dog under the sea", seed=1234, cumulative_time=False) -> np.ndarray:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        z = torch.randn(self.n_samples, *(self.z_shape), device=self.device).contiguous()  # 1,4,64,64
        clip_img = torch.randn(self.n_samples, 1, self.clip_img_dim, device=self.device).contiguous()  # 1,1,512
        for n in range(self.sample_steps):
            self.text_N[n] = torch.randn([1, 77, 64]).type(dtype=torch.float32).cuda()

        batch_encoding = self.tokenizer([prompt], truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        input_ids = batch_encoding["input_ids"].type(torch.int32).contiguous().cuda()

        input_ids_data_ptr = self.unidiffuser.inputs[0]["tensor"].data_ptr()
        input_ids_data_size = self.unidiffuser.inputs[0]["size"]
        z_data_ptr = self.unidiffuser.inputs[1]["tensor"].data_ptr()
        z_data_size = self.unidiffuser.inputs[1]["size"]
        clip_img_data_ptr = self.unidiffuser.inputs[2]["tensor"].data_ptr()
        clip_img_data_size = self.unidiffuser.inputs[2]["size"]
        text_N_data_ptr = self.unidiffuser.inputs[3]["tensor"].data_ptr()
        text_N_data_size = self.unidiffuser.inputs[3]["size"]

        cudart.cudaMemcpy(input_ids_data_ptr, input_ids.data_ptr(), input_ids_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(z_data_ptr, z.data_ptr(), z_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(clip_img_data_ptr, clip_img.data_ptr(), clip_img_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(text_N_data_ptr, self.text_N.contiguous().data_ptr(), text_N_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        self.unidiffuser.infer()
        return self.unidiffuser.outputs[0]["tensor"].cpu().numpy()
        
        
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
    print("mean end2end: {:.3f}ms".format(total_ms / len(prompts)))
