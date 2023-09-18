import torch
import os
import einops
import numpy as np
import libs.autoencoder
import libs.clip
from libs.uvit_multi_post_ln_v1 import UViT
from libs.caption_decoder import CaptionDecoder
from transformers import CLIPTokenizer, CLIPTextModel


if not os.path.exists("onnx"):
    os.makedirs("onnx")

class CLIP(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transformer = CLIPTextModel.from_pretrained("models/clip-vit-large-patch14")
        self.caption_decoder = CaptionDecoder(device="cpu", pretrained_path="models/caption_decoder.pth", hidden_dim=64)
    
    def forward(self, tokens):
        outputs = self.transformer(input_ids=tokens)
        contexts = outputs.last_hidden_state
        contexts_low_dim = self.caption_decoder.encode_prefix(contexts)
        return contexts_low_dim


n_samples = 1
prompt = "an elephant under the sea"
prompts = [prompt] * n_samples
tokenizer = CLIPTokenizer.from_pretrained("models/clip-vit-large-patch14")
batch_encoding = tokenizer(prompts, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
tokens = batch_encoding["input_ids"]
clip_net = CLIP()
clip_net.eval()
torch.onnx.export(clip_net, tokens, "onnx/CLIP.onnx", opset_version=18, input_names=["input_ids"], output_names=["contexts"])


class Decoder(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.autoencoder = libs.autoencoder.get_model(pretrained_path='models/autoencoder_kl.pth')
    
    def forward(self, x):
        samples = 0.5 * (self.autoencoder.decode(x) + 1.) 
        samples = samples.mul(255.).clamp_(0., 255.).permute((0, 2, 3, 1))[:, :, :, [2, 1, 0]].type(torch.uint8)
        return samples
    
x = torch.randn((1, 4, 64, 64), dtype=torch.float32)
decoder_net = Decoder()
decoder_net.eval()
torch.onnx.export(decoder_net, x, "onnx/decoder.onnx", opset_version=18, input_names=["z"], output_names=["samples"])


def split(x):
    C, H, W = 4, 64, 64
    z_dim = C * H * W
    z, clip_img = x.split([z_dim, 512], dim=1)
    z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
    clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=512)
    return z, clip_img
    
    
def combine(z, clip_img):
    z = einops.rearrange(z, 'B C H W -> B (C H W)')
    clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
    return torch.concat([z, clip_img], dim=-1)


class UViTNet(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
            "use_checkpoint": False}
    
        self.nnet = UViT(**nnet_dict)
        self.nnet.load_state_dict(torch.load("models/uvit_v1.pth", map_location='cpu'))

    def forward(self, x, timesteps, text, text_N, sigma, alpha):
        z, clip_img = split(x)
        t_text = torch.zeros(timesteps.size(0), dtype=torch.int)
        data_type = torch.zeros_like(t_text, dtype=torch.int) + 1
        _z = torch.cat([z, z], dim=0)
        _clip_img = torch.cat([clip_img, clip_img], dim=0)
        _text = torch.cat([text, text_N], dim=0)
        _t_img = torch.cat([timesteps, timesteps], dim=0)
        _t_text = torch.cat([t_text, torch.ones_like(timesteps) * 1000], dim=0)
        _data_type = torch.cat([data_type, data_type], dim=0)
        z_out, clip_img_out, _ = self.nnet(_z, _clip_img, text=_text, t_img=_t_img, t_text=_t_text, data_type=_data_type)
        x_out = combine(z_out, clip_img_out)
        noise = x_out[0] + 7. * (x_out[0] - x_out[1])
        x_out = (x - sigma * noise) / alpha
        return x_out
    

uvit_net = UViTNet()
uvit_net.eval()
x = torch.randn((1, 16896), dtype=torch.float32) 
timesteps = torch.ones((1,), dtype=torch.float32) 
text = torch.ones((1, 77, 64), dtype=torch.float32)
text_N = torch.ones((1, 77, 64), dtype=torch.float32)
sigma = torch.ones((1,), dtype=torch.float32)
alpha = torch.ones((1,), dtype=torch.float32)
torch.onnx.export(uvit_net, (x, timesteps, text, text_N, sigma, alpha), "onnx/uvit.onnx", opset_version=18, 
                  input_names=["x", "timesteps", "text", "text_N", "sigma", "alpha"], output_names=["x_out"])
