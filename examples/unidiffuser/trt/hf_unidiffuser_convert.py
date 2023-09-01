
import os
import sys
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pytorch.libs.caption_decoder import CaptionDecoder
from pytorch.libs.uvit_multi_post_ln_v1 import UViT

    
weight_dir = "weights"
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)
    
# clip
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
for name, param in transformer.named_parameters():
    if "weight" not in name and "bias" not in name:
        continue
    w = param.detach().cpu().numpy()
    print(name, w.shape, w.dtype)
    np.save("{}/{}.npy".format(weight_dir, name), w)
    
# caption decoder
caption_decoder = CaptionDecoder(device="cpu", pretrained_path="../pytorch/models/caption_decoder.pth", hidden_dim=64)
for name, param in caption_decoder.caption_model.named_parameters():
    if "weight" not in name and "bias" not in name:
        continue
    w = param.detach().cpu().numpy()
    print(name, w.shape, w.dtype)
    np.save("{}/{}.npy".format(weight_dir, name), w)


# uvit
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
nnet = UViT(**nnet_dict)
nnet.load_state_dict(torch.load("../pytorch/models/uvit_v1.pth", map_location='cpu'))
nnet.eval()
for name, param in nnet.named_parameters():
    if "pos_embed" in name:
        w = param.detach().cpu().numpy()
        print(name, w.shape, w.dtype)
        np.save("{}/{}.npy".format(weight_dir, name), w)
    if "weight" not in name and "bias" not in name:
        continue
    w = param.detach().cpu().numpy()
    print(name, w.shape, w.dtype)
    np.save("{}/{}.npy".format(weight_dir, name), w)