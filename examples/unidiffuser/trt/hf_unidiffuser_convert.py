
import os
import sys
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pytorch.libs.caption_decoder import CaptionDecoder

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
