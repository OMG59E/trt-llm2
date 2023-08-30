
import os
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel

weight_dir = "weights"
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

for name, param in transformer.named_parameters():
    if "weight" not in name and "bias" not in name:
        continue
    w = param.detach().cpu().numpy()
    print(name, w.shape, w.dtype)
    np.save("{}/{}.npy".format(weight_dir, name), w)
    


