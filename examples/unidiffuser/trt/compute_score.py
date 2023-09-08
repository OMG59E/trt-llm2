import os
import numpy as np
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3


block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx]).to("cuda")


def PD(base_img, new_img):
    inception_feature_ref, _ = fid_score.calculate_activation_statistics([base_img], model, batch_size=1, device="cuda")
    inception_feature, _ = fid_score.calculate_activation_statistics([new_img], model, batch_size=1, device="cuda")
    pd_value = np.linalg.norm(inception_feature - inception_feature_ref)
    return pd_value


torch_dir = "../pytorch/images"
trt_dir = "images"

for idx in range(40):
    base_path = os.path.join(torch_dir, "{}.jpg".format(str(idx).zfill(4)))
    new_path = os.path.join(trt_dir, "{}.jpg".format(str(idx).zfill(4)))
    score = PD(base_path, new_path)
    print("{}.jpg".format(str(idx).zfill(4)), "PD score: {:.3f}".format(score)) 

