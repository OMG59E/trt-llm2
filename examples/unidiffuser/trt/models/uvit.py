import os
import sys
import math
import torch
import numpy as np
import tensorrt as trt
from tensorrt_llm.parameter import Tensor, constant, Parameter
from tensorrt_llm.functional import matmul, softmax, silu, identity, gelu, concat, cos, sin, unsqueeze, interpolate, select, slice
from tensorrt_llm.layers import LayerNorm, Embedding, Linear, Conv2d
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm._utils import np_dtype_to_trt
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pytorch.dpm_solver_pp import NoiseScheduleVP


class Attention(Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype)
        self.proj = Linear(dim, dim, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(shape=[B, L, 3, self.num_heads, -1]).permute([2, 0, 3, 1, 4])
        _, B, H, L, D = qkv.shape
        q = select(qkv, 0, 0)
        k = select(qkv, 0, 1)
        v = select(qkv, 0, 2)
        attn = matmul(q, k.transpose(2, 3)) * self.scale 
        attn = softmax(attn, dim=-1)
        x = matmul(attn, v).transpose(1, 2).view(shape=[B, L, C])
        x = self.proj(x)
        return x
    

class MLP(Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, dtype=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features, dtype=dtype)
        self.act = act_layer
        self.fc2 = Linear(hidden_features, out_features, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
    
class Block(Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, act_layer=gelu, norm_layer=LayerNorm, skip=False, dtype=None) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim, dtype=dtype) if skip else None
        self.norm2 = norm_layer(dim, dtype=dtype)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, dtype=dtype)
        self.norm3 = norm_layer(dim, dtype=dtype)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, dtype=dtype)
        self.skip_linear = Linear(2 * dim, dim, dtype=dtype) if skip else None
        
    def forward(self, x: Tensor, skip=None) -> Tensor:
        if self.skip_linear is not None:
            x = self.skip_linear(concat([x, skip], dim=2))
            x = self.norm1(x)
        x = x + self.attn(x)
        x = self.norm2(x)
        x = x + self.mlp(x)
        x = self.norm3(x)
        return x
    
        
class PatchEmbed(Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768, dtype=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.view(shape=[B, C, H * W]).transpose(1, 2)
        return x


def timestep_embedding(timesteps: Tensor, dim, max_period=10000) -> Tensor :
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)[None]
    # freqs = constant(torch.cat([freqs, freqs], dim=0).detach().cpu().contiguous().numpy())
    freqs = constant(freqs.detach().cpu().contiguous().numpy())  # 1,768
    timesteps_shape = list(timesteps.size())
    timesteps_shape.append(1)  # [bs,1]
    # args = timesteps.view(shape=timesteps_shape).cast(trt.float32) * freqs  # [bs, 768]
    args = matmul(timesteps.view(shape=timesteps_shape).cast(trt.float32), freqs)  # [bs, 768]
    embedding = concat([cos(args), sin(args)], dim=1)  # [bs, 1536]
    if dim % 2:
        embedding = concat([embedding, constant(torch.zeros_like(embedding[:, :1]).detach().cpu().contiguous().numpy())], dim=1)
    return embedding


def interpolate_pos_emb(pos_emb: Tensor, old_shape, new_shape) -> Tensor:
    B, HW, C = pos_emb.shape
    H = old_shape[0]
    W = old_shape[1]
    assert H*W == HW
    pos_emb = pos_emb.view(shape=[B, H, W, C]).permute([0, 3, 1, 2])
    pos_emb = interpolate(pos_emb, size=new_shape, mode="bilinear")
    pos_emb = pos_emb.view(shape=[B, C, HW]).permute([0, 2, 1])
    return pos_emb


def unpatchify(x: Tensor, in_chans) -> Tensor:
    patch_size = int((x.shape[2] // in_chans) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    p1 = p2 = patch_size
    assert h * w == x.shape[1] and patch_size ** 2 * in_chans == x.shape[2]
    B, hw, p1p2C = x.shape
    C = p1p2C // p1 // p2
    x = x.view(shape=[B, h, w, p1, p2, C]).permute([0, 5, 1, 3, 2, 4]).view(shape=[B, C, h*p1, w*p2])
    return x


class UViT(Module):
    def __init__(self, img_size=64, in_chans=4, patch_size=2, embed_dim=1536, depth=30, num_heads=24, 
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=LayerNorm, 
                 mlp_time_embed=False, text_dim=None, num_text_tokens=None, clip_img_dim=None, np_dtype=np.float32):
        super().__init__()
        self.dtype = np_dtype_to_trt(np_dtype)
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, dtype=self.dtype)
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size  # the default img size
        assert self.img_size[0] % patch_size == 0 and self.img_size[1] % patch_size == 0
        self.num_patches = (self.img_size[0] // patch_size) * (self.img_size[1] // patch_size)
        
        self.time_img_embed = ModuleList([
            Linear(embed_dim, 4 * embed_dim, dtype=self.dtype),
            silu(),
            Linear(4 * embed_dim, embed_dim, dtype=self.dtype)]) if mlp_time_embed else identity
        
        self.time_text_embed = ModuleList([
            Linear(embed_dim, 4 * embed_dim, dtype=self.dtype),
            silu(),
            Linear(4 * embed_dim, embed_dim, dtype=self.dtype),
        ]) if mlp_time_embed else identity   
             
        self.text_embed = Linear(text_dim, embed_dim, dtype=self.dtype)
        # self.text_out = Linear(embed_dim, text_dim, dtype=self.dtype)
        
        self.clip_img_embed = Linear(clip_img_dim, embed_dim, dtype=self.dtype)
        self.clip_img_out = Linear(embed_dim, clip_img_dim, dtype=self.dtype)
        
        self.num_text_tokens = num_text_tokens   # 77
        self.num_tokens = 1 + 1 + num_text_tokens + 1 + self.num_patches  # 1105
                
        self.in_blocks = ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer, dtype=self.dtype) for _ in range(depth // 2)])
        
        self.mid_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer, dtype=self.dtype)
        
        self.out_blocks = ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer, skip=True, dtype=self.dtype) for _ in range(depth // 2)])
        
        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = Linear(embed_dim, self.patch_dim, bias=True, dtype=self.dtype)
        
        self.token_embedding = Embedding(2, embed_dim, dtype=self.dtype)
        
        self.pos_embed = Parameter(shape=(1, self.num_tokens, embed_dim), dtype=self.dtype)
        self.pos_embed_token = Parameter(shape=(1, 1, embed_dim), dtype=self.dtype)
        
    def forward(self, img, clip_img, text, t_img, t_text, data_type):
        """
        img shape:        bs, 4, 64, 64
        clip_img shape:   bs, 1, 512
        text shape:       2*bs, 77, 64
        t_img shape:      bs,
        t_text shape:     2*bs,
        data_type shape:  bs,
        """
        _, _, H, W = img.shape

        img = self.patch_embed(img)
        t_img_token = self.time_img_embed(timestep_embedding(t_img, self.embed_dim))
        t_text_token = self.time_text_embed(timestep_embedding(t_text, self.embed_dim))
        t_img_token = unsqueeze(t_img_token, axis=1)  # bs, 1, 1536
        t_text_token = unsqueeze(t_text_token, axis=1)  # 2*bs, 1, 1536

        text = self.text_embed(text)  # 2*bs, 77, 1536
        clip_img = self.clip_img_embed(clip_img)  # bs, 1, 1536
        token_embed = unsqueeze(self.token_embedding(data_type), axis=1)  # bs, 1, 1536

        # x = concat([t_img_token, t_text_token, token_embed, text, clip_img, img], dim=1)  # bs, 1105, 1536
        text0, text1 = text.split([1, 1], dim=0)
        t_text_token0, t_text_token1 = t_text_token.split([1, 1], dim=0)
        x0 = concat([t_img_token, t_text_token0, token_embed, text0, clip_img, img], dim=1)  # bs, 1105, 1536
        x1 = concat([t_img_token, t_text_token1, token_embed, text1, clip_img, img], dim=1)  # bs, 1105, 1536
        x = concat([x0, x1], dim=0)  # 2*bs, 1105, 1536

        num_text_tokens, num_img_tokens = text.size(1), img.size(1)
        pos_embed = concat([slice(self.pos_embed.value, starts=[0, 0, 0], sizes=[1, 2, self.embed_dim]), self.pos_embed_token.value,
                    slice(self.pos_embed.value, starts=[0, 2, 0], sizes=[1, self.num_tokens - 2, self.embed_dim])], dim=1)
        
        if not (H == self.img_size[0] and W == self.img_size[1]):
            # interpolate the positional embedding when the input image is not of the default shape
            pos_embed_others, pos_embed_patches = pos_embed.split([1 + 1 + 1 + num_text_tokens + 1, self.num_patches], dim=1)
            pos_embed_patches = interpolate_pos_emb(pos_embed_patches, 
                                                    (self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size), 
                                                    (H // self.patch_size, W // self.patch_size))
            pos_embed = concat([pos_embed_others, pos_embed_patches], dim=1)

        x = x + pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())
        x = self.norm(x)
        
        bs = x.size(0)
        img_out = slice(x, starts=[0, 1 + 1 + 1 + num_text_tokens + 1, 0], sizes=[bs, num_img_tokens, self.embed_dim])
        clip_img_out = slice(x, starts=[0, 1 + 1 + 1 + num_text_tokens, 0], sizes=[bs, 1, self.embed_dim])

        img_out = self.decoder_pred(img_out)
        img_out = unpatchify(img_out, self.in_chans)
        clip_img_out = self.clip_img_out(clip_img_out)
        return img_out, clip_img_out
    

def split(x: Tensor):
    C, H, W = 4, 64, 64
    z_dim = C * H * W
    z, clip_img = x.split([z_dim, 512], dim=1)
    B = z.shape[0]
    z = z.view(shape=[B, C, H, W])
    L = 1
    D = 512
    clip_img = clip_img.view(shape=[B, L, D])
    return z, clip_img
    
    
def combine(z: Tensor, clip_img: Tensor) -> Tensor:
    B, C, H, W = z.shape
    z = z.view(shape=[B, C*H*W])
    B, L, D = clip_img.shape
    clip_img = clip_img.view(shape=[B, L*D])
    return concat([z, clip_img], dim=1)


class UViTNet(Module):
    def __init__(self) -> None:
        super().__init__()
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
    
    def forward(self, x, timesteps, text, text_N, sigma, alpha):
        bs = timesteps.shape[0]
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


class UViTNet1(Module):
    def __init__(self) -> None:
        super().__init__()
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
        
        self._betas = torch.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, 1000, dtype=torch.float32) ** 2
        self.N = len(self._betas)
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(self._betas.numpy()).float())

        self.sample_steps = 50
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
    
    def forward(self, z: Tensor, clip_img: Tensor, idx: Tensor, text: Tensor, text_N: Tensor) -> Tensor:
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
    

class UViTNet2(Module):
    def __init__(self) -> None:
        super().__init__()
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
        
        self._betas = torch.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, 1000, dtype=torch.float32) ** 2
        self.N = len(self._betas)
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(self._betas.numpy()).float())

        self.sample_steps = 50
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
    
    def forward(self, z: Tensor, clip_img: Tensor, idx: Tensor, text: Tensor, text_N0: Tensor, text_N1: Tensor) -> Tensor:
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
    

class UViTNet3(Module):
    def __init__(self) -> None:
        super().__init__()
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
        
        self._betas = torch.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, 1000, dtype=torch.float32) ** 2
        self.N = len(self._betas)
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(self._betas.numpy()).float())

        self.sample_steps = 50
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
    
    def forward(self, z: Tensor, clip_img: Tensor, idx: Tensor, text: Tensor, text_N0: Tensor, text_N1: Tensor, text_N2: Tensor) -> Tensor:
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
