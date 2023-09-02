import math
import torch
import numpy as np
import tensorrt as trt
from tensorrt_llm.parameter import Tensor, constant, Parameter
from tensorrt_llm.functional import matmul, softmax, silu, clip, gelu, concat, cos, sin, unsqueeze, interpolate, select, gather, tanh
from tensorrt_llm.layers import LayerNorm, Embedding, Linear, Conv2d, GroupNorm
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm._utils import np_dtype_to_trt


class Upsample(Module):
    def __init__(self, in_channels, with_conv, dtype=None):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x
    
    
class ResnetBlock(Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, temb_channels=512, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = GroupNorm(32, in_channels, dtype=dtype)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dtype=dtype)
        if temb_channels > 0:
            self.temb_proj = Linear(temb_channels, out_channels, dtype=dtype)
        self.norm2 = GroupNorm(32, out_channels, dtype=dtype)
        
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dtype=dtype)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dtype=dtype)
            else:
                self.nin_shortcut = Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dtype=dtype)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + unsqueeze(unsqueeze(self.temb_proj(silu(temb)), axis=2), axis=3)

        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h
    

class AttnBlock(Module):
    def __init__(self, in_channels, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm(32, in_channels, dtype=dtype)
        self.q = Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dtype=dtype)
        self.k = Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dtype=dtype)
        self.v = Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dtype=dtype)
        self.proj_out = Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dtype=dtype)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.view(shape=[b, c, h*w])
        q = q.permute([0, 2, 1])    # b, hw, c
        k = k.view(shape=[b, c, h*w])  # b, c, hw
        w_ = matmul(q, k)     # b, hw, hw    w[b,i,j] = sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = softmax(w_, dim=2)

        # attend to values
        v = v.view(shape=[b, c, h*w])
        w_ = w_.permute([0, 2, 1])   # b,hw,hw (first hw of k, second of q)
        h_ = matmul(v, w_)         # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.view(shape=[b, c, h, w])
        h_ = self.proj_out(h_)
        return x + h_
    
    
class Decoder(Module):
    def __init__(self, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks=2, embed_dim=4, shape=[4, 64, 64], clip_img_dim=512,
                 attn_resolutions=[], resamp_with_conv=True, in_channels=3, resolution=256, z_channels=4, 
                 give_pre_end=False, tanh_out=False, use_linear_attn=False, scale_factor=0.18215, np_dtype=np.float32) -> None:
        super().__init__()
        self.dtype = np_dtype_to_trt(np_dtype)
        self.embed_dim = embed_dim
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.scale_factor = scale_factor
        self.clip_img_dim = clip_img_dim
        self.in_shape = shape
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))
        
        self.post_quant_conv = Conv2d(embed_dim, z_channels, kernel_size=(1, 1), dtype=self.dtype)

        self.conv_in = Conv2d(z_channels, block_in, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dtype=self.dtype)

        self.mid = Module()        
        self.mid._modules["block_1"]= ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dtype=self.dtype)
        self.mid._modules["attn_1"] = AttnBlock(block_in, dtype=self.dtype)
        self.mid._modules["block_2"] = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dtype=self.dtype)
        
        # upsampling
        self.up = list()
        for i_level in reversed(range(self.num_resolutions)):
            block = list()
            attn = list()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dtype=self.dtype))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, dtype=self.dtype))
            block = ModuleList(block)
            attn = ModuleList(attn)
            
            up = Module()
            up._modules["block"] = block
            up._modules["attn"] = attn
            if i_level != 0:
                up._modules["upsample"] = Upsample(block_in, resamp_with_conv, dtype=self.dtype)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order
        self.up = ModuleList(self.up)

        # end
        self.norm_out = GroupNorm(32, block_in, dtype=self.dtype)
        self.conv_out = Conv2d(block_in, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dtype=self.dtype)

    def forward(self, z: Tensor) -> Tensor:
        # C, H, W = self.in_shape
        # z_dim = C * H * W
        # z, _ = x.split([z_dim, self.clip_img_dim], dim=1)
        # z = z.view([-1, C, H, W])

        z = (1. / self.scale_factor) * z
        z = self.post_quant_conv(z)
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = silu(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = tanh(h)
        h = clip(127.5 * (h + 1.), alpha=0, beta=255).permute([0, 2, 3, 1])
        r, g, b = h.split([1, 1, 1], dim=3)
        h = concat([b, g, r], dim=3).cast(trt.uint8)  # BGR2RGB
        return h
        