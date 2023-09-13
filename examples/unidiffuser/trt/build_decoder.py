import time, os
import numpy as np
import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.functional import Tensor
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.network import net_guard
from tensorrt_llm.module import Module
from tensorrt_llm.layers import Linear
from models.decoder import Decoder


engine_path = "outputs"
precision = "float16"
engine_name = "decoder_{}.trt".format(precision)
if not os.path.exists(engine_path):
    os.makedirs(engine_path)
engine_path = os.path.join(engine_path, engine_name)

tensorrt_llm.logger.set_level("verbose")
builder = Builder()
builder_config = builder.create_builder_config(precision=precision)

tensorrt_llm_decoder = Decoder(
    ch=128, 
    out_ch=3, 
    ch_mult=(1, 2, 4, 4), 
    num_res_blocks=2, 
    embed_dim=4, 
    attn_resolutions=[], 
    resamp_with_conv=True, 
    in_channels=3, 
    resolution=256, 
    z_channels=4, 
    give_pre_end=False, 
    tanh_out=False, 
    use_linear_attn=False, 
    scale_factor=0.18215,
    np_dtype=np.float32)

tensorrt_llm_decoder.post_quant_conv.weight.value = np.ascontiguousarray(np.load("weights/post_quant_conv.weight.npy"))
tensorrt_llm_decoder.post_quant_conv.bias.value = np.ascontiguousarray(np.load("weights/post_quant_conv.bias.npy"))
tensorrt_llm_decoder.conv_in.weight.value = np.ascontiguousarray(np.load("weights/decoder.conv_in.weight.npy"))
tensorrt_llm_decoder.conv_in.bias.value = np.ascontiguousarray(np.load("weights/decoder.conv_in.bias.npy"))
tensorrt_llm_decoder.conv_out.weight.value = np.ascontiguousarray(np.load("weights/decoder.conv_out.weight.npy"))
tensorrt_llm_decoder.conv_out.bias.value = np.ascontiguousarray(np.load("weights/decoder.conv_out.bias.npy"))
tensorrt_llm_decoder.norm_out.weight.value = np.ascontiguousarray(np.load("weights/decoder.norm_out.weight.npy"))
tensorrt_llm_decoder.norm_out.bias.value = np.ascontiguousarray(np.load("weights/decoder.norm_out.bias.npy"))
tensorrt_llm_decoder.mid.block_1.norm1.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.norm1.weight.npy"))
tensorrt_llm_decoder.mid.block_1.norm1.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.norm1.bias.npy"))
tensorrt_llm_decoder.mid.block_1.conv1.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.conv1.weight.npy"))
tensorrt_llm_decoder.mid.block_1.conv1.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.conv1.bias.npy"))
tensorrt_llm_decoder.mid.block_1.norm2.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.norm2.weight.npy"))
tensorrt_llm_decoder.mid.block_1.norm2.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.norm2.bias.npy"))
tensorrt_llm_decoder.mid.block_1.conv2.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.conv2.weight.npy"))
tensorrt_llm_decoder.mid.block_1.conv2.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.conv2.bias.npy"))
tensorrt_llm_decoder.mid.attn_1.norm.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.norm.weight.npy"))
tensorrt_llm_decoder.mid.attn_1.norm.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.norm.bias.npy"))
tensorrt_llm_decoder.mid.attn_1.q.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.q.weight.npy"))
tensorrt_llm_decoder.mid.attn_1.q.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.q.bias.npy"))
tensorrt_llm_decoder.mid.attn_1.k.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.k.weight.npy"))
tensorrt_llm_decoder.mid.attn_1.k.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.k.bias.npy"))
tensorrt_llm_decoder.mid.attn_1.v.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.v.weight.npy"))
tensorrt_llm_decoder.mid.attn_1.v.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.v.bias.npy"))
tensorrt_llm_decoder.mid.attn_1.proj_out.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.proj_out.weight.npy"))
tensorrt_llm_decoder.mid.attn_1.proj_out.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.proj_out.bias.npy"))
tensorrt_llm_decoder.mid.block_2.norm1.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.norm1.weight.npy"))
tensorrt_llm_decoder.mid.block_2.norm1.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.norm1.bias.npy"))
tensorrt_llm_decoder.mid.block_2.conv1.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.conv1.weight.npy"))
tensorrt_llm_decoder.mid.block_2.conv1.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.conv1.bias.npy"))
tensorrt_llm_decoder.mid.block_2.norm2.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.norm2.weight.npy"))
tensorrt_llm_decoder.mid.block_2.norm2.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.norm2.bias.npy"))
tensorrt_llm_decoder.mid.block_2.conv2.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.conv2.weight.npy"))
tensorrt_llm_decoder.mid.block_2.conv2.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.conv2.bias.npy"))

num_up_block = len(tensorrt_llm_decoder.up)
for i in range(num_up_block):
    num_block = len(tensorrt_llm_decoder.up[i].block)
    for j in range(num_block):
        tensorrt_llm_decoder.up[i].block[j].norm1.weight.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.norm1.weight.npy".format(i, j)))
        tensorrt_llm_decoder.up[i].block[j].norm1.bias.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.norm1.bias.npy".format(i, j)))
        tensorrt_llm_decoder.up[i].block[j].conv1.weight.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.conv1.weight.npy".format(i, j)))
        tensorrt_llm_decoder.up[i].block[j].conv1.bias.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.conv1.bias.npy".format(i, j)))
        tensorrt_llm_decoder.up[i].block[j].norm2.weight.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.norm2.weight.npy".format(i, j)))
        tensorrt_llm_decoder.up[i].block[j].norm2.bias.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.norm2.bias.npy".format(i, j)))
        tensorrt_llm_decoder.up[i].block[j].conv2.weight.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.conv2.weight.npy".format(i, j)))
        tensorrt_llm_decoder.up[i].block[j].conv2.bias.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.conv2.bias.npy".format(i, j)))

    if i in [0, 1]:
        tensorrt_llm_decoder.up[i].block[0].nin_shortcut.weight.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.0.nin_shortcut.weight.npy".format(i)))
        tensorrt_llm_decoder.up[i].block[0].nin_shortcut.bias.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.0.nin_shortcut.bias.npy".format(i)))

    if i in [1, 2, 3]:
        tensorrt_llm_decoder.up[i].upsample.conv.weight.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.upsample.conv.weight.npy".format(i)))
        tensorrt_llm_decoder.up[i].upsample.conv.bias.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.upsample.conv.bias.npy".format(i))) 

network = builder.create_network()   

with net_guard(network):
    network.set_named_parameters(tensorrt_llm_decoder.named_parameters())
    x = Tensor(name='x', dtype=trt.float32, shape=[1, 4, 64, 64])
    z = tensorrt_llm_decoder(x)
    z.mark_output("z", trt.uint8)

engine = builder.build_engine(network, builder_config)

logger.info(f'Serializing engine to {engine_path}...')
tik = time.time()
with open(engine_path, 'wb') as f:
    f.write(bytearray(engine))
tok = time.time()
t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
logger.info(f'Engine serialized. Total time: {t}')
