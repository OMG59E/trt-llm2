import os
import time
import sys
import numpy as np
import torch
import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.functional import Tensor
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.network import net_guard
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.unidiffuser import Unidiffuser


engine_path = "outputs"
precision = "float16"
engine_name = "unidiffuser_{}.trt".format(precision)
if not os.path.exists(engine_path):
    os.makedirs(engine_path)
engine_path = os.path.join(engine_path, engine_name)


unidiffuser = Unidiffuser()

unidiffuser.caption_encode.encode_prefix.weight.value = np.ascontiguousarray(np.load("weights/encode_prefix.weight.npy"))
unidiffuser.caption_encode.encode_prefix.bias.value = np.ascontiguousarray(np.load("weights/encode_prefix.bias.npy"))

unidiffuser.clip.embeddings.position_embedding.weight.value = np.ascontiguousarray(np.load("weights/text_model.embeddings.position_embedding.weight.npy"))
unidiffuser.clip.embeddings.token_embedding.weight.value = np.ascontiguousarray(np.load("weights/text_model.embeddings.token_embedding.weight.npy"))
unidiffuser.clip.final_layer_norm.weight.value = np.ascontiguousarray(np.load("weights/text_model.final_layer_norm.weight.npy"))
unidiffuser.clip.final_layer_norm.bias.value = np.ascontiguousarray(np.load("weights/text_model.final_layer_norm.bias.npy"))

num_layers = len(unidiffuser.clip.encoder.layers)
for idx in range(num_layers):
    unidiffuser.clip.encoder.layers[idx].layer_norm1.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.layer_norm1.weight.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].layer_norm1.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.layer_norm1.bias.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].layer_norm2.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.layer_norm2.weight.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].layer_norm2.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.layer_norm2.bias.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].mlp.fc1.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.mlp.fc1.weight.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].mlp.fc1.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.mlp.fc1.bias.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].mlp.fc2.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.mlp.fc2.weight.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].mlp.fc2.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.mlp.fc2.bias.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].self_attn.q_proj.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.q_proj.weight.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].self_attn.q_proj.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.q_proj.bias.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].self_attn.k_proj.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.k_proj.weight.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].self_attn.k_proj.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.k_proj.bias.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].self_attn.v_proj.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.v_proj.weight.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].self_attn.v_proj.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.v_proj.bias.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].self_attn.out_proj.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.out_proj.weight.npy".format(idx)))
    unidiffuser.clip.encoder.layers[idx].self_attn.out_proj.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.out_proj.bias.npy".format(idx)))

unidiffuser.nnet.patch_embed.proj.weight.value = np.ascontiguousarray(np.load("weights/patch_embed.proj.weight.npy"))
unidiffuser.nnet.patch_embed.proj.bias.value = np.ascontiguousarray(np.load("weights/patch_embed.proj.bias.npy"))
unidiffuser.nnet.text_embed.weight.value = np.ascontiguousarray(np.load("weights/text_embed.weight.npy"))
unidiffuser.nnet.text_embed.bias.value = np.ascontiguousarray(np.load("weights/text_embed.bias.npy"))
unidiffuser.nnet.clip_img_embed.weight.value = np.ascontiguousarray(np.load("weights/clip_img_embed.weight.npy"))
unidiffuser.nnet.clip_img_embed.bias.value = np.ascontiguousarray(np.load("weights/clip_img_embed.bias.npy"))
unidiffuser.nnet.clip_img_out.weight.value = np.ascontiguousarray(np.load("weights/clip_img_out.weight.npy"))
unidiffuser.nnet.clip_img_out.bias.value = np.ascontiguousarray(np.load("weights/clip_img_out.bias.npy"))
unidiffuser.nnet.norm.weight.value = np.ascontiguousarray(np.load("weights/norm.weight.npy"))
unidiffuser.nnet.norm.bias.value = np.ascontiguousarray(np.load("weights/norm.bias.npy"))
unidiffuser.nnet.decoder_pred.weight.value = np.ascontiguousarray(np.load("weights/decoder_pred.weight.npy"))
unidiffuser.nnet.decoder_pred.bias.value = np.ascontiguousarray(np.load("weights/decoder_pred.bias.npy"))
unidiffuser.nnet.token_embedding.weight.value = np.ascontiguousarray(np.load("weights/token_embedding.weight.npy"))
unidiffuser.nnet.pos_embed.value = np.ascontiguousarray(np.load("weights/pos_embed.npy"))
unidiffuser.nnet.pos_embed_token.value = np.ascontiguousarray(np.load("weights/pos_embed_token.npy"))

num_in_blocks = len(unidiffuser.nnet.in_blocks)
for idx in range(num_in_blocks):
    unidiffuser.nnet.in_blocks[idx].norm2.weight.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.norm2.weight.npy".format(idx)))
    unidiffuser.nnet.in_blocks[idx].norm2.bias.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.norm2.bias.npy".format(idx)))
    unidiffuser.nnet.in_blocks[idx].attn.qkv.weight.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.attn.qkv.weight.npy".format(idx)))
    unidiffuser.nnet.in_blocks[idx].attn.proj.weight.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.attn.proj.weight.npy".format(idx)))
    unidiffuser.nnet.in_blocks[idx].attn.proj.bias.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.attn.proj.bias.npy".format(idx)))
    unidiffuser.nnet.in_blocks[idx].norm3.weight.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.norm3.weight.npy".format(idx)))
    unidiffuser.nnet.in_blocks[idx].norm3.bias.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.norm3.bias.npy".format(idx)))
    unidiffuser.nnet.in_blocks[idx].mlp.fc1.weight.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.mlp.fc1.weight.npy".format(idx)))
    unidiffuser.nnet.in_blocks[idx].mlp.fc1.bias.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.mlp.fc1.bias.npy".format(idx)))
    unidiffuser.nnet.in_blocks[idx].mlp.fc2.weight.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.mlp.fc2.weight.npy".format(idx)))
    unidiffuser.nnet.in_blocks[idx].mlp.fc2.bias.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.mlp.fc2.bias.npy".format(idx)))
    unidiffuser.nnet.mid_block.norm2.weight.value = np.ascontiguousarray(np.load("weights/mid_block.norm2.weight.npy"))
    unidiffuser.nnet.mid_block.norm2.bias.value = np.ascontiguousarray(np.load("weights/mid_block.norm2.bias.npy"))
    unidiffuser.nnet.mid_block.attn.qkv.weight.value = np.ascontiguousarray(np.load("weights/mid_block.attn.qkv.weight.npy"))
    unidiffuser.nnet.mid_block.attn.proj.weight.value = np.ascontiguousarray(np.load("weights/mid_block.attn.proj.weight.npy"))
    unidiffuser.nnet.mid_block.attn.proj.bias.value = np.ascontiguousarray(np.load("weights/mid_block.attn.proj.bias.npy"))
    unidiffuser.nnet.mid_block.norm3.weight.value = np.ascontiguousarray(np.load("weights/mid_block.norm3.weight.npy"))
    unidiffuser.nnet.mid_block.norm3.bias.value = np.ascontiguousarray(np.load("weights/mid_block.norm3.bias.npy"))
    unidiffuser.nnet.mid_block.mlp.fc1.weight.value = np.ascontiguousarray(np.load("weights/mid_block.mlp.fc1.weight.npy"))
    unidiffuser.nnet.mid_block.mlp.fc1.bias.value = np.ascontiguousarray(np.load("weights/mid_block.mlp.fc1.bias.npy"))
    unidiffuser.nnet.mid_block.mlp.fc2.weight.value = np.ascontiguousarray(np.load("weights/mid_block.mlp.fc2.weight.npy"))
    unidiffuser.nnet.mid_block.mlp.fc2.bias.value = np.ascontiguousarray(np.load("weights/mid_block.mlp.fc2.bias.npy"))

num_out_blocks = len(unidiffuser.nnet.out_blocks)
for idx in range(num_out_blocks):
    unidiffuser.nnet.out_blocks[idx].norm1.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.norm1.weight.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].norm1.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.norm1.bias.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].norm2.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.norm2.weight.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].norm2.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.norm2.bias.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].attn.qkv.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.attn.qkv.weight.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].attn.proj.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.attn.proj.weight.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].attn.proj.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.attn.proj.bias.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].norm3.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.norm3.weight.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].norm3.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.norm3.bias.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].mlp.fc1.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.mlp.fc1.weight.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].mlp.fc1.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.mlp.fc1.bias.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].mlp.fc2.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.mlp.fc2.weight.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].mlp.fc2.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.mlp.fc2.bias.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].skip_linear.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.skip_linear.weight.npy".format(idx)))
    unidiffuser.nnet.out_blocks[idx].skip_linear.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.skip_linear.bias.npy".format(idx)))

unidiffuser.decoder.post_quant_conv.weight.value = np.ascontiguousarray(np.load("weights/post_quant_conv.weight.npy"))
unidiffuser.decoder.post_quant_conv.bias.value = np.ascontiguousarray(np.load("weights/post_quant_conv.bias.npy"))
unidiffuser.decoder.conv_in.weight.value = np.ascontiguousarray(np.load("weights/decoder.conv_in.weight.npy"))
unidiffuser.decoder.conv_in.bias.value = np.ascontiguousarray(np.load("weights/decoder.conv_in.bias.npy"))
unidiffuser.decoder.conv_out.weight.value = np.ascontiguousarray(np.load("weights/decoder.conv_out.weight.npy"))
unidiffuser.decoder.conv_out.bias.value = np.ascontiguousarray(np.load("weights/decoder.conv_out.bias.npy"))
unidiffuser.decoder.norm_out.weight.value = np.ascontiguousarray(np.load("weights/decoder.norm_out.weight.npy"))
unidiffuser.decoder.norm_out.bias.value = np.ascontiguousarray(np.load("weights/decoder.norm_out.bias.npy"))
unidiffuser.decoder.mid.block_1.norm1.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.norm1.weight.npy"))
unidiffuser.decoder.mid.block_1.norm1.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.norm1.bias.npy"))
unidiffuser.decoder.mid.block_1.conv1.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.conv1.weight.npy"))
unidiffuser.decoder.mid.block_1.conv1.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.conv1.bias.npy"))
unidiffuser.decoder.mid.block_1.norm2.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.norm2.weight.npy"))
unidiffuser.decoder.mid.block_1.norm2.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.norm2.bias.npy"))
unidiffuser.decoder.mid.block_1.conv2.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.conv2.weight.npy"))
unidiffuser.decoder.mid.block_1.conv2.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_1.conv2.bias.npy"))
unidiffuser.decoder.mid.attn_1.norm.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.norm.weight.npy"))
unidiffuser.decoder.mid.attn_1.norm.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.norm.bias.npy"))
unidiffuser.decoder.mid.attn_1.q.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.q.weight.npy"))
unidiffuser.decoder.mid.attn_1.q.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.q.bias.npy"))
unidiffuser.decoder.mid.attn_1.k.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.k.weight.npy"))
unidiffuser.decoder.mid.attn_1.k.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.k.bias.npy"))
unidiffuser.decoder.mid.attn_1.v.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.v.weight.npy"))
unidiffuser.decoder.mid.attn_1.v.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.v.bias.npy"))
unidiffuser.decoder.mid.attn_1.proj_out.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.proj_out.weight.npy"))
unidiffuser.decoder.mid.attn_1.proj_out.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.attn_1.proj_out.bias.npy"))
unidiffuser.decoder.mid.block_2.norm1.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.norm1.weight.npy"))
unidiffuser.decoder.mid.block_2.norm1.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.norm1.bias.npy"))
unidiffuser.decoder.mid.block_2.conv1.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.conv1.weight.npy"))
unidiffuser.decoder.mid.block_2.conv1.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.conv1.bias.npy"))
unidiffuser.decoder.mid.block_2.norm2.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.norm2.weight.npy"))
unidiffuser.decoder.mid.block_2.norm2.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.norm2.bias.npy"))
unidiffuser.decoder.mid.block_2.conv2.weight.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.conv2.weight.npy"))
unidiffuser.decoder.mid.block_2.conv2.bias.value = np.ascontiguousarray(np.load("weights/decoder.mid.block_2.conv2.bias.npy"))

num_up_block = len(unidiffuser.decoder.up)
for i in range(num_up_block):
    num_block = len(unidiffuser.decoder.up[i].block)
    for j in range(num_block):
        unidiffuser.decoder.up[i].block[j].norm1.weight.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.norm1.weight.npy".format(i, j)))
        unidiffuser.decoder.up[i].block[j].norm1.bias.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.norm1.bias.npy".format(i, j)))
        unidiffuser.decoder.up[i].block[j].conv1.weight.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.conv1.weight.npy".format(i, j)))
        unidiffuser.decoder.up[i].block[j].conv1.bias.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.conv1.bias.npy".format(i, j)))
        unidiffuser.decoder.up[i].block[j].norm2.weight.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.norm2.weight.npy".format(i, j)))
        unidiffuser.decoder.up[i].block[j].norm2.bias.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.norm2.bias.npy".format(i, j)))
        unidiffuser.decoder.up[i].block[j].conv2.weight.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.conv2.weight.npy".format(i, j)))
        unidiffuser.decoder.up[i].block[j].conv2.bias.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.{}.conv2.bias.npy".format(i, j)))

    if i in [0, 1]:
        unidiffuser.decoder.up[i].block[0].nin_shortcut.weight.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.0.nin_shortcut.weight.npy".format(i)))
        unidiffuser.decoder.up[i].block[0].nin_shortcut.bias.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.block.0.nin_shortcut.bias.npy".format(i)))

    if i in [1, 2, 3]:
        unidiffuser.decoder.up[i].upsample.conv.weight.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.upsample.conv.weight.npy".format(i)))
        unidiffuser.decoder.up[i].upsample.conv.bias.value = np.ascontiguousarray(np.load("weights/decoder.up.{}.upsample.conv.bias.npy".format(i))) 


tensorrt_llm.logger.set_level("verbose")
builder = Builder()
builder_config = builder.create_builder_config(precision=precision)
network = builder.create_network()   
with net_guard(network):
    network.set_named_parameters(unidiffuser.named_parameters())
    input_ids = Tensor(name='input_ids', dtype=trt.int32, shape=[1, 77])
    z = Tensor(name='z', dtype=trt.float32, shape=[1, 4, 64, 64])
    clip_img = Tensor(name='clip_img', dtype=trt.float32, shape=[1, 1, 512])
    text_N = Tensor(name='text_N', dtype=trt.float32, shape=[50, 1, 77, 64])
    x_out = unidiffuser(input_ids, z, clip_img, text_N)
    x_out.mark_output("x_out", trt.uint8)

engine = builder.build_engine(network, builder_config)

logger.info(f'Serializing engine to {engine_path}...')
tik = time.time()
with open(engine_path, 'wb') as f:
    f.write(bytearray(engine))
tok = time.time()
t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
logger.info(f'Engine serialized. Total time: {t}')
