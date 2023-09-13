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
from models.uvit import UViTNet1, UViTNet2, UViTNet3


engine_path = "outputs"
precision = "float16"
engine1_name = "uvit1_{}.trt".format(precision)
engine2_name = "uvit2_{}.trt".format(precision)
engine3_name = "uvit3_{}.trt".format(precision)
if not os.path.exists(engine_path):
    os.makedirs(engine_path)
engine1_path = os.path.join(engine_path, engine1_name)
engine2_path = os.path.join(engine_path, engine2_name)
engine3_path = os.path.join(engine_path, engine3_name)


def init_weight(tensorrt_llm_uvit):
    tensorrt_llm_uvit.nnet.patch_embed.proj.weight.value = np.ascontiguousarray(np.load("weights/patch_embed.proj.weight.npy"))
    tensorrt_llm_uvit.nnet.patch_embed.proj.bias.value = np.ascontiguousarray(np.load("weights/patch_embed.proj.bias.npy"))
    tensorrt_llm_uvit.nnet.text_embed.weight.value = np.ascontiguousarray(np.load("weights/text_embed.weight.npy"))
    tensorrt_llm_uvit.nnet.text_embed.bias.value = np.ascontiguousarray(np.load("weights/text_embed.bias.npy"))
    # tensorrt_llm_uvit.nnet.text_out.weight.value = np.ascontiguousarray(np.load("weights/text_out.weight.npy"))
    # tensorrt_llm_uvit.nnet.text_out.bias.value = np.ascontiguousarray(np.load("weights/text_out.bias.npy"))
    tensorrt_llm_uvit.nnet.clip_img_embed.weight.value = np.ascontiguousarray(np.load("weights/clip_img_embed.weight.npy"))
    tensorrt_llm_uvit.nnet.clip_img_embed.bias.value = np.ascontiguousarray(np.load("weights/clip_img_embed.bias.npy"))
    tensorrt_llm_uvit.nnet.clip_img_out.weight.value = np.ascontiguousarray(np.load("weights/clip_img_out.weight.npy"))
    tensorrt_llm_uvit.nnet.clip_img_out.bias.value = np.ascontiguousarray(np.load("weights/clip_img_out.bias.npy"))
    tensorrt_llm_uvit.nnet.norm.weight.value = np.ascontiguousarray(np.load("weights/norm.weight.npy"))
    tensorrt_llm_uvit.nnet.norm.bias.value = np.ascontiguousarray(np.load("weights/norm.bias.npy"))
    tensorrt_llm_uvit.nnet.decoder_pred.weight.value = np.ascontiguousarray(np.load("weights/decoder_pred.weight.npy"))
    tensorrt_llm_uvit.nnet.decoder_pred.bias.value = np.ascontiguousarray(np.load("weights/decoder_pred.bias.npy"))
    tensorrt_llm_uvit.nnet.token_embedding.weight.value = np.ascontiguousarray(np.load("weights/token_embedding.weight.npy"))
    tensorrt_llm_uvit.nnet.pos_embed.value = np.ascontiguousarray(np.load("weights/pos_embed.npy"))
    tensorrt_llm_uvit.nnet.pos_embed_token.value = np.ascontiguousarray(np.load("weights/pos_embed_token.npy"))

    num_in_blocks = len(tensorrt_llm_uvit.nnet.in_blocks)
    for idx in range(num_in_blocks):
        tensorrt_llm_uvit.nnet.in_blocks[idx].norm2.weight.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.norm2.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.in_blocks[idx].norm2.bias.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.norm2.bias.npy".format(idx)))
        tensorrt_llm_uvit.nnet.in_blocks[idx].attn.qkv.weight.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.attn.qkv.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.in_blocks[idx].attn.proj.weight.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.attn.proj.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.in_blocks[idx].attn.proj.bias.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.attn.proj.bias.npy".format(idx)))
        tensorrt_llm_uvit.nnet.in_blocks[idx].norm3.weight.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.norm3.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.in_blocks[idx].norm3.bias.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.norm3.bias.npy".format(idx)))
        tensorrt_llm_uvit.nnet.in_blocks[idx].mlp.fc1.weight.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.mlp.fc1.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.in_blocks[idx].mlp.fc1.bias.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.mlp.fc1.bias.npy".format(idx)))
        tensorrt_llm_uvit.nnet.in_blocks[idx].mlp.fc2.weight.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.mlp.fc2.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.in_blocks[idx].mlp.fc2.bias.value = np.ascontiguousarray(np.load("weights/in_blocks.{}.mlp.fc2.bias.npy".format(idx)))

    tensorrt_llm_uvit.nnet.mid_block.norm2.weight.value = np.ascontiguousarray(np.load("weights/mid_block.norm2.weight.npy"))
    tensorrt_llm_uvit.nnet.mid_block.norm2.bias.value = np.ascontiguousarray(np.load("weights/mid_block.norm2.bias.npy"))
    tensorrt_llm_uvit.nnet.mid_block.attn.qkv.weight.value = np.ascontiguousarray(np.load("weights/mid_block.attn.qkv.weight.npy"))
    tensorrt_llm_uvit.nnet.mid_block.attn.proj.weight.value = np.ascontiguousarray(np.load("weights/mid_block.attn.proj.weight.npy"))
    tensorrt_llm_uvit.nnet.mid_block.attn.proj.bias.value = np.ascontiguousarray(np.load("weights/mid_block.attn.proj.bias.npy"))
    tensorrt_llm_uvit.nnet.mid_block.norm3.weight.value = np.ascontiguousarray(np.load("weights/mid_block.norm3.weight.npy"))
    tensorrt_llm_uvit.nnet.mid_block.norm3.bias.value = np.ascontiguousarray(np.load("weights/mid_block.norm3.bias.npy"))
    tensorrt_llm_uvit.nnet.mid_block.mlp.fc1.weight.value = np.ascontiguousarray(np.load("weights/mid_block.mlp.fc1.weight.npy"))
    tensorrt_llm_uvit.nnet.mid_block.mlp.fc1.bias.value = np.ascontiguousarray(np.load("weights/mid_block.mlp.fc1.bias.npy"))
    tensorrt_llm_uvit.nnet.mid_block.mlp.fc2.weight.value = np.ascontiguousarray(np.load("weights/mid_block.mlp.fc2.weight.npy"))
    tensorrt_llm_uvit.nnet.mid_block.mlp.fc2.bias.value = np.ascontiguousarray(np.load("weights/mid_block.mlp.fc2.bias.npy"))

    num_out_blocks = len(tensorrt_llm_uvit.nnet.out_blocks)
    for idx in range(num_out_blocks):
        tensorrt_llm_uvit.nnet.out_blocks[idx].norm1.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.norm1.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].norm1.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.norm1.bias.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].norm2.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.norm2.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].norm2.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.norm2.bias.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].attn.qkv.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.attn.qkv.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].attn.proj.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.attn.proj.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].attn.proj.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.attn.proj.bias.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].norm3.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.norm3.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].norm3.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.norm3.bias.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].mlp.fc1.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.mlp.fc1.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].mlp.fc1.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.mlp.fc1.bias.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].mlp.fc2.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.mlp.fc2.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].mlp.fc2.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.mlp.fc2.bias.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].skip_linear.weight.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.skip_linear.weight.npy".format(idx)))
        tensorrt_llm_uvit.nnet.out_blocks[idx].skip_linear.bias.value = np.ascontiguousarray(np.load("weights/out_blocks.{}.skip_linear.bias.npy".format(idx)))
    
    return tensorrt_llm_uvit


def build_engine1(tensorrt_llm_uvit):
    tensorrt_llm.logger.set_level("verbose")
    builder = Builder()
    builder_config = builder.create_builder_config(precision=precision)
    network = builder.create_network()   

    with net_guard(network):
        network.set_named_parameters(tensorrt_llm_uvit.named_parameters())
        z = Tensor(name='z', dtype=trt.float32, shape=[1, 4, 64, 64])
        clip_img = Tensor(name='clip_img', dtype=trt.float32, shape=[1, 1, 512])
        idx = Tensor(name='idx', dtype=trt.int32, shape=[1,])
        text = Tensor(name='text', dtype=trt.float32, shape=[1, 77, 64])
        text_N = Tensor(name='text_N', dtype=trt.float32, shape=[1, 77, 64])
        x_out, clip_img_out = tensorrt_llm_uvit(z, clip_img, idx, text, text_N)
        x_out.mark_output("x_out", trt.float32)
        clip_img_out.mark_output("clip_img_out", trt.float32)

    engine = builder.build_engine(network, builder_config)
    return engine


def build_engine2(tensorrt_llm_uvit):
    tensorrt_llm.logger.set_level("verbose")
    builder = Builder()
    builder_config = builder.create_builder_config(precision=precision)
    network = builder.create_network()   

    with net_guard(network):
        network.set_named_parameters(tensorrt_llm_uvit.named_parameters())
        z = Tensor(name='z', dtype=trt.float32, shape=[1, 4, 64, 64])
        clip_img = Tensor(name='clip_img', dtype=trt.float32, shape=[1, 1, 512])
        idx = Tensor(name='idx', dtype=trt.int32, shape=[1,])
        text = Tensor(name='text', dtype=trt.float32, shape=[1, 77, 64])
        text_N0 = Tensor(name='text_N0', dtype=trt.float32, shape=[1, 77, 64])
        text_N1 = Tensor(name='text_N1', dtype=trt.float32, shape=[1, 77, 64])
        x_out, clip_img_out = tensorrt_llm_uvit(z, clip_img, idx, text, text_N0, text_N1)
        x_out.mark_output("x_out", trt.float32)
        clip_img_out.mark_output("clip_img_out", trt.float32)

    engine = builder.build_engine(network, builder_config)
    return engine


def build_engine3(tensorrt_llm_uvit):
    tensorrt_llm.logger.set_level("verbose")
    builder = Builder()
    builder_config = builder.create_builder_config(precision=precision)
    network = builder.create_network()   

    with net_guard(network):
        network.set_named_parameters(tensorrt_llm_uvit.named_parameters())
        z = Tensor(name='z', dtype=trt.float32, shape=[1, 4, 64, 64])
        clip_img = Tensor(name='clip_img', dtype=trt.float32, shape=[1, 1, 512])
        idx = Tensor(name='idx', dtype=trt.int32, shape=[1,])
        text = Tensor(name='text', dtype=trt.float32, shape=[1, 77, 64])
        text_N0 = Tensor(name='text_N0', dtype=trt.float32, shape=[1, 77, 64])
        text_N1 = Tensor(name='text_N1', dtype=trt.float32, shape=[1, 77, 64])
        text_N2 = Tensor(name='text_N2', dtype=trt.float32, shape=[1, 77, 64])
        x_out, clip_img_out = tensorrt_llm_uvit(z, clip_img, idx, text, text_N0, text_N1, text_N2)
        x_out.mark_output("x_out", trt.float32)
        clip_img_out.mark_output("clip_img_out", trt.float32)

    engine = builder.build_engine(network, builder_config)
    return engine


def save_engine(engine, filepath):
    logger.info(f'Serializing engine to {filepath}...')
    tik = time.time()
    with open(filepath, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')
    

tensorrt_llm_uvit1 = UViTNet1()
tensorrt_llm_uvit2 = UViTNet2()
tensorrt_llm_uvit3 = UViTNet3()
tensorrt_llm_uvit1 = init_weight(tensorrt_llm_uvit1)
tensorrt_llm_uvit2 = init_weight(tensorrt_llm_uvit2)
tensorrt_llm_uvit3 = init_weight(tensorrt_llm_uvit3)

engine1 = build_engine1(tensorrt_llm_uvit1)
engine2 = build_engine2(tensorrt_llm_uvit2)
engine3 = build_engine3(tensorrt_llm_uvit3)

save_engine(engine1, engine1_path)
save_engine(engine2, engine2_path)
save_engine(engine3, engine3_path)