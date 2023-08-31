import time, os
import numpy as np
import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.functional import Tensor
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.network import net_guard

engine_path = "outputs"
precision = "float16"
engine_name = "clip_{}.trt".format(precision)
if not os.path.exists(engine_path):
    os.makedirs(engine_path)
engine_path = os.path.join(engine_path, engine_name)

tensorrt_llm.logger.set_level("verbose")
builder = Builder()
builder_config = builder.create_builder_config(precision=precision)

tensorrt_llm_clip = tensorrt_llm.models.CLIPTextTransformer(
        hidden_size=768, 
        vocab_size=49408, 
        max_position_embeddings=77, 
        num_attention_heads=12, 
        intermediate_size=3072, 
        hidden_act="quick_gelu", 
        num_hidden_layers=12, 
        output_attentions=False, 
        output_hidden_states=False, 
        use_return_dict=True, 
        np_dtype=np.float32)

network = builder.create_network()

tensorrt_llm_clip.embeddings.position_embedding.weight.value = np.ascontiguousarray(np.load("weights/text_model.embeddings.position_embedding.weight.npy"))
tensorrt_llm_clip.embeddings.token_embedding.weight.value = np.ascontiguousarray(np.load("weights/text_model.embeddings.token_embedding.weight.npy"))
tensorrt_llm_clip.final_layer_norm.weight.value = np.ascontiguousarray(np.load("weights/text_model.final_layer_norm.weight.npy"))
tensorrt_llm_clip.final_layer_norm.bias.value = np.ascontiguousarray(np.load("weights/text_model.final_layer_norm.bias.npy"))

num_layers = len(tensorrt_llm_clip.encoder.layers)
for idx in range(num_layers):
    tensorrt_llm_clip.encoder.layers[idx].layer_norm1.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.layer_norm1.weight.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].layer_norm1.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.layer_norm1.bias.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].layer_norm2.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.layer_norm2.weight.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].layer_norm2.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.layer_norm2.bias.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].mlp.fc1.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.mlp.fc1.weight.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].mlp.fc1.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.mlp.fc1.bias.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].mlp.fc2.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.mlp.fc2.weight.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].mlp.fc2.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.mlp.fc2.bias.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].self_attn.q_proj.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.q_proj.weight.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].self_attn.q_proj.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.q_proj.bias.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].self_attn.k_proj.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.k_proj.weight.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].self_attn.k_proj.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.k_proj.bias.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].self_attn.v_proj.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.v_proj.weight.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].self_attn.v_proj.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.v_proj.bias.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].self_attn.out_proj.weight.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.out_proj.weight.npy".format(idx)))
    tensorrt_llm_clip.encoder.layers[idx].self_attn.out_proj.bias.value = np.ascontiguousarray(np.load("weights/text_model.encoder.layers.{}.self_attn.out_proj.bias.npy".format(idx)))    

with net_guard(network):
    network.set_named_parameters(tensorrt_llm_clip.named_parameters())
    input_ids = Tensor(name='input_dis', dtype=trt.int32, shape=[1, 77])
    context = tensorrt_llm_clip(input_ids)
    context.mark_output("context", trt.float32)

engine = builder.build_engine(network, builder_config)

logger.info(f'Serializing engine to {engine_path}...')
tik = time.time()
with open(engine_path, 'wb') as f:
    f.write(bytearray(engine))
tok = time.time()
t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
logger.info(f'Engine serialized. Total time: {t}')