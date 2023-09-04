import ctypes
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt

from tensorrt_llm._common import default_trtnet
from tensorrt_llm.functional import Tensor, _create_tensor
from tensorrt_llm.module import Module, Parameter

TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'
LAYER_NAME = 'GroupNorm'


def _load_custom_plugin_lib():
    custom_plugin_dir = Path(__file__).parent.absolute()
    plugin_lib = custom_plugin_dir / 'plugin/build/libplugin.so'
    handle = ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    if handle is None:
        raise ImportError('TensorRT-LLM Custom Plugin is unavailable')
    handle.initLibNvInferPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.initLibNvInferPlugins.restype = ctypes.c_bool
    assert handle.initLibNvInferPlugins(None, TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))

_load_custom_plugin_lib()


def group_norm_op(x: Tensor, weight: Tensor, bias: Tensor, epsilon: float, bSwish: int) -> Tensor:
    # Create a plugin instance.
    plugin_creator = trt.get_plugin_registry().get_plugin_creator('GroupNorm', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None

    pfc = trt.PluginFieldCollection([
        trt.PluginField("epsilon", np.array([epsilon], np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("bSwish", np.array([bSwish], np.int32), trt.PluginFieldType.INT32),
    ])
    plugin = plugin_creator.create_plugin("GroupNorm", pfc)
    layer = default_trtnet().add_plugin_v2([x.trt_tensor, weight.trt_tensor, bias.trt_tensor], plugin)
    return _create_tensor(layer.get_output(0), layer)


class GroupNormLayer(Module):
    def __init__(self, num_groups, num_channels, epsilon: float = 1e-6, bSwish: int = 0, dtype=None):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.bSwish = bSwish
        self.weight = Parameter(shape=(self.num_channels, ), dtype=dtype)
        self.bias = Parameter(shape=(self.num_channels, ), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        out = group_norm_op(x, self.weight.value, self.bias.value, self.epsilon, self.bSwish)
        return out