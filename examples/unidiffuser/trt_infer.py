#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: trt_infer.py 
@time: 2023/07/16
@author: xingwg 
@contact: xwg031459@163.com
@software: PyCharm 
"""
import os
import sys
import torch
import numpy as np
import tensorrt as trt
from cuda import cudart


numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64
}


def check_cuda_err(err):
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


class TRTInfer(object):
    """Implements inference for the Model TensorRT engine.
    """
    def __init__(self, engine_path):
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.buffers = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            tensor = torch.zeros(list(shape), dtype=numpy_to_torch_dtype_dict[trt.nptype(dtype)]).cuda()
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "tensor": tensor,
                "size": size,
            }
            self.buffers.append(tensor.data_ptr())
            self.context.set_tensor_address(name, tensor.data_ptr())
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.buffers) > 0

        _, self.stream = cudart.cudaStreamCreate()
        # do inference before CUDA graph capture
        self.context.execute_async_v3(self.stream)
        # capture cuda graph
        cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        self.context.execute_async_v3(self.stream)
        e, self.graph = cudart.cudaStreamEndCapture(self.stream)
        e, self.instance = cudart.cudaGraphInstantiate(self.graph, 0)

    def infer(self):
        cudart.cudaGraphLaunch(self.instance, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

