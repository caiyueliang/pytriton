#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import OrderedDict
from typing import List
from copy import copy
import numpy as np
import onnx
import onnx_graphsurgeon as gs
# import os
# import math
# from PIL import Image
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
# import random
# from scipy import integrate
import tensorrt as trt
import torch
# import requests
# from io import BytesIO
# from cuda import cudart
import pycuda.driver as cuda
import pycuda.autoinit
from enum import Enum, auto
from loguru import logger

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

# def CUASSERT(cuda_ret):
#     err = cuda_ret[0]
#     if err != cudart.cudaError_t.cudaSuccess:
#          raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
#     if len(cuda_ret) > 1:
#         return cuda_ret[1]
#     return None

class PIPELINE_TYPE(Enum):
    TXT2IMG = auto()
    IMG2IMG = auto()
    INPAINT = auto()
    SD_XL_BASE = auto()
    SD_XL_REFINER = auto()

    def is_txt2img(self):
        return self == self.TXT2IMG

    def is_img2img(self):
        return self == self.IMG2IMG

    def is_inpaint(self):
        return self == self.INPAINT

    def is_sd_xl_base(self):
        return self == self.SD_XL_BASE

    def is_sd_xl_refiner(self):
        return self == self.SD_XL_REFINER

    def is_sd_xl(self):
        return self.is_sd_xl_base() or self.is_sd_xl_refiner()

class Engine():
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None # cuda graph

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, onnx_path, onnx_refit_path):
        def convert_int64(arr):
            # TODO: smarter conversion
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        logger.info(f"Refitting TensorRT engine with {onnx_refit_path} weights")
        refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

        # Construct mapping from weight names in refit model -> original model
        name_map = {}
        for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
            refit_node = refit_nodes[n]
            assert node.op == refit_node.op
            # Constant nodes in ONNX do not have inputs but have a constant output
            if node.op == "Constant":
                name_map[refit_node.outputs[0].name] = node.outputs[0].name
            # Handle scale and bias weights
            elif node.op == "Conv":
                if node.inputs[1].__class__ == gs.Constant:
                    name_map[refit_node.name+"_TRTKERNEL"] = node.name+"_TRTKERNEL"
                if node.inputs[2].__class__ == gs.Constant:
                    name_map[refit_node.name+"_TRTBIAS"] = node.name+"_TRTBIAS"
            # For all other nodes: find node inputs that are initializers (gs.Constant)
            else:
                for i, inp in enumerate(node.inputs):
                    if inp.__class__ == gs.Constant:
                        name_map[refit_node.inputs[i].name] = inp.name
        def map_name(name):
            if name in name_map:
                return name_map[name]
            return name

        # Construct refit dictionary
        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            # for speciailized roles, use a unique name in the map:
            if role == trt.WeightsRole.KERNEL:
                name = layer_name+"_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name+"_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None


        for n in refit_nodes:
            # Constant nodes in ONNX do not have inputs but have a constant output
            if n.op == "Constant":
                name = map_name(n.outputs[0].name)
                print(f"Add Constant {name}\n")
                add_to_map(refit_dict, name, n.outputs[0].values)

            # Handle scale and bias weights
            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = map_name(n.name+"_TRTKERNEL")
                    add_to_map(refit_dict, name, n.inputs[1].values)

                if n.inputs[2].__class__ == gs.Constant:
                    name = map_name(n.name+"_TRTBIAS")
                    add_to_map(refit_dict, name, n.inputs[2].values)

            # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
            else:
                for inp in n.inputs:
                    name = map_name(inp.name)
                    if inp.__class__ == gs.Constant:
                        add_to_map(refit_dict, name, inp.values)

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name+"_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name+"_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
            else:
                logger.warning(f"[W] No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            logger.error("Failed to refit!")
            exit(0)

    def build(self, onnx_path, fp16, input_profile=None, enable_refit=False, enable_preview=False, enable_all_tactics=False, timing_cache=None, update_output_names=None):
        logger.info(f"[build] TensorRT engine for {onnx_path} -> {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs['tactic_sources'] = []

        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        if update_output_names:
            logger.info(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)

        engine = engine_from_network(
            network,
            config=CreateConfig(fp16=fp16,
                refittable=enable_refit,
                profiles=[p],
                load_timing_cache=timing_cache,
                **config_kwargs
            ),
            save_timing_cache=timing_cache
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        logger.info(f"[load] TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    # def allocate_buffers(self, shape_dict=None, device='cuda'):
    #     for idx in range(self.engine.num_io_tensors):
    #         binding = self.engine[idx]
    #         # logger.info(f"[allocate_buffers] binding: {binding}")
    #         if shape_dict and binding in shape_dict:
    #             shape = shape_dict[binding]
    #         else:
    #             shape = self.engine.get_binding_shape(binding)

    #         dtype = trt.nptype(self.engine.get_binding_dtype(binding))
    #         if self.engine.binding_is_input(binding):
    #             self.context.set_binding_shape(idx, shape)
    #         # logger.info(f"[allocate_buffers] shape: {shape}")

    #         tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
    #         # logger.info(f"[allocate_buffers] tensor: {tensor.size()}")
    #         self.tensors[binding] = tensor
    def allocate_buffers(self, is_explicit_batch=False, shape_dict=None, device=False):
        inputs = []
        outputs = []
        bindings = []

        class HostDeviceMem(object):
            def __init__(self, host_mem, device_mem):
                self.host = host_mem
                self.device = device_mem

            def __str__(self):
                return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

            def __repr__(self):
                return self.__str__()
        
        # ====================================================================================
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            logger.info(f"[allocate_buffers] binding: {binding}")
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            logger.info(f"[allocate_buffers] shape: {shape}")

            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            logger.info(f"[allocate_buffers] dtype: {dtype}")

            size = trt.volume(shape)
            # logger.warning(f"[allocate_buffers] self.engine.max_batch_size: {self.engine.max_batch_size}")
            logger.warning(f"[allocate_buffers] size: {size}")
            # dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # logger.warning(f"[allocate_buffers] dtype: {dtype}")

            # if self.engine.binding_is_input(binding):
            #     self.context.set_binding_shape(idx, shape)
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            logger.warning(f"[allocate_buffers] host_mem: {host_mem}")
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):   # Determine whether a binding is an input binding.
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                
        # # ====================================================================================
        # input_shape = shape_dict["input_ids"]

        # for binding in self.engine:
        #     logger.warning(f"[allocate_buffers] binding: {binding}")
        #     dims = self.engine.get_binding_shape(binding)
        #     logger.warning(f"[allocate_buffers] dims: {dims}")
            
        #     # if dims[-1] == -1:
        #     #     assert(input_shape is not None)
        #     #     dims[-2], dims[-1] = input_shape

        #     size = trt.volume(dims) * self.engine.max_batch_size    # The maximum batch size which can be used for inference.
        #     logger.warning(f"[allocate_buffers] self.engine.max_batch_size: {self.engine.max_batch_size}")
        #     logger.warning(f"[allocate_buffers] size: {size}")
        #     dtype = trt.nptype(self.engine.get_binding_dtype(binding))
        #     logger.warning(f"[allocate_buffers] dtype: {dtype}")

        #     # Allocate host and device buffers
        #     host_mem = cuda.pagelocked_empty(size, dtype)
        #     device_mem = cuda.mem_alloc(host_mem.nbytes)
        #     # Append the device buffer to device bindings.
        #     bindings.append(int(device_mem))
        #     if self.engine.binding_is_input(binding):   # Determine whether a binding is an input binding.
        #         inputs.append(HostDeviceMem(host_mem, device_mem))
        #     else:
        #         outputs.append(HostDeviceMem(host_mem, device_mem))
        # # ====================================================================================
                
        logger.warning(f"[inputs] {inputs}, [outputs] {outputs}, [bindings] {bindings}")
        return inputs, outputs, bindings

    # def infer(self, feed_dict, stream, use_cuda_graph=False):
    #     # logger.warning(f"[feed_dict] {feed_dict}")
    #     for name, buf in feed_dict.items():
    #         # logger.info(f"[infer] name: {name}, buf: {buf.size()}")
    #         self.tensors[name].copy_(buf)

    #     for name, tensor in self.tensors.items():
    #         self.context.set_tensor_address(name, tensor.data_ptr())

    #     if use_cuda_graph:
    #         if self.cuda_graph_instance is not None:
    #             CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
    #             CUASSERT(cudart.cudaStreamSynchronize(stream))
    #         else:
    #             # do inference before CUDA graph capture
    #             noerror = self.context.execute_async_v3(stream)
    #             if not noerror:
    #                 raise ValueError(f"ERROR: inference failed.")
    #             # capture cuda graph
    #             CUASSERT(cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
    #             self.context.execute_async_v3(stream)
    #             self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream))
    #             self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
    #     else:
    #         noerror = self.context.execute_async_v3(stream)
    #         if not noerror:
    #             raise ValueError(f"ERROR: inference failed.")

    #     return self.tensors
    def infer(self, feed_dict, stream, use_cuda_graph=False):
        assert(self.engine is not None)  
        
        # input_image, input_shape = preprocess_image(imagepath)
        bs = feed_dict['input_ids'].size()[0]
        seq_len = feed_dict['input_ids'].size()[1]
        shape_dict = {
            'input_ids': (bs, seq_len), 
            'attention_mask': (bs, seq_len),
            'output_0': (bs, seq_len, 768),
            'output_1': (bs, seq_len)
        }
        
        segment_inputs, segment_outputs, segment_bindings = self.allocate_buffers(True, shape_dict=shape_dict)
        logger.info(f"[infer] segment_inputs: {segment_inputs}")
        logger.info(f"[infer] segment_outputs: {segment_outputs}")
        logger.info(f"[infer] segment_bindings: {segment_bindings}")

        stream = cuda.Stream()    
        with self.engine.create_execution_context() as context:
            context.active_optimization_profile = 0             # 增加部分
            origin_inputshape = context.get_binding_shape(0)
            logger.warning(f"[infer] origin_inputshape: {origin_inputshape}")

            # 增加部分
            if (origin_inputshape[-1]==-1):
                origin_inputshape[-2], origin_inputshape[-1]=(input_shape)
                context.set_binding_shape(0,(origin_inputshape))
            input_img_array = np.array([input_image] * batch_size)
            img = torch.from_numpy(input_img_array).float().numpy()
            logger.warning(f"[infer] img: {img}")
            segment_inputs[0].host = img

            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in segment_inputs]    # 数据异步复制: host -> device
            stream.synchronize()
            context.execute_async(bindings=segment_bindings, stream_handle=stream.handle)       # 异步执行推理。
            stream.synchronize()
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in segment_outputs]   # C数据异步复制: device -> host
            stream.synchronize()

            results = np.array(segment_outputs[0].host).reshape(batch_size, input_shape[0],input_shape[1]) 

        return results.transpose(1,2,0)
