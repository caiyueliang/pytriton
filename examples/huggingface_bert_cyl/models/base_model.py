from collections import OrderedDict
from copy import deepcopy
# from cuda import cudart
# from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
import numpy as np
import onnx
from onnx import shape_inference
import onnx_graphsurgeon as gs
import os
from polygraphy.backend.onnx.loader import fold_constants
import shutil
import tempfile
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from loguru import logger


class BaseModel(object):
    def __init__(self,
        device='cuda:0',
        verbose=True,
        fp16=False,
        min_batch_size=1,
        max_batch_size=128,
        opset_version=17
    ):
        self.name = self.__class__.__name__
        self.device = device
        self.verbose = verbose

        self.fp16 = fp16
        self.opset_version = opset_version

        self.min_batch = min_batch_size
        self.max_batch = max_batch_size

        # self.min_image_shape = 256   # min image resolution: 256x256
        # self.max_image_shape = 1024  # max image resolution: 1024x1024
        # self.min_latent_shape = self.min_image_shape // 8
        # self.max_latent_shape = self.max_image_shape // 8

        # self.text_maxlen = text_maxlen
        # self.embedding_dim = embedding_dim
        # self.extra_output_names = []

    def get_opset_version(self):
        return self.opset_version
    
    def get_model(self):
        raise NotImplementedError()

    def get_input_names(self):
        raise NotImplementedError()

    def get_output_names(self):
        raise NotImplementedError()

    def get_sample_input(self, text):
        raise NotImplementedError()

    def get_dynamic_axes(self):
        return None
    
    def get_input_profile(self, opt_batch_size=None, opt_seq_len=None):
        raise NotImplementedError()

    def get_shape_dict(self, batch_size, image_height, image_width):
        raise NotImplementedError()

    # def optimize(self, onnx_graph):
    #     opt = Optimizer(onnx_graph, verbose=self.verbose)
    #     opt.info(self.name + ': original')
    #     opt.cleanup()
    #     opt.info(self.name + ': cleanup')
    #     opt.fold_constants()
    #     opt.info(self.name + ': fold constants')
    #     opt.infer_shapes()
    #     opt.info(self.name + ': shape inference')
    #     onnx_opt_graph = opt.cleanup(return_onnx=True)
    #     opt.info(self.name + ': finished')
    #     return onnx_opt_graph

    # def check_dims(self, batch_size, image_height, image_width):
    #     # logger.info(f"{self.min_batch} < {batch_size} < {self.max_batch}")
    #     assert batch_size >= self.min_batch and batch_size <= self.max_batch
    #     assert image_height % 8 == 0 or image_width % 8 == 0
    #     latent_height = image_height // 8
    #     latent_width = image_width // 8
    #     assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
    #     assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
    #     return (latent_height, latent_width)

    # def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
    #     min_batch = batch_size if static_batch else self.min_batch
    #     max_batch = batch_size if static_batch else self.max_batch
    #     latent_height = image_height // 8
    #     latent_width = image_width // 8
    #     min_image_height = image_height if static_shape else self.min_image_shape
    #     max_image_height = image_height if static_shape else self.max_image_shape
    #     min_image_width = image_width if static_shape else self.min_image_shape
    #     max_image_width = image_width if static_shape else self.max_image_shape
    #     min_latent_height = latent_height if static_shape else self.min_latent_shape
    #     max_latent_height = latent_height if static_shape else self.max_latent_shape
    #     min_latent_width = latent_width if static_shape else self.min_latent_shape
    #     max_latent_width = latent_width if static_shape else self.max_latent_shape
    #     return (min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width)

