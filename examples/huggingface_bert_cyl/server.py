#!/usr/bin/env python3
# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple classifier example based on Hugging Face JAX BERT model."""

from loguru import logger
from typing import Any, List

import os
import argparse
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from pytriton.decorators import batch, first_value, group_by_values
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig


model_folder_embedding = os.environ["MODEL_PATH_EMBEDDING"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16

# @batch
# def _infer_fn_embedding(**inputs: np.ndarray):
#     # logger.info(f"[_infer_fn_embedding] inputs: {inputs}")
#     (sequence_batch,) = inputs.values()
#     logger.info(f"[_infer_fn_embedding] sequence_batch: {len(sequence_batch)}")
#     # need to convert dtype=object to bytes first
#     # end decode unicode bytes
#     sequence_batch = np.char.decode(sequence_batch.astype("bytes"), "utf-8")
#     # logger.info(f"[_infer_fn_embedding] sequence_batch: {sequence_batch}")

#     last_hidden_states = []
#     for sequence_item in sequence_batch:
#         # tokenized_sequence = tokenizer(sequence_item.item(), return_tensors="jax")
#         # results = model(**tokenized_sequence)
#         # logger.info(f"[_infer_fn_embedding] sequence_item: {sequence_item.item()}")
#         inputs = tokenizer(
#                         sequence_item.item(), 
#                         padding=True,
#                         truncation=True,
#                         max_length=512,
#                         return_tensors="pt"
#                     )
#         inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
#         results = model(**inputs_on_device, return_dict=True)
        
#         last_hidden_states.append(results.last_hidden_state.cpu().detach().numpy())
    
#         # logger.info(f"[_infer_fn_embedding] last_hidden_states: {last_hidden_states}")

#     last_hidden_states = np.array(last_hidden_states, dtype=np.float32)
#     logger.info(f"[_infer_fn_embedding] last_hidden_states: {last_hidden_states.shape}")
#     l = [last_hidden_states]
#     logger.info(f"[_infer_fn_embedding] len: {len(l)}; {l}")
#     return l

# @batch
# def _infer_fn_embedding(**inputs: np.ndarray):
#     # logger.info(f"[_infer_fn_embedding] inputs: {inputs}")
#     (sequence_batch,) = inputs.values()
#     logger.info(f"[_infer_fn_embedding] sequence_batch: {len(sequence_batch)}")
#     # need to convert dtype=object to bytes first
#     # end decode unicode bytes
#     sequence_batch = np.char.decode(sequence_batch.astype("bytes"), "utf-8")
#     sequence_batch = [s[0] for s in sequence_batch]
#     logger.info(f"[_infer_fn_embedding] sequence_batch: {sequence_batch}")

#     inputs = tokenizer(
#         sequence_batch, 
#         padding=True,
#         truncation=True,
#         max_length=512,
#         return_tensors="pt"
#     )
#     inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
#     results = model(**inputs_on_device, return_dict=True)
#     # logger.info(f"[_infer_fn_embedding] results: {results}")

#     last_hidden_states = results.last_hidden_state.unsqueeze(1).cpu().detach().numpy()
#     last_hidden_states = np.array(last_hidden_states, dtype=np.float32)
#     logger.info(f"[_infer_fn_embedding] last_hidden_states shape: {last_hidden_states.shape}")
#     return [last_hidden_states]


class _InferFuncWrapper:
    def __init__(self, model: Any, tokenizer: Any, device: str):
        self._model = model
        self._device = device
        self._tokenizer = tokenizer

    @batch
    @group_by_values("max_length", "pooler")
    @first_value("max_length", "pooler")
    def __call__(self, sequence: np.ndarray, max_length: np.int32, pooler: np.bytes_):
        sequence_batch = sequence
        logger.info(f"[_infer_fn_embedding] sequence_batch: {sequence_batch}")
        logger.info(f"[_infer_fn_embedding] max_length: {max_length}")
        pooler = pooler.decode("utf-8")
        logger.info(f"[_infer_fn_embedding] pooler: {pooler}")
        
        # need to convert dtype=object to bytes first
        # end decode unicode bytes
        sequence_batch = np.char.decode(sequence_batch.astype("bytes"), "utf-8")
        # logger.info(f"[_infer_fn_embedding] sequence_batch: {sequence_batch}")
        # import itertools
        # sequence_batch = list(itertools.chain(*sequence_batch))
        sequence_batch = [s[0] for s in sequence_batch]
        logger.info(f"[_infer_fn_embedding] sequence_batch: {sequence_batch}")

        inputs = self._tokenizer(
            sequence_batch, 
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
        results = self._model(**inputs_on_device, return_dict=True)
        # logger.info(f"[_infer_fn_embedding] results: {results}")

        last_hidden_states = results.last_hidden_state.unsqueeze(1).cpu().detach().numpy()
        last_hidden_states = np.array(last_hidden_states, dtype=np.float32)
        logger.info(f"[_infer_fn_embedding] last_hidden_states shape: {last_hidden_states.shape}")
        return {"last_hidden_state": last_hidden_states}
    
    # @batch
    # def __call__(self, **inputs: np.ndarray):
    #     logger.info(f"[_infer_fn_embedding] inputs: {inputs}")
    #     (sequence_batch, max_length, pooler) = inputs.values()
    #     logger.info(f"[_infer_fn_embedding] sequence_batch: {sequence_batch}")
    #     logger.info(f"[_infer_fn_embedding] sequence_batch: {len(sequence_batch)}")
    #     logger.info(f"[_infer_fn_embedding] max_length: {max_length}")
    #     logger.info(f"[_infer_fn_embedding] pooler: {pooler}")
    #     # need to convert dtype=object to bytes first
    #     # end decode unicode bytes
    #     sequence_batch = np.char.decode(sequence_batch.astype("bytes"), "utf-8")
    #     sequence_batch = [s[0] for s in sequence_batch]
    #     logger.info(f"[_infer_fn_embedding] sequence_batch: {sequence_batch}")

    #     inputs = self._tokenizer(
    #         sequence_batch, 
    #         padding=True,
    #         truncation=True,
    #         max_length=512,
    #         return_tensors="pt"
    #     )
    #     inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
    #     results = self._model(**inputs_on_device, return_dict=True)
    #     # logger.info(f"[_infer_fn_embedding] results: {results}")

    #     last_hidden_states = results.last_hidden_state.unsqueeze(1).cpu().detach().numpy()
    #     last_hidden_states = np.array(last_hidden_states, dtype=np.float32)
    #     logger.info(f"[_infer_fn_embedding] last_hidden_states shape: {last_hidden_states.shape}")
    #     return [last_hidden_states]

    
def _infer_function_factory(devices: List[str]):
    infer_funcs = []
    for device in devices:
        # 实例化embedding类
        logger.info("init model on: {}".format(str(device)))
        logger.info("embedding model folder: {}".format(model_folder_embedding))
        tokenizer = AutoTokenizer.from_pretrained(model_folder_embedding)
        model = AutoModel.from_pretrained(model_folder_embedding, torch_dtype=torch_dtype)
        model.to(device)
        model.eval()
        infer_funcs.append(_InferFuncWrapper(model=model, tokenizer=tokenizer, device=device))

    return infer_funcs

def parse_argvs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_batch_size", type=int, default=128, help="Batch size of request.", required=False)
    parser.add_argument("--max_queue_delay_microseconds", type=int, default=10000, help="Max queue delay microseconds.", required=False)
    parser.add_argument("--instances_number", type=int, default=1, help="Number of model instances.", required=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    logger.info('[parse_argvs] {}'.format(args))
    return args

if __name__ == "__main__":
    args = parse_argvs()

    log_verbose = 1 if args.verbose else 0
    config = TritonConfig(exit_on_error=True, log_verbose=log_verbose)

    devices = [device] * args.instances_number

    with Triton(config=config) as triton:
        logger.info(f"Loading BERT model on devices: {devices}")
        triton.bind(
            model_name="BERT",
            infer_func=_infer_function_factory(devices),
            inputs=[
                Tensor(name="sequence", dtype=np.bytes_, shape=(-1,)),
                Tensor(name="max_length", dtype=np.int32, shape=(1,)),
                Tensor(name="pooler", dtype=np.bytes_, shape=(1,)),
            ],
            outputs=[
                Tensor(
                    name="last_hidden_state",
                    dtype=np.float32,
                    shape=(-1, -1, -1),
                ),
            ],
            config=ModelConfig(
                max_batch_size=args.max_batch_size,
                batcher=DynamicBatcher(
                        max_queue_delay_microseconds=args.max_queue_delay_microseconds,
                    ),),
            strict=True,
        )
        logger.info("Serving inference")
        triton.serve()
