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

import os
import numpy as np
import torch
# from transformers import BertTokenizer, FlaxBertModel  # pytype: disable=import-error
from transformers import AutoModel, AutoTokenizer

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = FlaxBertModel.from_pretrained("bert-base-uncased")

model_folder_embedding = os.environ["MODEL_PATH_EMBEDDING"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16

# 实例化embedding类
logger.info("init model on: {}".format(str(device)))
logger.info("embedding model folder: {}".format(model_folder_embedding))

tokenizer = AutoTokenizer.from_pretrained(model_folder_embedding)
model = AutoModel.from_pretrained(model_folder_embedding, torch_dtype=torch_dtype)
model.to(device)
model.eval()


@batch
def _infer_fn(**inputs: np.ndarray):
    (sequence_batch,) = inputs.values()

    # need to convert dtype=object to bytes first
    # end decode unicode bytes
    sequence_batch = np.char.decode(sequence_batch.astype("bytes"), "utf-8")

    last_hidden_states = []
    for sequence_item in sequence_batch:
        # tokenized_sequence = tokenizer(sequence_item.item(), return_tensors="jax")
        # results = model(**tokenized_sequence)
        logger.info(f"[_infer_fn] sequence_item: {sequence_item.item()}")
        inputs = tokenizer(
                        sequence_item.item(), 
                        padding=True,
                        truncation=True,
                        max_length=1024,
                        return_tensors="pt"
                    )
        inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
        results = model(**inputs_on_device, return_dict=True)
        
        last_hidden_states.append(results.last_hidden_state.cpu().detach().numpy())
    
        logger.info(f"[_infer_fn] last_hidden_states: {last_hidden_states}")
    # last_hidden_states = last_hidden_states.detach().numpy()
    last_hidden_states = np.array(last_hidden_states, dtype=np.float32)
    return [last_hidden_states]


with Triton() as triton:
    logger.info("Loading BERT model.")
    triton.bind(
        model_name="BERT",
        infer_func=_infer_fn,
        inputs=[
            Tensor(name="sequence", dtype=np.bytes_, shape=(-1,)),
        ],
        outputs=[
            Tensor(
                name="last_hidden_state",
                dtype=np.float32,
                shape=(-1, -1, -1),
            ),
        ],
        config=ModelConfig(max_batch_size=16),
        strict=True,
    )
    logger.info("Serving inference")
    triton.serve()
