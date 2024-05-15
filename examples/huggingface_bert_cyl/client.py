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
"""Client for BERT classifier sample server."""

from loguru import logger

import numpy as np

from pytriton.client import ModelClient

init_timeout_s = 600  # increase default timeout to let model download from HF hub
sequence = np.array([b"Hello, my dog is cute"])
# sequence = np.array(["你好，介绍一下你自己"])

logger.info(f"Input: {sequence}")
logger.info("Sending request")

with ModelClient("localhost", "BERT", init_timeout_s=init_timeout_s) as client:
    result_dict = client.infer_sample(sequence)


for output_name, output_data in result_dict.items():
    output_data = np.array2string(output_data, max_line_width=np.inf, separator=",").replace("\n", "")
    logger.info(f"{output_name}: {output_data}")
