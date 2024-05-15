#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Client for Stable Diffusion 1.5."""

import argparse
import base64
import io
from loguru import logger
import pathlib
import time
import numpy as np
from PIL import Image  # pytype: disable=import-error
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

from pytriton.client import ModelClient

def infer(client, req_idx, prompts, img_size, results_path):
    prompt_id = req_idx % len(prompts)
    prompt = prompts[prompt_id]
    prompt_np = np.array([[prompt]])
    prompt_np = np.char.encode(prompt_np, "utf-8")
    logger.info(f"[infer] Prompt: ({req_idx}): {prompt}, {prompt_np}")
    logger.info(f"[infer] Image size: ({req_idx}): {img_size}")
    
    result_dict = client.infer_batch(prompt=prompt_np, img_size=img_size)

    for idx, image in enumerate(result_dict["image"]):
        # file_idx = req_idx + idx
        file_path = results_path / "image_{}_{}.jpeg".format(req_idx, prompt)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        msg = base64.b64decode(image[0])
        buffer = io.BytesIO(msg)
        image = Image.open(buffer)
        with file_path.open("wb") as fp:
            image.save(fp)
        logger.info(f"Image saved to {file_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="localhost",
        help=(
            "Url to Triton server (ex. grpc://localhost:8001)."
            "HTTP protocol with default port is used if parameter is not provided"
        ),
        required=False,
    )
    parser.add_argument(
        "--init-timeout-s",
        type=float,
        default=600.0,
        help="Server and model ready state timeout in seconds",
        required=False,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of requests per client.",
        required=False,
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results",
        help="Path to folder where images should be stored.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--num_thread",
        type=int,
        default=1,
        help="Number of requests per client.",
        required=False,
    )
    args = parser.parse_args()

    prompts = [
        "一个漂亮的女孩在看书",
        "一艘船在宇宙中漂浮",
        "一只鸟在水里游",
    ]

    img_size = np.array([[512]])
    results_path = pathlib.Path(args.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    start_t = time.time()
    with ModelClient(args.url, "StableDiffusion_1_5", init_timeout_s=args.init_timeout_s) as client:

        # with ThreadPool(args.num_thread) as pool:
        #     text_embedding_list = list(
        #         tqdm(
        #             pool.imap(
        #                 infer,
        #                 batched_text,
        #             ),
        #             total=len(args.iterations)
        #         )
        #     )
        # text_embedding_list = list(itertools.chain.from_iterable(text_embedding_list))
    
        for req_idx in range(1, args.iterations + 1):
            infer(client=client, 
                  req_idx=req_idx,
                  prompts=prompts, 
                  img_size=img_size,
                  results_path=results_path)
            # result_dict = client.infer_batch(prompt=prompt, img_size=img_size)
            # logger.debug(f"Result for for request ({req_idx}).")

            # for idx, image in enumerate(result_dict["image"]):
            #     # file_idx = req_idx + idx
            #     file_path = results_path / "image_{}_{}.jpeg".format(req_idx, prompts[prompt_id])
            #     file_path.parent.mkdir(parents=True, exist_ok=True)
            #     msg = base64.b64decode(image[0])
            #     buffer = io.BytesIO(msg)
            #     image = Image.open(buffer)
            #     with file_path.open("wb") as fp:
            #         image.save(fp)
            #     logger.info(f"Image saved to {file_path}")

    end_t = time.time()
    logger.warning(f"[time_used] {round(end_t-start_t, 5)} s")


if __name__ == "__main__":
    main()
