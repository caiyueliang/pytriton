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

import argparse
from loguru import logger
import time
from multiprocessing import Process, Queue, JoinableQueue
# from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import numpy as np

from pytriton.client import ModelClient

def infer(url, init_timeout_s, times, sequence, result_queue):
    times_list = []
    with ModelClient(url, "BERT", init_timeout_s=init_timeout_s) as client:
        for i in range(times):
            start = time.time() * 1000
            result_dict = client.infer_sample(sequence)

            for output_name, output_data in result_dict.items():
                output_data = np.array2string(output_data, max_line_width=np.inf, separator=",").replace("\n", "")
                # logger.info(f"{output_name}: len: {len(output_data)}; {output_data}")
            end = time.time() * 1000
            times_list.append(end-start)
    
    result_queue.put(times_list)
    # return True

def start_process(num_processes, url, init_timeout_s, times, sequence):
    # 创建一个 Queue 用于接收进程的返回数据
    # result_queue = Queue()
    result_queue = JoinableQueue()
    # 创建并启动多个进程
    processes = []
    for i in range(num_processes):
        # 可以为每个进程定制不同的参数
        # custom_params = {**base_params, f"param_{i}": f"value_{i}"}
        p = Process(target=infer, args=(url, init_timeout_s, times, sequence, result_queue))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    logger.info(f"11111")
    # 收集所有进程的返回数据
    results = []
    while not result_queue.empty():
        results += result_queue.get()

    logger.warning(f"[results] 请求数据量: {len(results)}, 平均请求耗时: {round(sum(results)/len(results), 2)} ms")


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

    # init_timeout_s = 600  # increase default timeout to let model download from HF hub
    sequence = np.array([b"Hello, my dog is cute"])
    # sequence = np.array(["你好，介绍一下你自己"])

    logger.info(f"Input: {sequence}")
    logger.info("Sending request")

    start_t = time.time()

    start_process(num_processes=args.num_thread, url=args.url, init_timeout_s=args.init_timeout_s, times=args.iterations, sequence=sequence)
    # infer(args.url, args.init_timeout_s, sequence)

    end_t = time.time()
    logger.warning(f"[time_used] {round(end_t-start_t, 5)} s")

if __name__ == "__main__":
    main()