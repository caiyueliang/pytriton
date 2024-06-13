# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   Description :    Concurrency Test
   Author :         caiyueliang
   Date :           2020-09-11
-------------------------------------------------
"""
import sys
import os
import argparse
import requests
import json
import time
import multiprocessing
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed
from pytriton.client import ModelClient

# 获取当前脚本的目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录
parent_directory = os.path.dirname(current_directory)
logger.warning(f"[parent_directory] {parent_directory}")
# 将上级目录添加到系统路径
sys.path.append(parent_directory)

from utils import utils

headers = {'Content-Type': 'application/json', 'authorization': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJhZG1pbiJ9.j6-hUMaFYdSIzfc6i6TJ5DaS96Z9I78SrjxAOg-71yE'}

def print_embedding(result_dict, compress=False):
    if compress is True:
        embeddings = np.frombuffer(result_dict['embedding'], dtype=np.float16).reshape(-1, 768)
        # embeddings = np.frombuffer(result_dict['embedding'], dtype=np.float32).reshape(-1, 768)
        embeddings = embeddings.tolist()
        for embed in embeddings:
            logger.info(f"[embedding] len: {len(embed)}\n{embed[:10]}; \n{embed[-10:]}")
    else:
        embeddings = np.array(result_dict['embedding'], dtype=np.float32).reshape(-1, 768)
        for embed in embeddings:
            logger.info(f"[embedding] len: {len(embed)}\n{embed[:10]}; \n{embed[-10:]}")

def infer(url, model_name, params, times):
    times_list = []
    result_list = []
    for i in range(times):
        try:
            start = time.time() * 1000
            response = requests.post(url, headers=headers, data=json.dumps(params))
            # if "compress" in model_name:
            #     print_embedding(result_dict=response.json(), compress=True)
            # else:
            #     print_embedding(result_dict=response.json())

            end = time.time() * 1000
            times_list.append(end-start)
            result_list.append(200)
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            times_list.append(10000)
            result_list.append(0)
    return times_list, result_list

def start_threads(url, model_name, works, times, params):
    """
    concurrency test start
    :param func:
    :param url:
    :param works:
    :param times:
    :return:
    """

    all_task = list()
    executor = ThreadPoolExecutor(max_workers=works)

    # start concurrent request
    start = utils.get_cur_millisecond()

    for i in range(works): 
        all_task.append(executor.submit(infer, url, model_name, params, times))

    wait(all_task, return_when=ALL_COMPLETED)
    end = utils.get_cur_millisecond()

    time_used = (float(end - start)/1000)
    tps_request = float(len(all_task) * times) / time_used

    time_list = list()
    result_list = list()

    for task in as_completed(all_task):
        time, result = task.result()
        # logger.warning(f"[time] {time}, [result] {result}")
        time_list += time
        result_list += result

    return time_list, result_list, time_used, tps_request


def start_multiprocessing(url, model_name, processes, num_thread, times, params):
    pool = multiprocessing.Pool(processes=processes)
    all_p = list()

    for i in range(processes):
        all_p.append(pool.apply_async(start_threads, (url, model_name, num_thread, times, params)))

    logger.info("[start_multiprocessing] start ...")
    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

    time_used_max = 0
    tps_request_total = 0
    total_time_p = list()
    total_result_p = list()

    for result in all_p:
        total_time_thread, total_result_thread, time_used, tps_request = result.get()

        time_used_max = time_used if time_used > time_used_max else time_used_max
        tps_request_total += tps_request

        total_time_p.extend(total_time_thread)
        total_result_p.extend(total_result_thread)

    logger.warning(f"[统计] 总共耗时: {time_used_max} s; QPS: {tps_request_total}")
    utils.calc_time_p99(time_list=total_time_p)
    utils.calc_success_rate(result_list=total_result_p)


# python stress_test.py --processes 1 --works 1 --times 10000
# python stress_test.py --processes 4 --works 2 --times 20000
# python stress_test.py --processes 8 --works 2 --times 20000
# python stress_test.py --processes 8 --works 4 --times 20000
def parse_argvs():
    """ parse argv """
    parser = argparse.ArgumentParser(description='test xbot')


    parser.add_argument("--url", default="http://localhost:8080/v2/models/embedding/generate", help=(
            "Url to Triton server (ex. grpc://localhost:8001)."
            "HTTP protocol with default port is used if parameter is not provided"
        ), required=False)
    
    parser.add_argument("--model_name", type=str, default="compress", required=False)
    parser.add_argument("--init-timeout-s", type=float, default=600.0, help="Server and model ready state timeout in seconds", required=False)
    parser.add_argument("--text", type=str, default="我是中国人", required=False)
    parser.add_argument("--use_trt", type=bool, default=False, required=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--processes", help="processes num", type=int, default=1)
    parser.add_argument("--num_thread", type=int, default=1, help="Number of requests per client.", required=False)
    parser.add_argument("--times", help="test times per processes", type=int, default=10)
    parser.add_argument("--pooler", help="pooler", type=str, default="cls")
    args = parser.parse_args()

    args = parser.parse_args()
    logger.info('当前参数: %s' % str(args))

    return parser, args


if __name__ == '__main__':
    parser, args = parse_argvs()

    logger.info("Sending request")

    # url = "{}/v2/models/{}/generate".format(args.url, args.model_name)
    # logger.info(f"[url] {url}")
    
    base_params = {
        "text": [args.text],
        # "text": [args.text, "你好，介绍一下你自己"],
        "max_length": 512,
        "batch_size": 1,
        "pooler": args.pooler,
    }
    start_multiprocessing(url=args.url, 
                          model_name=args.model_name,
                          processes=args.processes, 
                          num_thread=args.num_thread,
                          times=args.times, 
                          params=base_params)

