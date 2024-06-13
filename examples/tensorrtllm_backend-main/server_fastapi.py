"""
BCE文本抽向量服务 + Bce 精排服务

"""
#!/usr/bin/env python
# encoding: utf-8
import logging
import json
import os

import torch
import transformers
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer
# from rerankmodel import RerankerModel 


import uvicorn
from fastapi import FastAPI, Request
from typing import Optional, Union
from pydantic import BaseModel, Field
from utils.time_utils import TimeUtils
import base64
import pickle
from pytriton.client import ModelClient

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s [PID]%(process)d %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S %a')

app = FastAPI()

API_STATUS_CODE_OK = 200  # OK
API_STATUS_CODE_CLIENT_ERROR = 1001
API_STATUS_CODE_CLIENT_ERROR_FORMAT = 1002  # 请求数据格式错误
API_STATUS_CODE_CLIENT_ERROR_CONFIG = 1003  # 请求数据配置不支持
API_STATUS_CODE_SERVER_ERROR = 5000
API_STATUS_CODE_SERVER_ERROR_RUNNING = 5001  # 服务器运行中出错

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16
marker = "*" * 20
use_fp16=True


# 获取全局变量
@app.on_event("startup")
def startup():  
    global tokenizer
    global model_embedding
    global torch_dtype
    global reranker
    global TRT_URL
    global TRT_MODEL_NAME
    global TRT_MODEL_NAME_BATCH
    global client
    global client_batch

    TRT_URL = os.environ.get("TRT_URL", 'localhost:18080')
    TRT_MODEL_NAME = os.environ.get("TRT_MODEL_NAME", 'embedding')
    TRT_MODEL_NAME_BATCH = os.environ.get("TRT_MODEL_NAME_BATCH", 'embedding_batch')
    logging.info(f"[TRT_URL] {TRT_URL}; [TRT_MODEL_NAME] {TRT_MODEL_NAME}; [TRT_MODEL_NAME_BATCH] {TRT_MODEL_NAME_BATCH}")

    client = ModelClient(TRT_URL, TRT_MODEL_NAME, init_timeout_s=600)
    client_batch = ModelClient(TRT_URL, TRT_MODEL_NAME_BATCH, init_timeout_s=600)

    # model_folder_embedding = os.environ["MODEL_PATH_EMBEDDING"]

    # # 实例化embedding类
    # logging.info("init model on: {}".format(str(device)))
    # logging.info("embedding model folder: {}".format(model_folder_embedding))

    # tokenizer = AutoTokenizer.from_pretrained(model_folder_embedding)
    # model_embedding = AutoModel.from_pretrained(model_folder_embedding, torch_dtype=torch_dtype)
    # model_embedding.to(device)
    # model_embedding.eval()

    # # -------------------------------------------------------------------
    # # 实例化排序类
    # model_folder_rerank = os.environ["MODEL_PATH_RERANK"]
    # logging.info("reranker model folder: {}".format(model_folder_rerank))
    # reranker = RerankerModel(model_folder_rerank, use_fp16=use_fp16, device=device)  

# -------------------------------------------------------------------
class SentenceEmbeddingApiRequest(BaseModel):
    text: Union[list, str]
    encoding_format: Optional[str] = Field(description="float or base64 embedding", default="base64")
    batch_size: Optional[int] = Field(description="batch_size", default=1)
    max_length: Optional[int] = Field(description="max_length", default=512)
    pooler: Optional[str] = Field(description="pooler type", default="cls")

class RerankApiRequest(BaseModel):
    query: str
    candidate: list

class RerankApiResponse(object):
    """
    rerank http协议api响应类
    """
    def __init__(self, status_code, status_message='', score={}) -> None:
        self.status_code = status_code
        self.status_message = status_message
        self.score = score

    def to_json(self):
        """
        class to json
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, ensure_ascii=False)


class SentenceEmbeddingApiResponse(object):
    """
    embedding http协议api响应类
    """
    def __init__(self, status_code, status_message='', embedding=[], usage={"prompt_tokens": 0, "total_tokens": 0}) -> None:
        self.status_code = status_code
        self.status_message = status_message
        self.embedding = embedding
        self.usage = usage
    
    def to_json(self):
        """
        class to json
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


# def _get_embedding_trt(trt_model_name, text, max_length):
#     max_length = np.array([max_length], dtype=np.int32)   
#     sequence = np.array([t.encode('utf-8') for t in text])

#     with ModelClient(TRT_URL, trt_model_name, init_timeout_s=600) as client:
#         response = client.infer_sample(sequence, max_length)
#         token_information = pickle.loads(response["usage"])
#         embedding = np.array(response['embedding'], dtype=np.float32).reshape(-1, 768)

#         return embedding, token_information
def _get_embedding_trt(client, text, max_length):
    max_length = np.array([max_length], dtype=np.int32)   
    sequence = np.array([t.encode('utf-8') for t in text])

    # with ModelClient(TRT_URL, trt_model_name, init_timeout_s=600) as client:
    response = client.infer_sample(sequence, max_length)
    token_information = pickle.loads(response["usage"])
    embedding = np.array(response['embedding'], dtype=np.float32).reshape(-1, 768)

    return embedding, token_information
    
@app.post('/embedding')
async def sentence_embed_post(item: SentenceEmbeddingApiRequest):
    """
    embedding 路由服务
    """
    try:
        TimeUtils().start(task_name="embedding")
        logging.info(marker + "[embedding router]" + marker)
        request_data = item.model_dump()
        text = request_data['text']
        batch_size = request_data.get("batch_size", 1)
        max_length = request_data.get("max_length", 512)
        pooler = request_data.get("pooler", "cls")
        encoding_format = request_data.get("encoding_format", "base64")

        if batch_size > 1:
            infer_client = client_batch
        else:
            infer_client = client

        if isinstance(text, str):
            text = [text]

        logging.info("request text -- text: {}".format(text[:8]))
        logging.info("request args -- text len: {}, batch_size: {}, max_length: {}, pooler: {}".format(len(text), batch_size, max_length, pooler))

        if len(text) <= 0:
            json_data = SentenceEmbeddingApiResponse(API_STATUS_CODE_CLIENT_ERROR_FORMAT, "text is empty, should be list or list in string")
            buffer = json_data.to_json()
            logging.error("output: {}, error: {}".format(buffer, "query or candidate empty"))
            return json.loads(buffer)
       
        if isinstance(text, str):
            text_list = json.loads(text)
        else:
            text_list = text

        assert isinstance(text_list, list)

        # 输入的文本是空文本
        if len(text_list) <= 0:
            json_data = SentenceEmbeddingApiResponse(API_STATUS_CODE_CLIENT_ERROR_FORMAT, "text is empty")
            buffer = json_data.to_json()
            logging.error("output: {}, error: {}".format(buffer, "query or candidate empty"))
            return json.loads(buffer)

        TimeUtils().append("前处理", task_name="embedding")

        # inference
        embedding, token_information = _get_embedding_trt(client=infer_client, text=text, max_length=max_length)
        prompt_tokens = token_information.get("prompt_tokens", 0)
        total_tokens = token_information.get("total_tokens", 0)

        TimeUtils().append("推理", task_name="embedding")

        if encoding_format == "base64":
            embedding = base64.b64encode(embedding.tobytes()).decode('utf-8')
        elif encoding_format == "float":
            embedding = embedding.tolist()
        else:
            raise ValueError("unsupport encoding_format: {}, choose from base6 | float".format(encoding_format))
        
        usage = {"prompt_tokens": prompt_tokens, "total_tokens": total_tokens}
        json_data = SentenceEmbeddingApiResponse(API_STATUS_CODE_OK, 'ok', embedding=embedding, usage=usage)
        TimeUtils().append("后处理", task_name="embedding")
        TimeUtils().print(task_name="embedding")
        return json_data
    except Exception as e:
        json_data = SentenceEmbeddingApiResponse(API_STATUS_CODE_SERVER_ERROR, str(e))
        buffer = json_data.to_json()
        logging.error("output: {}, error: {}".format(buffer, e))
        return json.loads(buffer)


@app.post('/rerank')
async def rerank_retrieve_post(item: RerankApiRequest):
    """
    BCE排序服务
    """
    try:
        logging.info(marker + "[rerank router]" + marker)
        request_data = item.model_dump()
        query = request_data['query']
        candidate = request_data['candidate']

        if isinstance(candidate, str):
            candidate = json.loads(candidate)
        
        assert isinstance(candidate, list)
        logging.info(str(query))

        # 输入的文本是空文本
        if len(query) <= 0 or len(candidate) <=0:
            json_data = RerankApiResponse(API_STATUS_CODE_CLIENT_ERROR_FORMAT, "text is empty")
            buffer = json_data.to_json()
            logging.error("output: {}, error: {}".format(buffer, "query or candidate empty"))
            return json.loads(buffer)

        # ranking
        pairs = []
        for candi in candidate:
            pairs.append([query, candi])
        logging.info("query: {}".format(query))
        logging.info("candidate: {}".format(str(candidate)))

        result = reranker.compute_score(pairs)
        if isinstance(result, float):
            result = [result]
        logging.info("scores: {}".format(str(result)))

        score = {}

        for idx, res_score in enumerate(result):
            #logging.info("para: {} score: {}".format(para,str(score)))
            score[candidate[idx]] = round(res_score, 6)

        json_data = RerankApiResponse(API_STATUS_CODE_OK, 'ok')

        json_data.score = score

        buffer = json_data.to_json()
        return json.loads(buffer)

    except Exception as e:
        json_data = RerankApiResponse(API_STATUS_CODE_SERVER_ERROR, str(e))
        buffer = json_data.to_json()
        logging.error("output: {}, error: {}".format(buffer, e))
        return json.loads(buffer)


def argsparser():
    """
    args parser
    """
    parser = argparse.ArgumentParser(description="bce embedding + rerank server", add_help=False)
    parser.add_argument("--port", default=8080, type=int, help='port to listen')
    parser.add_argument("--host", default='0.0.0.0', type=str, help='host to listen')
    parser.add_argument("--worker", default=1, type=int, help='worker num')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    # for development env
    # app.run(host=args.host, port=args.port)
    # for production env
    uvicorn.run("server_fastapi:app", host=args.host, port=args.port, workers=args.worker)

