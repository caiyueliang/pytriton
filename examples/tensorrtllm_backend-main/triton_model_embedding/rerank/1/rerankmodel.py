import logging
import torch

import numpy as np

from tqdm import tqdm
from typing import List, Dict, Tuple, Type, Union
from copy import deepcopy

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import triton_python_backend_utils as pb_utils
import tensorrt as trt
from tensorrt_llm.runtime import ModelRunner, SamplingConfig
from tensorrt_llm.runtime import Session, TensorInfo


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)
    

class RerankerModel:
    def __init__(
            self,
            model_name_or_path: str='maidalun1020/yd-reranker-base_v1',
            use_fp16: bool=False,
            device: str=None,
            trt_model_name_or_path: str=None,
            **kwargs
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.sep_id = self.tokenizer.sep_token_id
        self.max_length = kwargs.get('max_length', 512)
        self.overlap_tokens = kwargs.get('overlap_tokens', 80)

        if trt_model_name_or_path:
            self.stream = torch.cuda.Stream()
            with open(trt_model_name_or_path, 'rb') as f:
                engine_buffer = f.read()
            self.session = Session.from_serialized_engine(engine_buffer)

    def compute_score(
            self, 
            sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], 
            batch_size: int = 128,
            max_length: int = 512,
            enable_tqdm: bool=False,
            **kwargs
        ):
        
        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        scores_collection = []
        for sentence_id in tqdm(range(0, len(sentence_pairs), batch_size), desc='Calculate scores', disable=not enable_tqdm):
            sentence_pairs_batch = sentence_pairs[sentence_id: sentence_id + batch_size]
            model_inputs = self.tokenizer(
                sentence_pairs_batch, 
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
            trt_input_ids = model_inputs.input_ids.int().cuda()
            trt_token_type_ids = model_inputs.attention_mask.int().cuda()
            trt_input_lengths = model_inputs.attention_mask.sum(dim=1).unsqueeze(0).int().cuda()
            # pb_utils.Logger.log_warn(f"shape input_ids_tensor: {trt_input_ids.shape}")
            # pb_utils.Logger.log_warn(f"shape token_type_ids_tensor: {trt_token_type_ids.shape}")
            # pb_utils.Logger.log_warn(f"shape input_lengths_tensor: {trt_input_lengths.shape}")

            inputs = {
                'input_ids': trt_input_ids,
                'token_type_ids': trt_token_type_ids,
                'input_lengths': trt_input_lengths
            }
            # pb_utils.Logger.log_warn(f"inputs: {inputs}")

            output_info = self.session.infer_shapes([
                    TensorInfo('input_ids', trt.DataType.INT32, inputs['input_ids'].shape),
                    TensorInfo('token_type_ids', trt.DataType.INT32, inputs['token_type_ids'].shape),
                    TensorInfo('input_lengths', trt.DataType.INT32, inputs['input_lengths'].squeeze(0).shape),
                ])
            # pb_utils.Logger.log_warn(f"output_info: {output_info}")
            outputs = {
                    t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device='cuda')
                    for t in output_info
                }
            session_state_code = self.session.run(inputs, outputs, self.stream.cuda_stream)
            if not session_state_code:
                raise RuntimeError("TRT-LLM Runtime execution failed")
            torch.cuda.Stream.synchronize(self.stream)

            # pb_utils.Logger.log_warn(f"[outputs] {outputs}")
            scores = outputs["logits"].view(-1,).float()
            # pb_utils.Logger.log_warn(f"[scores] 111 {scores}")
            scores = torch.sigmoid(scores)
            # pb_utils.Logger.log_warn(f"[scores] 222 {scores}")
            scores_collection.extend(scores.cpu().numpy().tolist())

        if len(scores_collection) == 1:
            return scores_collection[0]
        return scores_collection
    
    def _merge_inputs(self, chunk1_raw, chunk2):
        chunk1 = deepcopy(chunk1_raw)
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(self.sep_id)
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids'])+1)]
            chunk1['token_type_ids'].extend(token_type_ids)
        return chunk1

    def tokenize_preproc(
        self,
        query: str, 
        passages: List[str]
    ):
        query_inputs = self.tokenizer.encode_plus(query, truncation=False, padding=False)
        max_passage_inputs_length = self.max_length - len(query_inputs['input_ids']) - 1
        assert max_passage_inputs_length > 100, "Your query is too long! Please make sure your query less than 400 tokens!"
        overlap_tokens = min(self.overlap_tokens, max_passage_inputs_length//4)
        
        res_merge_inputs = []
        res_merge_inputs_pids = []
        for pid, passage in enumerate(passages):
            passage_inputs = self.tokenizer.encode_plus(passage, truncation=False, padding=False, add_special_tokens=False)
            passage_inputs_length = len(passage_inputs['input_ids'])

            if passage_inputs_length <= max_passage_inputs_length:
                qp_merge_inputs = self._merge_inputs(query_inputs, passage_inputs)
                res_merge_inputs.append(qp_merge_inputs)
                res_merge_inputs_pids.append(pid)
            else:
                start_id = 0
                while start_id < passage_inputs_length:
                    end_id = start_id + max_passage_inputs_length
                    sub_passage_inputs = {k:v[start_id:end_id] for k,v in passage_inputs.items()}
                    start_id = end_id - overlap_tokens if end_id < passage_inputs_length else end_id

                    qp_merge_inputs = self._merge_inputs(query_inputs, sub_passage_inputs)
                    res_merge_inputs.append(qp_merge_inputs)
                    res_merge_inputs_pids.append(pid)
        
        return res_merge_inputs, res_merge_inputs_pids
    
    def rerank(
            self,
            query: str,
            passages: List[str],
            batch_size: int=256,
            **kwargs
        ):
        # remove invalid passages
        passages = [p for p in passages if isinstance(p, str) and 0 < len(p) < 16000]
        if query is None or len(query) == 0 or len(passages) == 0:
            return {'rerank_passages': [], 'rerank_scores': []}
        
        # preproc of tokenization
        sentence_pairs, sentence_pairs_pids = self.tokenize_preproc(query, passages)

        # batch inference
        # if self.num_gpus > 1:
        #     batch_size = batch_size * self.num_gpus

        tot_scores = []
        with torch.no_grad():
            for k in range(0, len(sentence_pairs), batch_size):
                batch = self.tokenizer.pad(
                        sentence_pairs[k:k+batch_size],
                        padding=True,
                        max_length=None,
                        pad_to_multiple_of=None,
                        return_tensors="pt"
                    )
                batch_on_device = {k: v.to(self.device) for k, v in batch.items()}
                scores = self.model(**batch_on_device, return_dict=True).logits.view(-1,).float()
                scores = torch.sigmoid(scores)
                tot_scores.extend(scores.cpu().numpy().tolist())

        # ranking
        merge_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(sentence_pairs_pids, tot_scores):
            merge_scores[pid] = max(merge_scores[pid], score)

        merge_scores_argsort = np.argsort(merge_scores)[::-1]
        sorted_passages = []
        sorted_scores = []
        for mid in merge_scores_argsort:
            sorted_scores.append(merge_scores[mid])
            sorted_passages.append(passages[mid])
        
        return {
            'rerank_passages': sorted_passages,
            'rerank_scores': sorted_scores
        }
