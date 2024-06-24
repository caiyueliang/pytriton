import logging
import torch
import gc

import numpy as np

from tqdm import tqdm
from typing import List, Dict, Tuple, Type, Union
from copy import deepcopy

from transformers import AutoTokenizer, AutoModelForSequenceClassification



class RerankerModel:
    def __init__(
            self,
            model_name_or_path: str='maidalun1020/yd-reranker-base_v1',
            use_fp16: bool=False,
            device: str=None,
            **kwargs
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
        self.device = device

        # self.num_gpus = 1
        if use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.device)
        self.model = torch.compile(self.model, dynamic=True)

        # for advanced preproc of tokenization
        self.sep_id = self.tokenizer.sep_token_id
        self.max_length = kwargs.get('max_length', 512)
        self.overlap_tokens = kwargs.get('overlap_tokens', 80)

        logging.info(f"Execute device: {self.device};\t use fp16: {use_fp16}")
    
    def __del__(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
    def compute_score(
            self, 
            sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], 
            batch_size: int = 128,
            max_length: int = 512,
            enable_tqdm: bool=False,
            **kwargs
        ):
        # if self.num_gpus > 1:
        #     batch_size = batch_size * self.num_gpus
        
        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        with torch.no_grad():
            scores_collection = []
            for sentence_id in tqdm(range(0, len(sentence_pairs), batch_size), desc='Calculate scores', disable=not enable_tqdm):
                sentence_pairs_batch = sentence_pairs[sentence_id: sentence_id + batch_size]
                inputs = self.tokenizer(
                            sentence_pairs_batch, 
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                        )
                inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()} 
                # token_type_ids = inputs.attention_mask.sum(dim=1).unsqueeze(0).int().cuda()
                # token_type_ids = inputs.attention_mask.int().cuda()
                # inputs_on_device["token_type_ids"] = token_type_ids

                # logging.warning(f"[inputs_on_device] {inputs_on_device}")
                outputs = self.model(**inputs_on_device, return_dict=True)
                # logging.warning(f"[outputs] {outputs}")

                scores = outputs.logits.view(-1,).float()
                # logging.warning(f"[scores] 111 {scores}")
                scores = torch.sigmoid(scores)
                # logging.warning(f"[scores] 222 {scores}")
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
