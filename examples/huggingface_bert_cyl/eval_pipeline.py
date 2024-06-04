
#!/usr/bin/env python3
# coding=utf-8
"""
评价 召回 + 排序 全流程的效果
阈值搜索策略，不同阈值下的context召回结果，
与GT answer(人工标注)进行比对，计算召回率和准确率
召回率：Rough值（context 和 GT 两两计算Rough，然后全部取平均, 越大越好
精确率：Bleu数值，度量context和GT之间的精确度，也是越大越好
"""

import json
import sys
from functools import lru_cache

import jieba
from rouge import Rouge
import torch
import numpy as np
# import evaluate

sys.setrecursionlimit(8735 * 2080 + 10)

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from FlagEmbedding import FlagReranker
from BCEmbedding import RerankerModel 

from utils import cos_sim, dot_score
from test_reranker import Reranker
from get_embeddings import MyEmbedding


smooth = SmoothingFunction()


class RoughBleuEvaluator(object):
    """
    计算Rouge 和 Bleu值
    contexts: List(str)
    ground_truths: List(str)
    """
    def __init__(self):
        pass

    def rouge_score(self, prediction, ground_truth, score_type="f"):
        rouge = Rouge()
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        return scores["rouge-l"][score_type]
    
    @lru_cache(maxsize=100000)
    def cut_sentence(self, sentence):
        return list(jieba.cut(sentence, cut_all=False))

    def get_metrics(self, contexts, ground_truths):
        """
        计算指标 ——> 单条query样本 
        contexts: 召回的片段，list
        ground_truths: 标注的正确回复，list
        """
        if len(contexts) == 0:
            print("[calulate rough or bleu]: contexts list is empty")
            return 0, 0, 0, 0
        contexts_cut_list = [self.cut_sentence(item) for item in contexts]
        ground_truths_cut_list = [self.cut_sentence(item) for item in ground_truths]
        
        retreive_rouge_max = 0.0
        retreive_bleu_max = 0.0
        retreive_rouge_aver = 0.0
        retreive_bleu_aver = 0.0

        
        for context in contexts_cut_list:
            for ground_truth in ground_truths_cut_list:
                temp = self.rouge_zh_score(context, ground_truth, score_type="r")
                retreive_rouge_aver += temp
                retreive_rouge_max = max(
                    retreive_rouge_max, temp
                )
        for context in contexts_cut_list:
            for ground_truth in ground_truths_cut_list:
                temp = self.bleu_zh_score(context, ground_truth)
                retreive_bleu_aver += temp
                retreive_bleu_max = max(
                    retreive_bleu_max, temp
                )

        # print(retreive_rouge, retreive_bleu)
        total_pairs = len(contexts_cut_list) * len(ground_truths_cut_list)
        return (retreive_bleu_max, retreive_bleu_max, retreive_rouge_aver/total_pairs, retreive_bleu_aver/total_pairs)

    def rouge_zh_score(self, hypotheses, reference, score_type="f"):
        """
        hypotheses: List of word (after tokenize)
        reference: List of word (after tokenize)
        """
        hypotheses = " ".join(hypotheses)
        reference = " ".join(reference)
        score = self.rouge_score(hypotheses, reference, score_type)
        return score

    def bleu_zh_score(self, hypotheses, reference):
        """
        hypotheses: List of word (after tokenize)
        reference: List of word (after tokenize)
        """
        # hypotheses = " ".join(hypotheses)
        # reference = " ".join(reference)
        results = corpus_bleu([[reference]], [hypotheses], smoothing_function=smooth.method1)
        return results


## retrieve pipeline
# step1, 构建召回库，文档片段
def construct_or_load_context():
    pass


class RecallAndRerank(object):
    """
    输入测试样例，输出不同参数下的最终召回结果
    Args:
        index: id -> text
        candidates_embedding: 向量库，[{'id':  , 'text':  , "embedding": },,]
        query_embeddings: query向量：[{'id':  , 'text':  , "embedding": },,]
        rerank_model: 排序模型
    """
    def __init__(self, index, candidate_embeddings, 
                 query_embeddings, rerank_model) -> None:

        self.index = index
        self.candidate_embeddings = candidate_embeddings
        self.query_embeddings = query_embeddings
        self.rerank_model = rerank_model
    
    def recall(self, recall_args):
        """
        从候选库中按照recall_args进行召回
        """
        top_k = recall_args["top_k"]
        threshold = recall_args["threshold"]

        candidate_text_list = []
        candidate_embedding_list =[]

        for candidate_dict in self.candidate_embeddings:
            candidate_text_list.append(candidate_dict["text"])
            candidate_embedding_list.append(candidate_dict["embedding"])

        ## return recall data
        recall_data = []

        for query_dict in self.query_embeddings:
            query = query_dict["text"]
            recall_context = []
            query_embedding = query_dict["embedding"]

            pair_scores = cos_sim(query_embedding, candidate_embedding_list)

            # Get top-k values
            pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                pair_scores, min(top_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
            )
            pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
            pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

            for idx, score in enumerate(pair_scores_top_k_values[0]):
                if score < threshold:
                    break
                recall_context.append(candidate_text_list[pair_scores_top_k_idx[0][idx]])
            # print("recall context num: {}".format(len(recall_context)))
            if len(recall_context) == 0:
                print("[recall model]: recall nothing")
            recall_data.append({"query": query, "recall_context": recall_context})
        
        return recall_data

    def rerank(self, recall_data, rerank_args):
        """
        针对召回的数据，进一步进行重排过滤
        recall_data： [{"query": "", "recall_context": [a, b, ..]}]
        """
        top_k = rerank_args["top_k"]
        threshold = rerank_args["threshold"]

        rerank_data = []

        for recall_item in recall_data:
            query = recall_item["query"]
            recall_context = recall_item["recall_context"]

            rerank_context = []

            rerank_score = self.rerank_model.get_rerank_score(query, recall_context)
            sorted_indices = sorted(range(len(rerank_score)), key=lambda i: rerank_score[i], reverse=True)[:top_k]
            for idx in sorted_indices:
                if rerank_score[idx] < threshold or len(rerank_context) > top_k:
                    break

                rerank_context.append(recall_context[idx])
            if len(rerank_context) == 0:
                print("[rerank model]: after rerank, get nothing")
            rerank_data.append({"query": query, "recall_context": rerank_context})

        return rerank_data


if __name__ == "__main__":
    import os

    import pandas as pd

    data_path = "/home/wangkun/myprogram/RAG/data/pred.jsonl"
    
    evaluator = RoughBleuEvaluator()

    # ## test one sample
    # rouge, bleu = evaluator.get_metrics(["刘德华的老婆是朱丽倩"], ["刘德华的老婆是杨幂"])
    # print(rouge)
    # print(bleu)

    # ## read data
    # with open(data_path, 'r') as f_read:
    #     lines = f_read.readlines()

    # rouge_list, bleu_list = [], []
    # for line in lines:
    #     line = line.strip()
    #     line_obj = json.loads(line)
    #     rouge, bleu = evaluator.get_metrics(line_obj["contexts"], line_obj["ground_truths"])
    #     rouge_list.append(rouge)
    #     bleu_list.append(bleu)
    
    # rouge_mean = sum(rouge_list) /len(rouge_list)
    # bleu_mean = sum(bleu_list) / len(bleu_list)

    # print(rouge_mean)
    # print(bleu_mean)

    # test recall && rerank
    # candidate_embeddings = [{"id": "1", "text": "abbaabb", "embedding": [0.1, 0.2, 0.3]},
    #                        {"id": "2", "text": "abcabc", "embedding": [0.3, 0.3, 0.3]},
    #                        {"id": "3", "text": "dddddd", "embedding": [0.2, 0.2, 0.2]},]
    
    # query_embeddings =  [{"id": "4", "text": "kkkkkkk", "embedding": [0.7, 0.2, 0.3]},
    #                        {"id": "5", "text": "ffffff", "embedding": [0.7, 0.3, 0.3]},
    #                        {"id": "6", "text": "cccccc", "embedding": [0.6, 0.2, 0.2]},]
    
    # bce_model_folder = "/mnt/data/wangkun/huggface_models/bce-reranker-base_v1"
    # model = RerankerModel(bce_model_folder) 
    # reranker =  Reranker(model)
    
    # retrieve_sys = RecallAndRerank(None, candidate_embeddings,
    #                                query_embeddings, reranker)
    

    # recall_data = retrieve_sys.recall({"top_k": 10, "threshold": 0.1})
    # rerank_data = retrieve_sys.rerank(recall_data, {"top_k": 10, "threshold": 0.35})

    # print(recall_data)
    # print(rerank_data)

    ## BEGIN TEST pred.jsonl
    


    #  step1 解析query 和 GT
    querys = {}
    answer_GT = []
    with open(data_path, 'r') as f_read:
        lines = f_read.readlines()
    for idx, line in enumerate(lines):
        line = json.loads(line.strip())
        querys[str(idx)] = line["question"]
        answer_GT.append(line["ground_truths"])


    # load 候选库

    futian_512_piece = {}
    with open("/home/wangkun/myprogram/RAG/data/split_corpus/futian.jsonl") as f_read:
        lines = f_read.readlines()
    for idx, line in enumerate(lines):
        line = json.loads(line.strip())
        futian_512_piece["doc"+str(idx)] = line["text"]

    # load query的embedding 和 候选集合的embedding
    load_from_file = False
    myembed = MyEmbedding()
    if load_from_file:
        querys_with_embedding = None
        contexts_with_embedding = None
    else:
        querys_with_embedding = myembed.get_embeddings(querys, "bce-embedding-base_v1", text_type="query")
        contexts_with_embedding = myembed.get_embeddings(futian_512_piece, "bce-embedding-base_v1")

    # 实例化 reranK模型，bge or  bce  must in folder name  #bce-reranker-base_v1  bge-reranker-large
    rerank_model_folder = "/mnt/data/wangkun/huggface_models/bce-reranker-base_v1"
    reranker = Reranker(rerank_model_folder)

    # 设置不同召回排序参数进行相关context召回
    retrieve_sys = RecallAndRerank(None, contexts_with_embedding,
                                   querys_with_embedding, reranker)
    ## freeze threshold tune top_k
    ## freeze top_k tune threshold

    # draw picture x-axis: thresh y-axis: rough-aver
    import matplotlib.pyplot as plt
    evaluator = RoughBleuEvaluator()
    x_values = np.linspace(0.0, 1.0 , num=21).tolist()
    y_values = []
    for thresh in x_values:
        print("calculate under thresh: {}".format(thresh))
        recall_data = retrieve_sys.recall({"top_k": 10, "threshold": 0.2})
        rerank_data = retrieve_sys.rerank(recall_data, {"top_k": 5, "threshold": thresh})

        rouge_list, bleu_list = [], []
        for idx, data in enumerate(rerank_data):
            rouge_score_max, bleu_score_max, rouge_score_aver, bleu_score_aver = evaluator.get_metrics(data["recall_context"], answer_GT[idx])
            rouge_list.append(rouge_score_max)
            bleu_list.append(bleu_score_aver)
        rouge_mean = sum(rouge_list) /len(rouge_list)
        bleu_mean = sum(bleu_list) / len(bleu_list)

        y_values.append(rouge_mean)
    
    plt.plot(x_values, y_values)
    plt.title('top_k=5')
    plt.xlabel('rerank thresh')
    plt.ylabel('Rouge-max')
    plt.savefig('metric_max.png')



    # recall_data = retrieve_sys.recall({"top_k": 10, "threshold": 0.2})
    # rerank_data = retrieve_sys.rerank(recall_data, {"top_k": 5, "threshold": 0.0})

    # # 计算指标，Rough\Bleu
    # evaluator = RoughBleuEvaluator()
    

    # rouge_list, bleu_list = [], []
    # for idx, data in enumerate(rerank_data):
    #     rouge_score_max, bleu_score_max, rouge_score_aver, bleu_score_aver = evaluator.get_metrics(data["recall_context"], answer_GT[idx])
    #     rouge_list.append(rouge_score_max)
    #     bleu_list.append(bleu_score_max)

    # rouge_mean = sum(rouge_list) /len(rouge_list)
    # bleu_mean = sum(bleu_list) / len(bleu_list)

    # print(rouge_mean)
    # print(bleu_mean)

