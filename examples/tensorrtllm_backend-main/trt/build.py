# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from collections import OrderedDict

# isort: off
import torch
import tensorrt as trt
# isort: on

from transformers import BertConfig, BertForQuestionAnswering, BertForSequenceClassification, BertModel  # isort:skip
from transformers import RobertaConfig, RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaModel  # isort:skip
from transformers import XLMRobertaModel, XLMRobertaForSequenceClassification
from weight import (load_from_hf_cls_model, load_from_hf_model,
                    load_from_hf_qa_model)

from loguru import logger
import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='Tensor parallelism size')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument('--timing_cache', type=str, default='model.cache')
    parser.add_argument(
        '--profiling_verbosity',
        type=str,
        default='detailed',
        choices=['layer_names_only', 'detailed', 'none'],
        help=
        'The profiling verbosity for the generated TRT engine. Set to detailed can inspect tactic choices and kernel parameters.'
    )
    parser.add_argument('--log_level', type=str, default='verbose')
    parser.add_argument('--vocab_size', type=int, default=250002)       # 21128
    parser.add_argument('--n_labels', type=int, default=3)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_positions', type=int, default=514)         # 512
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--opt_batch_size', type=int, default=1)
    parser.add_argument('--max_batch_size', type=int, default=128)        # 1
    parser.add_argument('--opt_input_len', type=int, default=512)
    parser.add_argument('--max_input_len', type=int, default=512)
    parser.add_argument('--gpus_per_node', type=int, default=1)
    parser.add_argument('--type_vocab_size', type=int, default=1)       # 2
    parser.add_argument('--output_dir', type=str, default='/data/yangsheng/risk-control-master/train/bert_classify_torch/trt-llm-fp16-0/')
    parser.add_argument('--use_bert_attention_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32'])
    parser.add_argument('--use_gemm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32'])
    parser.add_argument('--enable_qk_half_accum',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_context_fmha',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        default=False,
                        action='store_true')
    parser.add_argument('--model_name',
                        default='BertForSequenceClassification',
                        choices=[
                            'BertModel',
                            'BertForQuestionAnswering',
                            'BertForSequenceClassification',
                            'RobertaModel',
                            'RobertaForQuestionAnswering',
                            'RobertaForSequenceClassification'
                        ])
    parser.add_argument('--model_path', type=str, default='/data/yangsheng/risk-control-master/train/bert_classify_torch/trt-llm-fp16-0/')
    args = parser.parse_args()
    logger.info(f"[parse_arguments] {args}")
    return args

def build(model_path, output_dir, model_name,
          vocab_size=250002, n_embd=768, n_layer=12, n_head=12, hidden_act='gelu', n_positions=514, type_vocab_size=1,
          max_batch_size=128, opt_batch_size=1, max_input_len=512, opt_input_len=512, dtype="float16", log_level="verbose",
          timing_cache='model.cache', profiling_verbosity='detailed', world_size=1, rank=0, n_labels=3,
          use_bert_attention_plugin=False, use_gemm_plugin=False, enable_qk_half_accum=False, enable_context_fmha=False, enable_context_fmha_fp32_acc=False):
    tensorrt_llm.logger.set_level(log_level)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
    # inlen_range = [1, (max_input_len + 1) // 2, max_input_len]
    bs_range = [1, opt_batch_size, max_batch_size]
    inlen_range = [1, opt_input_len, max_input_len]

    torch_dtype = torch.float16 if dtype == 'float16' else torch.float32
    trt_dtype = trt.float16 if dtype == 'float16' else trt.float32

    logger.info(f"[bs_range] {bs_range}; [inlen_range] {inlen_range}; [torch_dtype] {torch_dtype}; [trt_dtype] {trt_dtype}")

    builder = Builder()
    builder_config = builder.create_builder_config(
        name=model_name,
        precision=dtype,
        timing_cache=timing_cache,
        profiling_verbosity=profiling_verbosity,
        tensor_parallel=world_size,  # TP only
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
    )
    # Initialize model

    if 'Roberta' in model_name:
        model_type = 'Roberta'
    else:
        model_type = 'Bert'

    # logger.info(f"[globals] {globals()}")
    bert_config = globals()[f'{model_type}Config'](
        vocab_size=vocab_size,
        hidden_size=n_embd,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        intermediate_size=4 * n_embd,
        hidden_act=hidden_act,
        max_position_embeddings=n_positions,
        torch_dtype=torch_dtype,
        type_vocab_size=type_vocab_size
    )
    logger.info(f"[bert_config] {bert_config}")
    
    if model_name == 'BertModel' or model_name == 'RobertaModel':
        # hf_bert = globals()[f'{model_type}Model'](bert_config,
        #                                           add_pooling_layer=False)
        # bert_folder="/mnt/publish-data/pretrain_models/taichu-risk-control-bert/model/"
        hf_bert = BertModel.from_pretrained(model_path, trust_remote_code=True).cuda().to(torch_dtype).eval()
        tensorrt_llm_bert = tensorrt_llm.models.BertModel(
            num_layers=bert_config.num_hidden_layers,
            num_heads=bert_config.num_attention_heads,
            hidden_size=bert_config.hidden_size,
            vocab_size=bert_config.vocab_size,
            hidden_act=bert_config.hidden_act,
            max_position_embeddings=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            pad_token_id=bert_config.pad_token_id,
            is_roberta=(model_type == 'Roberta'),
            mapping=Mapping(world_size=world_size,
                            rank=rank,
                            tp_size=world_size),  # TP only
            dtype=trt_dtype)
        load_from_hf_model(
            tensorrt_llm_bert,
            hf_bert,
            bert_config,
            rank=rank,
            tensor_parallel=world_size,
            fp16=(dtype == 'float16'),
        )
        output_name = 'hidden_states'
    elif model_name == 'BertForQuestionAnswering' or model_name == 'RobertaForQuestionAnswering':
        # hf_bert = globals()[f'{model_type}ForQuestionAnswering'](bert_config)
        hf_bert = BertForQuestionAnswering.from_pretrained(model_path, trust_remote_code=True).cuda().to(torch_dtype).eval()

        tensorrt_llm_bert = tensorrt_llm.models.BertForQuestionAnswering(
            num_layers=bert_config.num_hidden_layers,
            num_heads=bert_config.num_attention_heads,
            hidden_size=bert_config.hidden_size,
            vocab_size=bert_config.vocab_size,
            hidden_act=bert_config.hidden_act,
            max_position_embeddings=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            pad_token_id=bert_config.pad_token_id,
            is_roberta=(model_type == 'Roberta'),
            num_labels=n_labels,  # TODO: this might just need to be a constant
            mapping=Mapping(world_size=world_size,
                            rank=rank,
                            tp_size=world_size),  # TP only
            dtype=trt_dtype)
        load_from_hf_qa_model(
            tensorrt_llm_bert,
            hf_bert,
            bert_config,
            rank=rank,
            tensor_parallel=world_size,
            fp16=(dtype == 'float16'),
        )
        output_name = 'logits'
    elif model_name == 'BertForSequenceClassification' or model_name == 'RobertaForSequenceClassification':
        # hf_bert = globals()[f'{model_type}ForSequenceClassification'](
        #     bert_config).cuda().to(torch_dtype).eval()
        # bert_folder="/mnt/publish-data/pretrain_models/taichu-risk-control-bert/model/"
        if model_name == 'BertForSequenceClassification':
            hf_bert = BertForSequenceClassification.from_pretrained(model_path, trust_remote_code=True).cuda().to(torch_dtype).eval()
        else:
            hf_bert = RobertaForSequenceClassification.from_pretrained(model_path, trust_remote_code=True).cuda().to(torch_dtype).eval()
        logger.warning(f"[hf_bert] {hf_bert}")

        tensorrt_llm_bert = tensorrt_llm.models.BertForSequenceClassification(
            num_layers=bert_config.num_hidden_layers,
            num_heads=bert_config.num_attention_heads,
            hidden_size=bert_config.hidden_size,
            vocab_size=bert_config.vocab_size,
            hidden_act=bert_config.hidden_act,
            max_position_embeddings=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            pad_token_id=bert_config.pad_token_id,
            is_roberta=(model_type == 'Roberta'),
            num_labels=n_labels,  # TODO: this might just need to be a constant
            mapping=Mapping(world_size=world_size,
                            rank=rank,
                            tp_size=world_size),  # TP only
            dtype=trt_dtype)
        load_from_hf_cls_model(
            tensorrt_llm_bert,
            hf_bert,
            bert_config,
            rank=rank,
            tensor_parallel=world_size,
            fp16=(dtype == 'float16'),
        )
        output_name = 'logits'
    else:
        assert False, f"Unknown BERT model: {model_name}"

    # Module -> Network
    network = builder.create_network()
    network.plugin_config.to_legacy_setting()
    if use_bert_attention_plugin:
        network.plugin_config.set_bert_attention_plugin(
            dtype=use_bert_attention_plugin)
    if use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=use_gemm_plugin)
    if enable_qk_half_accum:
        network.plugin_config.enable_qk_half_accum()
    assert not (enable_context_fmha and enable_context_fmha_fp32_acc)
    if enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(
            ContextFMHAType.enabled_with_fp32_acc)
    if world_size > 1:
        network.plugin_config.set_nccl_plugin(dtype)
    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_bert.named_parameters())

        # Forward
        input_ids = tensorrt_llm.Tensor(
            name='input_ids',
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict([('batch_size', [bs_range]),
                                   ('input_len', [inlen_range])]),
        )
        logger.info(f"[input_ids] {input_ids}")

        # also called segment_ids
        token_type_ids = tensorrt_llm.Tensor(
            name='token_type_ids',
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict([('batch_size', [bs_range]),
                                   ('input_len', [inlen_range])]),
        )
        logger.info(f"[token_type_ids] {token_type_ids}")

        input_lengths = tensorrt_llm.Tensor(name='input_lengths',
                                            dtype=trt.int32,
                                            shape=[-1],
                                            dim_range=OrderedDict([
                                                ('batch_size', [bs_range])
                                            ]))
        logger.info(f"[input_lengths] {input_lengths}")

        # logits for QA BERT, or hidden_state for vanilla BERT
        output = tensorrt_llm_bert(input_ids=input_ids,
                                   input_lengths=input_lengths,
                                   token_type_ids=token_type_ids)

        # Mark outputs
        output_dtype = trt.float16 if dtype == 'float16' else trt.float32
        output.mark_output(output_name, output_dtype)

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    assert engine is not None, 'Failed to build engine.'
    engine_file = os.path.join(output_dir, get_engine_name(model_name, dtype, world_size, rank))
    with open(engine_file, 'wb') as f:
        f.write(engine)
    builder.save_config(builder_config, os.path.join(output_dir, 'config.json'))    

if __name__ == '__main__':
    args = parse_arguments()
    build(model_path=args.model_path, 
          output_dir=args.output_dir,
          model_name=args.model_name,
          vocab_size=args.vocab_size,
          n_embd=args.n_embd,
          n_layer=args.n_layer,
          n_head=args.n_head,
          hidden_act=args.hidden_act,
          n_positions=args.n_positions,
          type_vocab_size=args.type_vocab_size,
          max_batch_size=args.max_batch_size, 
          opt_batch_size=args.opt_batch_size, 
          max_input_len=args.max_input_len,
          opt_input_len=args.opt_input_len,
          dtype=args.dtype,
          log_level=args.log_level,
          timing_cache=args.timing_cache,
          profiling_verbosity=args.profiling_verbosity,
          world_size=args.world_size,
          rank=args.rank,
          n_labels=args.n_labels,
          use_bert_attention_plugin=args.use_bert_attention_plugin,
          use_gemm_plugin=args.use_gemm_plugin,
          enable_qk_half_accum=args.enable_qk_half_accum,
          enable_context_fmha=args.enable_context_fmha,
          enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc)
