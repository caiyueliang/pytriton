import os
import json
import pickle
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
from torch import from_numpy
from transformers import AutoModel, AutoTokenizer

import tensorrt as trt
from tensorrt_llm.runtime import ModelRunner, SamplingConfig
from tensorrt_llm.runtime import Session, TensorInfo

from rerankmodel import RerankerModel 

def mpi_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


def mpi_rank():
    return mpi_comm().Get_rank()


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def get_input_tensor_by_name(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is not None:
        # Triton tensor -> numpy tensor -> PyTorch tensor
        return from_numpy(tensor.as_numpy())
    else:
        return tensor


def get_input_scalar_by_name(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is not None:
        # Triton tensor -> numpy tensor -> first scalar
        tensor = tensor.as_numpy()
        return tensor.reshape((tensor.size, ))[0]
    else:
        return tensor

def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


def tensor_group(tensor, group_sizes):
    groups = []
    start_index = 0
    for size in group_sizes:
        group = tensor[start_index: start_index + size]
        groups.append(group)
        start_index += size
    return groups

def generage_usage(input_lengths, group_sizes):
    usage_list = []
    input_len_group = tensor_group(tensor=input_lengths, group_sizes=group_sizes)
    # pb_utils.Logger.log_warn(f"[input_len_group] {input_len_group}")
    for input_len in input_len_group:
        total_token = torch.sum(input_len).item()
        prompt_tokens = (total_token - 2 * input_len.shape[0])
        usage_list.append({"total_tokens": total_token, "prompt_tokens": prompt_tokens})
    return usage_list


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        model_config = json.loads(args['model_config'])

        tokenizer_dir = model_config['parameters']['tokenizer_dir']['string_value']
        pb_utils.Logger.log_warn(f"[tokenizer_dir] {tokenizer_dir}")
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

        # engine_dir = model_config['parameters']['engine_dir']['string_value']
        # model_path = engine_dir
        # pb_utils.Logger.log_warn(f"[model_path] {model_path}")
        # self.stream = torch.cuda.Stream()
        # with open(model_path, 'rb') as f:
        #     engine_buffer = f.read()
        # self.session = Session.from_serialized_engine(engine_buffer)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16=True
        self.reranker = RerankerModel(model_name_or_path=tokenizer_dir, use_fp16=use_fp16, device=device) 

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        group_list = []
        pairs = [] 
        candidate_list = []

        for request in requests:
            query = pb_utils.get_input_tensor_by_name(request, 'query').as_numpy()[0, 0].decode("utf-8")
            candidate = pb_utils.get_input_tensor_by_name(request, 'candidate').as_numpy()

            group_list.append(len(candidate[0]))
            for candi_bytes in candidate[0]:
                candi = candi_bytes.decode("utf-8")
                pairs.append([query, candi])
                candidate_list.append(candi)

        pb_utils.Logger.log_warn(f"group_list: {group_list}; text len: {len(pairs)}; pairs: {pairs[:4]}")

        result = self.reranker.compute_score(sentence_pairs=pairs)
        if isinstance(result, float):
            result = [result]
        # pb_utils.Logger.log_warn("score: {}".format(result))

        result_g = tensor_group(tensor=result, group_sizes=group_list)
        for result in result_g:
            score = {}
            for idx, res_score in enumerate(result):
                #logging.info("para: {} score: {}".format(para,str(score)))
                score[candidate_list[idx]] = round(res_score, 6)

            # pb_utils.Logger.log_warn("score: {}".format(score))
            score_bytes = np.array([pickle.dumps(score)])
            inference_response = pb_utils.InferenceResponse(
                output_tensors = [
                    pb_utils.Tensor("score", score_bytes)
                ]
            )
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        return