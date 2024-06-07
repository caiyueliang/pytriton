import os
import json
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
from torch import from_numpy
from transformers import AutoModel, AutoTokenizer

import tensorrt as trt
from tensorrt_llm.runtime import ModelRunner, SamplingConfig
from tensorrt_llm.runtime import Session, TensorInfo


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


def embeddings_group(embeddings, group_sizes):
    groups = []
    start_index = 0
    for size in group_sizes:
        group = embeddings[start_index: start_index + size]
        groups.append(group)
        start_index += size
    return groups


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
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

        engine_dir = model_config['parameters']['engine_dir']['string_value']
        model_path = engine_dir
        pb_utils.Logger.log_warn(f"[model_path] {model_path}")
        self.stream = torch.cuda.Stream()
        with open(model_path, 'rb') as f:
            engine_buffer = f.read()
        self.session = Session.from_serialized_engine(engine_buffer)

        # self.comm = mpi_comm()
        # self.rank = mpi_rank()
        # self.runner = ModelRunner.from_dir(engine_dir=engine_dir,
        #                                    rank=self.rank)
        # if self.rank != 0:
        #     while (True):
        #         self.execute([None])

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
        text_batch = []
        group_list = []
        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.

        # pb_utils.Logger.log_warn(f"requests len: {len(requests)}")
        for idx, request in enumerate(requests):
            max_length = pb_utils.get_input_tensor_by_name(request, 'max_length').as_numpy()
            max_length = max_length[0][0]
            # pb_utils.Logger.log_warn(f"[execute] text: {text}; max_length: {max_length}")

            # Get input tensors
            text_per_req = pb_utils.get_input_tensor_by_name(request, 'text').as_numpy()
            text_len_req = len(text_per_req[0])
            # pb_utils.Logger.log_warn(f"idx: {idx}, text: {text_per_req}, type: {type(text_per_req)}, len: {text_len_req}")
            group_list.append(text_len_req)

            for text_bytes in text_per_req[0]:
                text = text_bytes.decode("utf-8")
                # pb_utils.Logger.log_warn(f"text: {text}")
                text_batch.append(text)

        pb_utils.Logger.log_warn(f"group_list: {group_list}; text len: {len(text_batch)}; max_length: {max_length}; text_batch: {text_batch[:8]}; ")

        # pb_utils.Logger.log_warn(f"text_batch len: {len(requests)}")

        model_input = self.tokenizer(text_batch, truncation=True, padding=True, return_tensors="pt", max_length=max_length)
        # model_input = self.tokenizer(text, truncation=True, padding="max_length", return_tensors="pt", max_length=max_length)
        # pb_utils.Logger.log_warn(f"[model_input] {model_input}")

        trt_input_ids = model_input.input_ids.int().cuda()
        trt_token_type_ids = model_input.attention_mask.int().cuda()
        trt_input_lengths = model_input.attention_mask.sum(dim=1).unsqueeze(0).int().cuda()

        inputs = {
            'input_ids': trt_input_ids,
            'token_type_ids': trt_token_type_ids,
            'input_lengths': trt_input_lengths
        }
        # pb_utils.Logger.log_warn(f"shape input_ids_tensor: {trt_input_ids.shape}")
        # pb_utils.Logger.log_warn(f"shape token_type_ids_tensor: {trt_token_type_ids.shape}")
        # pb_utils.Logger.log_warn(f"shape input_lengths_tensor: {trt_input_lengths.shape}")

        output_info = self.session.infer_shapes([
                TensorInfo('input_ids', trt.DataType.INT32, inputs['input_ids'].shape),
                TensorInfo('input_lengths', trt.DataType.INT32, inputs['input_lengths'].squeeze(0).shape),
                TensorInfo('token_type_ids', trt.DataType.INT32, inputs['token_type_ids'].shape),
            ])
        outputs = {
                t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device='cuda')
                for t in output_info
            }
        session_state_code = self.session.run(inputs, outputs, self.stream.cuda_stream)
        if not session_state_code:
            raise RuntimeError("TRT-LLM Runtime execution failed")
        torch.cuda.Stream.synchronize(self.stream)

        # if pooler == "cls":
        #     embeddings = outputs["hidden_states"][:, 0]
        # elif pooler == "mean":
        #     attention_mask = inputs['attention_mask'].to(device)
        #     last_hidden = outputs["hidden_states"]
        #     embeddings = (last_hidden * attention_mask.unsqueeze(-1).float()).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        # else:
        #     raise NotImplementedError
        embeddings = outputs["hidden_states"][:, 0]
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        embeddings = embeddings.cpu().detach().numpy()
        # pb_utils.Logger.log_warn(f"embeddings: {type(embeddings)}, {embeddings}")

        embeddings_g = embeddings_group(embeddings=embeddings, group_sizes=group_list)
        # pb_utils.Logger.log_warn(f"embeddings_g: {embeddings_g}")
        for embedding in embeddings_g:
            # pb_utils.Logger.log_warn(f"embedding: {embedding.shape}")
            embedding = embedding.tolist()
            # pb_utils.Logger.log_warn(f"[embedding] len: {len(embedding)}")
            embedding = np.array([[embedding]], dtype=np.float32)
            # pb_utils.Logger.log_warn(f"[embedding] {embedding}")
            inference_response = pb_utils.InferenceResponse(
                output_tensors = [pb_utils.Tensor("embedding", embedding)]
            )
            responses.append(inference_response)
        
        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        return