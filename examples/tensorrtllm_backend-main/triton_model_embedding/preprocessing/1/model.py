import json
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import triton_python_backend_utils as pb_utils
# from transformers import BertTokenizer

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
        # Parse model configs
        # model_config = json.loads(args['model_config'])
        model_config = json.loads(args['model_config'])
        tokenizer_dir = model_config['parameters']['tokenizer_dir']['string_value']
        # tokenizer_dir = "/mnt/publish-data/pretrain_models/taichu-risk-control-bert/bert-base-chinese"
        pb_utils.Logger.log_warn(f"[tokenizer_dir] {tokenizer_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

        # Parse model output configs and convert Triton types to numpy types
        output_names = [
            "input_ids", "input_lengths", "token_type_ids"
        ]
        input_names = ["text", "max_length"]
        for input_name in input_names:
            setattr(
                self,
                input_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_input_config_by_name(
                        model_config, input_name)['data_type']))

        for output_name in output_names:
            setattr(
                self,
                output_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(
                        model_config, output_name)['data_type']))

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
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

        pb_utils.Logger.log_warn(f"[execute] requests: {requests}")
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get input tensors
            # text = pb_utils.get_input_tensor_by_name(request, 'text').as_numpy()
            text = pb_utils.get_input_tensor_by_name(request, 'text').as_numpy()
            text = text[0][0].decode("utf-8")
            # pb_utils.Logger.log_warn(f"text: {text}")
            # pb_utils.Logger.log_warn(f"type text: {type(text[0][0])}")
            
            max_length = pb_utils.get_input_tensor_by_name(request, 'max_length').as_numpy()
            max_length = max_length[0][0]
            pb_utils.Logger.log_warn(f"[execute] text: {text}; max_length: {max_length}")

            model_input = self.tokenizer(text, truncation=True, padding=True, \
                                         return_tensors="pt", max_length=max_length)
            trt_input_ids = model_input.input_ids.int().numpy()
            trt_token_type_ids = model_input.attention_mask.int().numpy()
            trt_input_lengths = model_input.attention_mask.sum(dim=1).unsqueeze(0).int().numpy()

            pb_utils.Logger.log_warn(f"shape input_ids_tensor: {trt_input_ids.shape}")
            pb_utils.Logger.log_warn(f"shape token_type_ids_tensor: {trt_token_type_ids.shape}")
            pb_utils.Logger.log_warn(f"shape input_lengths_tensor: {trt_input_lengths.shape}")

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            input_ids_tensor = pb_utils.Tensor(
                'input_ids', trt_input_ids.astype(self.input_ids_dtype))
            input_lengths_tensor = pb_utils.Tensor(
                'input_lengths', trt_input_lengths.astype(self.input_lengths_dtype))
            token_type_ids_tensor = pb_utils.Tensor(
                'token_type_ids', trt_token_type_ids.astype(self.token_type_ids_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                input_ids_tensor, input_lengths_tensor, token_type_ids_tensor
            ])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
