from loguru import logger
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from .base_model import BaseModel


class BCEEmbedding(BaseModel):
    def __init__(self,
        model_path,
        device='cuda:0',
        verbose=True,
        fp16=True,
        min_batch_size=1,
        max_batch_size=128,
        min_seq_len=1,
        max_seq_len=512,
        opset_version=17,
    ):
        super(BCEEmbedding, self).__init__(device=device, verbose=verbose, fp16=fp16, min_batch_size=min_batch_size, max_batch_size=max_batch_size, opset_version=opset_version)
        
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        self.model = XLMRobertaModel.from_pretrained(model_path)
        self.model.to(device)

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

    def get_model(self):
        return self.model

    def get_input_names(self):
        return ['input_ids', 'attention_mask']

    def get_sample_input(self, text="这是个测试用例"):
        if self.tokenizer is not None:
            inputs = self.tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)
            inputs = inputs.to(self.device)
            logger.info(f"[get_sample_input] inputs: {inputs}")
            return (inputs['input_ids'],                               # model input (or a tuple for multiple inputs)
                    inputs['attention_mask'])
        else:
            raise AttributeError(f"[get_sample_input] model Bert except self.tokenizer, but get None...")
    
    def get_output_names(self):
       return ['output_0', 'output_1']

    def get_dynamic_axes(self):
        return {
        'input_ids': {0: "batch", 1: "sequence"}, 
        "attention_mask": {0: "batch", 1: "sequence"}, 
        'output_0': {0: "batch", 1: "sequence"},
        'output_1': {0: "batch"}
    }

    def get_input_profile(self, opt_batch_size=None, opt_seq_len=None):
        logger.info(f"[get_input_profile] opt_batch_size: {opt_batch_size}, opt_seq_len: {opt_seq_len}")
        # self.check_dims(batch_size, image_height, image_width)
        # min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        opt_batch_size = opt_batch_size if opt_batch_size else self.max_batch
        opt_seq_len = opt_seq_len if opt_seq_len else self.max_seq_len

        return {
            'input_ids': [(self.min_batch, self.min_seq_len), 
                          (opt_batch_size, opt_seq_len), 
                          (self.max_batch, self.max_seq_len)],
            'attention_mask': [(self.min_batch, self.min_seq_len), 
                               (opt_batch_size, opt_seq_len), 
                               (self.max_batch, self.max_seq_len)]
        }

    def get_shape_dict(self, batch_size, seq_len):
        # self.check_dims(batch_size, image_height, image_width)
        output = {
            'input_ids': (batch_size, seq_len),
            'attention_mask': (batch_size, seq_len)
        }
        return output

    # def optimize(self, onnx_graph):
    #     opt = Optimizer(onnx_graph, verbose=self.verbose)
    #     opt.info(self.name + ': original')
    #     opt.select_outputs([0]) # delete graph output#1
    #     opt.cleanup()
    #     opt.info(self.name + ': remove output[1]')
    #     opt.fold_constants()
    #     opt.info(self.name + ': fold constants')
    #     opt.infer_shapes()
    #     opt.info(self.name + ': shape inference')
    #     opt.select_outputs([0], names=['text_embeddings']) # rename network output
    #     opt.info(self.name + ': remove output[0]')
    #     opt_onnx_graph = opt.cleanup(return_onnx=True)
    #     if 'hidden_states' in self.extra_output_names:
    #         opt_onnx_graph = opt.clip_add_hidden_states(return_onnx=True)
    #         opt.info(self.name + ': added hidden_states')
    #     opt.info(self.name + ': finished')
    #     return opt_onnx_graph
