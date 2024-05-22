import argparse
import torch
import tensorrt as trt
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
import os
from loguru import logger


def export_onnx_model(model, onnx_model_path):
    with torch.no_grad():
        torch.onnx.export(
            model=model.get_model(),
            args=model.get_sample_input(),
            f=onnx_model_path,
            verbose=True,
            opset_version=model.get_opset_version(),            # the ONNX version to export the model to
            do_constant_folding=True,                           # whether to execute constant folding for optimization
            input_names=model.get_input_names(),
            output_names=model.get_output_names(),
            dynamic_axes=model.get_dynamic_axes())      
        logger.warning(f"[export_onnx_model] ONNX Model exported to: {onnx_model_path}")


def export_tensorrt_model(model, onnx_model_path, trt_model_path, fp16=False, input_profile=None, enable_refit=False, 
                          enable_preview=False, enable_all_tactics=False, timing_cache=None, update_output_names=None):
    logger.info(f"[export_tensorrt_model] Building TensorRT engine for: {onnx_model_path} -> {trt_model_path}")

    input_profile = model.get_input_profile()
    logger.warning(f"[export_tensorrt_model] input_profile: {input_profile}")
    p = Profile()
    if input_profile:
        for name, dims in input_profile.items():
            assert len(dims) == 3
            p.add(name, min=dims[0], opt=dims[1], max=dims[2])

    config_kwargs = {}
    if not enable_all_tactics:
        config_kwargs['tactic_sources'] = []

    network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
    if update_output_names:
        logger.info(f"[export_tensorrt_model] updating network outputs to: {update_output_names}")
        network = ModifyNetworkOutputs(network, update_output_names)

    engine = engine_from_network(
            network,
            config=CreateConfig(fp16=fp16,
                refittable=enable_refit,
                profiles=[p],
                load_timing_cache=timing_cache,
                **config_kwargs
            ),
            save_timing_cache=timing_cache
    )
    save_engine(engine, path=trt_model_path)


def parse_argvs():
    parser = argparse.ArgumentParser(description='exchange_torch2trt')
    parser.add_argument("--model_type", type=str, choices=["bert", "bce_embedding", "bce_reranker"], default="bert")
    parser.add_argument("--model_path", type=str, default="/mnt/publish-data/pretrain_models/bert/chinese-bert-wwm/")
    parser.add_argument("--onnx_path", type=str, default="./onnx/bert-base-chinese-1.onnx")
    parser.add_argument("--trt_path", type=str, default="./engines/bert-base-chinese-1.engine")
    parser.add_argument("--opset_version", type=int, default=17)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--force_export", type=bool, default=False)

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()

    model_path = args.model_path
    onnx_path = args.onnx_path
    trt_path = args.trt_path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.model_type == "bert":
        from models.bert import Bert
        export_model = Bert(model_path=model_path, 
                            device=device, 
                            fp16=args.fp16, 
                            opset_version=args.opset_version)
    elif args.model_type == "bce_embedding":
        from models.bce_embedding import BCEEmbedding
        export_model = BCEEmbedding(model_path=model_path, 
                                    device=device, 
                                    fp16=args.fp16, 
                                    opset_version=args.opset_version)
    elif args.model_type == "bce_reranker":
        from models.bce_reranker import BCEReranker
        export_model = BCEReranker(model_path=model_path, 
                                   device=device, 
                                   fp16=args.fp16, 
                                   opset_version=args.opset_version)
    else:
        raise KeyError(f"model_type: {args.model_type} not support ...")

    if os.path.exists(onnx_path) and args.force_export is False:
        logger.warning(f"[onnx_path] {onnx_path} already exists ...")
    else:
        logger.info(f"[onnx_path] need to export_onnx_model: {onnx_path}")
        export_onnx_model(model=export_model,
                          onnx_model_path=onnx_path)

    if os.path.exists(trt_path) and args.force_export is False:
        logger.warning(f"[trt_path] {trt_path} already exists ...")
    else:
        logger.info(f"[trt_path] need to export_tensorrt_model: {trt_path}")
        export_tensorrt_model(model=export_model, 
                              onnx_model_path=onnx_path,
                              trt_model_path=trt_path,
                              fp16=args.fp16)
        