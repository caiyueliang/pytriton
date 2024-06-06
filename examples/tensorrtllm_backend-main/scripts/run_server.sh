#!/bin/bash

set -e

HOST=${HOST:=0.0.0.0}
PORT=${PORT:=8080}
GRPC_PORT=${GRPC_PORT:=8081}
METRICS_PORT=${METRICS_PORT:=8082}
MODEL_PATH_EMBEDDING=${MODEL_PATH_EMBEDDING:="/mnt/publish-data/pretrain_models/bert/bce-embedding-base_v1/"}
MODEL_PATH_EMBEDDING_TRT=${MODEL_PATH_EMBEDDING_TRT:="/mnt/publish-data/pretrain_models/trt/bce-embedding-base_v1_trt/"}
MODEL_TRT_NAME=${MODEL_TRT_NAME:="RobertaModel_float16_tp1_rank0.engine"}
OPT_BATCH_SIZE=${OPT_BATCH_SIZE:=1}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:=128}
OPT_INPUT_LEN=${OPT_INPUT_LEN:=256}
MAX_INPUT_LEN=${MAX_INPUT_LEN:=512}

TRT_FILE=${MODEL_PATH_EMBEDDING_TRT}${MODEL_TRT_NAME}

echo "======================================================================"
echo "[PORT] ${PORT}"
echo "[GRPC_PORT] ${GRPC_PORT}"
echo "[METRICS_PORT] ${METRICS_PORT}"
echo "[MODEL_PATH_EMBEDDING] ${MODEL_PATH_EMBEDDING}"
echo "[MODEL_PATH_EMBEDDING_TRT] ${MODEL_PATH_EMBEDDING_TRT}"
echo "[MODEL_TRT_NAME] ${MODEL_TRT_NAME}"
echo "[TRT_FILE] ${TRT_FILE}"
echo "[OPT_BATCH_SIZE] ${OPT_BATCH_SIZE}"
echo "[MAX_BATCH_SIZE] ${MAX_BATCH_SIZE}"
echo "[OPT_INPUT_LEN] ${OPT_INPUT_LEN}"
echo "[MAX_INPUT_LEN] ${MAX_INPUT_LEN}"

echo "======================================================================"
if [ ! -f "${TRT_FILE}" ]; then
    # 文件不存在
    echo "[init] TRT_FILE: ${TRT_FILE} no exists, building ..."
    python ./trt/build.py \
        --model_path ${MODEL_PATH_EMBEDDING} \
        --output_dir ${MODEL_PATH_EMBEDDING_TRT} \
        --model_name "RobertaModel" \
        --opt_batch_size ${OPT_BATCH_SIZE} \
        --max_batch_size ${MAX_BATCH_SIZE} \
        --opt_input_len ${OPT_INPUT_LEN} \
        --max_input_len ${MAX_INPUT_LEN}
else
    echo "[init] TRT_FILE: ${TRT_FILE} already exist, do nothing."
fi


echo "======================================================================"
echo "start triton server ..."

/opt/tritonserver/bin/tritonserver \
    --model-repository=/data/caiyueliang/pytriton/examples/tensorrtllm_backend-main/triton_model_embedding/ \
    --http-port=${PORT} \
    --grpc-port=${GRPC_PORT} \
    --metrics-port=${METRICS_PORT}
