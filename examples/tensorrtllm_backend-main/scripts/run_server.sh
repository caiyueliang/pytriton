#!/bin/bash

set -e

HOST=${HOST:=0.0.0.0}
PORT=${PORT:=8080}
GRPC_PORT=${GRPC_PORT:=8081}
METRICS_PORT=${METRICS_PORT:=8082}

MODEL_REPOSITORY=${MODEL_REPOSITORY:="/home/server/triton_models/"}

MODEL_PATH_EMBEDDING=${MODEL_PATH_EMBEDDING:="/mnt/publish-data/train_data/embedding/bce-embedding-base_v1/"}
MODEL_PATH_EMBEDDING_TRT=${MODEL_PATH_EMBEDDING_TRT:="/mnt/publish-data/train_data/embedding/bce-embedding-base_v1_triton_trt/"}
MODEL_NAME_EMBEDDING_TRT=${MODEL_NAME_EMBEDDING_TRT:="RobertaModel_float16_tp1_rank0.engine"}
TRT_FILE_EMBEDDING=${MODEL_PATH_EMBEDDING_TRT}${MODEL_NAME_EMBEDDING_TRT}

MODEL_PATH_RERANK=${MODEL_PATH_RERANK:="/mnt/publish-data/train_data/rerank/bce-reranker-base_v1/"}
MODEL_PATH_RERANK_TRT=${MODEL_PATH_RERANK_TRT:="/mnt/publish-data/train_data/rerank/bce-reranker-base_v1_triton_trt/"}
MODEL_NAME_RERANK_TRT=${MODEL_NAME_RERANK_TRT:="RobertaForSequenceClassification_float16_tp1_rank0.engine"}
TRT_FILE_RERANK=${MODEL_PATH_RERANK_TRT}${MODEL_NAME_RERANK_TRT}

OPT_BATCH_SIZE=${OPT_BATCH_SIZE:=1}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:=128}
OPT_INPUT_LEN=${OPT_INPUT_LEN:=256}
MAX_INPUT_LEN=${MAX_INPUT_LEN:=512}


echo "======================================================================"
echo "[PORT] ${PORT}"
echo "[GRPC_PORT] ${GRPC_PORT}"
echo "[METRICS_PORT] ${METRICS_PORT}"
echo "[MODEL_REPOSITORY] ${MODEL_REPOSITORY}"
echo "[OPT_BATCH_SIZE] ${OPT_BATCH_SIZE}"
echo "[MAX_BATCH_SIZE] ${MAX_BATCH_SIZE}"
echo "[OPT_INPUT_LEN] ${OPT_INPUT_LEN}"
echo "[MAX_INPUT_LEN] ${MAX_INPUT_LEN}"

echo "======================================================================"
echo "[build] embedding model start ..."
echo "[build] [MODEL_PATH_EMBEDDING] ${MODEL_PATH_EMBEDDING}"
echo "[build] [MODEL_PATH_EMBEDDING_TRT] ${MODEL_PATH_EMBEDDING_TRT}"
echo "[build] [MODEL_NAME_EMBEDDING_TRT] ${MODEL_NAME_EMBEDDING_TRT}"
echo "[build] [TRT_FILE_EMBEDDING] ${TRT_FILE_EMBEDDING}"
if [ ! -f "${TRT_FILE_EMBEDDING}" ]; then
    # 文件不存在
    echo "[build] TRT_FILE: ${TRT_FILE_EMBEDDING} no exists, building ..."
    python ./trt/build.py \
        --model_path ${MODEL_PATH_EMBEDDING} \
        --output_dir ${MODEL_PATH_EMBEDDING_TRT} \
        --model_name "RobertaModel" \
        --opt_batch_size ${OPT_BATCH_SIZE} \
        --max_batch_size ${MAX_BATCH_SIZE} \
        --opt_input_len ${OPT_INPUT_LEN} \
        --max_input_len ${MAX_INPUT_LEN}
else
    echo "[build] TRT_FILE_EMBEDDING: ${TRT_FILE_EMBEDDING} already exist, do nothing."
fi

echo "======================================================================"
echo "[build] rerank model start ..."
echo "[build] [MODEL_PATH_RERANK] ${MODEL_PATH_RERANK}"
echo "[build] [MODEL_PATH_RERANK_TRT] ${MODEL_PATH_RERANK_TRT}"
echo "[build] [MODEL_NAME_RERANK_TRT] ${MODEL_NAME_RERANK_TRT}"
echo "[build] [TRT_FILE_RERANK] ${TRT_FILE_RERANK}"

if [ ! -f "${TRT_FILE_RERANK}" ]; then
    # 文件不存在
    echo "[build] TRT_FILE: ${TRT_FILE_RERANK} no exists, building ..."
    python ./trt/build.py \
        --model_path ${MODEL_PATH_RERANK} \
        --output_dir ${MODEL_PATH_RERANK_TRT} \
        --model_name "RobertaForSequenceClassification" \
        --n_labels 1 \
        --opt_batch_size ${OPT_BATCH_SIZE} \
        --max_batch_size ${MAX_BATCH_SIZE} \
        --opt_input_len ${OPT_INPUT_LEN} \
        --max_input_len ${MAX_INPUT_LEN}
else
    echo "[build] TRT_FILE_RERANK: ${TRT_FILE_RERANK} already exist, do nothing."
fi

echo "======================================================================"
echo "start triton server ..."

/opt/tritonserver/bin/tritonserver \
    --model-repository=${MODEL_REPOSITORY} \
    --http-port=${PORT} \
    --grpc-port=${GRPC_PORT} \
    --metrics-port=${METRICS_PORT}
