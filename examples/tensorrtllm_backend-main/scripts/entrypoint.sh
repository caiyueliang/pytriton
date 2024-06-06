#!/bin/bash

HOST=${HOST:=0.0.0.0}
PORT=${PORT:=8080}
MODEL_PATH=${MODEL_PATH:=/lustre/wangjiancheng/Taichu_Chat_20231115}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:=2}
SEED=${SEED:=24}
SWAP_SPACE=${SWAP_SPACE:=4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:=0.9}

cmd="python -m vllm.entrypoints.openai.api_server \
--host ${HOST} \
--port ${PORT} \
--model ${MODEL_PATH} \
--trust-remote-code \
--tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
--seed ${SEED} \
--swap-space ${SWAP_SPACE} \
--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}
"

if [ -v MAX_MODEL_LEN ]; then
    cmd+=" --max-model-len ${MAX_MODEL_LEN}"
fi

if [ -v MAX_PARALLEL_LOADING_WORKERS ]; then
    cmd+=" --max-parallel-loading-workers ${MAX_PARALLEL_LOADING_WORKERS}"
fi

# export TAICHU_PREFIX=$'###答案：接下来是一个给我的提问或指令，我会解答这个问题并按照指令要求进行回复。</s>\n\n'

if [ -v TAICHU_CHAT_TEMPLATE ]; then
    cmd+=" --chat-template ${TAICHU_CHAT_TEMPLATE}"
else
    if [ -e "${MODEL_PATH}/template.jinja" ]; then
        cmd+=" --chat-template ${MODEL_PATH}/template.jinja"
    fi
fi

echo $cmd

$cmd
