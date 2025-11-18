source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate vllm


VLLM_PORT=$1
MODEL_ID=$2

EXTRA_ARGS=()
if [[ "$MODEL_PATH" == *"GPTQ"* ]]; then
  EXTRA_ARGS+=(--quantization gptq)
fi

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_ID \
    --max-model-len 16384 \
    --gpu_memory_utilization 0.8 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0  \
    --port $VLLM_PORT \
    "${EXTRA_ARGS[@]}"