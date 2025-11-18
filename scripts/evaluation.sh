source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

model_id="models/Qwen2.5-32B-Instruct-GPTQ-Int8"
port=8001

if nc -zv localhost $port 2>&1 | grep -q 'succeeded'; then
  echo "vllm service is already running on port $port. Skipping startup."
else
  echo "Starting vllm service..."
  CUDA_VISIBLE_DEVICES=4,5 bash scripts/start_vllm_server.sh $port $model_id &

  echo "Waiting for vllm service to be ready..."
  while ! nc -zv localhost $port; do
    sleep 60
  done
  echo "vllm service is ready!"
fi

export PYTHONPATH=./src

# closed models
python3 src/pipelines/closed_models_experiments.py
# open source models
python3 src/pipelines/open_source_models_experiments.py
# domain specific models
python3 src/pipelines/domain_specific_models_experiments.py
# note2chat models
python3 src/pipelines/note2chat_experiments.py