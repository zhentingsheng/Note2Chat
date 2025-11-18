source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

model_id="models/Qwen2.5-32B-Instruct-GPTQ-Int8"
port=8001

if nc -zv localhost $port 2>&1 | grep -q 'succeeded'; then
  echo "patient agent service is already running on port $port. Skipping startup."
else
  echo "Starting patient agent service..."
  CUDA_VISIBLE_DEVICES=4,5 bash scripts/start_vllm_server.sh $port $model_id &

  echo "Waiting for patient agent service to be ready..."
  while ! nc -zv localhost $port; do
    sleep 60
  done
  echo "patient agent service is ready!"
fi

model_id="models/Qwen2.5-7B-Instruct"
port=8002

if nc -zv localhost $port 2>&1 | grep -q 'succeeded'; then
  echo "patient agent service is already running on port $port. Skipping startup."
else
  echo "Starting patient agent service..."
  CUDA_VISIBLE_DEVICES=6,7 bash scripts/start_vllm_server.sh $port $model_id &

  echo "Waiting for patient agent service to be ready..."
  while ! nc -zv localhost $port; do
    sleep 60
  done
  echo "patient agent service is ready!"
fi

export PYTHONPATH='./src'

python src/evaluation/check_single_turn_questions.py \
  --single_turn_sample_dir data/single_turn_sampling \
  --model_id models/Qwen2.5-7B-Instruct \
  --num_threads 10 &

python src/evaluation/evaluate_single_turn_ddx.py \
  --single_turn_sample_dir data/single_turn_sampling \
  --model_id models/Qwen2.5-32B-Instruct-GPTQ-Int8 \
  --num_threads 10 &
