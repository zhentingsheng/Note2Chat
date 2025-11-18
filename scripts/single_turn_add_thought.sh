source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

# Step 0: Starting vllm service...
model_id="models/Qwen2.5-32B-Instruct-GPTQ-Int8"
port=8001

if nc -zv localhost $port 2>&1 | grep -q 'succeeded'; then
  echo "vllm service is already running on port $port. Skipping startup."
else
  echo "Starting vllm service..."
  bash scripts/start_vllm_server.sh $port $model_id &

  echo "Waiting for vllm service to be ready..."
  while ! nc -zv localhost $port; do
    sleep 60
  done
  echo "vllm service is ready!"
fi

export PYTHONPATH=./src

multi_turn_sample_dir=data/gpt_dialogues_and_sampling_best_rollout
single_turn_sample_dir=data/single_turn_gpt_dialogues_and_sampling_best_rollout
num_threads=4

python src/dialogue_synthesis/single_turn/add_thought.py --multi_turn_sample_dir $multi_turn_sample_dir --single_turn_sample_dir $single_turn_sample_dir --num_threads $num_threads --model_id $model_id


