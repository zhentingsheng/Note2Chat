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
  bash scripts/start_vllm_server.sh $port $model_id &

  echo "Waiting for patient agent service to be ready..."
  while ! nc -zv localhost $port; do
    sleep 60
  done
  echo "patient agent service is ready!"
fi


export PYTHONPATH=./src

# Step 1: Starting sampling...
python src/pipelines/sampling.py

echo "done"