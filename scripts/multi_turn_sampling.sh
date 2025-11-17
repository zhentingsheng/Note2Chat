source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

model_id=models/Qwen2.5-32B-Instruct-GPTQ-Int8

# Step 0: Starting patient agent service...
echo "Starting patient agent service..."
port=8001
bash scripts/start_vllm_server.sh $port $model_id &

echo "Waiting for patient agent service to be ready..."
while ! nc -zv localhost $port; do
  sleep 60
done

echo "patient agent service is ready!"

export PYTHONPATH=./src

# Step 1: Starting sampling...
python src/pipelines/sampling.py

