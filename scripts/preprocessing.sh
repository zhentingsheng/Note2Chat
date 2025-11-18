source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

export PYTHONPATH=./src

model_id=models/Qwen2.5-32B-Instruct-GPTQ-Int8
note_path=data/notes/notes.csv
note_hpi_path=data/processed/note_hpi.jsonl
hpi_sentences_path=data/processed/hpi_sentences.jsonl
hpi_sentence_categories_path=data/processed/hpi_sentence_categories.json

# Step 0: Starting vllm service...
port=8001

if nc -zv localhost $port 2>&1 | grep -q 'succeeded'; then
  echo "patient agent service is already running on port $port. Skipping startup."
else
  echo "Starting patient agent service..."
  CUDA_VISIBLE_DEVICES=4,5 bash scripts/start_vllm_server.sh $port $model_id &

  echo "Waiting for vllm service to be ready..."
  while ! nc -zv localhost $port; do
    sleep 60
  done
  echo "vllm service is ready!"
fi

# Step 1: Split HPI section from discharge notes using LLM
python3 src/preprocessing/hpi_splitter.py --model_id $model_id --note_path $note_path --note_hpi_path $note_hpi_path

# Step 2: Split HPI into sentences using LLM
python3 src/preprocessing/hpi_sentences_splitter.py --model_id $model_id --note_hpi_path $note_hpi_path --hpi_sentences_path $hpi_sentences_path

# Step 3: Classify sentences using LLM
python3 src/preprocessing/hpi_sentences_classifier.py --model_id $model_id --hpi_sentences_path $hpi_sentences_path --hpi_sentence_categories_path  $hpi_sentence_categories_path --num_threads 10


echo "done"