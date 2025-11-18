source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

export PYTHONPATH=./src

note_path=data/notes/notes.csv
note_hpi_path=data/processed/note_hpi.jsonl
trainset_note_ids_path=data/note_ids/trainset_note_ids.csv
sample_dir=data/gpt_dialogues
model_id=o4-mini

# Step 1: generate decision tree and dialogue
python3 src/dialogue_synthesis/generation.py --note_path $note_path --note_hpi_path $note_hpi_path  --sample_dir $sample_dir  --model_id $model_id

# Step 2: evaluate chief complaint using LLM
#!/bin/bash

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

gpt_dialogues_dir=data/gpt_dialogues
chief_complaint_hit_sentences_dir=data/processed/chief_complaint_hit_sentences

echo "chief complaint evaluating"
python3 src/preprocessing/chief_complaint_evaluator.py --model_id $model_id --gpt_dialogues_dir $gpt_dialogues_dir --chief_complaint_hit_sentences_dir $chief_complaint_hit_sentences_dir

# Step 3: refinement
python3 src/dialogue_synthesis/check.py --sample_dir $sample_dir --model_id $model_id --conv_key "conv" --eval_key "eval"

model_id=gpt-4o
python3 src/dialogue_synthesis/refinement.py --sample_dir $sample_dir  --model_id $model_id

model_id=models/Qwen2.5-32B-Instruct-GPTQ-Int8
python3 src/dialogue_synthesis/check.py --sample_dir $sample_dir --model_id $model_id --conv_key "conv_revised" --eval_key "eval_revised"

echo "done"