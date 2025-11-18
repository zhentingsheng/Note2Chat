source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"

conda activate vllm

export PYTHONPATH=./src

base_model_path=models/Qwen2.5-7B-Instruct
adapter_path=adapters/my_multi_turn_qwen2.5_7b_sft_sampling
merged_model_path=models/my_multi_turn_qwen2.5_7b_sft_sampling

python src/merge_adapter.py --base_model_path $base_model_path --adapter_path $adapter_path --merged_model_path $merged_model_path

conda activate dpo

TRAIN_RAW=data/gpt_dialogues
SAMPLING=data/sampling
TARGET=data/trainset/multi_turn_dpo

# Only run if TARGET directory does not exist
if [ ! -d "$TARGET" ]; then
  echo "Target directory $TARGET does not exist. Creating and copying files..."

  mkdir -p "$TARGET"

  # Copy JSON files from train_raw with suffix
  for FILE in "$TRAIN_RAW"/*.json; do
    BASENAME=$(basename "$FILE" .json)
    cp "$FILE" "$TARGET/${BASENAME}_gpt_dialogue.json"
  done

  # Loop through all subdirectories in sampling
  for FOLDER in "$SAMPLING"/*; do
    if [ -d "$FOLDER" ]; then
      FOLDER_NAME=$(basename "$FOLDER")
      for FILE in "$FOLDER"/*.json; do
        if [ -f "$FILE" ]; then
          BASENAME=$(basename "$FILE" .json)
          cp "$FILE" "$TARGET/${BASENAME}_${FOLDER_NAME}.json"
          # echo "Copied $FILE -> ${BASENAME}_${FOLDER_NAME}.json"
        fi
      done
    else
      echo "Warning: $FOLDER is not a directory. Skipping."
    fi
  done

  echo "All JSON files have been copied to $TARGET with folder name suffixes."
else
  echo "Target directory $TARGET already exists. Skipping copy."
fi

note2chat_dir=/home

dataset_dir=$TARGET
model_name=$note2chat_dir/Note2Chat/models/my_multi_turn_qwen2.5_7b_sft_sampling
trainset_name=multi_turn
each_note_max_samples=1

datetime=$(date +"%Y%m%d_%H%M%S")
model_name_basename=$(basename "$model_name" | tr '[:upper:]' '[:lower:]')

output_dir=adapters/my_multi_turn_qwen2.5_7b_sft_sampling_dpo

mkdir -p $output_dir
echo $output_dir

config_dir=$output_dir/config
mkdir -p $config_dir

adapter_dir=$output_dir/output
mkdir -p $adapter_dir

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num-processes 4 --main_process_port 29505 --config-file dpo/configs/zero3.yaml dpo/dpo_multi_turn.py \
    --dataset_dir $dataset_dir \
    --model_name $model_name \
    --trainset_name $trainset_name \
    --each_note_max_samples $each_note_max_samples \
    --output_dir $output_dir