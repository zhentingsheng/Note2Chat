source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate dpo

cd /mnt/data/zy/zhenting/final_version/history_taking


ROOT_DIR="data/dataset/multi_turn"
TRAIN_RAW="$ROOT_DIR/train_raw"
SAMPLING="$ROOT_DIR/sampling"
TARGET="$ROOT_DIR/dpo"

# Only run if TARGET directory does not exist
if [ ! -d "$TARGET" ]; then
  echo "Target directory $TARGET does not exist. Creating and copying files..."

  mkdir -p "$TARGET"

  # Copy JSON files from train_raw with suffix
  for FILE in "$TRAIN_RAW"/*.json; do
    BASENAME=$(basename "$FILE" .json)
    cp "$FILE" "$TARGET/${BASENAME}_train_raw.json"
    # echo "Copied $FILE -> ${BASENAME}_train_raw.json"
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


dataset_dir=$TARGET
model_name=/mnt/data/zy/zhenting/final_version/history_taking/models/multi_turn_qwen2.5_7b_instruct_gpt
# len_penalty=0.025
trainset_name=multi_turn
# margin_min=0.3
# repeated_penalty=0.1
each_note_max_samples=1

datetime=$(date +"%Y%m%d_%H%M%S")
model_name_basename=$(basename "$model_name" | tr '[:upper:]' '[:lower:]')

# output_dir="outputs/dpo_"${model_name_basename}'_'${trainset_name}_${datetime}
output_dir="dpo/outputs/dpo_multi_turn_qwen2.5_7b_instruct_gpt_${each_note_max_samples}_${datetime}"

mkdir -p $output_dir
echo $output_dir

config_dir=$output_dir/config
mkdir -p $config_dir

adapter_dir=$output_dir/output
mkdir -p $adapter_dir

CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --num-processes 4 --main_process_port 29505 --config-file dpo/configs/zero3.yaml dpo/dpo_multi_turn.py \
    --dataset_dir $dataset_dir \
    --model_name $model_name \
    --trainset_name $trainset_name \
    --each_note_max_samples $each_note_max_samples \
    --output_dir $output_dir
    # --len_penalty $len_penalty \
    # --repeated_penalty $repeated_penalty\
    # --output_dir $output_dir > $output_dir/train.log 2>&1
    # --output_dir $output_dir > $output_dir/train.log 2>&1