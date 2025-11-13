# source ~/.bashrc
# # Initialize Conda environment
# eval "$(conda shell.bash hook)"
# conda activate dpo

# cd /mnt/data/zy/zhenting/final_version/history_taking

model_name="/home/zhouyang/history_taking/models/single_turn_summary_plan"
trainset_name=note_single_turn
margin_min=-1
max_samples=45000

datetime=$(date +"%Y%m%d_%H%M%S")
model_name_basename=$(basename "$model_name" | tr '[:upper:]' '[:lower:]')

output_dir="dpo/outputs_single_turn/dpo_"${model_name_basename}'_'${trainset_name}_${datetime}

mkdir -p $output_dir
echo $output_dir

config_dir=$output_dir/config
mkdir -p $config_dir

adapter_dir=$output_dir/output
mkdir -p $adapter_dir

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --num-processes 8 --main_process_port 29504 \
    dpo/dpo_single_turn.py \
    --model_name $model_name \
    --trainset_name $trainset_name \
    --margin_min $margin_min \
    --max_samples $max_samples \
    --output_dir $output_dir

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
#     --num-processes 8 --main_process_port 29504 \
#     --config-file dpo/configs/my_zero3.yaml \
#     --deepspeed_config_file dpo/configs/ds_bf16_zero3.json \
#     dpo/dpo_single_turn.py \
#     --model_name $model_name \
#     --trainset_name $trainset_name \
#     --margin_min $margin_min \
#     --max_samples $max_samples \
#     --output_dir $output_dir