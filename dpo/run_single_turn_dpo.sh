source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"

conda activate vllm

export PYTHONPATH=./src

base_model_path=models/Qwen2.5-7B-Instruct
adapter_path=adapters/my_single_turn_qwen2.5_7b_sft_sampling
merged_model_path=models/my_single_turn_qwen2.5_7b_sft_sampling

python src/merge_adapter.py --base_model_path $base_model_path --adapter_path $adapter_path --merged_model_path $merged_model_path

conda activate dpo

note2chat_dir=/home
model_name=$note2chat_dir/Note2Chat/$merged_model_path
trainset_name=note_single_turn
dataset_dir=$note2chat_dir/data/single_turn_sampling
margin_min=-1.0
max_samples=45000
output_dir=adapters/my_single_turn_qwen2.5_7b_sft_sampling_dpo

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num-processes 4 --main_process_port 29505 --config-file dpo/configs/zero3.yaml dpo/dpo_single_turn.py \
    --model_name $model_name \
    --trainset_name $trainset_name \
    --dataset_dir $dataset_dir \
    --margin_min $margin_min \
    --max_samples $max_samples \
    --output_dir $output_dir