source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"

conda activate vllm
export PYTHONPATH=./src

dataset_dir=data/single_turn_gpt_dialogues_and_sampling_best_rollout
trainset_name=note2chat_single_turn_sft_gpt
trainset_path=data/trainset/$note2chat_single_turn_sft_gpt.json

python src/dataset_generate/single_turn_generate_dataset_sft.py --dataset_dir $dataset_dir --trainset_path $trainset_path --with_sampling

data_info_path=/home/LLaMA-Factory/data/dataset_info.json

python3 src/dataset_generate/update_data_info.py --trainset_name $trainset_name --filename /home/Note2Chat/$trainset_path --formatting alpaca --data_info_path $data_info_path

conda activate LLaMA-Factory

cd LLaMA-Factory

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port 25000 \
    src/train.py \
    Note2Chat/sft/configs/single_turn_qwen2.5_7b_instruct_gpt.yaml