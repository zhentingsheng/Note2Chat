source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"

conda activate vllm
export PYTHONPATH=./src

dataset_dir=data/gpt_dialogues
trainset_name=note2chat_multi_turn_sft_gpt
trainset_path=data/trainset/$trainset_name.json
conv_key=conv_revised

python3 src/dataset_generate/generate_dataset_sft.py --dataset_dir $dataset_dir --trainset_path $trainset_path --conv_key $conv_key

data_info_path=/home/LLaMA-Factory/data/dataset_info.json

python3 src/dataset_generate/update_data_info.py --trainset_name $trainset_name --filename /home/Note2Chat/$trainset_path --formatting sharegpt --data_info_path $data_info_path


cd /home/LLaMA-Factory

conda activate llamafactory

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port 25000 \
    src/train.py \
    /home/Note2Chat/sft/configs/multi_turn_qwen2.5_7b_instruct_gpt.yaml