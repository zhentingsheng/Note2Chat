source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"

conda activate vllm
export PYTHONPATH=./src

llamafactory_dir=/home
note2chat_dir=/home

dataset_dir=data/single_turn_gpt_dialogues_and_sampling_best_rollout
trainset_name=note2chat_single_turn_sft_gpt_sampling
trainset_path=data/trainset/$trainset_name.json

python src/dataset_generate/single_turn_generate_dataset_sft.py --dataset_dir $dataset_dir --trainset_path $trainset_path --with_sampling true

data_info_path=$llamafactory_dir/LLaMA-Factory/data/dataset_info.json

python3 src/dataset_generate/update_data_info.py --trainset_name $trainset_name --filename $note2chat_dir/Note2Chat/$trainset_path --formatting alpaca --data_info_path $data_info_path

conda activate llamafactory

cd $llamafactory_dir/LLaMA-Factory

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --master_port 25000 \
    src/train.py \
    $note2chat_dir/Note2Chat/sft/configs/single_turn_qwen2.5_7b_instruct_gpt_sampling.yaml