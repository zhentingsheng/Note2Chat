source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"

conda activate LLaMA-Factory

cd LLaMA-Factory

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port 25000 \
    src/train.py \
    Note2Chat/sft/configs/single_turn_qwen2.5_7b_instruct_gpt_sampling.yaml