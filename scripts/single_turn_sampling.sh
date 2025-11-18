source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

model_id=models/Qwen2.5-7B-Instruct
lora_path=adapters/my_single_turn_qwen2.5_7b_sft_sampling
dataset_dir=data/single_turn_gpt_dialogues_and_sampling_best_rollout
output_dir=data/single_turn_sampling
world_size=4

export PYTHONPATH=./src


for rank in $(seq 0 $((world_size - 1)))
do
    echo "Running with rank $rank..."
    
    python src/dialogue_synthesis/single_turn/single_turn_sampling.py \
        --model_id $model_id \
        --lora_path $lora_path \
        --dataset_dir $dataset_dir \
        --output_dir $output_dir \
        --rank $rank \
        --world_size $world_size &
done

# Wait for all background jobs to finish
wait

echo "All processes have finished."
