#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate sr

source .env

################################################
########### Batch VLM Eval for Intra ###########
################################################

output_dir="result/ea/s1"
dataset_id="kdeng03/intra-image-sr"

vlm_list=(
    # "llava-hf/llama3-llava-next-8b-hf"
    # "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    # "HuggingFaceM4/Idefics3-8B-Llama3"
    # "remyxai/SpaceQwen2.5-VL-3B-Instruct"
    
    "Qwen/Qwen2.5-VL-3B-Instruct"
    # "Qwen/Qwen2.5-VL-7B-Instruct"
    # "Qwen/Qwen2.5-VL-32B-Instruct"
    # "Qwen/Qwen2.5-VL-72B-Instruct"
    # "Qwen/Qwen3-VL-4B-Instruct"
    # "Qwen/Qwen3-VL-8B-Instruct"
    # "Qwen/Qwen3-VL-32B-Instruct"

    # "Qwen/Qwen3-VL-8B-Thinking"
    # "THUDM/GLM-4.1V-9B-Thinking"
    
    # "gpt-4o"
    # "gpt-5"
)

for vlm in "${vlm_list[@]}"; do
    echo "Evaluating VLM: $vlm"
    python eval-ea/intra_vlm.py \
        --output_dir $output_dir \
        --vlm_id $vlm \
        --dataset_id $dataset_id
done


################################################
########### Batch VLM Eval for Cross ###########
################################################

output_dir="result/ea/s2"
dataset_id="kdeng03/cross-image-sr"

vlm_list=(
    # "llava-hf/llama3-llava-next-8b-hf"
    # "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    # "HuggingFaceM4/Idefics3-8B-Llama3"
    # "remyxai/SpaceQwen2.5-VL-3B-Instruct"
    
    "Qwen/Qwen2.5-VL-3B-Instruct"
    # "Qwen/Qwen2.5-VL-7B-Instruct"
    # "Qwen/Qwen2.5-VL-32B-Instruct"
    # "Qwen/Qwen2.5-VL-72B-Instruct"
    # "Qwen/Qwen3-VL-4B-Instruct"
    # "Qwen/Qwen3-VL-8B-Instruct"
    # "Qwen/Qwen3-VL-32B-Instruct"

    # "Qwen/Qwen3-VL-8B-Thinking"
    # "THUDM/GLM-4.1V-9B-Thinking"
    
    # "gpt-4o"
    # "gpt-5"
)

for vlm in "${vlm_list[@]}"; do
    echo "Evaluating VLM: $vlm"
    python eval-ea/cross_vlm.py \
        --output_dir $output_dir \
        --vlm_id $vlm \
        --dataset_id $dataset_id
done


# ################################################
# ######### Batch VLM Eval for Ablation ##########
# ################################################

output_dir="result/ea/s3"
dataset_id="kdeng03/ablation-VRRPI-Diag"

vlm_list=(
    # "llava-hf/llama3-llava-next-8b-hf"
    # "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    # "HuggingFaceM4/Idefics3-8B-Llama3"
    # "remyxai/SpaceQwen2.5-VL-3B-Instruct"
    
    "Qwen/Qwen2.5-VL-3B-Instruct"
    # "Qwen/Qwen2.5-VL-7B-Instruct"
    # "Qwen/Qwen2.5-VL-32B-Instruct"
    # "Qwen/Qwen2.5-VL-72B-Instruct"
    # "Qwen/Qwen3-VL-4B-Instruct"
    # "Qwen/Qwen3-VL-8B-Instruct"
    # "Qwen/Qwen3-VL-32B-Instruct"

    # "Qwen/Qwen3-VL-8B-Thinking"
    # "THUDM/GLM-4.1V-9B-Thinking"
    
    # "gpt-4o"
    # "gpt-5"
)

for vlm in "${vlm_list[@]}"; do
    echo "Evaluating VLM: $vlm"
    python eval-ea/ablation_vlm.py \
        --output_dir $output_dir \
        --vlm_id $vlm \
        --dataset_id $dataset_id
done
