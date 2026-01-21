#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate sr

source .env

################################################
########### Batch VLM Eval for Bench ###########
################################################
dataset_id="kdeng03/VRRPI-Bench"
output_dir="result/bench"

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

prompt_mode=(
    "zero-shot"
    # "w/ trap"
)

for vlm_id in "${vlm_list[@]}"; do
    for pm in "${prompt_mode[@]}"; do
        echo "Evaluating VLM: $vlm_id with prompt mode: $pm"
        python eval-bench/vlm_eval.py \
            --output_dir $output_dir \
            --vlm_id $vlm_id \
            --dataset_id $dataset_id \
            --prompt_mode "${pm}"
    done
done

################################################
############ Batch VLM Eval for Diag ###########
################################################

# output_dir="result/diag"
# dataset_id="kdeng03/VRRPI-Diag"

# vlm_list=(
#     # "llava-hf/llama3-llava-next-8b-hf"
#     # "llava-hf/llava-onevision-qwen2-7b-ov-hf"
#     # "HuggingFaceM4/Idefics3-8B-Llama3"
#     # "remyxai/SpaceQwen2.5-VL-3B-Instruct"
    
#     "Qwen/Qwen2.5-VL-3B-Instruct"
#     # "Qwen/Qwen2.5-VL-7B-Instruct"
#     # "Qwen/Qwen2.5-VL-32B-Instruct"
#     # "Qwen/Qwen2.5-VL-72B-Instruct"
#     # "Qwen/Qwen3-VL-4B-Instruct"
#     # "Qwen/Qwen3-VL-8B-Instruct"
#     # "Qwen/Qwen3-VL-32B-Instruct"

#     # "Qwen/Qwen3-VL-8B-Thinking"
#     # "THUDM/GLM-4.1V-9B-Thinking"
    
#     # "gpt-4o"
#     # "gpt-5"
# )

# prompt_mode=(
#     "zero-shot"
#     # "CoT"
# )

# for vlm in "${vlm_list[@]}"; do
#     for pm in "${prompt_mode[@]}"; do
#         echo "Evaluating VLM: $vlm with prompt mode: $pm"
#         python eval-diag/vlm.py \
#             --output_dir $output_dir \
#             --vlm_id $vlm \
#             --dataset_id $dataset_id \
#             --prompt_mode "${pm}"
#     done
# done