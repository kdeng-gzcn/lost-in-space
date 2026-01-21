#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate sr

source .env

dataset_id="kdeng03/VRRPI-Bench"
output_dir="result/bench"
model=(
    SIFT
    LoFTR
)

for m in "${model[@]}"; do
    echo "Evaluating: $m"
    python eval-bench/cv_eval.py \
        --output_dir $output_dir \
        --model $m \
        --dataset_id $dataset_id
done