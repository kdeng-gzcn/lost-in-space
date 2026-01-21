#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate sr

source .env

#################### Bench ######################

# 7 Scenes
# ----------------------------------------
yaml_path=config/bench/ss_tau_15_v1.yaml
# yaml_path=config/bench/ss_tau_30_v1.yaml
# yaml_path=config/bench/ss_tau_45_v1.yaml
# yaml_path=config/bench/ss_tau_60_v3.yaml

python bench-gen/bench/ss_filter.py \
    --output_dir benchmark/bench \
    --data_dir data/rgb-d-dataset-7-scenes \
    --yaml_path $yaml_path


# ScanNet
# ----------------------------------------
# yaml_path=config/bench/sn_tau_15_v1.yaml
# yaml_path=config/bench/sn_tau_30_v1.yaml
# yaml_path=config/bench/sn_tau_45_v1.yaml
# yaml_path=config/bench/sn_tau_60_v1.yaml

# python bench-gen/bench/sn_filter.py \
#     --output_dir benchmark/bench \
#     --data_dir data/scannet-v2/scans_test \
#     --yaml_path $yaml_path


#################### Diag ######################

# 7 Scenes
# ----------------------------------------
# yaml_path=config/diag/ss_v1.yaml

# python bench-gen/diag/ss_filter.py \
#     --output_dir benchmark/diag \
#     --data_dir data/rgb-d-dataset-7-scenes \
#     --yaml_path $yaml_path

# ScanNet
# ----------------------------------------
# yaml_path=config/diag/sn_v1.yaml

# python bench-gen/diag/sn_filter.py \
#     --output_dir benchmark/diag \
#     --data_dir data/scannet-v2/scans_test \
#     --yaml_path $yaml_path

# ScanNet++
# ----------------------------------------
# yaml_path=config/diag/snpp_v1.yaml

# python bench-gen/diag/snpp_filter.py \
#     --output_dir benchmark/diag \
#     --data_dir data/scannetpp \
#     --yaml_path $yaml_path