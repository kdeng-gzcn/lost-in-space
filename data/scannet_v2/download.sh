#!/bin/bash
source $(conda info --base)/bin/activate sr

test_split_path="ScanNet/Tasks/Benchmark/scannetv2_test.txt"

while IFS= read -r id; do
    python data/scannet-v2/download-scannet.py -o data/scannet-v2 --id "$id" --type .sens
done < "$test_split_path"