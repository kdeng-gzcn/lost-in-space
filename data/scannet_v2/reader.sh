#!/bin/bash
source $(conda info --base)/bin/activate sr

for dir in data/scannet-v2/scans_test/*; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        # Check if the .sens file exists
        sens_file="$dir/$(basename "$dir").sens"
        if [ -f "$sens_file" ]; then
            echo "Found .sens file: $sens_file"
            # Run the Python script with the .sens file
            python ScanNet/SensReader/python/reader.py --filename "$sens_file" --output_path "$dir" --export_depth_images --export_color_images --export_poses --export_intrinsics
        else
            echo "No .sens file found in $dir"
        fi
    fi
done
# python ScanNet/SensReader/python/reader.py --filename data/scannet-v2/demo/scans/scene0000_00/scene0000_00.sens --output_path data/scannet-v2/demo/scans/scene0000_00/reader_demo --export_depth_images --export_color_images --export_poses --export_intrinsics
# Options:
# --export_depth_images: export all depth frames as 16-bit pngs (depth shift 1000)
# --export_color_images: export all color frames as 8-bit rgb jpgs
# --export_poses: export all camera poses (4x4 matrix, camera to world)
# --export_intrinsics: export camera intrinsics (4x4 matrix)