#!/bin/bash

cd data/rgb-d-dataset-7-scenes

wget https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip
wget https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip
wget https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip
wget https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip
wget https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip
wget https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip
wget https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip

for f in *.zip; do
    unzip "$f" -d "./"
    dir_name="${f%.zip}"
    cd "$dir_name"
    for subf in *.zip; do
        unzip "$subf" -d "./"
        rm "$subf"
    done
    cd ..
    rm "$f"
done

echo "All zip files have been extracted."