#!/bin/bash

file_dir="./102_mogu/"
file=$file_dir"prompt.txt"

file_contents=$(cat "$file")

echo "$file_contents"
vol_dir=$file_dir"con_volume_lod_150.pth"
echo "$vol_dir"

python -u launch.py --config custom/threestudio-geodream/configs/geodream-neus.yaml --train --gpu 1 system.prompt_processor.prompt="$file_contents" system.geometry.init_volume_path="$vol_dir"
