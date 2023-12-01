#!/bin/bash

### mv diffusion -> file_dir

file_dir="/share/project/denghaoge/janus_problem/40_A_photograph_of_an_astronaut_riding_a_horse_2022-08-28/"
file=$file_dir"prompt.txt"

file_contents=$(cat "$file")

echo "$file_contents"
vol_dir=$file_dir"con_volume_lod_150.pth"
echo "$vol_dir"

python -u launch.py --config configs/geodream-neus.yaml --train --gpu 0 system.prompt_processor.prompt="$file_contents" system.geometry.init_volume_path="$vol_dir"