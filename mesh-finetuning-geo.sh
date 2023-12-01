#!/bin/bash

file_dir="/share/project/denghaoge/zeroplus/113_dog/"
file=$file_dir"prompt.txt"

file_contents=$(cat "$file")
echo "$file_contents"

ckpt_dir="/share/project/mabaorui/GeoDream-github/outputs/geodream/A_high_quality_photo_of_a_furry_dog@20231127-181730/"
c_dir="/ckpts/last.ckpt"
python -u launch.py --config configs/geodream-dmtet-geometry.yaml --train system.geometry_convert_from=$ckpt_dir$c_dir --gpu 0 system.prompt_processor.prompt="$file_contents" system.renderer.context_type=cuda system.geometry_convert_override.isosurface_threshold=0.0