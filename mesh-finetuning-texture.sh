#!/bin/bash

file_dir="/share/project/denghaoge/zeroplus/111_building/"
file=$file_dir"prompt.txt"
echo "$file_contents"

file_contents=$(cat "$file")

echo "$file_contents"
ckpt_dir='/share/project/mabaorui/GeoDream-github/outputs/geodream-dmtnet-geometry/3d_stylized_game_little_building@20231127-181204'
c_dir='/ckpts/last.ckpt'
python -u launch.py --config configs/geodream-dmtet-texture.yaml system.geometry.isosurface_resolution=256 --train data.batch_size=2 system.renderer.context_type=cuda --gpu 0 system.geometry_convert_from="$ckpt_dir$c_dir" system.prompt_processor.prompt="$file_contents" \