#!/bin/bash

root_dir="/home/mabaorui2023/threestudio-20231027/outputs/prolificdreamer-texture/a_DSLR_photo_of_a_squirrel_playing_guitar@20231122-202437/"
config_dir="/configs/parsed.yaml"
resume_dir="/ckpts/last.ckpt"
python -u launch.py --config "$root_dir$config_dir" --test --gpu 0 resume="$root_dir$resume_dir" 