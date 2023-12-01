#!/bin/bash

resume_dir_root="/home/mabaorui2023/threestudio-20231027/outputs/prolificdreamer-texture/A_bald_eagle_carved_out_of_wood.@20231103-225632"
config_dir="/configs/parsed.yaml"
ckpt_dir="/ckpts/epoch=0-step=15000.ckpt"

python -u launch.py --config $resume_dir_root$config_dir  --export --gpu 0 resume=$resume_dir_root$ckpt_dir system.exporter_type=mesh-exporter system.exporter.context_type=cuda