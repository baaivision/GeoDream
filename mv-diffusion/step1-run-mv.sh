#!/bin/bash
# conda activate Geodream

# MVdream : text -> 4 views
TEXT="$1"
FILE_NAME=$(echo "$TEXT" | sed 's/ /_/g')

cd MVDream
echo "[Text Prompt] "$TEXT
echo "[Save Project Name] "$FILE_NAME

echo "[Start] MVDream"
python t2i_single.py \
--text "$TEXT" \
--num_frames 4 \
--camera_elev 15 \
--camera_azim 180 \
--camera_azim_span 360

# zero123 plus : 4 views -> 40 views
echo "[Start] zero123plus"
cp -r remove_bg ../zero123plus/img
cd ../zero123plus/examples
python img_to_6mv.py

cd ../rank_again
python rank_and_resize.py --project_name $FILE_NAME

cd ../../zero123plus/result
cp -r $FILE_NAME ../../One-2-3-45/exp
cd ../../One-2-3-45/exp
cp pose.json $FILE_NAME/pose.json
