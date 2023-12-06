# conda activate Geodream
TEXT="$1"
VIEW_PATH="$2"
FILE_NAME=$(echo "$TEXT" | sed 's/ /_/g')

echo "[Save Project Name] "$FILE_NAME
echo "[Reference View Path] GeoDream/mv-diffusion/One-2-3-45-by-view/"$VIEW_PATH

cd One-2-3-45-by-view
CUDA_VISIBLE_DEVICES=0 python run.py \
--img_path $VIEW_PATH \
--text "$TEXT" \
--half_precision 

echo "[Cost Volume save at] : " GeoDream/volume/$FILE_NAME