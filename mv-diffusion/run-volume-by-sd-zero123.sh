# conda activate Geodream
TEXT="$1"
VIEW_PATH="$2"
FILE_NAME=$(echo "$TEXT" | sed 's/ /_/g')

echo "[Save Project Name] "$FILE_NAME
echo "[Reference View Path] GeoDream/mv-diffusion/One-2-3-45-by-view/"$VIEW_PATH

cd One-2-3-45-by-view
CUDA_VISIBLE_DEVICES=0 python run.py \
--model_type "sd-zero123" \
--img_path $VIEW_PATH \
--text "$TEXT" \
--half_precision \
# --save_vis

echo "[Cost Volume save at] : " GeoDream/mv-diffusion/volume/$FILE_NAME