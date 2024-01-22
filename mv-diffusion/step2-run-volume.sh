# conda activate One2345_py38_2
TEXT="$1"
FILE_NAME=$(echo "$TEXT" | sed 's/ /_/g')

if [ -e "volume/"$FILE_NAME ]; then
    echo $FILE_NAME
    echo "[Error] Directory volume/"$FILE_NAME "already exists."
    exit 1
fi

echo "[Save Project Name] "$FILE_NAME

cd One-2-3-45/reconstruction
CUDA_VISIBLE_DEVICES=0 python exp_runner_generic_blender_val.py \
--specific_dataset_name exp/$FILE_NAME \
--mode export_mesh \
--conf confs/one2345_lod0_val_demo.conf \
--resolution 256

directory=../../volume/$FILE_NAME
if [ ! -d "$directory" ]; then
    mkdir "$directory"
    echo "Directory created."
else
    echo "Directory already exists."
fi

cp ../exp/$FILE_NAME/con_volume_lod_150.pth ../../volume/$FILE_NAME/con_volume_lod_150.pth

echo "[Final Save Project Name] : " GeoDream/mv-diffusion/volume/$FILE_NAME
