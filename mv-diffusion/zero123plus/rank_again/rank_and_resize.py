import shutil
import os

import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
from rembg import remove

# step 1 : args
parser = argparse.ArgumentParser()
parser.add_argument("--project_name", type=str, default="cat", help="saved project name")
args = parser.parse_args()

img_name = "tmp"

# create path
print('[Zero123 Plus : Create path...]')
save_name = f'../result/{args.project_name}'
if os.path.exists(save_name):
    shutil.rmtree(save_name)
os.makedirs(save_name)

destination_folder = f'../result/{img_name}'
if os.path.exists(destination_folder):
    shutil.rmtree(destination_folder)
os.makedirs(destination_folder)

# mvdreamer
print('[Zero123 Plus : Pick up images from MVDreamer...]')
source_name = ['0.png']*6 + [f'{i}.png' for i in range(4)]
new_filename = ['0.png']+ [f'{i}.png' for i in range(1, 6)] + [f'{i*7+12}.png' for i in range(4)]

for i in range(len(source_name)):
    source_file = "../../MVDream/remove_bg/"+source_name[i]
    shutil.copy(source_file, os.path.join(destination_folder, new_filename[i]))

# zeroplus
# 6~11
print('[Zero123 Plus : Pick up images from Zero123 Plus...]')
root_path = '../../zero123plus/result/'
folder_path = root_path+'0'
source_names =  [f'{folder_path}/{i}.png' for i in range(1, 7)]
new_filename = [f'{6+i}.png' for i in range(6)]

for i in range(len(source_names)):
    source_name = source_names[i]
    shutil.copy(source_name, os.path.join(destination_folder, new_filename[i]))

# 13~18
root_path = '../../zero123plus/result/'
folder_path = root_path+'0'
source_names =  [f'{folder_path}/{i}.png' for i in range(1, 7)]
new_filename = [f'{13+i}.png' for i in range(6)]

for i in range(len(source_names)):
    source_name = source_names[i]
    shutil.copy(source_name, os.path.join(destination_folder, new_filename[i]))

# 20~25
root_path = '../../zero123plus/result/'
folder_path = root_path+'1'
source_names =  [f'{folder_path}/{i}.png' for i in range(1, 7)]
new_filename = [f'{20+i}.png' for i in range(6)]

for i in range(len(source_names)):
    source_name = source_names[i]
    shutil.copy(source_name, os.path.join(destination_folder, new_filename[i]))

# 27~32
root_path = '../../zero123plus/result/'
i = 2
folder_path = root_path+'2'
source_names =  [f'{folder_path}/{i}.png' for i in range(1, 7)]
new_filename = [f'{27+i}.png' for i in range(6)]

for i in range(len(source_names)):
    source_name = source_names[i]
    shutil.copy(source_name, os.path.join(destination_folder, new_filename[i]))

# 34~39
root_path = '../../zero123plus/result/'
folder_path = root_path+'3'
source_names =  [f'{folder_path}/{i}.png' for i in range(1, 7)]
new_filename = [f'{34+i}.png' for i in range(6)]

for i in range(len(source_names)):
    source_name = source_names[i]
    shutil.copy(source_name, os.path.join(destination_folder, new_filename[i]))


# resize and remove background
print('[Zero123 Plus : Resize and remove background...]')
threshold = 128
for i in range(40):
    image = Image.open(f'../result/{img_name}/{i}.png')

    # resize
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])
    resized_image = transform(image)

    # remove background
    resized_image.save(f'../result/{img_name}/{i}.png')
    input = cv2.imread(f'../result/{img_name}/{i}.png')
    output = remove(input)
    output = np.array(output)
    condition = output[:, :, 3] <= threshold
    output[condition, :3] = 255
            
    # save
    cv2.imwrite(f'../result/{args.project_name}/{i}.png', output[:, :, :3])

shutil.rmtree(f'../result/{img_name}')

# adapt to one2345
print('[Zero123 Plus : Adapt to one2345...]')
image_dir = f"../result/{args.project_name}"
output_dir_base = f"../result/{args.project_name}"

for stage in range(1, 3):
    output_dir = f"{output_dir_base}/stage{stage}_8"
    os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        index = int(filename.split(".")[0])
        stage = 1 if index < 8 else 2
        source_path = os.path.join(image_dir, filename)
        dest_path = os.path.join(output_dir_base, f"stage{stage}_8", filename)
        shutil.move(source_path, dest_path)