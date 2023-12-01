import torch
import requests
from PIL import Image
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import rembg
from tqdm import tqdm
import os

def crop_and_save(image, root_path):
    width, height = image.size
    subimage_width = width // 2
    subimage_height = height // 3
    subimages = []
    
    for j in range(3):
        for i in range(2):
            left = i * subimage_width
            upper = j * subimage_height
            right = left + subimage_width
            lower = upper + subimage_height
            subimage = image.crop((left, upper, right, lower))
            subimages.append(subimage)

    for idx, subimage in enumerate(subimages):
        subimage.save(root_path + f"/{idx + 1}.png")


# args
img_names = ['{i}.png'.format(i=i) for i in range(4)]

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "../weight/zero123plus-v1.1", 
    custom_pipeline="../weight/zero123plus-pipeline",
    torch_dtype=torch.float16
)
# Feel free to tune the scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
pipeline.to('cuda:0')

for img_name in tqdm(img_names):
    # import pdb;pdb.set_trace()
    cond = Image.open("../img/remove_bg/" + img_name)
    result = pipeline(cond, num_inference_steps=75).images[0]
    
    # result.save(f"../result/{img_name.split('.')[0]}_mv.png")
    # os.makedirs(f"../result/{img_name.split('.')[0]}", exist_ok=True)
    # crop_and_save(result, f"../result/{img_name.split('.')[0]}")

    result = rembg.remove(result)
    os.makedirs(f"../result/{img_name.split('.')[0]}", exist_ok=True)
    crop_and_save(result, f"../result/{img_name.split('.')[0]}")