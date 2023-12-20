# mv-diffusion
- The purpose of the current directory is predicting source views.

## Installation
### Manually Install Using `pip`.
```bash
conda create --name geodream_mv python=3.8
conda activate geodream_mv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install inplace_abn
sudo apt-get install libsparsehash-dev
pip install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
pip install -r requirements.txt

```

## Model Card
### Manually download.
| Model      | Weight Link | Path |
| ----------- | ----------- | ----------- |
| MVdream   | [sd-v2.1-base-4view.pt](https://huggingface.co/MVDream/MVDream/blob/main/sd-v2.1-base-4view.pt) | GeoDream/mv-diffusion/MVDream/weight/sd-v2.1-base-4view.pt
| zero123plus        | [zero123plus-v1.1](https://huggingface.co/sudo-ai/zero123plus-v1.1/tree/main)             | GeoDream/mv-diffusion/zero123plus/weight/zero123plus-v1.1
| zero123plus        | [zero123plus-pipeline](https://huggingface.co/sudo-ai/zero123plus-pipeline/tree/main)             | GeoDream/mv-diffusion/zero123plus/weight/zero123plus-pipeline
| CLIP-ViT-H-14-laion2B-s32B-b79K        | [CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)             | GeoDream/mv-diffusion/MVDream/CLIP-ViT-H-14-laion2B-s32B-b79K
| Zero123-xl        | [Zero123-xl](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)             | GeoDream/mv-diffusion/One-2-3-45-by-view/zero123-xl.ckpt
| SAM        | [SAM](https://huggingface.co/One-2-3-45/code/resolve/main/sam_vit_h_4b8939.pth)             | GeoDream/mv-diffusion/One-2-3-45-by-view/sam_vit_h_4b8939.pth

## Run
### Construct cost volume by prompt
```bash
conda activate geodream_mv
cd GeoDream/mv-diffusion
sh step1-run-mv.sh "An astronaut riding a horse"
conda deactivate

. venv/bin/activate
sh step2-run-volume.sh "An astronaut riding a horse"
```
---

### Construct cost volume by reference view
```bash
conda activate geodream_mv
cd GeoDream/mv-diffusion
# use zero123
sh run-volume-by-zero123 "An astronaut riding a horse" "ref_imges/demo.png"
conda deactivate
```


## Acknowledgement
This repository is heavily based on [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), [MVdream](https://github.com/bytedance/MVDream), [One-2-3-45](https://github.com/One-2-3-45/One-2-3-45), [zero123plus](https://github.com/SUDO-AI-3D/zero123plus). We would like to thank the authors of these work for publicly releasing their code.