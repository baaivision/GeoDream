import os
import sys
import random
import argparse
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch 
from rembg import remove
import shutil
import cv2

from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1):
    '''
        model, args.size,   t,     uc, sampler, step=50, scale=10,  batch_size=batch_size, ddim_eta=0.0, dtype=dtype, device=device, camera=camera, num_frames=args.num_frames
    '''
    # ---------------------------------------------------------
    # step 1 : pormpt
    if type(prompt)!=list:
        prompt = [prompt]
    
    # ---------------------------------------------------------
    # step 2 : infer
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        # ---------------------------------------------------------
        # step 2.1 : context
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size,1,1)}
        uc_ = {"context": uc.repeat(batch_size,1,1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        # ---------------------------------------------------------
        # step 2.2 : sample
        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=None)
        
        # ---------------------------------------------------------
        # step 2.3 : decode
        x_sample = model.decode_first_stage(samples_ddim)

        # ---------------------------------------------------------
        # step 2.4 : norm 
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # step 1 : args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default="mvdream/configs/sd-v2-base.yaml", help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default="weight/sd-v2.1-base-4view.pt", help="path to local checkpoint")
    parser.add_argument("--text", type=str, default="an astronaut riding a horse")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=4, help="num of frames (views) to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = max(4, args.num_frames)

    # -----------------------------------------------------------------------
    # step 2 : model
    print("load t2i model ... ")
    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    else:
        assert args.ckpt_path is not None, "ckpt_path must be specified!"
        config = OmegaConf.load(args.config_path)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model.device = device
    model.to(device)
    model.eval()

    # -----------------------------------------------------------------------
    # step 3 : sampler
    sampler = DDIMSampler(model)
    uc = model.get_learned_conditioning( [""] ).to(device)
    print("load t2i model done . ")

    # -----------------------------------------------------------------------
    # step 4 : camera
    # pre-compute camera matrices
    # import pdb;pdb.set_trace()
    if args.use_camera:
        camera = get_camera(args.num_frames, elevation=args.camera_elev, 
                azimuth_start=args.camera_azim, azimuth_span=args.camera_azim_span)
        camera = camera.repeat(batch_size//args.num_frames,1).to(device)
    else:
        camera = None
    
    # -----------------------------------------------------------------------
    # step 5 : t2i
    t = args.text + args.suffix
    set_seed(args.seed)
    images = []
    # for j in range(3):
    img = t2i(model, args.size, t, uc, sampler, step=50, scale=10, batch_size=batch_size, ddim_eta=0.0, 
            dtype=dtype, device=device, camera=camera, num_frames=args.num_frames)
    img = np.concatenate(img, 1)
    images = np.array_split(img, args.num_frames, axis=1)
    
    # images = np.concatenate(images, 0)
    
    os.makedirs("remove_bg", exist_ok=True)
    for idx, image in enumerate(images):
        Image.fromarray(image).save(f"{idx}.png")

    
    # -----------------------------------------------------------------------
    # step 6 : remove bg
    threshold = 128
    for i in range(4):
        input_path = f'{i}.png'
        output_path = f'remove_bg/{(i+1)%4}.png'

        input_img = cv2.imread(input_path)
        output = remove(input_img)
        output = np.array(output)

        condition = output[:, :, 3] <= threshold
        output[condition, :3] = 255
                
        cv2.imwrite(output_path, output[:, :, :3])
        os.remove(input_path)