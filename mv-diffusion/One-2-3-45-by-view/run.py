import os
import shutil
import torch
import argparse
from PIL import Image
from utils.zero123_utils import init_model, predict_stage1_gradio, zero123_infer
from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import pred_bbox, image_preprocess_nosave, gen_poses, convert_mesh_format



def preprocess(predictor, raw_im, lower_contrast=False):
    raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), pred_bbox(raw_im))
    input_256 = image_preprocess_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
    torch.cuda.empty_cache()
    return input_256

def stage1_run(model, device, exp_dir,
               input_im, scale, ddim_steps):
    stage1_dir = os.path.join(exp_dir, "stage1_8")
    os.makedirs(stage1_dir, exist_ok=True)

    output_ims = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale)

    stage2_steps = 50
    zero123_infer(model, exp_dir, indices=[0], device=device, ddim_steps=stage2_steps, scale=scale)

    polar_angle = 90
    gen_poses(exp_dir, polar_angle)

    output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(8,12)), device=device, ddim_steps=ddim_steps, scale=scale)
    torch.cuda.empty_cache()
    return 90-polar_angle, output_ims+output_ims_2
    
def stage2_run(model, device, exp_dir,
               elev, scale, stage2_steps=50):
    if 90-elev <= 75:
        zero123_infer(model, exp_dir, indices=list(range(1,8)), device=device, ddim_steps=stage2_steps, scale=scale)
    else:
        zero123_infer(model, exp_dir, indices=list(range(1,4))+list(range(8,12)), device=device, ddim_steps=stage2_steps, scale=scale)

def reconstruct(exp_dir, text="", output_format=".ply", device_idx=0, 
    resolution=256, save_vis=False):
    exp_dir = os.path.abspath(exp_dir)
    main_dir_path = os.path.abspath(os.path.dirname("./"))
    os.chdir('reconstruction/')

    if save_vis:
        bash_script = f'CUDA_VISIBLE_DEVICES={device_idx} python exp_runner_generic_blender_val.py \
                        --specific_dataset_name {exp_dir} \
                        --mode export_mesh \
                        --conf confs/one2345_lod0_val_demo.conf \
                        --resolution {resolution} \
                        --save_vis'
    else :
        bash_script = f'CUDA_VISIBLE_DEVICES={device_idx} python exp_runner_generic_blender_val.py \
                        --specific_dataset_name {exp_dir} \
                        --mode export_mesh \
                        --conf confs/one2345_lod0_val_demo.conf \
                        --resolution {resolution}'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(exp_dir)))

    src_path = exp_dir
    tgt_path = os.path.join(parent_dir, f"volume")
    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)
    tgt_path = os.path.join(tgt_path, f"{text.replace(' ', '_')}")
    shutil.move(src_path, tgt_path)



def predict_multiview(shape_dir, args):
    device = f"cuda:{args.gpu_idx}"
    models = init_model(device, 'zero123-xl.ckpt', half_precision=args.half_precision)
    model_zero123 = models["turncam"]
    
    predictor = sam_init(args.gpu_idx)
    input_raw = Image.open(args.img_path)

    input_256 = preprocess(predictor, input_raw)

    elev, stage1_imgs = stage1_run(model_zero123, device, shape_dir, input_256, scale=3, ddim_steps=75)
    stage2_run(model_zero123, device, shape_dir, elev, scale=3, stage2_steps=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--img_path', type=str, default="./demo/demo_examples/01_wild_hydrant.png", help='Path to the input image')
    parser.add_argument('--text', type=str, default="An astronaut riding a horse", help='prompt text')
    parser.add_argument('--mesh_output_path', type=str, default="", help='Path to the input image')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--half_precision', action='store_true', help='Use half precision')
    parser.add_argument('--mesh_resolution', type=int, default=256, help='Mesh resolution')
    parser.add_argument('--output_format', type=str, default=".ply", help='Output format: .ply, .obj, .glb')
    parser.add_argument('--save_vis', default=False, action="store_true")

    args = parser.parse_args()

    assert(torch.cuda.is_available())

    shape_id = args.img_path.split('/')[-1].split('.')[0]
    shape_dir = f"./exp/{shape_id}"
    os.makedirs(shape_dir, exist_ok=True)

    predict_multiview(shape_dir, args)

    reconstruct(
        shape_dir, 
        text=args.text,
        output_format=args.output_format, 
        device_idx=args.gpu_idx, 
        resolution=args.mesh_resolution,
        save_vis=args.save_vis)
