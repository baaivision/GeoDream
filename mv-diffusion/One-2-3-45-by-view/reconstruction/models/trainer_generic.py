"""
decouple the trainer with the renderer
"""
import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import trimesh

from utils.misc_utils import visualize_depth_numpy

from utils.training_utils import numpy2tensor

from loss.depth_loss import DepthLoss, DepthSmoothLoss

from models.sparse_neus_renderer import SparseNeuSRenderer


class GenericTrainer(nn.Module):
    def __init__(self,
                 rendering_network_outside,
                 pyramid_feature_network_lod0,
                 pyramid_feature_network_lod1,
                 sdf_network_lod0,
                 sdf_network_lod1,
                 variance_network_lod0,
                 variance_network_lod1,
                 rendering_network_lod0,
                 rendering_network_lod1,
                 n_samples_lod0,
                 n_importance_lod0,
                 n_samples_lod1,
                 n_importance_lod1,
                 n_outside,
                 perturb,
                 alpha_type='div',
                 conf=None,
                 timestamp="",
                 mode='train',
                 base_exp_dir=None,
                 ):
        super(GenericTrainer, self).__init__()

        self.conf = conf
        self.timestamp = timestamp

        
        self.base_exp_dir = base_exp_dir 
        

        self.anneal_start = self.conf.get_float('train.anneal_start', default=0.0)# 0
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)# 25000
        self.anneal_start_lod1 = self.conf.get_float('train.anneal_start_lod1', default=0.0)# 0
        self.anneal_end_lod1 = self.conf.get_float('train.anneal_end_lod1', default=0.0)# 15000

        # network setups
        self.rendering_network_outside = rendering_network_outside# !!!!None
        self.pyramid_feature_network_geometry_lod0 = pyramid_feature_network_lod0  # 2D pyramid feature network for geometry
        self.pyramid_feature_network_geometry_lod1 = pyramid_feature_network_lod1  # use differnet networks for the two lods !!!!!None

        # when num_lods==2, may consume too much memeory
        self.sdf_network_lod0 = sdf_network_lod0# 有网络
        self.sdf_network_lod1 = sdf_network_lod1# None

        # - warpped by ModuleList to support DataParallel
        self.variance_network_lod0 = variance_network_lod0
        self.variance_network_lod1 = variance_network_lod1# None

        self.rendering_network_lod0 = rendering_network_lod0
        self.rendering_network_lod1 = rendering_network_lod1# None

        self.n_samples_lod0 = n_samples_lod0# 64
        self.n_importance_lod0 = n_importance_lod0# 64
        self.n_samples_lod1 = n_samples_lod1# 64
        self.n_importance_lod1 = n_importance_lod1# 64
        self.n_outside = n_outside# 0
        self.num_lods = conf.get_int('model.num_lods')  # the number of octree lods 1
        self.perturb = perturb
        self.alpha_type = alpha_type

        # - the two renderers
        self.sdf_renderer_lod0 = SparseNeuSRenderer(
            self.rendering_network_outside,# None
            self.sdf_network_lod0,# 有网络
            self.variance_network_lod0,# 有网络
            self.rendering_network_lod0,# 有网络
            self.n_samples_lod0,
            self.n_importance_lod0,
            self.n_outside,
            self.perturb,
            alpha_type='div',
            conf=self.conf)

        self.sdf_renderer_lod1 = SparseNeuSRenderer(
            self.rendering_network_outside,
            self.sdf_network_lod1,# None
            self.variance_network_lod1,# None
            self.rendering_network_lod1,# None
            self.n_samples_lod1,
            self.n_importance_lod1,
            self.n_outside,
            self.perturb,
            alpha_type='div',
            conf=self.conf)

        self.if_fix_lod0_networks = self.conf.get_bool('train.if_fix_lod0_networks')# False

        # sdf network weights
        self.sdf_igr_weight = self.conf.get_float('train.sdf_igr_weight')# 0.1 
        self.sdf_sparse_weight = self.conf.get_float('train.sdf_sparse_weight', default=0)# 0.02
        self.sdf_decay_param = self.conf.get_float('train.sdf_decay_param', default=100)# 100
        self.fg_bg_weight = self.conf.get_float('train.fg_bg_weight', default=0.00)# 0.1
        self.bg_ratio = self.conf.get_float('train.bg_ratio', default=0.0)# 0.3

        self.depth_loss_weight = self.conf.get_float('train.depth_loss_weight', default=1.00)# 0

        print("depth_loss_weight: ", self.depth_loss_weight)
        self.depth_criterion = DepthLoss()

        # - DataParallel mode, cannot modify attributes in forward()
        # self.iter_step = 0
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')

        # - True for finetuning; False for general training
        self.if_fitted_rendering = self.conf.get_bool('train.if_fitted_rendering', default=False)

        self.prune_depth_filter = self.conf.get_bool('model.prune_depth_filter', default=False)

        self.save_volume = self.conf.get_bool('train.save_volume', default=False)

    
    def get_trainable_params(self):
        # set trainable params

        self.params_to_train = []

        if not self.if_fix_lod0_networks:
            #  load pretrained featurenet
            self.params_to_train += list(self.pyramid_feature_network_geometry_lod0.parameters())
            self.params_to_train += list(self.sdf_network_lod0.parameters())
            self.params_to_train += list(self.variance_network_lod0.parameters())

            if self.rendering_network_lod0 is not None:
                self.params_to_train += list(self.rendering_network_lod0.parameters())

        if self.sdf_network_lod1 is not None:
            #  load pretrained featurenet
            self.params_to_train += list(self.pyramid_feature_network_geometry_lod1.parameters())

            self.params_to_train += list(self.sdf_network_lod1.parameters())
            self.params_to_train += list(self.variance_network_lod1.parameters())
            if self.rendering_network_lod1 is not None:
                self.params_to_train += list(self.rendering_network_lod1.parameters())

        return self.params_to_train


    def export_mesh_step(self, sample,
                        iter_step=0,
                        chunk_size=512,
                        resolution=360,
                        save_vis=False,
                        ):
        # * only support batch_size==1
        # ! attention: the list of string cannot be splited in DataParallel
        batch_idx = sample['batch_idx'][0]
        meta = sample['meta'][batch_idx] 

        sizeW = sample['img_wh'][0][0]# 256
        sizeH = sample['img_wh'][0][1]# 256
        H, W = sizeH, sizeW

        partial_vol_origin = sample['partial_vol_origin']
        near, far = sample['query_near_far'][0, :1], sample['query_near_far'][0, 1:]

        # the ray variables
        sample_rays = sample['rays']
        rays_o = sample_rays['rays_o'][0]
        rays_d = sample_rays['rays_v'][0]

        imgs = sample['images'][0]
        intrinsics = sample['intrinsics'][0]
        intrinsics_l_4x = intrinsics.clone()
        intrinsics_l_4x[:, :2] *= 0.25
        w2cs = sample['w2cs'][0]
        proj_matrices = sample['affine_mats']

        scale_mat = sample['scale_mat'] 
        trans_mat = sample['trans_mat']
        query_c2w = sample['query_c2w']
        query_w2c = sample['query_w2c'] 
        
        with torch.no_grad():
            geometry_feature_maps = self.obtain_pyramid_feature_maps(imgs, lod=0)# torch.Size([32, 56, 256, 256])

            conditional_features_lod0 = self.sdf_network_lod0.get_conditional_volume(
                feature_maps=geometry_feature_maps[None, :, :, :, :],
                partial_vol_origin=partial_vol_origin,
                proj_mats=proj_matrices,
                sizeH=sizeH,
                sizeW=sizeW,
                lod=0,
            )
        con_volume_lod0 = conditional_features_lod0['dense_volume_scale0']
        con_valid_mask_volume_lod0 = conditional_features_lod0['valid_mask_volume_scale0']
        coords_lod0 = conditional_features_lod0['coords_scale0']  # [1,3,wX,wY,wZ] torch.Size([1, 3, 96, 96, 96])

        # print(f"cost volume save at {self.base_exp_dir}")
        coords_lod0_150 = F.interpolate(con_volume_lod0, size=(150, 150, 150), mode='trilinear', align_corners=False)
        torch.save(coords_lod0_150, self.base_exp_dir+'/con_volume_lod_150.pth')
        print('')

    def forward(self, sample,
                perturb_overwrite=-1,
                background_rgb=None,
                alpha_inter_ratio_lod0=0.0,
                alpha_inter_ratio_lod1=0.0,
                iter_step=0,
                mode='train',
                save_vis=False,
                resolution=360,
                ):
        result =  self.export_mesh_step(sample,
                                iter_step=iter_step,
                                save_vis=save_vis,
                                resolution=resolution,
                                )

    def obtain_pyramid_feature_maps(self, imgs, lod=0):
        """
        get feature maps of all conditional images
        :param imgs:
        :return:
        """

        if lod == 0:
            extractor = self.pyramid_feature_network_geometry_lod0
        elif lod >= 1:
            extractor = self.pyramid_feature_network_geometry_lod1

        pyramid_feature_maps = extractor(imgs)

        # * the pyramid features are very important, if only use the coarst features, hard to optimize
        fused_feature_maps = torch.cat([
            F.interpolate(pyramid_feature_maps[0], scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(pyramid_feature_maps[1], scale_factor=2, mode='bilinear', align_corners=True),
            pyramid_feature_maps[2]
        ], dim=1)

        return fused_feature_maps

        # render_out dict_keys(['depth', 'color_fine', 'color_fine_mask', 'color_outside', 'color_outside_mask', 'color_mlp', 'color_mlp_mask', 'variance', 'cdf_fine', 'depth_variance', 'weights_sum', 'weights_max', 'alpha_sum', 'alpha_mean', 'gradients', 'weights', 'gradient_error_fine', 'inside_sphere', 'sdf', 'sdf_random', 'blended_color_patch', 'blended_color_patch_mask', 'weights_sum_fg'])
        # loss weight schedule; the regularization terms should be added in later training stage
        def get_weight(iter_step, weight):
            if lod == 1:
                anneal_start = self.anneal_end if lod == 0 else self.anneal_end_lod1
                anneal_end = self.anneal_end if lod == 0 else self.anneal_end_lod1
                anneal_end = anneal_end * 2
            else:
                anneal_start = self.anneal_start if lod == 0 else self.anneal_start_lod1
                anneal_end = self.anneal_end if lod == 0 else self.anneal_end_lod1
                anneal_end = anneal_end * 2

            if iter_step < 0:
                return weight

            if anneal_end == 0.0:
                return weight
            elif iter_step < anneal_start:
                return 0.0
            else:
                return np.min(
                    [1.0,
                     (iter_step - anneal_start) / (anneal_end - anneal_start)]) * weight
        # sample_rays.keys()=dict_keys(['rays_o', 'rays_v', 'rays_ndc_uv', 'rays_color', 'rays_mask', 'rays_norm_XYZ_cam', 'rays_patch_color', 'rays_patch_mask', 'rays_depth'])
        rays_o = sample_rays['rays_o'][0]
        rays_d = sample_rays['rays_v'][0]
        true_rgb = sample_rays['rays_color'][0]# torch.Size([512, 3])

        if 'rays_depth' in sample_rays.keys():
            true_depth = sample_rays['rays_depth'][0]
        else:
            true_depth = None
        mask = sample_rays['rays_mask'][0]

        color_fine = render_out['color_fine']# torch.Size([512  , 3])
        color_fine_mask = render_out['color_fine_mask']# torch.Size([512, 1])
        depth_pred = render_out['depth']# torch.Size([512, 1])

        variance = render_out['variance']
        cdf_fine = render_out['cdf_fine']# torch.Size([512, 128])
        weight_sum = render_out['weights_sum']# torch.Size([512, 1])

        gradient_error_fine = render_out['gradient_error_fine']

        sdf = render_out['sdf']# torch.Size([65536, 1])

        # * color generated by mlp
        color_mlp = render_out['color_mlp']# None
        color_mlp_mask = render_out['color_mlp_mask']# None
        # step 2 : Color loss
        if color_fine is not None:# True
            # Color loss
            color_mask = color_fine_mask if color_fine_mask is not None else mask# torch.Size([512, 1])
            color_mask = color_mask[..., 0]# torch.Size([512])
            color_error = (color_fine[color_mask] - true_rgb[color_mask])# torch.Size([512, 3])
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error).to(color_error.device),
                                        reduction='mean')
            psnr = 20.0 * torch.log10(
                1.0 / (((color_fine[color_mask] - true_rgb[color_mask]) ** 2).mean() / (3.0)).sqrt())
        else:# False
            color_fine_loss = 0.
            psnr = 0.

        if color_mlp is not None:# False
            # Color loss
            color_mlp_mask = color_mlp_mask[..., 0]
            color_error_mlp = (color_mlp[color_mlp_mask] - true_rgb[color_mlp_mask])
            color_mlp_loss = F.l1_loss(color_error_mlp,
                                       torch.zeros_like(color_error_mlp).to(color_error_mlp.device),
                                       reduction='mean')

            psnr_mlp = 20.0 * torch.log10(
                1.0 / (((color_mlp[color_mlp_mask] - true_rgb[color_mlp_mask]) ** 2).mean() / (3.0)).sqrt())
        else:# True
            color_mlp_loss = 0.
            psnr_mlp = 0.
        # step 3 : depth loss
        # depth loss is only used for inference, not included in total loss
        if true_depth is not None:# True
            # depth_loss = self.depth_criterion(depth_pred, true_depth, mask)
            depth_loss = self.depth_criterion(depth_pred, true_depth)

            depth_statis = None
        else:
            depth_loss = 0.
            depth_statis = None

        sparse_loss_1 = torch.exp(
            -1 * torch.abs(render_out['sdf_random']) * self.sdf_decay_param).mean()  # - should equal render_out['sdf_random'].shape([1024, 1])
        sparse_loss_2 = torch.exp(-1 * torch.abs(sdf) * self.sdf_decay_param).mean()# sdf.shape=([65536, 1])
        sparse_loss = (sparse_loss_1 + sparse_loss_2) / 2

        sdf_mean = torch.abs(sdf).mean()
        sparseness_1 = (torch.abs(sdf) < 0.01).to(torch.float32).mean()
        sparseness_2 = (torch.abs(sdf) < 0.02).to(torch.float32).mean()

        # Eikonal loss
        gradient_error_loss = gradient_error_fine
        # ! the first 50k, don't use bg constraint
        fg_bg_weight = 0.0 if iter_step < 50000 else get_weight(iter_step, self.fg_bg_weight)

        # Mask loss, optional
        # The images of DTU dataset contain large black regions (0 rgb values),
        # can use this data prior to make fg more clean
        background_loss = 0.0
        fg_bg_loss = 0.0
        if self.fg_bg_weight > 0 and torch.mean((mask < 0.5).to(torch.float32)) > 0.02:# True
            weights_sum_fg = render_out['weights_sum_fg']# torch.Size([512, 1])
            fg_bg_error = (weights_sum_fg - mask)
            fg_bg_loss = F.l1_loss(fg_bg_error,
                                   torch.zeros_like(fg_bg_error).to(fg_bg_error.device),
                                   reduction='mean')


        loss = self.depth_loss_weight * depth_loss + color_fine_loss + color_mlp_loss + \
               sparse_loss * get_weight(iter_step, self.sdf_sparse_weight) + \
               fg_bg_loss * fg_bg_weight + \
               gradient_error_loss * self.sdf_igr_weight  # ! gradient_error_loss need a mask
        # self.depth_loss_weight=0  color_fine_loss=0.3618 color_mlp_loss=0.0 get_weight=0 fg_bg_weight=0 self.sdf_igr_weight=0.1 
        # 综上:color_fine_loss gradient_error_loss是有用的
        losses = {
            "loss": loss,
            "depth_loss": depth_loss,
            "color_fine_loss": color_fine_loss,
            "color_mlp_loss": color_mlp_loss,
            "gradient_error_loss": gradient_error_loss,
            "background_loss": background_loss,
            "sparse_loss": sparse_loss,
            "sparseness_1": sparseness_1,
            "sparseness_2": sparseness_2,
            "sdf_mean": sdf_mean,
            "psnr": psnr,
            "psnr_mlp": psnr_mlp,
            "weights_sum": render_out['weights_sum'],
            "weights_sum_fg": render_out['weights_sum_fg'],
            "alpha_sum": render_out['alpha_sum'],
            "variance": render_out['variance'],
            "sparse_weight": get_weight(iter_step, self.sdf_sparse_weight),
            "fg_bg_weight": fg_bg_weight,
            "fg_bg_loss": fg_bg_loss, 
        }
        losses = numpy2tensor(losses, device=rays_o.device)
        return loss, losses, depth_statis