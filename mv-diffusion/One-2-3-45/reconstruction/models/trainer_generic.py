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
        

        self.anneal_start = self.conf.get_float('train.anneal_start', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.anneal_start_lod1 = self.conf.get_float('train.anneal_start_lod1', default=0.0)
        self.anneal_end_lod1 = self.conf.get_float('train.anneal_end_lod1', default=0.0)

        # network setups
        self.rendering_network_outside = rendering_network_outside
        self.pyramid_feature_network_geometry_lod0 = pyramid_feature_network_lod0  # 2D pyramid feature network for geometry
        self.pyramid_feature_network_geometry_lod1 = pyramid_feature_network_lod1  # use differnet networks for the two lods

        # when num_lods==2, may consume too much memeory
        self.sdf_network_lod0 = sdf_network_lod0
        self.sdf_network_lod1 = sdf_network_lod1

        # - warpped by ModuleList to support DataParallel
        self.variance_network_lod0 = variance_network_lod0
        self.variance_network_lod1 = variance_network_lod1

        self.rendering_network_lod0 = rendering_network_lod0
        self.rendering_network_lod1 = rendering_network_lod1

        self.n_samples_lod0 = n_samples_lod0
        self.n_importance_lod0 = n_importance_lod0
        self.n_samples_lod1 = n_samples_lod1
        self.n_importance_lod1 = n_importance_lod1
        self.n_outside = n_outside
        self.num_lods = conf.get_int('model.num_lods')  # the number of octree lods
        self.perturb = perturb
        self.alpha_type = alpha_type

        # - the two renderers
        self.sdf_renderer_lod0 = SparseNeuSRenderer(
            self.rendering_network_outside,
            self.sdf_network_lod0,
            self.variance_network_lod0,
            self.rendering_network_lod0,
            self.n_samples_lod0,
            self.n_importance_lod0,
            self.n_outside,
            self.perturb,
            alpha_type='div',
            conf=self.conf)

        self.sdf_renderer_lod1 = SparseNeuSRenderer(
            self.rendering_network_outside,
            self.sdf_network_lod1,
            self.variance_network_lod1,
            self.rendering_network_lod1,
            self.n_samples_lod1,
            self.n_importance_lod1,
            self.n_outside,
            self.perturb,
            alpha_type='div',
            conf=self.conf)

        self.if_fix_lod0_networks = self.conf.get_bool('train.if_fix_lod0_networks')

        # sdf network weights
        self.sdf_igr_weight = self.conf.get_float('train.sdf_igr_weight')
        self.sdf_sparse_weight = self.conf.get_float('train.sdf_sparse_weight', default=0)
        self.sdf_decay_param = self.conf.get_float('train.sdf_decay_param', default=100)
        self.fg_bg_weight = self.conf.get_float('train.fg_bg_weight', default=0.00)
        self.bg_ratio = self.conf.get_float('train.bg_ratio', default=0.0)

        self.depth_loss_weight = self.conf.get_float('train.depth_loss_weight', default=1.00)

        print("depth_loss_weight: ", self.depth_loss_weight)
        self.depth_criterion = DepthLoss()

        # - DataParallel mode, cannot modify attributes in forward()
        # self.iter_step = 0
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')

        # - True for finetuning; False for general training
        self.if_fitted_rendering = self.conf.get_bool('train.if_fitted_rendering', default=False)

        self.prune_depth_filter = self.conf.get_bool('model.prune_depth_filter', default=False)

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
        sizeW = sample['img_wh'][0][0]
        sizeH = sample['img_wh'][0][1]

        partial_vol_origin = sample['partial_vol_origin']  # [B, 3]

        imgs = sample['images'][0]
        # target_candidate_w2cs = sample['target_candidate_w2cs'][0]
        proj_matrices = sample['affine_mats']

        # - obtain conditional features
        with torch.no_grad():
            # - obtain conditional features
            geometry_feature_maps = self.obtain_pyramid_feature_maps(imgs, lod=0)
            # - lod 0
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
        coords_lod0 = conditional_features_lod0['coords_scale0']  # [1,3,wX,wY,wZ]

        save_path = self.base_exp_dir+'/con_volume_lod_150.pth'
        coords_lod0_150 = F.interpolate(con_volume_lod0, size=(150, 150, 150), mode='trilinear', align_corners=False)
        torch.save(coords_lod0_150, save_path)
        
        # torch.save(con_valid_mask_volume_lod0, '/home/bitterdhg/Code/nerf/Learn/One-2-3-45-master/con_valid_mask_volume_lod0.pth')
        print("save_path: " + save_path)

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
        import time
        begin = time.time()
        result =  self.export_mesh_step(sample,
                                iter_step=iter_step,
                                save_vis=save_vis,
                                resolution=resolution,
                                )
        end = time.time()

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
