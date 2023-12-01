from dataclasses import dataclass

import nerfacc
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.utils.ops import chunk_batch, validate_empty_rays
from threestudio.utils.typing import *


def volsdf_density(sdf, inv_std):
    beta = 1 / inv_std
    alpha = inv_std
    return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))


class LearnedVariance(nn.Module):
    def __init__(self, init_val):
        super(LearnedVariance, self).__init__()
        self.register_parameter("_inv_std", nn.Parameter(torch.tensor(init_val)))

    @property
    def inv_std(self):
        val = torch.exp(self._inv_std * 10.0)
        return val

    def forward(self, x):
        return torch.ones_like(x) * self.inv_std.clamp(1.0e-6, 1.0e6)


@threestudio.register("geodream-volume-renderer")
class GeodreamVolumeRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        randomized: bool = True
        eval_chunk_size: int = 16000
        grid_prune: bool = True
        prune_alpha_threshold: bool = True
        learned_variance_init: float = 0.3
        cos_anneal_end_steps: int = 0
        use_volsdf: bool = False

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.variance = LearnedVariance(self.cfg.learned_variance_init)
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb=self.bbox.view(-1), resolution=32, levels=1
        )
        if not self.cfg.grid_prune:
            self.estimator.occs.fill_(True)
            self.estimator.binaries.fill_(True)
        self.render_step_size = (
            1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
        )
        self.randomized = self.cfg.randomized
        self.cos_anneal_ratio = 1.0

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]

        def alpha_fn(t_starts, t_ends, ray_indices):
            t_starts, t_ends = t_starts[..., None], t_ends[..., None]
            t_origins = rays_o_flatten[ray_indices]
            t_positions = (t_starts + t_ends) / 2.0
            t_dirs = rays_d_flatten[ray_indices]
            positions = t_origins + t_dirs * t_positions
            if self.training:
                sdf, feature_vector = self.geometry.sdf(positions)
            else:
                sdf, feature_vector = chunk_batch(
                    self.geometry.sdf,
                    self.cfg.eval_chunk_size,
                    positions,
                )

            inv_std = self.geometry.deviation_network(feature_vector)[:, :1].clip(1e-6, 1e6)
            if self.cfg.use_volsdf:
                alpha = self.render_step_size * volsdf_density(sdf, inv_std)
            else:
                estimated_next_sdf = sdf - self.render_step_size * 0.5
                estimated_prev_sdf = sdf + self.render_step_size * 0.5
                prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
                next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)
                p = prev_cdf - next_cdf
                c = prev_cdf
                alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

            return alpha[..., 0]

        if not self.cfg.grid_prune:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                    rays_o_flatten,
                    rays_d_flatten,
                    alpha_fn=None,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.0,
                    stratified=self.randomized,
                    cone_angle=0.0,
                    early_stop_eps=0,
                    training = self.training,
                )
        else:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                    rays_o_flatten,
                    rays_d_flatten,
                    alpha_fn=alpha_fn if self.cfg.prune_alpha_threshold else None,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                    stratified=self.randomized,
                    cone_angle=0.0,
                    training = self.training,
                )
                
        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts
        
        if self.training:
            geo_out = self.geometry(
                positions, viewdirs=t_dirs, dists = t_intervals, output_normal=self.material.requires_normal
            )
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out,
                **kwargs
            )
            comp_rgb_bg = self.background(dirs=rays_d)
        else:
            with torch.no_grad():
                geo_out = self.geometry(
                    positions, viewdirs=t_dirs, dists = t_intervals, output_normal=self.material.requires_normal
                )
                rgb_fg_all = self.material(
                    viewdirs=t_dirs,
                    positions=positions,
                    light_positions=t_light_positions,
                    **geo_out,
                    **kwargs
                )
                comp_rgb_bg = self.background(dirs=rays_d)
                
        alpha: Float[Tensor, "Nr 1"] = geo_out["ALPHA"]

        weights: Float[Tensor, "Nr 1"]
        weights_, _ = nerfacc.render_weight_from_alpha(
            alpha[..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )

        if bg_color is None:
            bg_color = comp_rgb_bg

        if bg_color.shape[:-1] == (batch_size, height, width):
            bg_color = bg_color.reshape(batch_size * height * width, -1)

        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)

        t_depth = depth[ray_indices]
        z_variance = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=(t_positions - t_depth) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        
        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "depth": depth.view(batch_size, height, width, 1),
            "z_variance": z_variance.view(batch_size, height, width, 1),
        }
        

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )
        else:
            if "normal" in geo_out:
                comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                out.update(
                    {
                        "comp_normal": comp_normal.view(batch_size, height, width, 3),
                    }
                )
        out.update({"inv_std": self.geometry.deviation_network.get_variance()})
        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        self.cos_anneal_ratio = (
            1.0
            if self.cfg.cos_anneal_end_steps == 0
            else min(1.0, global_step / self.cfg.cos_anneal_end_steps)
        )
        if self.cfg.grid_prune:

            def occ_eval_fn(x):
                sdf, feature_vector = self.geometry.sdf(x)
                inv_std = self.geometry.deviation_network(feature_vector)[:, :1].clip(1e-6, 1e6)
                if self.cfg.use_volsdf:
                    alpha = self.render_step_size * volsdf_density(sdf, inv_std)
                else:
                    estimated_next_sdf = sdf - self.render_step_size * 0.5
                    estimated_prev_sdf = sdf + self.render_step_size * 0.5
                    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
                    next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)
                    p = prev_cdf - next_cdf
                    c = prev_cdf
                    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
                return alpha[...,0]

            if self.training and not on_load_weights:
                self.estimator.update_every_n_steps(
                    step=global_step, occ_eval_fn=occ_eval_fn
                )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()
