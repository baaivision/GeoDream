from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseGeometry,
    BaseImplicitGeometry,
    contract_to_unisphere,
)
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *
from .grid_sampler import grid_sample_3d, tricubic_sample_3d


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True, normalize=False):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
        self.normalize = normalize

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                if self.normalize:
                    out += [func(freq * x) / freq]
                else:
                    out += [func(freq * x)]

        return torch.cat(out, -1)

@threestudio.register("geodream-geometry")
class GeodreamGeometryVolume(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        density_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = "blob_magic3d"
        density_blob_scale: float = 10.0
        density_blob_std: float = 0.5
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: float = 0.01

        # automatically determine the threshold
        isosurface_threshold: Union[float, str] = 25.0
        init_volume_path: str = "con_volume_lod0.pth"
        one2345_weight: str = "pretrain.pth"
        sdf_network_grad: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )
        
        
        self.sdf_layers = SdfLayer()
        self.deviation_network = SingleVarianceNetwork(self.cfg.one2345_weight)

        # sdf_layers weight
        sdf_layers_weight = torch.load(self.cfg.one2345_weight)['sdf_network_lod0']
        selected_state_dict = {}
        prefix = 'sdf_layer'
        for key, value in sdf_layers_weight.items():
            if key.startswith(prefix):
                selected_state_dict[key[10:]] = value# key need remove sdf_layer prefix
        self.sdf_layers.load_state_dict(selected_state_dict)
        print("sdf_layers is loading weight at " + self.cfg.one2345_weight)
        
        # sdf_layers freeze 
        if self.cfg.sdf_network_grad:
            print("sdf_layers network is training")
        else:
            for p in self.sdf_layers.parameters():
                p.requires_grad_(False)
            print("sdf_layers network is freezeing")

        # volume weight
        volume_weight = torch.load(self.cfg.init_volume_path)

        self.volume = nn.Parameter(volume_weight, requires_grad=True)
        print("volume network is loading weight at " + self.cfg.init_volume_path)

    def get_activated_density(
        self, points: Float[Tensor, "*N Di"], density: Float[Tensor, "*N 1"]
    ) -> Tuple[Float[Tensor, "*N 1"], Float[Tensor, "*N 1"]]:
        density_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.density_bias == "blob_dreamfusion":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * torch.exp(
                    -0.5 * (points**2).sum(dim=-1) / self.cfg.density_blob_std**2
                )[..., None]
            )
        elif self.cfg.density_bias == "blob_magic3d":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * (
                    1
                    - torch.sqrt((points**2).sum(dim=-1)) / self.cfg.density_blob_std
                )[..., None]
            )
        elif isinstance(self.cfg.density_bias, float):
            density_bias = self.cfg.density_bias
        else:
            raise ValueError(f"Unknown density bias {self.cfg.density_bias}")
        raw_density: Float[Tensor, "*N 1"] = density + density_bias
        density = get_activation(self.cfg.density_activation)(raw_density)
        return raw_density, density

    def forward(
        self, points: Float[Tensor, "*N Di"], viewdirs, dists, output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        grad_enabled = torch.is_grad_enabled()

        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        
        sdf, feature_vector = self.sdf(points.view(-1, self.cfg.n_input_dims))

        output = {
            "density": sdf,
        }
        
        g = self.gradient(points.view(-1, self.cfg.n_input_dims))
        alphas = self.get_alpha(points.view(-1, self.cfg.n_input_dims), viewdirs, dists, feature_vector, sdf, g)
        output.update({"ALPHA": alphas})

        
        points_norm = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)

        
        enc = self.encoding(points_norm.view(-1, self.cfg.n_input_dims))
        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            output.update({"features": features})

        
        torch.set_grad_enabled(grad_enabled)
        return output

    def forward_density(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        density, _ = self.sdf(points.view(-1, self.cfg.n_input_dims))
        density = density.reshape(*points.shape[:-1], 1)
        return density
    
    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        sdf, _ = self.sdf(points.view(-1, self.cfg.n_input_dims))
        sdf = sdf.reshape(*points.shape[:-1], 1)
        deformation: Optional[Float[Tensor, "*N 3"]] = None
        return sdf, deformation

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return field - threshold
    

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "GeodreamGeometryVolume":
        if isinstance(other, GeodreamGeometryVolume):
            instance = GeodreamGeometryVolume(cfg, **kwargs)
            instance.encoding.load_state_dict(other.encoding.state_dict())
            instance.density_network.load_state_dict(other.density_network.state_dict())
            if copy_net:
                if (
                    instance.cfg.n_feature_dims > 0
                    and other.cfg.n_feature_dims == instance.cfg.n_feature_dims
                ):
                    instance.feature_network.load_state_dict(
                        other.feature_network.state_dict()
                    )
                if (
                    instance.cfg.normal_type == "pred"
                    and other.cfg.normal_type == "pred"
                ):
                    instance.normal_network.load_state_dict(
                        other.normal_network.state_dict()
                    )
            return instance
        else:
            raise TypeError(
                f"Cannot create {GeodreamGeometryVolume.__name__} from {other.__class__.__name__}"
            )
    
    def forward_sdf(self, pts):
        sdf, _ = self.sdf(pts)
        return sdf
    
    def sdf(self, pts, lod=0):
        conditional_volume = self.volume
        num_pts = pts.shape[0]
        device = pts.device
        pts_ = pts.clone()
        pts = pts.view(1, 1, 1, num_pts, 3)  # - should be in range (-1, 1)

        pts = torch.flip(pts, dims=[-1])
        sampled_feature = grid_sample_3d(conditional_volume, pts)  # [1, c, 1, 1, num_pts]
        sampled_feature = sampled_feature.view(-1, num_pts).permute(1, 0).contiguous().to(device)

        sdf_pts = self.sdf_layers(pts_, sampled_feature)

        return sdf_pts[:, :1], sdf_pts[:, 1:]
    
    def get_alpha(self, ray_samples, rays_d, dists, feature_vector, sdf=None, gradients=None):
        """compute alpha from sdf as in NeuS"""
        inv_variance = self.deviation_network(feature_vector)[:, :1].clip(1e-6, 1e6)  # Single parameter

    
        #gradients = torch.ones_like(rays_d, requires_grad=False, device=rays_d.device)
        true_dot_val = (rays_d * gradients).sum(-1, keepdim=True)  # * calculate
        alpha_inter_ratio = 0.0 
        iter_cos = -(F.relu(-true_dot_val * 0.5 + 0.5) * (1.0 - alpha_inter_ratio) + F.relu(
            -true_dot_val) * alpha_inter_ratio)  # always non-positive

        true_estimate_sdf_half_next = sdf + iter_cos.clip(-10.0, 10.0) * dists.reshape(-1, 1) * 0.5
        true_estimate_sdf_half_prev = sdf - iter_cos.clip(-10.0, 10.0) * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(true_estimate_sdf_half_prev * inv_variance)
        next_cdf = torch.sigmoid(true_estimate_sdf_half_next * inv_variance)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alpha
    
    def gradient(self, x):
        
        x.requires_grad_(True)
        with torch.enable_grad():
            sdf, _ = self.sdf(x)
        y = sdf

        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        # ! Distributed Data Parallel doesn’t work with torch.autograd.grad()
        # ! (i.e. it will only work if gradients are to be accumulated in .grad attributes of parameters).
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients
    
    
class SdfLayer(nn.Module):
    def __init__(self):
        super(SdfLayer, self).__init__()
        self.d_in=3
        self.d_out=129# self.hidden_dim + 1,
        self.d_hidden=128# self.hidden_dim,
        self.n_layers=4# num_sdf_layers,
        self.skip_in=(4,)
        self.multires=6# multires,
        self.bias=0.5
        self.geometric_init=True
        self.weight_norm=True
        self.activation='softplus'
        self.d_conditional_feature=16
        #self.d_conditional_feature = self.d_conditional_feature

        # concat latent code for ench layer input excepting the first layer and the last layer
        dims_in = [self.d_in] + [self.d_hidden + self.d_conditional_feature for _ in range(self.n_layers - 2)] + [self.d_hidden]
        dims_out = [self.d_hidden for _ in range(self.n_layers - 1)] + [self.d_out]

        self.embed_fn_fine = None

        if self.multires > 0:
            embed_fn = Embedding(in_channels=self.d_in, N_freqs=self.multires)  # * include the input
            self.embed_fn_fine = embed_fn
            dims_in[0] = embed_fn.out_channels

        self.num_layers = self.n_layers
        self.skip_in = self.skip_in
        self.sdf_layers = []
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                in_dim = dims_in[l] + dims_in[0]
            else:
                in_dim = dims_in[l]

            out_dim = dims_out[l]
            lin = nn.Linear(in_dim, out_dim)
            
            if self.geometric_init:  # - from IDR code,
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -self.bias)
                    # the channels for latent codes are set to 0
                    torch.nn.init.constant_(lin.weight[:, -self.d_conditional_feature:], 0.0)
                    torch.nn.init.constant_(lin.bias[-self.d_conditional_feature:], 0.0)

                elif self.multires > 0 and l == 0:  # the first layer
                    torch.nn.init.constant_(lin.bias, 0.0)
                    # * the channels for position embeddings are set to 0
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    # * the channels for the xyz coordinate (3 channels) for initialized by normal distribution
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    # * the channels for position embeddings (and conditional_feature) are initialized to 0
                    torch.nn.init.constant_(lin.weight[:, -(dims_in[0] - 3 + self.d_conditional_feature):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    # the channels for latent code are initialized to 0
                    torch.nn.init.constant_(lin.weight[:, -self.d_conditional_feature:], 0.0)

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)

            #self.sdf_layers += [lin]
            setattr(self, "lin" + str(l), lin)

        if self.activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert self.activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs, latent):
            inputs = inputs
            if self.embed_fn_fine is not None:
                inputs = self.embed_fn_fine(inputs)

            # - only for lod1 network can use the pretrained params of lod0 network
            if latent.shape[1] != self.d_conditional_feature:
                latent = torch.cat([latent, latent], dim=1)

            x = inputs
            for l in range(0, self.num_layers - 1):
                lin = getattr(self, "lin" + str(l))

                # * due to the conditional bias, different from original neus version
                if l in self.skip_in:
                    x = torch.cat([x, inputs], 1) / np.sqrt(2)

                if 0 < l < self.num_layers - 1:
                    x = torch.cat([x, latent], 1)

                x = lin(x)
                if l < self.num_layers - 2:
                    x = self.activation(x)

            return x

    
class SingleVarianceNetwork(nn.Module):
    def __init__(self, pretrain_path, init_val=1.0):
        super(SingleVarianceNetwork, self).__init__()
        variance_weight = torch.load(pretrain_path)['variance_network_lod0']['variance']
        self.register_parameter('variance', nn.Parameter(variance_weight, requires_grad=False))
        print("variance network is loading weight at " + pretrain_path)
        print("variance network is freezeing")

    def forward(self, x):
        return torch.ones([len(x), 1]).to(x.device) * torch.exp(self.variance * 10.0)
    
    def get_variance(self):
        return self.variance