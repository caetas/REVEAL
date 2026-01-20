# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import accelerate
from tqdm import trange, tqdm
import os
from config import models_dir
from collections import OrderedDict
from diffusers.models import AutoencoderKL, AutoencoderKLFlux2
import copy
from torchvision.utils import make_grid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .vision_encoder import load_encoders
from .loss import SILoss


@torch.no_grad()
def update_ema(ema_model, model, decay=0.5):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        # if name contains "module" then remove module
        if "module" in name:
            name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def create_checkpoint_dir():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(os.path.join(models_dir, "iREPA")):
        os.makedirs(os.path.join(models_dir, "iREPA"))


def build_mlp(hidden_size, projector_dim, z_dim, **kwargs):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Build Projection Layers                                       #
#################################################################################
ALL_PROJECTION_LAYER_TYPES = ["mlp", "linear", "conv"]
class ProjectionLayer(nn.Module):
    def __init__(self, projection_layer_type="mlp", **kwargs):
        super().__init__()
        assert projection_layer_type in ALL_PROJECTION_LAYER_TYPES, f"Unsupported projection layer type: {projection_layer_type}. Must be one of {ALL_PROJECTION_LAYER_TYPES}"
        # self.kwargs = kwargs
        self.projection_layer_type = projection_layer_type 
        self.build_projection_layer(projection_layer_type, **kwargs)

    def build_projection_layer(self, projection_layer_type, **kwargs):
        if projection_layer_type == "mlp":
            self.projection_layer = build_mlp(**kwargs)
        elif projection_layer_type == "linear":
            in_dim  = kwargs.pop("hidden_size")
            out_dim = kwargs.pop("z_dim")
            self.projection_layer = nn.Linear(in_dim, out_dim)
        elif projection_layer_type == "conv":
            in_ch  = kwargs.pop("hidden_size")
            out_ch = kwargs.pop("z_dim")
            kernel_size = kwargs.pop("proj_kwargs_kernel_size")
            padding = kernel_size // 2 # to keep spatial dimension
            self.projection_layer = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding)
        else:
            raise ValueError(f"Unsupported projection layer type: {projection_layer_type}")

    def forward(self, x, hw: tuple[int, int] | None = None):
        """
        x: [B, T, D]
        hw: optional (H, W) for non-square token grids (mostly not used).
        """
        B, T, D = x.shape
        if self.projection_layer_type in ("mlp", "linear"):
            x_ = self.projection_layer(x.reshape(B * T, D))
            return x_.reshape(B, T, -1)

        elif self.projection_layer_type == "conv":
            if hw is None:
                H = W = int(math.isqrt(T))
                assert H * W == T, f"conv projector needs square grid or pass hw; got T={T}"
            else:
                H, W = hw
                assert H * W == T, f"Provided hw={hw} but T={T}"

            # [B, T, D] -> [B, D, H, W]
            x_ = x.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
            y  = self.projection_layer(x_)                  # [B, z_dim, H, W]
            y  = y.permute(0, 2, 3, 1).contiguous()         # [B, H, W, z_dim]
            return y.reshape(B, T, -1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################            
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        eval_mode=False,
        projector_dim=2048,
        projection_layer_type="mlp",
        proj_kwargs_kernel_size=3,
        old_projector_version=False,
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth
        self.eval_mode = eval_mode
        self.projection_layer_type = projection_layer_type
        self.old_projector_version = old_projector_version

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])

        # create projectors only in non-eval mode
        if not self.eval_mode:
            if self.old_projector_version:
                self.projectors = nn.ModuleList([
                    build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
                ])
            else:
                self.projectors = nn.ModuleList([
                    ProjectionLayer(projection_layer_type, hidden_size=hidden_size, z_dim=z_dim, projector_dim=projector_dim, proj_kwargs_kernel_size=proj_kwargs_kernel_size) for z_dim in z_dims
                ])
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def forward(self, x, t, y, return_logvar=False):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                                # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)

            if (i + 1) == self.encoder_depth:
                if self.old_projector_version:
                    zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors] if not self.eval_mode else None
                else:
                    zs = [projector(x) for projector in self.projectors] if not self.eval_mode else None
                # zs = [projector(x) for projector in self.projectors] if not self.eval_mode else None
                # zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors] if not self.eval_mode else None
                zs_original = [x.clone() for _ in self.projectors] if not self.eval_mode else None
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x, zs, zs_original

    def forward_features(self, x, t, y, encoder_depths=[8], proj=False):
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                                # (N, D)

        zs = []
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if (i + 1) in encoder_depths:
                if (i + 1) == self.encoder_depth and proj:
                    # z = self.projectors[0](x.reshape(-1, D)).reshape(N, T, -1) if not self.eval_mode else None
                    z = self.projectors[0](x) if not self.eval_mode else None
                    zs.append(z)
                else:
                    zs.append(x)
            # Short-cut the compute, make it faster
            if (i + 1) == max(encoder_depths):
                break
        return zs

    def forward_feature_proj(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                                # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if (i + 1) == self.encoder_depth:
                z = self.projectors[0](x) if not self.eval_mode else None
                # z = self.projectors[0](x.reshape(-1, D)).reshape(N, T, -1) if not self.eval_mode else None
                return x, z
        return None

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}


#################################################################################
# spatial normalization
#################################################################################
ALL_SPNORM_METHODS = ["none", "zscore"]


def spatial_zscore(feat: torch.Tensor, alpha: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Z-score normalization along spatial dimension.

    Args:
        feat: (B, T, D) patch tokens
        alpha: scaling factor for mean subtraction (default 1.0)
        eps: small constant for numerical stability

    Returns:
        Normalized features (B, T, D)
    """
    mean = feat.mean(dim=1, keepdim=True)
    std = feat.std(dim=1, keepdim=True)
    return (feat - alpha * mean) / (std + eps)


class SpatialNormalization:
    """
    Spatial normalization wrapper for backward compatibility.

    Only supports "none" and "zscore" methods.
    """
    def __init__(self, method: str, *, eps: float = 1e-6):
        assert method in ALL_SPNORM_METHODS, f"Invalid method: {method}. Must be one of {ALL_SPNORM_METHODS}"
        self.method = method
        self.eps = eps

    def __call__(self, feat: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.method == "none":
            return feat
        alpha = kwargs.get('zscore_alpha', 1.0)
        return spatial_zscore(feat, alpha=alpha, eps=self.eps)
    
    
##################################################################################
#                                 Denoiser Model                                #
##################################################################################

class DenoiserREPA(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_cls = SiT_models[args.model]
        self.args = args

        if args.vae == "SD2":
            self.vae =  AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse")
            self.channels = 4
            self.img_size = args.img_size // 8
            self.scale = 0.18215
            self.bias = 0.0

        elif args.vae == "SD3":
            self.vae =  AutoencoderKL.from_pretrained(f"stabilityai/stable-diffusion-3.5-medium", subfolder='vae')
            self.channels = 16
            self.img_size = args.img_size // 8
            self.scale = 1.5305
            self.bias = 0.0609

        elif args.vae == "Flux1":
            self.vae =  AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder='vae')
            self.channels = 16
            self.img_size = args.img_size // 8
            self.scale = 0.3611
            self.bias = 0.1159

        elif args.vae == "Flux2":
            self.vae =  AutoencoderKLFlux2.from_pretrained(f"black-forest-labs/FLUX.2-dev", subfolder='vae')
            self.channels = 128
            self.img_size = args.img_size // 16
            self.scale = 1.0
            self.bias = 0.0

        self.num_classes = args.class_num
        self.label_drop_prob = args.label_drop_prob
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.num_classes == 0:
            self.label_drop_prob = 1.0
        
        if args.train:
            if self.args.repa_loss:
                self.encoders = load_encoders(
                    self.args.enc_type, self.device, self.args.img_size, checkpoint_path=self.args.enc_ckpt_path
                )
            else:
                self.encoders = []

            z_dims = [encoder.embed_dim for encoder in self.encoders]
        else:
            z_dims = [""] # dummy value for eval mode
            self.encoders = []

        block_kwargs = {
            "fused_attn": args.fused_attn, 
            "qk_norm": args.qk_norm,
        }

        self.model = model_cls(
            input_size=self.img_size,
            in_channels=self.channels,
            class_dropout_prob=self.label_drop_prob,
            num_classes=self.num_classes,
            use_cfg = (self.num_classes > 0),
            z_dims=z_dims,
            encoder_depth=args.encoder_depth,
            projection_layer_type=args.projection_layer_type,
            proj_kwargs_kernel_size=args.proj_kwargs_kernel_size,
            eval_mode=not args.train,
            **block_kwargs
        )

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps

        # ema
        self.ema_decay = args.ema_decay
        self.ema = None
        self.dataset = args.dataset

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.snapshot = self.n_epochs // args.snapshot
        self.sample_and_save_freq = args.sample_and_save_freq
        self.no_wandb = args.no_wandb

        if args.train:
            self.ema = copy.deepcopy(self.model)
            self.ema = self.ema.eval()
            for param in self.ema.parameters():
                param.requires_grad = False

    @staticmethod
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents
    
    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @torch.no_grad()
    def normalize_lat(self, z):
        '''
        Normalize the input image
        :param x: input image
        '''
        if self.args.vae == "Flux2":
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.vae.module.bn.eval()
                return self.vae.module.bn(self._patchify_latents(z.latent_dist.mode()))
            else:
                self.vae.bn.eval()
                return self.vae.bn(self._patchify_latents(z.latent_dist.mode()))
        else:
            return (z.latent_dist.sample()-self.bias) * self.scale
    
    @torch.no_grad()
    def inverse_normalize_lat(self, z):
        '''
        Inverse normalize the input image
        :param x: input image
        '''
        if self.args.vae == "Flux2":
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.vae.module.bn.eval()
                s = torch.sqrt(self.vae.module.bn.running_var.view(1, -1, 1, 1) + self.vae.module.config.batch_norm_eps)
                m = self.vae.module.bn.running_mean.view(1, -1, 1, 1)
            else:
                self.vae.bn.eval()
                s = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
                m = self.vae.bn.running_mean.view(1, -1, 1, 1)
            return self._unpatchify_latents(z * s + m)

        else:
            return z / self.scale + self.bias
    
    @torch.no_grad()
    def encode(self, x):
        '''
        Encode the input image
        :param x: input image
        '''
        # check if it is a distributted model or not
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.normalize_lat(self.vae.module.encode(x))
        else:
            return self.normalize_lat(self.vae.encode(x))
        
    @torch.no_grad()    
    def decode(self, z):
        '''
        Decode the input image
        :param z: input image
        '''
        z = self.inverse_normalize_lat(z)
        # check if it is a distributted model or not
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.vae.module.decode(z).sample
        else:
            return self.vae.decode(z).sample
        
    @torch.no_grad()
    def sample(self):
        self.model.eval()
        bsz = self.args.num_samples
        labels = torch.arange(0, bsz) % self.num_classes if self.num_classes > 0 else torch.zeros(bsz, dtype=torch.long)
        accelerator = accelerate.Accelerator()
        self.model, self.vae = accelerator.prepare(self.model, self.vae)
        labels = labels.to(accelerator.device)

        samples = self.generate(labels, accelerator=accelerator).float()

        samples = samples*0.5 + 0.5
        samples = torch.clamp(samples, 0.0, 1.0)

        grid = make_grid(samples, nrow=int(bsz**0.5))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.show()

    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.isfile(self.args.checkpoint):
                state_dict = torch.load(self.args.checkpoint, map_location=self.device)
                # delete projectors keys if present
                state_dict = {k: v for k, v in state_dict.items() if "projectors" not in k}
                self.model.load_state_dict(state_dict)
                #self.model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.device))
                print(f"Loaded checkpoint from {self.args.checkpoint}")
        
    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, y, t):
        '''
        Forward pass of the model
        :param x: input image
        :param y: class labels
        :param t: diffusion timesteps
        '''
        if self.args.train:
            v_pred, _, _ = self.ema(x, t, y)
        else:
            v_pred, _, _ = self.model(x, t, y)
        return v_pred
    
    def forward_with_cfg(self, x, y, t, cfg_scale):
        '''
        Forward pass of the model with classifier-free guidance
        :param x: input image
        :param y: class labels
        :param t: diffusion timesteps
        :param cfg_scale: classifier-free guidance scale
        '''
        bsz = x.size(0)
        # duplicate for cfg
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        y_in = torch.cat([y, torch.ones_like(y)*self.num_classes], dim=0)

        if self.args.train:
            v_pred = self.ema(x_in, t_in, y_in)[0]
        else:
            v_pred = self.model(x_in, t_in, y_in)[0]

        v_pred_uncond, v_pred_cond = v_pred.chunk(2, dim=0)
        v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
        return v_pred
    
    @torch.no_grad()
    def generate(self, labels, accelerator=None):
        bsz = labels.size(0)
        z = torch.randn(bsz, self.channels, self.img_size, self.img_size, device=accelerator.device if accelerator is not None else self.device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=accelerator.device if accelerator is not None else self.device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError
    
        # ode
        for i in tqdm(range(self.steps - 1), desc="Sampling Steps"):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            with accelerator.autocast():
                z = stepper(z, t, t_next, labels)
        # last step euler
        with accelerator.autocast():
            z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
            if self.vae is not None:
                z = self.decode(z)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        if self.num_classes > 0 and self.cfg_scale > 0:
            v_pred = self.forward_with_cfg(z, labels, t.flatten(), cfg_scale_interval)
        else:
            v_pred = self.forward(z, torch.ones_like(labels)*self.num_classes, t.flatten())
        return v_pred

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next
        
    def train_model(self, train_loader, verbose=True):
        '''
        Train the model
        :param train_loader: PyTorch DataLoader object
        :param verbose: bool, whether to display progress bar
        '''
        appendix = self.args.enc_ckpt_path.split('/')[-1].replace('.pth', '') if self.args.enc_ckpt_path is not None else self.args.enc_type
        accelerator = accelerate.Accelerator(log_with="wandb")
        if not self.no_wandb:
            accelerator.init_trackers("iREPA", config=self.args, init_kwargs={"wandb": {"name": f"{self.args.vae}_{appendix}_{self.args.model}_{self.dataset}"}})
        create_checkpoint_dir()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.n_epochs * len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor= self.lr / self.args.final_lr,
            final_div_factor=1,
            cycle_momentum=False
        )

        epoch_bar = trange(self.n_epochs, desc="Epochs")

        train_loader, self.model, optimizer, scheduler, self.ema, self.vae = accelerator.prepare(train_loader, self.model, optimizer, scheduler, self.ema, self.vae)

        if self.args.repa_loss:
                self.encoders = load_encoders(
                    self.args.enc_type, self.device, self.args.img_size, checkpoint_path=self.args.enc_ckpt_path, accelerator=accelerator
                )

        loss_fn = SILoss(
            accelerator = accelerator,
            projection_loss_type=self.args.projection_loss_type,
            proj_coeff=self.args.proj_coeff,
        )


        self.vae.eval()

        update_ema(self.ema, self.model, 0)

        spnorm = SpatialNormalization(self.args.spnorm_method)

        best_loss = float("inf")
        for epoch in epoch_bar:
            self.model.train()
            train_loss = 0
            train_loss_vel = 0
            train_loss_proj = 0
            for (x, cond) in tqdm(train_loader, desc='Batches', leave=False):
                #x = x.to(self.device)
                #cond = cond.to(self.device)

                with torch.no_grad():
                    zs = []

                    with accelerator.autocast():
                        for encoder in self.encoders:
                            encoder.eval()

                            raw_image_ = encoder.preprocess(x*127.5 + 127.5)

                            features = encoder.forward_features(raw_image_)

                            # normalize spatial features
                            spnorm_kwargs = {
                                'feat': features['x_norm_patchtokens'],
                                'cls': features['x_norm_clstoken'],
                                'cls_weight': self.args.cls_token_weight,
                                'zscore_alpha': self.args.zscore_alpha,
                                'zscore_proj_skip_std': self.args.zscore_proj_skip_std,
                            }
                            z = spnorm(**spnorm_kwargs)

                            # append to list
                            zs.append(z)

                with accelerator.autocast():

                    if self.vae is not None:
                        with torch.no_grad():
                            # if x has one channel, make it 3 channels
                            if x.shape[1] == 1:
                                x = torch.cat((x, x, x), dim=1)
                            x = self.encode(x)

                    optimizer.zero_grad()

                    model_kwargs = dict(y=cond)
                    loss, proj_loss, loss_dict = loss_fn(self.model, x, model_kwargs, zs=zs)
                    loss_mean = loss.mean()
                    proj_loss_mean = proj_loss.mean()
                    loss = loss_mean + proj_loss_mean

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = self.model.parameters()
                        grad_norm = accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()*x.shape[0]
                train_loss_vel += loss_mean.item()*x.shape[0]
                train_loss_proj += proj_loss_mean.item()*x.shape[0]
                if accelerator.sync_gradients:
                    update_ema(self.ema, self.model, self.ema_decay)

            accelerator.wait_for_everyone()
            
            if not self.no_wandb:
                accelerator.log({"train_loss": train_loss / len(train_loader.dataset)}, step=epoch)
                accelerator.log({"train_loss_vel": train_loss_vel / len(train_loader.dataset)}, step=epoch)
                accelerator.log({"train_loss_proj": train_loss_proj / len(train_loader.dataset)}, step=epoch)
                accelerator.log({"lr": scheduler.get_last_lr()[0]}, step=epoch)
                
            epoch_bar.set_postfix(loss=train_loss / len(train_loader.dataset))

            if train_loss/len(train_loader.dataset) < best_loss:
                best_loss = train_loss/len(train_loader.dataset)

            if (epoch+1) % self.snapshot == 0:
                ema_to_save = accelerator.unwrap_model(self.ema)
                accelerator.save(ema_to_save.state_dict(), os.path.join(models_dir, "iREPA", f"{self.args.vae}__{appendix}_{self.args.model.replace('/', '_')}_{self.dataset}_epoch{epoch+1}.pt"))
        
            if epoch == 0 or ((epoch+1) % self.sample_and_save_freq == 0):
                self.model.eval()
                self.ema.eval()
                with torch.no_grad():
                    sample_labels = (torch.arange(0, 16, device=accelerator.device) % self.num_classes) if self.num_classes > 0 else torch.zeros(16, device=accelerator.device, dtype=torch.long)
                    samples = self.generate(sample_labels, accelerator=accelerator).float()
                    samples = samples*0.5 + 0.5
                    samples = torch.clamp(samples, 0.0, 1.0)
                    grid = make_grid(samples, nrow=4)
                    fig = plt.figure(figsize=(8, 8))
                    plt.axis('off')
                    plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
                    if not self.args.no_wandb:
                        accelerator.log({"samples": fig}, step=epoch)
                    plt.close(fig)
                self.model.train()
        
        accelerator.end_training()


    @torch.no_grad()
    def fid_sample(self):

        # extract epoch from checkpoint path
        name = self.args.checkpoint.split('/')[-1].replace('.pt', '')
        print(f"Generating FID samples for {name}...")

        if not os.path.exists('../../fid_samples'):
            os.makedirs('../../fid_samples')
        if not os.path.exists('../../fid_samples/' + self.dataset):
            os.makedirs('../../fid_samples/' + self.dataset)
        fid_dir = f"../../fid_samples/{self.dataset}/{name}_steps{self.steps}_solver{self.method}_cfg{self.cfg_scale}"
        if not os.path.exists(fid_dir):
            os.makedirs(fid_dir)

        self.model.eval()
        accelerator = accelerate.Accelerator()
        self.model, self.vae = accelerator.prepare(self.model, self.vae)
        device = accelerator.device
        bsz = self.args.batch_size
        labels = torch.arange(0, 50000, device=device) % self.num_classes if self.num_classes > 0 else torch.zeros(50000, device=device, dtype=torch.long)

        for i in tqdm(range(0, 50000, bsz), desc="FID Sampling", leave=False):
            current_bsz = min(bsz, 50000 - i)
            batch_labels = labels[i:i+current_bsz]
            samples = self.generate(batch_labels, accelerator=accelerator)

            samples = samples*0.5 + 0.5
            samples = torch.clamp(samples, 0.0, 1.0).float()

            for j in range(current_bsz):
                idx = i + j
                plt.imsave(
                    os.path.join(fid_dir, f"{idx:05d}.png"),
                    samples[j].permute(1, 2, 0).cpu().numpy()
                )

    @torch.no_grad()
    def feature_extractor(self, dataloader, t=1.0, depths=[8]):
        '''
        Extract features from the model
        :param dataloader: PyTorch DataLoader object
        :param t: diffusion timestep (from 0.0 to 1.0, default 1.0 i.e., clean input)
        :param depths: list of encoder depths to extract features from
        :return: features and labels
        '''
        self.model.eval()
        self.vae.eval()
        accelerator = accelerate.Accelerator()
        self.model, self.vae, dataloader = accelerator.prepare(self.model, self.vae, dataloader)
        device = accelerator.device
        all_features = []
        all_labels = []

        for (x, label) in tqdm(dataloader, desc="Feature Extraction", leave=False):

            with torch.no_grad():
                if self.vae is not None:
                    # if x has one channel, make it 3 channels
                    if x.shape[1] == 1:
                        x = torch.cat((x, x, x), dim=1)
                    x = self.encode(x)

                t_batch = torch.ones(x.size(0), device=device) * t

                features = self.model.forward_features(x, t_batch, torch.ones(x.size(0), device=device)*self.num_classes, encoder_depths=depths)

                all_features.append(features.cpu().float())
                all_labels.append(label.cpu())