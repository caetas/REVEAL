import inspect
import torch
import torch.nn.functional as F
from typing import Callable, Dict
import numpy as np

# =========================================
# Registry / Factory with kwarg filtering
# =========================================

_LOSS_REGISTRY: Dict[str, Callable[..., "ProjectionLoss"]] = {}

def register_loss(name: str):
    def deco(cls):
        _LOSS_REGISTRY[name] = cls
        cls.__loss_name__ = name
        return cls
    return deco

def available_losses():
    return sorted(_LOSS_REGISTRY.keys())

def _apply_aliases(cls, kwargs: dict) -> dict:
    # Optional per-class alias map, e.g. {"temperature": "tau", "t": "tau"}
    aliases = getattr(cls, "KWARG_ALIASES", None) or {}
    out = dict(kwargs)
    for a, target in aliases.items():
        if a in out and target not in out:
            out[target] = out.pop(a)
    return out

def make_projection_loss(name: str, strict: bool = False, **kwargs) -> "ProjectionLoss":
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {available_losses()}")
    cls = _LOSS_REGISTRY[name]
    kw = _apply_aliases(cls, kwargs)
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in kw.items() if k in sig.parameters}
    unused = {k: v for k, v in kw.items() if k not in sig.parameters}
    if strict and unused:
        raise TypeError(f"Unused kwargs for loss '{name}': {sorted(unused)}")
    return cls(**valid)

# =========================================
# Base
# =========================================

class ProjectionLoss:
    """All projection losses implement __call__(zs, zs_tilde, **kwargs) with tensors shaped [B, T, D]."""
    def _check(self, zs, zs_tilde):
        if zs.ndim != 3 or zs_tilde.ndim != 3:
            raise ValueError(f"zs and zs_tilde must be [B,T,D]; got {zs.shape=} {zs_tilde.shape=}")
        if zs.shape != zs_tilde.shape:
            raise ValueError(f"Shape mismatch: {zs.shape=} vs {zs_tilde.shape=}")

    def __call__(self, zs, zs_tilde, **kwargs):
        raise NotImplementedError

# =========================================
# Cosine
# =========================================

@register_loss("cosine")
class CosineProjectionLoss(ProjectionLoss):
    # accepts only these kwargs; others will be ignored by factory unless strict=True
    def __init__(self, **kwargs):
        pass

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)
        # normalize zs and zs_tilde
        zs = F.normalize(zs, dim=-1)
        zs_tilde = F.normalize(zs_tilde, dim=-1)
        # compute cosine similarity
        cos_sim = (zs * zs_tilde).sum(dim=-1)    # [B,T]
        loss = -cos_sim
        return loss.mean()

########################################################
# Loss for the denoising step
########################################################
def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


class SILoss:
    def __init__(
            self,
            accelerator=None, 
            projection_loss_type="cosine",
            projection_loss_kwargs={},
            proj_coeff=[0.5],
        ):
        self.accelerator = accelerator
        # parse projection loss type and coeff
        self.projection_loss_type = [elem.strip() for elem in projection_loss_type.split(",") if elem.strip()]
        self.proj_coeff = [float(elem.strip()) for elem in proj_coeff.split(",") if elem.strip()]
        assert len(self.projection_loss_type) == len(self.proj_coeff), \
            f"len(self.projection_loss_type) - {len(self.projection_loss_type)} != len(self.proj_coeff) - {len(self.proj_coeff)}"
        self.projection_loss_kwargs = projection_loss_kwargs
        # create projection loss
        self.projection_loss = [
            make_projection_loss(projection_loss_type, **projection_loss_kwargs)
            for projection_loss_type in self.projection_loss_type
        ]
        assert len(self.projection_loss) == len(self.proj_coeff), \
            f"len(self.projection_loss) - {len(self.projection_loss)} != len(self.proj_coeff) - {len(self.proj_coeff)}"
        

    def interpolant(self, t):
        # linear path
        alpha_t = t
        sigma_t = 1 - t
        d_alpha_t = 1
        d_sigma_t =  -1

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=FileNotFoundError):
        if model_kwargs is None:
            model_kwargs = {}

        # sample timesteps
        time_input = torch.randn((images.shape[0], 1, 1, 1), device=images.device)*0.8 - 0.8
        time_input = torch.sigmoid(time_input)
                
        time_input = time_input.to(dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        
        model_target = d_alpha_t * images + d_sigma_t * noises

        model_output, zs_tilde, zs_tilde_original = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection loss
        total_proj_loss = 0.
        proj_loss_dict = {}
        # loop across different projection losses [e.g. cosine, nt-xent, p2p-gram-cossim]
        for proj_loss_name, proj_loss_fn, coeff in zip(self.projection_loss_type, self.projection_loss, self.proj_coeff):
            proj_loss = torch.tensor(0.0, device=images.device, dtype=images.dtype)
            if len(zs) > 0 and len(zs_tilde) > 0:
                # loop across different encoders
                for z, z_tilde, z_tilde_original in zip(zs, zs_tilde, zs_tilde_original):
                    # NOTE: We pass vision_feats, projected_sit_feats, and unprojected_sit_feats, but the last one might not be used
                    proj_loss = proj_loss + proj_loss_fn(z, z_tilde, z_tilde_original)
                proj_loss /= len(zs)
            proj_loss_dict[proj_loss_name] = proj_loss.detach().item()
            proj_loss_dict[f"{proj_loss_name}_weighted"] = proj_loss.detach().item() * coeff
            total_proj_loss = total_proj_loss + coeff * proj_loss
        return denoising_loss, total_proj_loss, proj_loss_dict