import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from pathlib import Path
from torchvision.transforms import Normalize
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import os
from torchvision import transforms
from pathlib import Path
from config import models_dir

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

# Compute pretrained models directory relative to this file
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(os.path.join(models_dir, "pretrained_models")):
    os.makedirs(os.path.join(models_dir, "pretrained_models"))

PRETRAINED_DIR = os.path.join(models_dir, "pretrained_models")

def make_dinov3_transform(resize_size: int = 224):
    to_tensor = transforms.Lambda(lambda x: x / 255.)
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])


MODEL_NAMES = {
    'dinov3_vits16',
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
}
SHA_CHECKSUM = {
    "dinov3_vits16": "08c60483",
    "dinov3_vits16plus": "4057cbaa",
    "dinov3_vitb16": "73cec8be",
    "dinov3_vitl16": "8aa4cbdd",
    "dinov3_vith16plus": "7c1da9a5",
    "dinov3_vit7b16": "a955f4ea",
}
METHOD_ROOT = Path(__file__).resolve().parents[1]
print(f"using METHOD_ROOT: {METHOD_ROOT}")
REPO_DIR = os.environ.get("DINOV3_REPO_DIR", os.path.join(METHOD_ROOT.parent, "dinov3"))
CHECKPOINT_DIR = os.environ.get("DINOV3_CKPT_DIR", os.path.join(METHOD_ROOT, "..", "pretrained_models"))

def load_dinov3(model_name, checkpoint_path: Optional[str] = None):
    assert model_name in MODEL_NAMES
    model = torch.hub.load(
        REPO_DIR,
        model_name,
        source='local',
        untie_global_and_local_cls_norm=True if checkpoint_path is not None else False,
        weights=checkpoint_path if checkpoint_path is not None else os.path.join(PRETRAINED_DIR, f"{model_name}_pretrain_lvd1689m-{SHA_CHECKSUM[model_name]}.pth")
    )
    return model


class VisionEncoder(ABC):
    """Base class for all vision encoders"""
    
    def __init__(self, encoder_type: str, architecture: str, model_config: str, 
                 device: torch.device, resolution: int = 256, accelerator=None):
        self.encoder_type = encoder_type
        self.architecture = architecture
        self.model_config = model_config
        self.device = device
        self.resolution = resolution
        self.accelerator = accelerator
        self._embed_dim = None
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """Load and initialize the encoder model"""
        pass
        
    @abstractmethod
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess raw images
        Args:
            x: Raw images tensor (B, C, H, W) in range [0, 255]
        Returns:
            Preprocessed tensor ready for encoder
        """
        pass
        
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass through encoder
        Args:
            x: Preprocessed images
        Returns:
            Dictionary with:
                - 'x_norm_clstoken': (B, D) CLS token or None if not available
                - 'x_norm_patchtokens': (B, T, D) patch tokens
        """
        # Default implementation - subclasses should override if needed
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            return out
        else:
            # Assume it's just patch tokens
            return {
                'x_norm_clstoken': None,
                'x_norm_patchtokens': out
            }
    
    @property
    def embed_dim(self) -> int:
        return self._embed_dim
    
    def eval(self):
        """Set model to eval mode"""
        if self.model is not None:
            self.model.eval()
        return self
    
    def to(self, device):
        """Move model to device"""
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self


class DINOEncoder(VisionEncoder):
    """DINO encoder implementation"""
    
    def load_model(self):
        import timm
        
        # Load model from torch hub
        model_name = f'dino_vit{self.model_config}16'
        
        if self.accelerator is not None:
            with self.accelerator.main_process_first():
                self.model = torch.hub.load('facebookresearch/dino:main', model_name)
        else:
            self.model = torch.hub.load('facebookresearch/dino:main', model_name)
        
        # Remove head
        del self.model.head
        self.model.head = torch.nn.Identity()
        
        # Resample position embeddings if needed
        patch_resolution = 16 * (self.resolution // 256)
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution],
        )
        
        # Set embed dim
        self._embed_dim = self.model.embed_dim
        
        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        # Interpolate if needed
        x = torch.nn.functional.interpolate(x, self.resolution, mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # DINO returns a dictionary with cls and patch tokens
        out = self.model.get_intermediate_layers(x)[0]
        return {
            'x_norm_clstoken': out[:, 0],
            'x_norm_patchtokens': out[:, 1:]
        }


class DINOv2Encoder(VisionEncoder):
    """DINOv2 encoder implementation"""
    
    def load_model(self):
        import timm
        
        # Determine if using register tokens
        use_reg = 'reg' in self.encoder_type
        
        # Load model from torch hub
        model_name = f'dinov2_vit{self.model_config}14{"_reg" if use_reg else ""}'
        
        if self.accelerator is not None:
            with self.accelerator.main_process_first():
                self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # Remove head
        del self.model.head
        self.model.head = torch.nn.Identity()
        
        # Resample position embeddings if needed
        patch_resolution = 16 * (self.resolution // 256)
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution],
        )
        
        # Set embed dim
        self._embed_dim = self.model.embed_dim
        
        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        # Interpolate if needed
        x = torch.nn.functional.interpolate(x, 224 * (self.resolution // 256), mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # DINOv2 returns a dictionary with cls and patch tokens
        out = self.model.forward_features(x)
        return {
            'x_norm_clstoken': out.get('x_norm_clstoken'),
            'x_norm_patchtokens': out.get('x_norm_patchtokens')
        }


class DINOv2MixedEncoder(DINOv2Encoder):
    """DINOv2 encoder with mixed CLS and patch tokens"""
    
    def __init__(self, *args, alpha: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward_features(x)
        cls_token = out['x_norm_clstoken']
        patch_tokens = out['x_norm_patchtokens']
        
        # Mix CLS token into patch tokens
        mixed_patch_tokens = cls_token[:, None, :] * self.alpha + patch_tokens * (1 - self.alpha)
        
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': mixed_patch_tokens
        }


class DINOv3Encoder(VisionEncoder):
    """DINOv3 encoder implementation"""
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        
        self.model = load_dinov3(f"dinov3_vit{self.model_config}", checkpoint_path=checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set embed dim
        self._embed_dim = self.model.embed_dim
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        transform_func = make_dinov3_transform(resize_size=self.resolution)
        return transform_func(x)
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward_features(x)
        return {
            'x_norm_clstoken': out.get('x_norm_clstoken'),
            'x_norm_patchtokens': out.get('x_norm_patchtokens')
        }


class SAMEncoder(VisionEncoder):
    def load_model(self):
        from transformers import SamModel

        if self.model_config == "b":
            model_name = "facebook/sam-vit-base"
        elif self.model_config == "l":
            model_name = "facebook/sam-vit-large"
        elif self.model_config == "h":
            model_name = "facebook/sam-vit-huge"
        else:
            raise NotImplementedError(f"model size {self.model_config} not supported")

        sam_model = SamModel.from_pretrained(model_name).to(self.device).eval()
        self.model = sam_model.vision_encoder
        self._embed_dim = self.model.config.output_channels

    def preprocess(self, x):
        # SAM only takes 1024 input
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 1024, mode='bicubic')
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        return x

    def forward_features(self, x):
        out = self.model(x)
        hidden_states = out.last_hidden_state
        if hidden_states.shape[-1] != self.resolution // 16:
            hidden_states = torch.nn.functional.interpolate(
                hidden_states, 
                size=(self.resolution // 16, self.resolution // 16), 
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        return {
        'x_norm_clstoken': None,
        'x_norm_patchtokens': hidden_states.view(hidden_states.shape[0], -1, hidden_states.shape[-1]),
    }


class SAM2Encoder(VisionEncoder):
    def load_model(self):
        from transformers import Sam2Model

        if self.model_config == "s":
            model_name = "facebook/sam2-hiera-small"
        elif self.model_config == "b":
            model_name = "facebook/sam2-hiera-base-plus"
        elif self.model_config == "l":
            model_name = "facebook/sam2-hiera-large"
        else:
            raise NotImplementedError(f"model size {self.model_config} not supported")

        sam_model = Sam2Model.from_pretrained(model_name).to(self.device).eval()
        self.model = sam_model.vision_encoder
        self._embed_dim = self.model.config.backbone_config.embed_dim_per_stage[-1]

    def preprocess(self, x):
        # SAM2 has 32x downsample rate
        x = x / 255.
        x = torch.nn.functional.interpolate(x, self.resolution * 2, mode='bicubic')
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        return x

    def forward_features(self, x):
        out = self.model(x)
        hidden_states = out.last_hidden_state
        return {
        'x_norm_clstoken': None,
        'x_norm_patchtokens': hidden_states.view(hidden_states.shape[0], -1, hidden_states.shape[-1]),
    }


class SAM2LogitEncoder(VisionEncoder):
    @staticmethod
    def make_grids(grid_size, H=1024, W=1024):
        y_coords = torch.linspace(0, H-1, grid_size)
        x_coords = torch.linspace(0, W-1, grid_size)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [grid_size*grid_size, 2]
        return grid_points

    def load_model(self):
        from transformers import Sam2Model, Sam2Processor

        model_config = self.model_config[0]
        grid_size = int(self.model_config[1:])

        if model_config == "s":
            model_name = "facebook/sam2-hiera-small"
        elif model_config == "b":
            model_name = "facebook/sam2-hiera-base-plus"
        elif model_config == "l":
            model_name = "facebook/sam2-hiera-large"
        else:
            raise NotImplementedError(f"model size {model_config} not supported")

        self.model = Sam2Model.from_pretrained(model_name).to(self.device).eval()
        self.processor = Sam2Processor.from_pretrained(model_name)
        self._embed_dim = grid_size * grid_size
        self.grid_points = self.make_grids(grid_size)
        self.target_resolution = self.resolution // 16

    def preprocess(self, x):
        # Preprocess is in the forward
        return x

    def forward_features(self, x):
        B = x.shape[0]
        num_points = self.grid_points.shape[0]
        max_batch = 64  # NOTE: Points per batch, should be smaller on 80G VRAM

        all_results = {i: [] for i in range(B)}  # Store results per image

        # Process points in batches
        for point_idx in range(0, num_points, max_batch):
            end_idx = min(point_idx + max_batch, num_points)
            batch_points = self.grid_points[point_idx:end_idx]
            current_batch = end_idx - point_idx
            
            # Format for batch processing: each image gets same points as separate objects
            input_points = []
            input_labels = []
            
            for img_idx in range(B):
                # Each point is a separate object with 1 point
                # Format: [[[x1, y1]], [[x2, y2]], ...] for image
                points_for_image = []
                labels_for_image = []
                
                for i in range(current_batch):
                    points_for_image.append([[batch_points[i, 0].item(), batch_points[i, 1].item()]])
                    labels_for_image.append([1])
                
                input_points.append(points_for_image)
                input_labels.append(labels_for_image)
            
            with torch.no_grad():
                inputs = self.processor(
                    images=[x[i] for i in range(B)],
                    input_points=input_points,
                    input_labels=input_labels,
                    return_tensors="pt"
                ).to("cuda")
                
                outputs = self.model(**inputs, multimask_output=False)
                # Shape: [B, current_batch, 1, H, W] - one mask per point per image
                mask_logits = outputs.pred_masks.squeeze(2)  # Remove the channel dimension -> [B, current_batch, H, W]
            
            # Distribute masks to correct image
            for img_idx in range(B):
                # Get masks for this image from this batch
                image_masks = mask_logits[img_idx]  # [current_batch, H, W]
                all_results[img_idx].append(image_masks)

        # Combine results per image: [B, num_points, H, W]
        dense_features = torch.stack([
            torch.cat(all_results[i], dim=0) for i in range(B)
        ], dim=0)

        # Downsample if needed
        dense_features = torch.nn.functional.interpolate(
            dense_features, 
            size=(self.target_resolution, self.target_resolution), 
            mode='bilinear',
            align_corners=False
        )
        return {
            'x_norm_clstoken': None,
            'x_norm_patchtokens': dense_features.view(B, dense_features.shape[1], -1).permute(0, 2, 1),
        }


class PEEncoder(VisionEncoder):
    """PE (Perceptual Encoder) implementation"""
    
    def load_model(self):
        import models.pe as pe
        
        # Check if using normalization
        self.use_norm = self.model_config.endswith("norm")
        if self.use_norm:
            config_name = self.model_config[:-4]
        else:
            config_name = self.model_config
        
        # Map config to model name
        if self.encoder_type == "pe":
            config_map = {
                "t": "PE-Core-T16-384",
                "s": "PE-Core-S16-384",
                "b": "PE-Core-B16-224",
                "l": "PE-Core-L14-336",
                "g": "PE-Core-G14-448"
            }
        elif self.encoder_type == "spatialpe":
            config_map = {
                "b": "PE-Spatial-B16-512",
                "l": "PE-Spatial-L14-448",
                "g": "PE-Spatial-G14-448"
            }
        elif self.encoder_type == "langpe":
            config_map = {
                "l": "PE-Lang-L14-448",
                "g": "PE-Lang-G14-448"
            }
        else:
            raise ValueError(f"Unknown PE encoder type: {self.encoder_type}")
        
        if config_name not in config_map:
            raise ValueError(f"Unknown PE model config: {config_name}")
        
        self.model = pe.VisionTransformer.from_config(config_map[config_name], pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self._embed_dim = self.model.width
        
        # Get patch size for preprocessing
        if config_name in {"t", "s", "b", "tnorm", "snorm", "bnorm"}:
            self.patch_size = 16
        elif config_name in {"l", "g", "lnorm", "gnorm"}:
            self.patch_size = 14
        else:
            raise NotImplementedError()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = torch.nn.functional.interpolate(
            x, self.patch_size * (self.resolution // 16), mode='bilinear'
        )
        x = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # PE returns patch tokens without CLS
        out = self.model.forward_features(x, norm=self.use_norm, strip_cls_token=False)
        if self.model.use_cls_token:
            cls_token = out[:, 0]
            patch_tokens = out[:, 1:]
        else:
            cls_token = None
            patch_tokens = out
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens
        }



# Registry mapping encoder types to classes
ENCODER_REGISTRY = {
    'dino': DINOEncoder,
    # dinov2 and dinov3 encoders
    'dinov2': DINOv2Encoder,
    'dinov2reg': DINOv2Encoder,
    'dinov2mixed': DINOv2MixedEncoder,
    'dinov2mixedreg': DINOv2MixedEncoder,
    'dinov3': DINOv3Encoder,
    # PE encoders
    'pe': PEEncoder,
    'spatialpe': PEEncoder,
    'langpe': PEEncoder,
    # sam encoders
    "sam": SAMEncoder,
    "sam2": SAM2Encoder,
    "sam2logit": SAM2LogitEncoder,
}


def create_encoder(encoder_string: str, device: torch.device, 
                   resolution: int = 256, accelerator=None, checkpoint_path: Optional[str] = None) -> VisionEncoder:
    """
    Factory function to create encoder from string specification
    
    Args:
        encoder_string: Format "encoder_type-architecture-model_config"
        device: torch device
        resolution: Input image resolution
        accelerator: Optional accelerator for distributed training
    
    Returns:
        VisionEncoder instance
    """
    parts = encoder_string.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid encoder string format: {encoder_string}. "
                        f"Expected format: encoder_type-architecture-model_config")
    
    encoder_type, architecture, model_config = parts
    
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Available types: {list(ENCODER_REGISTRY.keys())}")
    
    encoder_class = ENCODER_REGISTRY[encoder_type]
    encoder = encoder_class(encoder_type, architecture, model_config, 
                            device, resolution, accelerator)
    if checkpoint_path is None:
        encoder.load_model()
    else:
        encoder.load_model(checkpoint_path=checkpoint_path)
    
    return encoder


@torch.no_grad()
def load_encoders(enc_type: str, device: torch.device, resolution: int = 256, 
                  accelerator=None, checkpoint_path: Optional[str] = None) -> List[VisionEncoder]:
    """
    Load multiple encoders from comma-separated string
    
    Args:
        enc_type: Comma-separated encoder specifications
        device: torch device
        resolution: Input image resolution
        use_cls_token: Whether to return CLS tokens (for compatibility)
        accelerator: Optional accelerator for distributed training
    
    Returns:
        List of VisionEncoder instances
    """
    # if resolution not in [256, 512]:
    #     raise ValueError(f"Resolution must be 256 or 512, got {resolution}")

    enc_names = enc_type.split(',')
    encoders = []
    
    for enc_name in enc_names:
        # Parse encoder specification
        parts = enc_name.split('-')
        if len(parts) != 3:
            raise ValueError(f"Invalid encoder format: {enc_name}")
        
        encoder = create_encoder(enc_name, device, resolution, accelerator, checkpoint_path=checkpoint_path)
        encoder.eval()
        encoders.append(encoder)
    
    return encoders