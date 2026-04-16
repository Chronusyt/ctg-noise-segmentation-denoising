"""Model definitions used across CTG denoising experiments."""

from .modern_tcn_backbone import ModernTCNBackbone1D
from .multiscale_tcn_unet_backbone import (
    ConvNeXtUNetBackbone1D,
    ModernTCNUNetBackbone1D,
    MultiscaleModernTCNUNetBackbone1D,
    MultiscaleTCNUNetBackbone1D,
    MultiscaleUNetBackbone1D,
    TCNUNetBackbone1D,
    UNetBackbone1D,
)
from .unet1d_denoiser import UNet1DDenoiser
from .unet1d_mask_guided_denoiser import UNet1DMaskGuidedDenoiser
from .unet1d_multilabel_segmentation import UNet1DMultilabelSegmentation
from .unet1d_physiological_multitask import UNet1DPhysiologicalMultitask
from .unet1d_segmentation import UNet1DSegmentation

__all__ = [
    "ModernTCNBackbone1D",
    "ConvNeXtUNetBackbone1D",
    "ModernTCNUNetBackbone1D",
    "MultiscaleTCNUNetBackbone1D",
    "MultiscaleModernTCNUNetBackbone1D",
    "MultiscaleUNetBackbone1D",
    "TCNUNetBackbone1D",
    "UNetBackbone1D",
    "UNet1DDenoiser",
    "UNet1DMaskGuidedDenoiser",
    "UNet1DMultilabelSegmentation",
    "UNet1DPhysiologicalMultitask",
    "UNet1DSegmentation",
]
