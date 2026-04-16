"""
1D physiological multitask model family with configurable backbone and residual heads.

Default behavior remains compatible with the historical shared U-Net model:
- backbone_type = "unet"
- model_variant = "legacy_single_residual"

Supported inputs:
    no-mask:
        noisy FHR [B, 1, L]
    gt-mask / pred-mask:
        noisy FHR + 5-class masks [B, 6, L]

Supported residual heads:
    legacy_single_residual:
        one shared residual decoder
    expert_residual:
        five artifact-specific residual experts fused by mask channels
        (falls back to the legacy path when no mask channels are present)
    typed_scale_residual:
        halving / doubling affine-style scale heads + shared residual head for
        mhr / missing / spike
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from .modern_tcn_backbone import ModernTCNBackbone1D
    from .multiscale_tcn_unet_backbone import (
        ConvNeXtUNetBackbone1D,
        ModernTCNUNetBackbone1D,
        MultiscaleConvNeXtUNetBackbone1D,
        MultiscaleModernTCNUNetBackbone1D,
        MultiscaleTCNUNetBackbone1D,
        MultiscaleUNetBackbone1D,
        TCNUNetBackbone1D,
    )
    from .unet1d_denoiser import DoubleConv1D
except ImportError:  # Allow standalone smoke tests.
    from modern_tcn_backbone import ModernTCNBackbone1D
    from multiscale_tcn_unet_backbone import (
        ConvNeXtUNetBackbone1D,
        ModernTCNUNetBackbone1D,
        MultiscaleConvNeXtUNetBackbone1D,
        MultiscaleModernTCNUNetBackbone1D,
        MultiscaleTCNUNetBackbone1D,
        MultiscaleUNetBackbone1D,
        TCNUNetBackbone1D,
    )
    from unet1d_denoiser import DoubleConv1D


ARTIFACT_CLASS_ORDER = ("halving", "doubling", "mhr", "missing", "spike")
NUM_EXPERTS = len(ARTIFACT_CLASS_ORDER)
NUM_TYPED_BRANCHES = 3
MODEL_VARIANTS = {"legacy_single_residual", "expert_residual", "typed_scale_residual"}
BACKBONE_TYPES = {
    "unet",
    "modern_tcn",
    "multiscale_unet",
    "tcn_unet",
    "multiscale_tcn_unet",
    "convnext1d_unet",
    "multiscale_convnext1d_unet",
    "modern_tcn_unet",
    "multiscale_modern_tcn_unet",
}
BOTTLENECK_TYPES = {"none", "mhsa"}


def _normalize_model_variant(model_variant: str) -> str:
    if model_variant in MODEL_VARIANTS:
        return model_variant
    return "legacy_single_residual"


class ScalarRegressionHead(nn.Module):
    """Global average pooling + small MLP scalar regression head."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(x)).squeeze(-1)


class UNet1DPhysiologicalMultitask(nn.Module):
    """
    Shared-encoder physiological multitask model family.

    Public outputs are preserved:
    - reconstruction [B, 1, L]
    - raw_residual [B, 1, L]
    - gated_residual [B, 1, L]
    - acc_logits / dec_logits [B, 1, L]
    - baseline / stv / ltv / baseline_variability [B]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        scalar_hidden_channels: int = 128,
        dropout: float = 0.0,
        residual_reconstruction: bool = True,
        model_variant: str = "legacy_single_residual",
        backbone_type: str = "unet",
        bottleneck_type: str = "none",
        modern_tcn_blocks_per_stage: int = 2,
        modern_tcn_kernel_size: int = 5,
        modern_tcn_expansion: int = 2,
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.residual_reconstruction = residual_reconstruction
        self.model_variant = _normalize_model_variant(model_variant)
        if backbone_type not in BACKBONE_TYPES:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        if bottleneck_type not in BOTTLENECK_TYPES:
            raise ValueError(f"Unsupported bottleneck_type: {bottleneck_type}")
        self.backbone_type = backbone_type
        self.bottleneck_type = bottleneck_type
        self.num_experts = NUM_EXPERTS
        self.shared_backbone = None
        self.multiscale_tcn_unet_backbone = None

        if self.backbone_type == "unet":
            self.enc = nn.ModuleList()
            ch = in_channels
            for i in range(depth):
                out_ch = base_channels * (2**i)
                self.enc.append(DoubleConv1D(ch, out_ch))
                ch = out_ch

            bottleneck_ch = base_channels * (2**depth)
            self.bottleneck = DoubleConv1D(ch, bottleneck_ch)

            self.dec = nn.ModuleList()
            self.up = nn.ModuleList()
            ch = bottleneck_ch
            for i in range(depth - 1, -1, -1):
                out_ch = base_channels * (2**i)
                self.up.append(nn.ConvTranspose1d(ch, out_ch, kernel_size=2, stride=2))
                self.dec.append(DoubleConv1D(out_ch * 2, out_ch))
                ch = out_ch

            time_channels = ch
            global_channels = bottleneck_ch
        elif self.backbone_type == "modern_tcn":
            self.shared_backbone = ModernTCNBackbone1D(
                in_channels=in_channels,
                base_channels=base_channels,
                depth=depth,
                blocks_per_stage=modern_tcn_blocks_per_stage,
                kernel_size=modern_tcn_kernel_size,
                expansion=modern_tcn_expansion,
                dropout=dropout,
            )
            time_channels = self.shared_backbone.out_channels
            global_channels = self.shared_backbone.out_channels
        elif self.backbone_type == "multiscale_unet":
            self.shared_backbone = MultiscaleUNetBackbone1D(
                in_channels=in_channels,
                base_channels=base_channels,
                depth=depth,
                dropout=dropout,
            )
            time_channels = self.shared_backbone.time_channels
            global_channels = self.shared_backbone.global_channels
        elif self.backbone_type == "tcn_unet":
            self.shared_backbone = TCNUNetBackbone1D(
                in_channels=in_channels,
                base_channels=base_channels,
                depth=depth,
                dropout=dropout,
                bottleneck_type=bottleneck_type,
            )
            time_channels = self.shared_backbone.time_channels
            global_channels = self.shared_backbone.global_channels
        elif self.backbone_type == "multiscale_tcn_unet":
            self.multiscale_tcn_unet_backbone = MultiscaleTCNUNetBackbone1D(
                in_channels=in_channels,
                base_channels=base_channels,
                depth=depth,
                dropout=dropout,
                bottleneck_type=bottleneck_type,
            )
            time_channels = self.multiscale_tcn_unet_backbone.time_channels
            global_channels = self.multiscale_tcn_unet_backbone.global_channels
        elif self.backbone_type == "convnext1d_unet":
            self.shared_backbone = ConvNeXtUNetBackbone1D(
                in_channels=in_channels,
                base_channels=base_channels,
                depth=depth,
                dropout=dropout,
                bottleneck_type=bottleneck_type,
                blocks_per_stage=modern_tcn_blocks_per_stage,
            )
            time_channels = self.shared_backbone.time_channels
            global_channels = self.shared_backbone.global_channels
        elif self.backbone_type == "multiscale_convnext1d_unet":
            self.shared_backbone = MultiscaleConvNeXtUNetBackbone1D(
                in_channels=in_channels,
                base_channels=base_channels,
                depth=depth,
                dropout=dropout,
                bottleneck_type=bottleneck_type,
                blocks_per_stage=modern_tcn_blocks_per_stage,
            )
            time_channels = self.shared_backbone.time_channels
            global_channels = self.shared_backbone.global_channels
        elif self.backbone_type == "modern_tcn_unet":
            self.shared_backbone = ModernTCNUNetBackbone1D(
                in_channels=in_channels,
                base_channels=base_channels,
                depth=depth,
                dropout=dropout,
                bottleneck_type=bottleneck_type,
                blocks_per_stage=modern_tcn_blocks_per_stage,
                kernel_size=modern_tcn_kernel_size,
                expansion=modern_tcn_expansion,
            )
            time_channels = self.shared_backbone.time_channels
            global_channels = self.shared_backbone.global_channels
        else:
            self.shared_backbone = MultiscaleModernTCNUNetBackbone1D(
                in_channels=in_channels,
                base_channels=base_channels,
                depth=depth,
                dropout=dropout,
                bottleneck_type=bottleneck_type,
                blocks_per_stage=modern_tcn_blocks_per_stage,
                kernel_size=modern_tcn_kernel_size,
                expansion=modern_tcn_expansion,
            )
            time_channels = self.shared_backbone.time_channels
            global_channels = self.shared_backbone.global_channels

        # Keep the historical single residual head for compatibility and for the
        # multiscale-backbone shared-residual ablation.
        self.reconstruction_head = nn.Conv1d(time_channels, 1, kernel_size=1)
        if self.model_variant == "expert_residual":
            self.expert_reconstruction_heads = nn.ModuleList(
                [nn.Conv1d(time_channels, 1, kernel_size=1) for _ in range(self.num_experts)]
            )
        else:
            self.expert_reconstruction_heads = None

        if self.model_variant == "typed_scale_residual":
            self.halving_scale_head = nn.Conv1d(time_channels, 1, kernel_size=1)
            self.halving_bias_head = nn.Conv1d(time_channels, 1, kernel_size=1)
            self.doubling_scale_head = nn.Conv1d(time_channels, 1, kernel_size=1)
            self.doubling_bias_head = nn.Conv1d(time_channels, 1, kernel_size=1)
            self.other_shared_head = nn.Conv1d(time_channels, 1, kernel_size=1)
        else:
            self.halving_scale_head = None
            self.halving_bias_head = None
            self.doubling_scale_head = None
            self.doubling_bias_head = None
            self.other_shared_head = None

        self.acceleration_head = nn.Conv1d(time_channels, 1, kernel_size=1)
        self.deceleration_head = nn.Conv1d(time_channels, 1, kernel_size=1)

        self.baseline_head = ScalarRegressionHead(global_channels, scalar_hidden_channels, dropout)
        self.stv_head = ScalarRegressionHead(global_channels, scalar_hidden_channels, dropout)
        self.ltv_head = ScalarRegressionHead(global_channels, scalar_hidden_channels, dropout)
        self.baseline_variability_head = ScalarRegressionHead(global_channels, scalar_hidden_channels, dropout)

    def _forward_unet_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skips = []
        h = x
        for i, block in enumerate(self.enc):
            h = block(h)
            skips.append(h)
            if i < self.depth - 1:
                h = nn.functional.max_pool1d(h, 2)

        bottleneck = self.bottleneck(h)
        h = bottleneck

        for i, (up, block) in enumerate(zip(self.up, self.dec)):
            h = up(h)
            s = skips[-(i + 1)]
            if h.shape[-1] != s.shape[-1]:
                h = h[..., : s.shape[-1]]
            h = torch.cat([h, s], dim=1)
            h = block(h)
        return h, bottleneck

    def _forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone_type == "unet":
            return self._forward_unet_features(x)
        if self.backbone_type == "multiscale_tcn_unet":
            shared = self.multiscale_tcn_unet_backbone(x)
            return shared["time_features"], shared["global_features"]
        shared = self.shared_backbone(x)
        return shared["time_features"], shared["global_features"]

    def _expert_routing_weights(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.model_variant != "expert_residual":
            return None
        if x.shape[1] < 1 + self.num_experts:
            return None
        return x[:, 1 : 1 + self.num_experts, :]

    def _typed_routing_weights(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.model_variant != "typed_scale_residual":
            return None
        if x.shape[1] < 1 + self.num_experts:
            return None
        halving_mask = x[:, 1:2, :]
        doubling_mask = x[:, 2:3, :]
        other_mask = x[:, 3:6, :].amax(dim=1, keepdim=True)
        weights = torch.cat([halving_mask, doubling_mask, other_mask], dim=1)
        weight_sum = weights.sum(dim=1, keepdim=True)
        normalized = torch.where(weight_sum > 0, weights / weight_sum.clamp(min=1e-8), torch.zeros_like(weights))
        return normalized

    def _typed_branch_residuals(self, noisy_signal: torch.Tensor, time_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if (
            self.halving_scale_head is None
            or self.halving_bias_head is None
            or self.doubling_scale_head is None
            or self.doubling_bias_head is None
            or self.other_shared_head is None
        ):
            raise RuntimeError("typed_scale_residual heads are not initialized")

        halving_delta_scale = self.halving_scale_head(time_features)
        halving_delta_bias = self.halving_bias_head(time_features)
        doubling_delta_scale = self.doubling_scale_head(time_features)
        doubling_delta_bias = self.doubling_bias_head(time_features)
        halving_residual = noisy_signal * halving_delta_scale + halving_delta_bias
        doubling_residual = noisy_signal * doubling_delta_scale + doubling_delta_bias
        other_residual = self.other_shared_head(time_features)
        branch_residuals = torch.cat([halving_residual, doubling_residual, other_residual], dim=1)
        return {
            "halving_residual": halving_residual,
            "doubling_residual": doubling_residual,
            "other_residual": other_residual,
            "typed_branch_residuals": branch_residuals,
        }

    def _fused_residual(self, x: torch.Tensor, time_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.model_variant == "typed_scale_residual":
            routing_weights = self._typed_routing_weights(x)
            if routing_weights is None:
                raise ValueError("typed_scale_residual requires noisy + 5-class mask input [B, 6, L]")
            typed_branches = self._typed_branch_residuals(x[:, :1, :], time_features)
            raw_residual = torch.sum(typed_branches["typed_branch_residuals"] * routing_weights, dim=1, keepdim=True)
            return {
                "raw_residual": raw_residual,
                "halving_residual": typed_branches["halving_residual"],
                "doubling_residual": typed_branches["doubling_residual"],
                "other_residual": typed_branches["other_residual"],
                "typed_routing_weights": routing_weights,
            }

        routing_weights = self._expert_routing_weights(x)
        if routing_weights is None or self.expert_reconstruction_heads is None:
            raw_residual = self.reconstruction_head(time_features)
            return {
                "raw_residual": raw_residual,
            }

        expert_residuals = torch.cat([head(time_features) for head in self.expert_reconstruction_heads], dim=1)
        raw_residual = torch.sum(expert_residuals * routing_weights, dim=1, keepdim=True)
        return {
            "raw_residual": raw_residual,
            "expert_residuals": expert_residuals,
            "routing_weights": routing_weights,
        }

    def gradnorm_reference_parameter(self) -> torch.nn.Parameter:
        if self.backbone_type == "unet":
            return self.enc[0].conv[0].weight
        if self.backbone_type == "multiscale_tcn_unet":
            return self.multiscale_tcn_unet_backbone.reference_parameter()
        if hasattr(self.shared_backbone, "reference_parameter"):
            return self.shared_backbone.reference_parameter()
        return self.shared_backbone.stem[0].weight

    def forward(self, x: torch.Tensor, edit_gate: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        time_features, global_features = self._forward_features(x)
        residual_outputs = self._fused_residual(x, time_features)
        raw_residual = residual_outputs["raw_residual"]
        gated_residual = raw_residual if edit_gate is None else raw_residual * edit_gate
        reconstruction = gated_residual
        if self.residual_reconstruction:
            reconstruction = reconstruction + x[:, :1, :]

        outputs = {
            "reconstruction": reconstruction,
            "raw_residual": raw_residual,
            "gated_residual": gated_residual,
            "acc_logits": self.acceleration_head(time_features),
            "dec_logits": self.deceleration_head(time_features),
            "baseline": self.baseline_head(global_features),
            "stv": self.stv_head(global_features),
            "ltv": self.ltv_head(global_features),
            "baseline_variability": self.baseline_variability_head(global_features),
        }
        outputs.update({k: v for k, v in residual_outputs.items() if k != "raw_residual"})
        return outputs


def _test() -> None:
    common_backbones = (
        "unet",
        "modern_tcn",
        "multiscale_unet",
        "tcn_unet",
        "multiscale_tcn_unet",
        "convnext1d_unet",
        "multiscale_convnext1d_unet",
        "modern_tcn_unet",
        "multiscale_modern_tcn_unet",
    )
    for backbone_type in common_backbones:
        for in_channels in (1, 6):
            model = UNet1DPhysiologicalMultitask(
                in_channels=in_channels,
                base_channels=16,
                depth=4,
                model_variant="legacy_single_residual",
                backbone_type=backbone_type,
            )
            x = torch.randn(2, in_channels, 240)
            gate = torch.rand(2, 1, 240)
            y = model(x, edit_gate=gate)
            assert y["reconstruction"].shape == (2, 1, 240)
            assert y["raw_residual"].shape == (2, 1, 240)
            assert y["gated_residual"].shape == (2, 1, 240)
            assert y["acc_logits"].shape == (2, 1, 240)
            assert y["dec_logits"].shape == (2, 1, 240)
            for key in ("baseline", "stv", "ltv", "baseline_variability"):
                assert y[key].shape == (2,), f"{key}: {y[key].shape}"

    for backbone_type in ("unet", "modern_tcn", "tcn_unet", "convnext1d_unet", "multiscale_convnext1d_unet", "modern_tcn_unet"):
        model = UNet1DPhysiologicalMultitask(
            in_channels=6,
            base_channels=16,
            depth=4,
            model_variant="expert_residual",
            backbone_type=backbone_type,
        )
        y = model(torch.randn(2, 6, 240), edit_gate=torch.rand(2, 1, 240))
        assert y["expert_residuals"].shape == (2, 5, 240)
        assert y["routing_weights"].shape == (2, 5, 240)

    for bottleneck_type in ("none", "mhsa"):
        model = UNet1DPhysiologicalMultitask(
            in_channels=6,
            base_channels=16,
            depth=4,
            model_variant="typed_scale_residual",
            backbone_type="multiscale_tcn_unet",
            bottleneck_type=bottleneck_type,
        )
        y = model(torch.randn(2, 6, 240), edit_gate=torch.rand(2, 1, 240))
        assert y["halving_residual"].shape == (2, 1, 240)
        assert y["doubling_residual"].shape == (2, 1, 240)
        assert y["other_residual"].shape == (2, 1, 240)
        assert y["typed_routing_weights"].shape == (2, 3, 240)
    print("UNet1DPhysiologicalMultitask test OK")


if __name__ == "__main__":
    _test()
