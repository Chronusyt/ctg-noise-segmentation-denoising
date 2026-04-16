"""Lightweight same-parent neighboring-chunk context conditioning modules."""
from __future__ import annotations

import torch
import torch.nn as nn


class ContextChunkEncoder(nn.Module):
    """Encode one 1min context chunk into a compact vector."""

    def __init__(self, in_channels: int = 6, hidden_channels: int = 32, embedding_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ContextConditioner(nn.Module):
    """
    Encode neighboring 1min chunks and modulate center-window features.

    Inputs:
    - context_noisy_signal: [B, K, 1, L]
    - context_pred_mask: [B, K, 5, L]
    - context_valid: [B, K]
    """

    def __init__(
        self,
        time_channels: int,
        global_channels: int,
        context_in_channels: int = 6,
        context_embedding_dim: int = 64,
        context_fusion: str = "film",
    ) -> None:
        super().__init__()
        if context_fusion != "film":
            raise ValueError(f"Unsupported context_fusion: {context_fusion}")

        self.context_fusion = context_fusion
        self.encoder = ContextChunkEncoder(
            in_channels=context_in_channels,
            hidden_channels=32,
            embedding_dim=context_embedding_dim,
        )
        self.time_film = nn.Linear(context_embedding_dim, time_channels * 2)
        self.global_delta = nn.Linear(context_embedding_dim, global_channels)

    def encode_context(
        self,
        context_noisy_signal: torch.Tensor,
        context_pred_mask: torch.Tensor,
        context_valid: torch.Tensor,
    ) -> torch.Tensor:
        if context_noisy_signal.ndim != 4 or context_pred_mask.ndim != 4:
            raise ValueError("Context tensors must have shape [B, K, C, L]")
        if context_noisy_signal.shape[:2] != context_pred_mask.shape[:2]:
            raise ValueError("context_noisy_signal/context_pred_mask batch and K must match")
        if context_valid.shape != context_noisy_signal.shape[:2]:
            raise ValueError("context_valid must have shape [B, K]")

        bsz, num_chunks, _, signal_len = context_noisy_signal.shape
        context_input = torch.cat([context_noisy_signal, context_pred_mask], dim=2)
        encoded = self.encoder(context_input.reshape(bsz * num_chunks, -1, signal_len))
        encoded = encoded.reshape(bsz, num_chunks, -1)

        valid = context_valid.to(encoded.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = (encoded * valid).sum(dim=1) / denom
        has_valid = (context_valid.sum(dim=1, keepdim=True) > 0).to(encoded.dtype)
        return pooled * has_valid

    def condition_features(
        self,
        time_features: torch.Tensor,
        global_features: torch.Tensor,
        context_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        film = self.time_film(context_global)
        gamma, beta = torch.chunk(film, 2, dim=1)
        time_features = time_features * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

        delta = self.global_delta(context_global)
        if global_features.ndim == 3:
            global_features = global_features + delta.unsqueeze(-1)
        elif global_features.ndim == 2:
            global_features = global_features + delta
        else:
            raise ValueError(f"Unsupported global_features ndim: {global_features.ndim}")
        return time_features, global_features

    def forward(
        self,
        time_features: torch.Tensor,
        global_features: torch.Tensor,
        context_noisy_signal: torch.Tensor,
        context_pred_mask: torch.Tensor,
        context_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_global = self.encode_context(context_noisy_signal, context_pred_mask, context_valid)
        time_features, global_features = self.condition_features(time_features, global_features, context_global)
        return time_features, global_features, context_global
