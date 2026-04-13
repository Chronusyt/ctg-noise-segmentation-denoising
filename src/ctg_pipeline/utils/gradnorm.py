"""GradNorm loss balancing utilities for multitask training."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradNormBalancer(nn.Module):
    """GradNorm task balancer with normalized positive task weights."""

    def __init__(
        self,
        task_keys: Iterable[str],
        initial_weights: Dict[str, float],
        alpha: float = 1.5,
    ) -> None:
        super().__init__()
        self.task_keys = tuple(task_keys)
        if not self.task_keys:
            raise ValueError("GradNorm requires at least one task")
        init = torch.tensor(
            [max(float(initial_weights.get(key, 1.0)), 1e-6) for key in self.task_keys],
            dtype=torch.float32,
        )
        init = init * (len(self.task_keys) / init.sum())
        self.log_task_weights = nn.Parameter(init.log())
        self.alpha = float(alpha)
        self.register_buffer("initial_losses", torch.zeros(len(self.task_keys), dtype=torch.float32))
        self.register_buffer("has_initial_losses", torch.tensor(False, dtype=torch.bool))

    def normalized_weights(self) -> torch.Tensor:
        weights = torch.exp(self.log_task_weights)
        return weights * (len(self.task_keys) / weights.sum())

    def task_vector(self, task_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([task_losses[key] for key in self.task_keys])

    def maybe_initialize(self, task_losses: Dict[str, torch.Tensor]) -> None:
        if bool(self.has_initial_losses.item()):
            return
        with torch.no_grad():
            self.initial_losses.copy_(self.task_vector(task_losses).detach().clamp(min=1e-8))
            self.has_initial_losses.fill_(True)

    def weighted_total(self, task_losses: Dict[str, torch.Tensor], detach_weights: bool) -> tuple[torch.Tensor, OrderedDict[str, float]]:
        weights = self.normalized_weights()
        if detach_weights:
            weights = weights.detach()
        weighted = OrderedDict()
        total = torch.zeros((), device=weights.device, dtype=weights.dtype)
        for idx, key in enumerate(self.task_keys):
            total = total + weights[idx] * task_losses[key]
            weighted[key] = float(weights[idx].detach().cpu())
        return total, weighted

    def gradnorm_loss(self, task_losses: Dict[str, torch.Tensor], reference_parameter: torch.nn.Parameter) -> torch.Tensor:
        self.maybe_initialize(task_losses)
        weights = self.normalized_weights()
        task_vector = self.task_vector(task_losses)
        weighted_losses = weights * task_vector

        norms = []
        for weighted_loss in weighted_losses:
            grad = torch.autograd.grad(
                weighted_loss,
                reference_parameter,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )[0]
            if grad is None:
                norms.append(torch.zeros((), device=weighted_loss.device, dtype=weighted_loss.dtype))
            else:
                norms.append(grad.norm(p=2))
        norms = torch.stack(norms)

        with torch.no_grad():
            relative_losses = (task_vector.detach().clamp(min=1e-8) / self.initial_losses.clamp(min=1e-8))
            inverse_train_rates = relative_losses / relative_losses.mean()
            target_norms = norms.detach().mean() * (inverse_train_rates ** self.alpha)
        return F.l1_loss(norms, target_norms, reduction="sum")

    def renormalize_(self) -> None:
        with torch.no_grad():
            weights = self.normalized_weights()
            self.log_task_weights.copy_(weights.log())
