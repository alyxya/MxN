#!/usr/bin/env python3
from typing import Any, Dict

import torch

from memory_matrix_network import MemoryMatrixNetwork
from memory_utils import apply_rotation, newton_schulz_orthogonalize


class MemoryMatrixNetworkOptimizer:
    def __init__(
        self,
        model: MemoryMatrixNetwork,
        *,
        momentum_decay: float,
        base_lr: float,
        token_lr: float,
        momentum_weight: float = 0.0,
        update_noise_scale: float = 0.0,
        orthogonalize_period: int = 0,
    ):
        if not 0.0 <= momentum_decay < 1.0:
            raise ValueError("momentum_decay must be >= 0 and < 1")
        if not 0.0 <= momentum_weight <= 1.0:
            raise ValueError("momentum_weight must be >= 0 and <= 1")

        self.model = model
        self.momentum_decay = momentum_decay
        self.base_lr = base_lr
        self.token_lr = token_lr
        self.momentum_weight = momentum_weight
        self.update_noise_scale = update_noise_scale
        self.orthogonalize_period = orthogonalize_period
        self.update_count = 0
        self.base_momentum = None
        self.token_momentum = None
        if self.momentum_weight > 0.0:
            self.base_momentum = torch.zeros_like(model.base_mat)
            self.token_momentum = torch.zeros_like(model.token_mats)

    @torch.no_grad()
    def step(self, base_update_terms: torch.Tensor, token_update_terms: torch.Tensor) -> None:
        if self.base_momentum is None or self.token_momentum is None:
            base_update = base_update_terms
            token_update = token_update_terms
        else:
            self.base_momentum.mul_(self.momentum_decay).add_(base_update_terms * (1.0 - self.momentum_decay))
            self.token_momentum.mul_(self.momentum_decay).add_(token_update_terms * (1.0 - self.momentum_decay))

            current_update_weight = 1.0 - self.momentum_weight
            base_update = base_update_terms * current_update_weight + self.base_momentum * self.momentum_weight
            token_update = token_update_terms * current_update_weight + self.token_momentum * self.momentum_weight

        self.model.base_mat.copy_(
            apply_rotation(
                self.model.base_mat,
                base_update,
                self.base_lr,
                self.update_noise_scale,
            )
        )
        self.model.token_mats.copy_(
            apply_rotation(
                self.model.token_mats,
                token_update,
                self.token_lr,
                self.update_noise_scale,
            )
        )
        self.update_count += 1
        if self.orthogonalize_period > 0 and self.update_count % self.orthogonalize_period == 0:
            self.model.base_mat.copy_(newton_schulz_orthogonalize(self.model.base_mat))
            self.model.token_mats.copy_(newton_schulz_orthogonalize(self.model.token_mats))
        self.model.reset_state()

    def state_dict(self) -> Dict[str, Any]:
        if self.base_momentum is None or self.token_momentum is None:
            return {}
        return {
            "base_momentum": self.base_momentum,
            "token_momentum": self.token_momentum,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if self.base_momentum is None or self.token_momentum is None:
            return
        if not state:
            return
        self.base_momentum.copy_(state["base_momentum"].to(self.model.base_mat.device))
        self.token_momentum.copy_(state["token_momentum"].to(self.model.base_mat.device))
