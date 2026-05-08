#!/usr/bin/env python3
from typing import Any, Dict

import torch

from matrix_network import MatrixNetwork


def orthogonalize(w: torch.Tensor) -> torch.Tensor:
    return 1.5 * w - 0.5 * (w @ w.transpose(-1, -2) @ w)


def rotate_matrix(current: torch.Tensor, generator: torch.Tensor, lr: float) -> torch.Tensor:
    eye = torch.eye(current.shape[-1], device=current.device, dtype=current.dtype)
    while eye.ndim < current.ndim:
        eye = eye.unsqueeze(0)
    return orthogonalize((eye + lr * generator) @ current)


class MatrixNetworkOptimizer:
    def __init__(
        self,
        model: MatrixNetwork,
        *,
        momentum_decay: float,
        base_lr: float,
        token_lr: float,
        current_update_weight: float = 0.0,
    ):
        self.model = model
        self.momentum_decay = momentum_decay
        self.base_lr = base_lr
        self.token_lr = token_lr
        self.current_update_weight = current_update_weight
        self.base_momentum = torch.zeros_like(model.base_mat)
        self.token_momentum = torch.zeros_like(model.token_mats)

    @torch.no_grad()
    def step(self, base_delta: torch.Tensor, token_delta: torch.Tensor, update_count: int) -> None:
        scale = 1.0 / max(update_count, 1)
        base_delta = base_delta * scale
        token_delta = token_delta * scale

        self.base_momentum.mul_(self.momentum_decay).add_(base_delta * (1.0 - self.momentum_decay))
        self.token_momentum.mul_(self.momentum_decay).add_(token_delta * (1.0 - self.momentum_decay))

        base_update = self.base_momentum + base_delta * self.current_update_weight
        token_update = self.token_momentum + token_delta * self.current_update_weight
        self.model.base_mat.copy_(rotate_matrix(self.model.base_mat, base_update, self.base_lr))
        self.model.token_mats.copy_(rotate_matrix(self.model.token_mats, token_update, self.token_lr))
        self.model.reset_state()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "momentum_decay": self.momentum_decay,
            "base_lr": self.base_lr,
            "token_lr": self.token_lr,
            "current_update_weight": self.current_update_weight,
            "base_momentum": self.base_momentum,
            "token_momentum": self.token_momentum,
        }

    def load_state_dict(self, state: Dict[str, Any] | None) -> None:
        if not state:
            return
        self.momentum_decay = float(state.get("momentum_decay", self.momentum_decay))
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self.token_lr = float(state.get("token_lr", self.token_lr))
        self.current_update_weight = float(state.get("current_update_weight", self.current_update_weight))

        base_momentum = state.get("base_momentum")
        if isinstance(base_momentum, torch.Tensor):
            self.base_momentum.copy_(base_momentum.to(self.model.device))

        token_momentum = state.get("token_momentum")
        if isinstance(token_momentum, torch.Tensor):
            self.token_momentum.copy_(token_momentum.to(self.model.device))
