#!/usr/bin/env python3
import math
from typing import Any, Dict

import torch

from matrix_network import MatrixNetwork


def eye_like_mats(mats: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(mats.shape[-1], device=mats.device, dtype=mats.dtype)
    while eye.ndim < mats.ndim:
        eye = eye.unsqueeze(0)
    return eye


def skew(update_terms: torch.Tensor) -> torch.Tensor:
    return update_terms - update_terms.transpose(-1, -2)


def exp_rotation(generator: torch.Tensor, lr: float, max_step_norm: float = 0.001) -> torch.Tensor:
    a = generator * lr
    norm = torch.linalg.matrix_norm(a, ord="fro", dim=(-2, -1)).max()
    norm_ratio = float((norm / max_step_norm).item())
    squarings = 0 if norm_ratio <= 1.0 else math.ceil(math.log2(norm_ratio))

    r = eye_like_mats(a) + a / (2 ** squarings)
    for _ in range(squarings):
        r = r @ r
    return r


def apply_rotation(current: torch.Tensor, update_terms: torch.Tensor, lr: float) -> torch.Tensor:
    return exp_rotation(skew(update_terms), lr) @ current


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
    def step(self, base_update_terms: torch.Tensor, token_update_terms: torch.Tensor, update_count: int) -> None:
        scale = 1.0 / max(update_count, 1)
        base_update_terms = base_update_terms * scale
        token_update_terms = token_update_terms * scale

        self.base_momentum.mul_(self.momentum_decay).add_(base_update_terms * (1.0 - self.momentum_decay))
        self.token_momentum.mul_(self.momentum_decay).add_(token_update_terms * (1.0 - self.momentum_decay))

        base_update = self.base_momentum + base_update_terms * self.current_update_weight
        token_update = self.token_momentum + token_update_terms * self.current_update_weight
        self.model.base_mat.copy_(apply_rotation(self.model.base_mat, base_update, self.base_lr))
        self.model.token_mats.copy_(apply_rotation(self.model.token_mats, token_update, self.token_lr))
        self.model.reset_state()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "update_format": "outer_product_terms",
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
        legacy_skew_momentum = state.get("update_format") != "outer_product_terms"

        base_momentum = state.get("base_momentum")
        if isinstance(base_momentum, torch.Tensor):
            loaded = base_momentum.to(self.model.base_mat.device)
            self.base_momentum.copy_(loaded * 0.5 if legacy_skew_momentum else loaded)

        token_momentum = state.get("token_momentum")
        if isinstance(token_momentum, torch.Tensor):
            loaded = token_momentum.to(self.model.base_mat.device)
            self.token_momentum.copy_(loaded * 0.5 if legacy_skew_momentum else loaded)
