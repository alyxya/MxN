#!/usr/bin/env python3
import torch

from matrix_network import MatrixNetwork
from matrix_network_utils import apply_rotation, newton_schulz_orthogonalize


class MatrixNetworkOptimizer:
    def __init__(
        self,
        model: MatrixNetwork,
        *,
        base_lr: float,
        token_lr: float,
        update_noise_scale: float = 0.0,
        orthogonalize_period: int = 0,
    ):
        self.model = model
        self.base_lr = base_lr
        self.token_lr = token_lr
        self.update_noise_scale = update_noise_scale
        self.orthogonalize_period = orthogonalize_period
        self.update_count = 0

    @torch.no_grad()
    def step(self, base_update_terms: torch.Tensor, token_update_terms: torch.Tensor) -> None:
        self.model.base_mat.copy_(
            apply_rotation(
                self.model.base_mat,
                base_update_terms,
                self.base_lr,
                self.update_noise_scale,
            )
        )
        self.model.token_mats.copy_(
            apply_rotation(
                self.model.token_mats,
                token_update_terms,
                self.token_lr,
                self.update_noise_scale,
            )
        )
        self.update_count += 1
        if self.orthogonalize_period > 0 and self.update_count % self.orthogonalize_period == 0:
            self.model.base_mat.copy_(newton_schulz_orthogonalize(self.model.base_mat))
            self.model.token_mats.copy_(newton_schulz_orthogonalize(self.model.token_mats))
        # The learned matrices changed, so the cached inference state is stale.
        self.model.reset_state()
