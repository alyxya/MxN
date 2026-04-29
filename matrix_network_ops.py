#!/usr/bin/env python3
from typing import Dict, Tuple

import torch


EPS = 1e-12
INIT_ORTHOGONALIZE_STEPS = 4
_EYE_CACHE: Dict[Tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}


def normalize_columns(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=0, keepdim=True) + eps)


def normalize_last_dim(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def one_hot_vectors(count: int, dim: int, device: torch.device) -> torch.Tensor:
    vectors = torch.zeros((count, dim), device=device)
    vectors[:, :count] = cached_eye(count, device, vectors.dtype)
    return vectors


def initialize_rotation_like(shape: Tuple[int, ...], device: torch.device, strength: float) -> torch.Tensor:
    n = shape[-1]
    eye = cached_eye(n, device, torch.float32)
    is_batched = len(shape) > 2
    if is_batched:
        eye = eye.expand(*shape[:-2], n, n).clone()
    if strength == 0.0:
        return eye if is_batched else eye.clone()
    noise = torch.randn(shape, device=device) / (n**0.5)
    w = (eye + strength * noise) / ((1.0 + strength**2) ** 0.5)
    return orthogonalize_steps(w, INIT_ORTHOGONALIZE_STEPS)


def orthogonalize_steps(w: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(max(steps, 0)):
        w = orthogonalize_newton_schulz(w)
    return w


def orthogonalize_newton_schulz(w: torch.Tensor) -> torch.Tensor:
    return 1.5 * w - 0.5 * (w @ w.transpose(-1, -2) @ w)


def cached_eye(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n, device.type, device.index, dtype)
    eye = _EYE_CACHE.get(key)
    if eye is None:
        eye = torch.eye(n, device=device, dtype=dtype)
        _EYE_CACHE[key] = eye
    return eye
