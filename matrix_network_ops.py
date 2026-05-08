#!/usr/bin/env python3
from typing import Dict, Tuple

import torch


EPS = 1e-12
_EYE_CACHE: Dict[Tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}


def normalize_columns(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=0, keepdim=True) + eps)


def normalize_last_dim(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def orthogonalize_steps(w: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(max(steps, 0)):
        w = orthogonalize_newton_schulz(w)
    return w


def orthogonalize_newton_schulz(w: torch.Tensor) -> torch.Tensor:
    return 1.5 * w - 0.5 * (w @ w.transpose(-1, -2) @ w)


def cached_eye(n: int, device: torch.device | str | None, dtype: torch.dtype) -> torch.Tensor:
    if device is None:
        device = torch.empty((), dtype=dtype).device
    else:
        device = torch.device(device)
    key = (n, device.type, device.index, dtype)
    eye = _EYE_CACHE.get(key)
    if eye is None:
        eye = torch.eye(n, device=device, dtype=dtype)
        _EYE_CACHE[key] = eye
    return eye
