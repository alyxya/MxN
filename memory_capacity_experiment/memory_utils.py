#!/usr/bin/env python3
import math

import torch


def skew(update_terms: torch.Tensor) -> torch.Tensor:
    return update_terms - update_terms.transpose(-1, -2)


def newton_schulz_orthogonalize(x: torch.Tensor, *, steps: int = 1) -> torch.Tensor:
    for _ in range(max(steps, 0)):
        x = 1.5 * x - 0.5 * (x @ x.transpose(-1, -2) @ x)
    return x


def exp_rotation(generator: torch.Tensor, max_step_norm: float = 0.001) -> torch.Tensor:
    norm = torch.linalg.matrix_norm(generator, ord="fro", dim=(-2, -1)).max()
    norm_ratio = float((norm / max_step_norm).item())
    squarings = 0 if norm_ratio <= 1.0 else math.ceil(math.log2(norm_ratio))

    eye = torch.eye(generator.shape[-1], device=generator.device, dtype=generator.dtype)
    while eye.ndim < generator.ndim:
        eye = eye.unsqueeze(0)
    r = eye + generator / (2 ** squarings)
    for _ in range(squarings):
        r = r @ r
    return r


def generator_noise(generator: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 0.0:
        return torch.zeros_like(generator)

    n = generator.shape[-1]
    off_diagonal_count = n * (n - 1)
    generator_mean_square = generator.square().sum(dim=(-2, -1), keepdim=True) / off_diagonal_count
    noise_std = (generator_mean_square / 2.0).sqrt() * scale
    return skew(torch.empty_like(generator).normal_() * noise_std)


def apply_rotation(
    current: torch.Tensor,
    update_terms: torch.Tensor,
    lr: float,
    update_noise_scale: float = 0.0,
) -> torch.Tensor:
    learned_generator = skew(update_terms)
    generator = learned_generator * lr + generator_noise(learned_generator, update_noise_scale)
    return exp_rotation(generator) @ current


def subspace_summary(label: str, vectors: torch.Tensor) -> str:
    samples, dim = vectors.shape
    if samples < 2:
        return f"{label}[samples={samples} dim={dim}]"
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    energy = torch.linalg.svdvals(centered).square()
    total = float(energy.sum().item())
    if total <= 1e-12:
        return f"{label}[samples={samples} dim={dim} rank=0]"
    rank_eps = int((energy > energy.max() * 1e-6).sum().item())
    cdf = torch.cumsum(energy / total, dim=0)
    rank_99 = int(torch.searchsorted(cdf, torch.tensor(0.99, device=cdf.device)).item()) + 1
    pr = float((energy.sum() ** 2 / energy.square().sum().clamp_min(1e-12)).item())
    p = energy / total
    erank = float((-(p * p.clamp_min(1e-12).log()).sum()).exp().item())
    return (
        f"{label}[samples={samples} dim={dim} rank_eps={rank_eps} "
        f"rank99={rank_99} pr={pr:.1f} erank={erank:.1f}]"
    )
