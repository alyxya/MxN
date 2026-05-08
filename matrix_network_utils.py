#!/usr/bin/env python3
import torch

from matrix_network_training import EPS


def subspace_summary(label: str, vectors: torch.Tensor) -> str:
    samples, dim = vectors.shape
    if samples < 2:
        return f"{label}[samples={samples} dim={dim}]"
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    energy = torch.linalg.svdvals(centered).square()
    total = float(energy.sum().item())
    if total <= EPS:
        return f"{label}[samples={samples} dim={dim} rank=0]"
    rank_eps = int((energy > energy.max() * 1e-6).sum().item())
    cdf = torch.cumsum(energy / total, dim=0)
    rank_99 = int(torch.searchsorted(cdf, torch.tensor(0.99, device=cdf.device)).item()) + 1
    pr = float((energy.sum() ** 2 / energy.square().sum().clamp_min(EPS)).item())
    p = energy / total
    erank = float((-(p * p.clamp_min(EPS).log()).sum()).exp().item())
    return (
        f"{label}[samples={samples} dim={dim} rank_eps={rank_eps} "
        f"rank99={rank_99} pr={pr:.1f} erank={erank:.1f}]"
    )
