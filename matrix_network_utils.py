#!/usr/bin/env python3
from typing import Dict

import torch

from matrix_network_ops import EPS


def subspace_summary(vectors: torch.Tensor) -> Dict[str, float]:
    if vectors.ndim != 2:
        raise ValueError(f"expected 2D tensor for subspace summary, got shape={tuple(vectors.shape)}")
    sample_count, dim = vectors.shape
    if sample_count == 0:
        return {"rank_eps": 0.0, "rank_99": 0.0, "pr": 0.0, "erank": 0.0, "dim": float(dim), "samples": 0.0}
    if sample_count == 1:
        return {"rank_eps": 1.0, "rank_99": 1.0, "pr": 1.0, "erank": 1.0, "dim": float(dim), "samples": 1.0}

    centered = vectors - vectors.mean(dim=0, keepdim=True)
    svals = torch.linalg.svdvals(centered)
    energy = svals.square()
    total_energy = float(energy.sum().item())
    if total_energy <= EPS:
        return {"rank_eps": 0.0, "rank_99": 0.0, "pr": 0.0, "erank": 0.0, "dim": float(dim), "samples": float(sample_count)}

    max_energy = float(energy.max().item())
    rank_eps = int((energy > (max_energy * 1e-6)).sum().item())
    normalized = energy / total_energy
    cdf = torch.cumsum(normalized, dim=0)
    rank_99 = int(torch.searchsorted(cdf, torch.tensor(0.99, device=cdf.device)).item()) + 1
    pr = float((energy.sum().square() / energy.square().sum().clamp_min(EPS)).item())
    entropy = -(normalized * normalized.clamp_min(EPS).log()).sum()
    erank = float(entropy.exp().item())
    return {
        "rank_eps": float(rank_eps),
        "rank_99": float(rank_99),
        "pr": pr,
        "erank": erank,
        "dim": float(dim),
        "samples": float(sample_count),
    }


def format_subspace_summary(label: str, summary: Dict[str, float]) -> str:
    dim = int(summary.get("dim", 0.0))
    samples = int(summary.get("samples", 0.0))
    rank_eps = summary["rank_eps"]
    rank_99 = summary["rank_99"]
    pr = summary["pr"]
    erank = summary["erank"]
    return (
        f"{label}[samples={samples} dim={dim} "
        f"rank_eps={rank_eps:.0f} rank99={rank_99:.0f} "
        f"pr={pr:.1f} erank={erank:.1f}]"
    )
