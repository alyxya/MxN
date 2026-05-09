#!/usr/bin/env python3
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple

import torch

if TYPE_CHECKING:
    from matrix_network import MatrixNetwork
    from matrix_network_optimizer import MatrixNetworkOptimizer

OptimizerState = Dict[str, Any]


def skew(update_terms: torch.Tensor) -> torch.Tensor:
    return update_terms - update_terms.transpose(-1, -2)


def newton_schulz_orthogonalize_step(x: torch.Tensor) -> torch.Tensor:
    return 1.5 * x - 0.5 * (x @ x.transpose(-1, -2) @ x)


def exp_rotation(generator: torch.Tensor, lr: float, max_step_norm: float = 0.001) -> torch.Tensor:
    a = generator * lr
    norm = torch.linalg.matrix_norm(a, ord="fro", dim=(-2, -1)).max()
    norm_ratio = float((norm / max_step_norm).item())
    squarings = 0 if norm_ratio <= 1.0 else math.ceil(math.log2(norm_ratio))

    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    while eye.ndim < a.ndim:
        eye = eye.unsqueeze(0)
    r = eye + a / (2 ** squarings)
    for _ in range(squarings):
        r = r @ r
    return r


def apply_rotation(current: torch.Tensor, update_terms: torch.Tensor, lr: float) -> torch.Tensor:
    return exp_rotation(skew(update_terms), lr) @ current


def save_checkpoint(
    model: "MatrixNetwork",
    optimizer: "MatrixNetworkOptimizer",
    path: str,
    *,
    metadata: Dict[str, Any] | None = None,
) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = save_path.with_suffix(save_path.suffix + ".tmp")
    torch.save(
        {
            "n": model.n,
            "vocab": model.vocab,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": metadata or {},
        },
        tmp,
    )
    tmp.replace(save_path)


def load_checkpoint(path: str, device: torch.device | str | None) -> Tuple["MatrixNetwork", OptimizerState, Dict[str, Any]]:
    from matrix_network import MatrixNetwork

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MatrixNetwork(n=int(ckpt["n"]), vocab=ckpt["vocab"], device=device)
    model.load_state_dict(ckpt["model_state"])
    model.reset_state()
    return model, ckpt["optimizer_state"], dict(ckpt["metadata"])


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
