import torch


EPS = 1e-12


def normalize_last_dim(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def orthogonalize_steps(w: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(max(steps, 0)):
        w = orthogonalize_newton_schulz(w)
    return w


def orthogonalize_newton_schulz(w: torch.Tensor) -> torch.Tensor:
    return 1.5 * w - 0.5 * (w @ w.transpose(-1, -2) @ w)
