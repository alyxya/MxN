#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

from matrix_network import MatrixNetwork


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def orthogonalize(w: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(steps):
        w = 1.5 * w - 0.5 * (w @ w.transpose(-1, -2) @ w)
    return w


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rotate_matrix(current: torch.Tensor, generator: torch.Tensor, lr: float, ortho_steps: int) -> torch.Tensor:
    eye = torch.eye(current.shape[-1], device=current.device, dtype=current.dtype)
    while eye.ndim < current.ndim:
        eye = eye.unsqueeze(0)
    return orthogonalize((eye + lr * generator) @ current, ortho_steps)


@dataclass
class OptimizerState:
    momentum_decay: float
    base_momentum: torch.Tensor
    token_momentum: torch.Tensor

    @classmethod
    def for_model(cls, model: MatrixNetwork, momentum_decay: float) -> "OptimizerState":
        return cls(
            momentum_decay=momentum_decay,
            base_momentum=torch.zeros_like(model.base_mat),
            token_momentum=torch.zeros_like(model.token_mats),
        )

    @classmethod
    def from_dict(cls, state: Dict[str, Any] | None, model: MatrixNetwork, momentum_decay: float) -> "OptimizerState":
        if not state:
            return cls.for_model(model, momentum_decay)
        base_mom = state.get("base_momentum")
        token_mom = state.get("token_momentum")
        return cls(
            momentum_decay=float(state.get("momentum_decay", momentum_decay)),
            base_momentum=base_mom.to(model.device) if isinstance(base_mom, torch.Tensor) else torch.zeros_like(model.base_mat),
            token_momentum=token_mom.to(model.device) if isinstance(token_mom, torch.Tensor) else torch.zeros_like(model.token_mats),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "momentum_decay": self.momentum_decay,
            "base_momentum": self.base_momentum,
            "token_momentum": self.token_momentum,
        }


def save_checkpoint(
    model: MatrixNetwork,
    optimizer: OptimizerState,
    path: str,
    *,
    completed_iters: int = 0,
    metadata: Dict[str, Any] | None = None,
) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": model.n,
        "vocab": model.vocab,
        "token_mats": model.token_mats,
        "base_mat": model.base_mat,
        "optimizer_state": optimizer.to_dict(),
        "completed_iters": completed_iters,
        "metadata": metadata or {},
    }
    tmp = save_path.with_suffix(save_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(save_path)


def load_checkpoint(path: str, device: torch.device, *, momentum_decay: float) -> Tuple[MatrixNetwork, OptimizerState, int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MatrixNetwork(n=int(ckpt["n"]), vocab=ckpt["vocab"], device=device)
    model.token_mats = ckpt["token_mats"].to(device)
    model.base_mat = ckpt["base_mat"].to(device)
    model.reset_state()
    optimizer = OptimizerState.from_dict(ckpt.get("optimizer_state"), model, momentum_decay)
    completed_iters = int(ckpt.get("completed_iters") or 0)
    metadata = dict(ckpt.get("metadata") or {})
    return model, optimizer, completed_iters, metadata


@torch.no_grad()
def apply_batch_update(
    model: MatrixNetwork,
    optimizer: OptimizerState,
    sequences: Sequence[Sequence[int]],
    prompt_lens: Sequence[int],
    *,
    token_lr: float,
    base_lr: float,
    target_noise: float,
    ortho_steps: int,
    current_update_weight: float,
) -> Tuple[float, float]:
    base_delta = torch.zeros_like(model.base_mat)
    token_delta = torch.zeros_like(model.token_mats)
    correct = total = mistakes = 0
    target_score_sum = 0.0
    eye = torch.eye(model.n, device=model.device, dtype=model.base_mat.dtype)
    decay = optimizer.momentum_decay

    for full_ids, prompt_len in zip(sequences, prompt_lens):
        prefix_op = eye
        for tid in full_ids[:prompt_len]:
            prefix_op = prefix_op @ model.token_mats[tid]

        for pos in range(prompt_len, len(full_ids)):
            target_id = full_ids[pos]
            state = model.base_mat @ (prefix_op @ model.query)
            state_n = normalize(state)
            scores = model.unembed_vectors @ state_n

            total += 1
            target_score_sum += float(scores[target_id].item())
            if int(scores.argmax().item()) == target_id:
                correct += 1
            else:
                mistakes += 1
                target = model.unembed_vectors[target_id]
                if target_noise > 0.0:
                    target = normalize(target + torch.randn_like(target) * target_noise)

                u = state_n.unsqueeze(1)
                v = target.unsqueeze(1)
                base_delta.add_(v @ u.T - u @ v.T)

                prefix_query = (prefix_op @ model.query).unsqueeze(1)
                base_target = (model.base_mat.T @ target).unsqueeze(1)
                prev_op = eye
                for tid in full_ids[:pos]:
                    pt = prev_op.T
                    pu = normalize((pt @ prefix_query).squeeze(1)).unsqueeze(1)
                    pv = normalize((pt @ base_target).squeeze(1)).unsqueeze(1)
                    token_delta[tid].add_(pv @ pu.T - pu @ pv.T)
                    prev_op = prev_op @ model.token_mats[tid]

            prefix_op = prefix_op @ model.token_mats[target_id]

    scale = 1.0 / max(mistakes, 1)
    base_d = base_delta * scale
    token_d = token_delta * scale
    optimizer.base_momentum.mul_(decay).add_(base_d * (1.0 - decay))
    optimizer.token_momentum.mul_(decay).add_(token_d * (1.0 - decay))

    base_update = optimizer.base_momentum + base_d * current_update_weight
    token_update = optimizer.token_momentum + token_d * current_update_weight
    model.base_mat = rotate_matrix(model.base_mat, base_update, base_lr, ortho_steps)
    model.token_mats = rotate_matrix(model.token_mats, token_update, token_lr, ortho_steps)
    model.reset_state()

    return target_score_sum / max(total, 1), correct / max(total, 1)


def train(
    *,
    model: MatrixNetwork,
    optimizer: OptimizerState,
    sample_batch: Callable[[], Tuple[List[List[int]], List[int]]],
    iters: int,
    token_lr: float,
    base_lr: float,
    target_noise: float,
    ortho_steps: int,
    current_update_weight: float,
    log_every: int,
    eval_every: int = 0,
    evaluate: Callable[[MatrixNetwork, int], None] | None = None,
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[int, MatrixNetwork, OptimizerState], None] | None = None,
) -> None:
    for it in range(1, iters + 1):
        sequences, prompt_lens = sample_batch()
        score, acc = apply_batch_update(
            model, optimizer, sequences, prompt_lens,
            token_lr=token_lr, base_lr=base_lr, target_noise=target_noise,
            ortho_steps=ortho_steps, current_update_weight=current_update_weight,
        )
        if it == 1 or it % log_every == 0:
            print(f"iter={it:5d} mean_target_score={score:.4f} token_acc={acc:.3f}")
        if evaluate is not None and eval_every > 0 and (it % eval_every == 0 or it == iters):
            evaluate(model, it)
        if on_checkpoint is not None and checkpoint_every > 0 and it % checkpoint_every == 0:
            on_checkpoint(it, model, optimizer)
