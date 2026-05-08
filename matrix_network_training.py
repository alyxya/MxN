#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

from matrix_network import MatrixNetwork
from matrix_network_ops import (
    EPS,
    normalize_last_dim,
    orthogonalize_steps,
)


DEFAULT_UPDATE_ORTHOGONALIZE_STEPS = 1


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def matrix_rotation_generator(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return v @ u.transpose(-1, -2) - u @ v.transpose(-1, -2)


def apply_matrix_rotation(
    current: torch.Tensor,
    momentum_generator: torch.Tensor,
    learning_rate: float,
    orthogonalize_steps_count: int,
) -> torch.Tensor:
    eye = torch.eye(current.shape[-1], device=current.device, dtype=current.dtype)
    if current.ndim == 2:
        updated = (eye + learning_rate * momentum_generator) @ current
    else:
        updated = (eye.unsqueeze(0) + learning_rate * momentum_generator) @ current
    return orthogonalize_steps(updated, orthogonalize_steps_count)


def _zeros_like_shape(shape: Sequence[int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(tuple(shape), device=device, dtype=dtype)


@dataclass
class MatrixNetworkOptimizerState:
    momentum_decay: float
    base_momentum: torch.Tensor | None = None
    token_momentum: torch.Tensor | None = None

    def ensure_initialized(self, model: MatrixNetwork) -> None:
        dtype = model.base_mat.dtype
        device = model.device
        n = model.n
        vocab_size = model.vocab_size

        if self.base_momentum is None or self.base_momentum.shape != (n, n):
            self.base_momentum = _zeros_like_shape((n, n), device=device, dtype=dtype)
        if self.token_momentum is None or self.token_momentum.shape != (vocab_size, n, n):
            self.token_momentum = _zeros_like_shape((vocab_size, n, n), device=device, dtype=dtype)

        self.base_momentum = self.base_momentum.to(device=device, dtype=dtype)
        self.token_momentum = self.token_momentum.to(device=device, dtype=dtype)

    def state_dict(self) -> Dict[str, torch.Tensor | float]:
        result: Dict[str, torch.Tensor | float] = {"momentum_decay": float(self.momentum_decay)}
        if self.base_momentum is not None:
            result["base_momentum"] = self.base_momentum
        if self.token_momentum is not None:
            result["token_momentum"] = self.token_momentum
        return result

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, Any] | None,
        *,
        momentum_decay: float,
        device: torch.device,
    ) -> "MatrixNetworkOptimizerState":
        state = cls(momentum_decay=momentum_decay)
        if not state_dict:
            return state
        if "momentum_decay" in state_dict:
            state.momentum_decay = float(state_dict["momentum_decay"])
        elif "current_update_weight" in state_dict:
            state.momentum_decay = 1.0 - float(state_dict["current_update_weight"])
        for attr in ("base_momentum", "token_momentum"):
            value = state_dict.get(attr)
            if isinstance(value, torch.Tensor):
                setattr(state, attr, value.to(device))
        return state


def model_from_checkpoint_dict(ckpt: Dict[str, Any], device: torch.device) -> MatrixNetwork:
    n = int(ckpt["n"])
    model = MatrixNetwork(
        n=n,
        device=device,
        vocab=ckpt["vocab"],
    )
    model.token_mats = ckpt["token_mats"].to(device)
    model.base_mat = ckpt["base_mat"].to(device)
    model.reset_state()
    return model


def load_checkpoint_dict(path: str, device: torch.device) -> Dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint at {path} is not a dict")
    return ckpt


def load_training_checkpoint(
    path: str,
    device: torch.device,
    *,
    momentum_decay: float,
) -> Tuple[MatrixNetwork, MatrixNetworkOptimizerState, int, Dict[str, Any]]:
    ckpt = load_checkpoint_dict(path, device)
    model = model_from_checkpoint_dict(ckpt, device)
    optimizer_state = MatrixNetworkOptimizerState.from_state_dict(
        ckpt.get("optimizer_state"),
        momentum_decay=momentum_decay,
        device=device,
    )
    optimizer_state.ensure_initialized(model)
    completed_iters = int(ckpt.get("completed_iters") or 0)
    metadata = dict(ckpt.get("metadata") or {})
    return model, optimizer_state, completed_iters, metadata


def save_checkpoint(
    model: MatrixNetwork,
    save_path: str,
    *,
    optimizer_state: MatrixNetworkOptimizerState | None = None,
    update_orthogonalize_steps: int = DEFAULT_UPDATE_ORTHOGONALIZE_STEPS,
    completed_iters: int | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    path = Path(save_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": model.n,
        "vocab": model.vocab,
        "token_mats": model.token_mats,
        "base_mat": model.base_mat,
        "optimizer_state": None if optimizer_state is None else optimizer_state.state_dict(),
        "update_orthogonalize_steps": int(update_orthogonalize_steps),
        "completed_iters": None if completed_iters is None else int(completed_iters),
        "metadata": dict(metadata or {}),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


@torch.no_grad()
def apply_batch_update(
    model: MatrixNetwork,
    optimizer_state: MatrixNetworkOptimizerState,
    sequences: Sequence[Sequence[int]],
    prompt_lens: Sequence[int],
    token_learning_rate: float,
    base_learning_rate: float,
    target_randomize_scale: float,
    update_orthogonalize_steps: int,
    current_update_weight: float,
) -> Tuple[float, float]:
    base_delta = torch.zeros_like(model.base_mat)
    token_delta = torch.zeros_like(model.token_mats)
    correct = 0
    total = 0
    mistakes = 0
    mean_target_score = 0.0

    optimizer_state.ensure_initialized(model)
    momentum_decay = optimizer_state.momentum_decay
    momentum_update_weight = 1.0 - momentum_decay

    assert optimizer_state.base_momentum is not None
    assert optimizer_state.token_momentum is not None

    eye = torch.eye(model.n, device=model.device, dtype=model.base_mat.dtype)

    def randomized_target(target_id: int) -> torch.Tensor:
        target = model.unembed_vectors[target_id]
        if target_randomize_scale > 0.0:
            target = normalize_last_dim(target + torch.randn_like(target) * target_randomize_scale)
        return target

    def add_rotation_updates(
        prefix_ids: Sequence[int],
        prefix_op: torch.Tensor,
        target: torch.Tensor,
        state: torch.Tensor,
    ) -> None:
        target_col = target.unsqueeze(1)
        base_delta.add_(
            matrix_rotation_generator(
                normalize_last_dim(state).unsqueeze(1),
                target_col,
            )
        )

        query_col = model.query.unsqueeze(1)
        prefix_query = prefix_op @ query_col
        base_target = model.base_mat.transpose(-1, -2) @ target_col
        previous_tokens_op = eye
        for token_id in prefix_ids:
            previous_tokens_t = previous_tokens_op.transpose(-1, -2)
            u = normalize_last_dim((previous_tokens_t @ prefix_query).squeeze(1)).unsqueeze(1)
            v = normalize_last_dim((previous_tokens_t @ base_target).squeeze(1)).unsqueeze(1)
            token_delta[token_id].add_(matrix_rotation_generator(u, v))
            previous_tokens_op = previous_tokens_op @ model.token_mats[token_id]

    for full_ids, prompt_len in zip(sequences, prompt_lens):
        prefix_op = eye
        for token_id in full_ids[:prompt_len]:
            prefix_op = prefix_op @ model.token_mats[token_id]

        for target_pos in range(prompt_len, len(full_ids)):
            prefix_ids = full_ids[:target_pos]
            target_id = full_ids[target_pos]
            state = model.base_mat @ (prefix_op @ model.query)
            state_normed = normalize_last_dim(state)
            scores = model.unembed_vectors @ state_normed

            total += 1
            mean_target_score += float(scores[target_id].item())
            if int(scores.argmax().item()) == target_id:
                correct += 1
            else:
                mistakes += 1
                add_rotation_updates(prefix_ids, prefix_op, randomized_target(target_id), state)
            prefix_op = prefix_op @ model.token_mats[target_id]

    update_scale = 1.0 / float(max(mistakes, 1))
    mean_base_delta = base_delta * update_scale
    optimizer_state.base_momentum.mul_(momentum_decay).add_(mean_base_delta * momentum_update_weight)
    base_update = optimizer_state.base_momentum + mean_base_delta * current_update_weight
    if float(base_update.abs().amax().item()) > EPS:
        model.base_mat = apply_matrix_rotation(
            model.base_mat,
            base_update,
            base_learning_rate,
            update_orthogonalize_steps,
        )
        model.reset_state()

    mean_token_delta = token_delta * update_scale
    optimizer_state.token_momentum.mul_(momentum_decay).add_(mean_token_delta * momentum_update_weight)
    token_update = optimizer_state.token_momentum + mean_token_delta * current_update_weight
    token_hist_active = token_update.abs().amax(dim=(1, 2)) > EPS
    if token_hist_active.any():
        model.token_mats[token_hist_active] = apply_matrix_rotation(
            model.token_mats[token_hist_active],
            token_update[token_hist_active],
            token_learning_rate,
            update_orthogonalize_steps,
        )
        model.reset_state()

    return (
        mean_target_score / max(total, 1),
        correct / max(total, 1),
    )


def train(
    *,
    model: MatrixNetwork,
    optimizer_state: MatrixNetworkOptimizerState,
    sample_batch: Callable[[], Tuple[List[List[int]], List[int]]],
    evaluate_callback: Callable[[MatrixNetwork, int], None] | None,
    iters: int,
    token_learning_rate: float,
    base_learning_rate: float,
    target_randomize_scale: float,
    log_every: int,
    eval_every: int,
    update_orthogonalize_steps: int,
    current_update_weight: float,
    checkpoint_every: int = 0,
    checkpoint_callback: Callable[[int, MatrixNetwork, MatrixNetworkOptimizerState], None] | None = None,
) -> Tuple[MatrixNetwork, MatrixNetworkOptimizerState]:
    for iter_idx in range(1, iters + 1):
        sequences, prompt_lens = sample_batch()
        mean_target_score, token_acc = apply_batch_update(
            model,
            optimizer_state,
            sequences=sequences,
            prompt_lens=prompt_lens,
            token_learning_rate=token_learning_rate,
            base_learning_rate=base_learning_rate,
            target_randomize_scale=target_randomize_scale,
            update_orthogonalize_steps=update_orthogonalize_steps,
            current_update_weight=current_update_weight,
        )

        if iter_idx % log_every == 0 or iter_idx == 1:
            print(
                f"iter={iter_idx:5d} mean_target_score={mean_target_score:.4f} "
                f"token_acc={token_acc:.3f}"
            )

        if evaluate_callback is not None and (iter_idx % eval_every == 0 or iter_idx == iters):
            evaluate_callback(model, iter_idx)

        if checkpoint_callback is not None and checkpoint_every > 0 and iter_idx % checkpoint_every == 0:
            checkpoint_callback(iter_idx, model, optimizer_state)

    return model, optimizer_state
