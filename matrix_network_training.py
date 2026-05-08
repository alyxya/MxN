#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

from matrix_network import MatrixNetwork
from matrix_network_ops import (
    EPS,
    cached_eye,
    normalize_columns,
    normalize_last_dim,
    one_hot_vectors,
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


def matrix_rotation_generator(u: torch.Tensor, v: torch.Tensor, weight: float) -> torch.Tensor:
    return (v @ u.transpose(-1, -2) - u @ v.transpose(-1, -2)) * weight


def apply_matrix_rotation(
    current: torch.Tensor,
    momentum_generator: torch.Tensor,
    learning_rate: float,
    orthogonalize_steps_count: int,
) -> torch.Tensor:
    eye = cached_eye(current.shape[-1], current.device, current.dtype)
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


@dataclass
class BatchUpdateWorkspace:
    base_delta: torch.Tensor | None = None
    token_delta: torch.Tensor | None = None
    seq_middle_ops: torch.Tensor | None = None
    seq_pre_token_ops: torch.Tensor | None = None

    def prepare(self, model: MatrixNetwork) -> None:
        dtype = model.base_mat.dtype
        device = model.device
        n = model.n
        vocab_size = model.vocab_size

        def ensure_zero(name: str, shape: Tuple[int, ...]) -> torch.Tensor:
            tensor = getattr(self, name)
            if tensor is None or tensor.shape != shape or tensor.device != device or tensor.dtype != dtype:
                tensor = torch.zeros(shape, device=device, dtype=dtype)
                setattr(self, name, tensor)
            else:
                tensor.zero_()
            return tensor

        ensure_zero("base_delta", (n, n))
        ensure_zero("token_delta", (vocab_size, n, n))

    def ensure_sequence_buffers(self, *, max_prefix_len: int, n: int, device: torch.device, dtype: torch.dtype) -> None:
        def ensure(name: str) -> torch.Tensor:
            tensor = getattr(self, name)
            shape = (max_prefix_len, n, n)
            if tensor is None or tensor.shape != shape or tensor.device != device or tensor.dtype != dtype:
                tensor = torch.empty(shape, device=device, dtype=dtype)
                setattr(self, name, tensor)
            return tensor

        ensure("seq_middle_ops")
        ensure("seq_pre_token_ops")


@dataclass
class PrefixOperatorCache:
    prefix_op: torch.Tensor
    prefix_transpose_op: torch.Tensor
    middle_op: torch.Tensor


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
    model.query = one_hot_vectors(1, n, device)[0]
    model.unembed_vectors = one_hot_vectors(model.vocab_size, n, device)
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


def prefix_operator_from_ids(model: MatrixNetwork, token_ids: Sequence[int]) -> torch.Tensor:
    prefix_op = cached_eye(model.n, model.device, model.base_mat.dtype)
    for tid in token_ids:
        prefix_op = prefix_op @ model.token_mats[tid]
    return prefix_op


def advance_prefix_operator(
    model: MatrixNetwork,
    prefix_op: torch.Tensor,
    token_id: int,
) -> torch.Tensor:
    return prefix_op @ model.token_mats[token_id]


def state_from_prefix_op(
    model: MatrixNetwork,
    prefix_op: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    return model.base_mat @ (prefix_op @ query)


def predict_next_id_from_prefix_op(
    model: MatrixNetwork,
    prefix_op: torch.Tensor,
) -> int:
    state = state_from_prefix_op(model, prefix_op, model.query)
    scores = model.unembed_vectors @ normalize_columns(state.unsqueeze(1))
    return int(scores[:, 0].argmax().item())


def generate_until_token_id(model: MatrixNetwork, prompt_ids: Sequence[int], stop_token_id: int, max_len: int) -> Tuple[List[int], bool]:
    prefix_op = prefix_operator_from_ids(model, prompt_ids)
    pred_ids: List[int] = []
    for _ in range(max_len):
        next_id = predict_next_id_from_prefix_op(model, prefix_op)
        if next_id == stop_token_id:
            return pred_ids, True
        pred_ids.append(next_id)
        prefix_op = advance_prefix_operator(model, prefix_op, next_id)
    return pred_ids, False


def generate_until_token(model: MatrixNetwork, prompt_text: str, stop_token: str, max_len: int) -> Tuple[str, bool]:
    prefix_ids = model.encode(prompt_text)
    pred_ids, did_stop = generate_until_token_id(model, prefix_ids, model.stoi[stop_token], max_len)
    return "".join(model.itos[tid] for tid in pred_ids), did_stop


@torch.no_grad()
def apply_batch_update(
    model: MatrixNetwork,
    optimizer_state: MatrixNetworkOptimizerState,
    workspace: BatchUpdateWorkspace,
    sequences: Sequence[Sequence[int]],
    prompt_lens: Sequence[int],
    token_learning_rate: float,
    base_learning_rate: float,
    target_randomize_scale: float,
    update_orthogonalize_steps: int,
    current_update_weight: float,
) -> Tuple[float, float]:
    n = model.n
    workspace.prepare(model)
    base_delta = workspace.base_delta
    token_delta = workspace.token_delta
    correct = 0
    total = 0
    mistake_weight = 0.0
    mean_target_score = 0.0

    optimizer_state.ensure_initialized(model)
    momentum_decay = optimizer_state.momentum_decay
    momentum_update_weight = 1.0 - momentum_decay

    assert optimizer_state.base_momentum is not None
    assert optimizer_state.token_momentum is not None

    base = model.base_mat
    base_t = base.transpose(-1, -2)
    eye = cached_eye(n, model.device, base.dtype)

    def extend_prefix_cache(
        parent_cache: PrefixOperatorCache | None,
        token_id: int,
        prefix_len: int,
        pre_token_buffer: torch.Tensor,
    ) -> PrefixOperatorCache:
        token_mat = model.token_mats[token_id]
        token_mat_t = token_mat.transpose(-1, -2)
        if parent_cache is None:
            prefix_op = token_mat
            prefix_transpose_op = token_mat_t
            pre_token_buffer[0] = eye
        else:
            prefix_op = parent_cache.prefix_op @ token_mat
            prefix_transpose_op = token_mat_t @ parent_cache.prefix_transpose_op
            pre_token_buffer[prefix_len - 1] = parent_cache.prefix_transpose_op

        middle_op = base @ prefix_op
        return PrefixOperatorCache(
            prefix_op=prefix_op,
            prefix_transpose_op=prefix_transpose_op,
            middle_op=middle_op,
        )

    def accumulate_sequence_updates(
        full_ids: Sequence[int],
        prompt_len: int,
        prefix_caches: Sequence[PrefixOperatorCache],
        middle_ops_buffer: torch.Tensor,
    ) -> Tuple[float, int, int, int]:
        total_prefixes = len(full_ids) - 1
        target_start = prompt_len - 1
        total_count = total_prefixes - target_start
        if total_count <= 0:
            return 0.0, 0, 0, 0

        target_ids_tensor = torch.tensor(full_ids[prompt_len:], device=model.device, dtype=torch.long)
        query_mat = model.query.unsqueeze(1)
        middle_ops = middle_ops_buffer[target_start:total_prefixes]
        middle_states = middle_ops @ query_mat
        state_vecs = normalize_last_dim(middle_states.squeeze(-1))
        scores = model.unembed_vectors @ state_vecs.transpose(0, 1)
        prefix_indices = torch.arange(total_count, device=model.device)
        target_scores = scores[target_ids_tensor, prefix_indices]
        predicted_ids = scores.argmax(dim=0)
        mistaken_prefix_mask = predicted_ids != target_ids_tensor
        mistake_count = int(mistaken_prefix_mask.sum().item())
        correct_count = total_count - mistake_count
        if mistake_count == 0:
            return float(target_scores.sum().item()), correct_count, total_count, 0

        target_vectors = model.unembed_vectors[target_ids_tensor]
        if target_randomize_scale > 0.0:
            target_vectors = target_vectors + torch.randn_like(target_vectors) * target_randomize_scale
            target_vectors = normalize_last_dim(target_vectors)
        target_mats = target_vectors.unsqueeze(-1)

        u_base = normalize_last_dim(middle_states[mistaken_prefix_mask].squeeze(-1)).unsqueeze(-1)
        v_base = target_mats[mistaken_prefix_mask]
        base_delta.add_(matrix_rotation_generator(u_base, v_base, 1.0).sum(dim=0))

        target_caches = prefix_caches[target_start:total_prefixes]
        base_query_targets = base_t @ target_mats
        for idx in range(total_prefixes):
            active_start = max(0, idx - prompt_len + 1)
            if active_start >= total_count:
                continue
            active_mask = mistaken_prefix_mask[active_start:]
            if not active_mask.any():
                continue
            tid = full_ids[idx]
            v_ops = pre_token_buffer[idx].unsqueeze(0).expand(total_count - active_start, -1, -1)[active_mask]
            prefix_ops = torch.stack(
                [cache.prefix_op for cache in target_caches[active_start:]],
                dim=0,
            )[active_mask]
            u_ops = v_ops @ prefix_ops
            active_count = int(active_mask.sum().item())
            u = normalize_last_dim((u_ops @ query_mat.unsqueeze(0).expand(active_count, -1, -1)).squeeze(-1)).unsqueeze(-1)
            v = normalize_last_dim((v_ops @ base_query_targets[active_start:][active_mask]).squeeze(-1)).unsqueeze(-1)
            token_delta[tid].add_(matrix_rotation_generator(u, v, 1.0).sum(dim=0))

        return float(target_scores.sum().item()), correct_count, total_count, mistake_count

    for full_ids, prompt_len in zip(sequences, prompt_lens):
        seq_max_prefix_len = len(full_ids) - 1
        workspace.ensure_sequence_buffers(max_prefix_len=seq_max_prefix_len, n=n, device=model.device, dtype=base.dtype)
        middle_ops_buffer = workspace.seq_middle_ops[:seq_max_prefix_len]
        pre_token_buffer = workspace.seq_pre_token_ops[:seq_max_prefix_len]

        cache: PrefixOperatorCache | None = None
        prefix_caches: List[PrefixOperatorCache] = []
        for prefix_len in range(1, len(full_ids)):
            cache = extend_prefix_cache(cache, full_ids[prefix_len - 1], prefix_len, pre_token_buffer)
            prefix_caches.append(cache)
            middle_ops_buffer[prefix_len - 1] = cache.middle_op

        score_sum, correct_count, total_count, mistake_count = accumulate_sequence_updates(
            full_ids,
            prompt_len,
            prefix_caches,
            middle_ops_buffer,
        )
        mistake_weight += float(mistake_count)
        mean_target_score += score_sum
        correct += correct_count
        total += total_count

    update_scale = 1.0 / float(max(mistake_weight, 1.0))
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
    workspace = BatchUpdateWorkspace()

    for iter_idx in range(1, iters + 1):
        sequences, prompt_lens = sample_batch()
        mean_target_score, token_acc = apply_batch_update(
            model,
            optimizer_state,
            workspace,
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
