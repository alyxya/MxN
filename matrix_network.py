#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch


EPS = 1e-12
INIT_ORTHOGONALIZE_STEPS = 4
DEFAULT_UPDATE_ORTHOGONALIZE_STEPS = 1
_EYE_CACHE: Dict[Tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}


def normalize_columns(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=0, keepdim=True) + eps)


def normalize_last_dim(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cached_eye(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n, device.type, device.index, dtype)
    eye = _EYE_CACHE.get(key)
    if eye is None:
        eye = torch.eye(n, device=device, dtype=dtype)
        _EYE_CACHE[key] = eye
    return eye


def matrix_rotation_generator(u: torch.Tensor, v: torch.Tensor, weight: float) -> torch.Tensor:
    return (v @ u.transpose(-1, -2) - u @ v.transpose(-1, -2)) * weight


def orthogonalize_newton_schulz(w: torch.Tensor) -> torch.Tensor:
    return 1.5 * w - 0.5 * (w @ w.transpose(-1, -2) @ w)


def orthogonalize_steps(w: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(max(steps, 0)):
        w = orthogonalize_newton_schulz(w)
    return w


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


def initialize_rotation_like(shape: Tuple[int, ...], device: torch.device, strength: float) -> torch.Tensor:
    n = shape[-1]
    eye = cached_eye(n, device, torch.float32)
    if len(shape) > 2:
        eye = eye.expand(*shape[:-2], n, n).clone()
    noise = torch.randn(shape, device=device) / (n**0.5)
    w = (eye + strength * noise) / ((1.0 + strength**2) ** 0.5)
    for _ in range(INIT_ORTHOGONALIZE_STEPS):
        w = orthogonalize_newton_schulz(w)
    return w


def one_hot_vectors(count: int, dim: int, device: torch.device) -> torch.Tensor:
    vectors = torch.zeros((count, dim), device=device)
    vectors[:, :count] = cached_eye(count, device, vectors.dtype)
    return vectors


def explored_targets(
    token_ids: torch.Tensor,
    *,
    vocab_size: int,
    dim: int,
    device: torch.device,
    exploration_directions: torch.Tensor | None,
    exploration_strength: float,
) -> torch.Tensor:
    if vocab_size > dim:
        raise ValueError(f"vocab_size must be <= dim for one-hot targets, got vocab_size={vocab_size} dim={dim}")
    targets = torch.zeros((token_ids.shape[0], dim), device=device)
    targets.scatter_(1, token_ids.unsqueeze(1), 1.0)
    if exploration_strength > 0 and exploration_directions is not None and exploration_directions.numel() > 0:
        directions = normalize_last_dim(exploration_directions.to(device=device, dtype=targets.dtype))
        coeffs = torch.randn((token_ids.shape[0], directions.shape[0]), device=device, dtype=targets.dtype)
        targets.add_((coeffs @ directions) * (exploration_strength / (directions.shape[0] ** 0.5)))
    return normalize_last_dim(targets)


@dataclass
class StateExplorationCache:
    samples: torch.Tensor | None = None
    directions: torch.Tensor | None = None

    def add_samples(self, states: torch.Tensor, *, max_samples: int) -> None:
        if max_samples <= 0 or states.numel() == 0:
            return
        states = normalize_last_dim(states.detach())
        if self.samples is None:
            self.samples = states[-max_samples:].clone()
            return
        self.samples = torch.cat([self.samples.to(states.device), states], dim=0)[-max_samples:].clone()

    def refresh(self, *, rank: int) -> None:
        if self.samples is None or self.samples.numel() == 0 or rank <= 0:
            self.directions = None
            return
        states = normalize_last_dim(self.samples)
        _, _, vh = torch.linalg.svd(states, full_matrices=True)
        keep = min(rank, vh.shape[0])
        self.directions = normalize_last_dim(vh[-keep:].detach())


def _zeros_like_shape(shape: Sequence[int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(tuple(shape), device=device, dtype=dtype)


@dataclass
class MatrixNetworkOptimizerState:
    momentum_decay: float
    base_momentum: torch.Tensor | None = None
    token_momentum: torch.Tensor | None = None

    def ensure_initialized(self, model: "MatrixNetwork") -> None:
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
        state.momentum_decay = float(state_dict.get("momentum_decay", momentum_decay))
        for attr in ("base_momentum", "token_momentum"):
            value = state_dict.get(attr)
            if isinstance(value, torch.Tensor):
                setattr(state, attr, value.to(device))
        return state


@dataclass
class BatchUpdateWorkspace:
    base_primary_delta: torch.Tensor | None = None
    token_primary_delta: torch.Tensor | None = None
    seq_middle_ops: torch.Tensor | None = None
    seq_pre_token_ops: torch.Tensor | None = None

    def prepare(self, model: "MatrixNetwork") -> None:
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

        ensure_zero("base_primary_delta", (n, n))
        ensure_zero("token_primary_delta", (vocab_size, n, n))

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
    prefix_ids: Tuple[int, ...]
    prefix_op: torch.Tensor
    prefix_transpose_op: torch.Tensor
    middle_op: torch.Tensor
    pre_token_ops: torch.Tensor


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MatrixNetwork:
    def __init__(
        self,
        *,
        n: int,
        device: torch.device,
        vocab: str,
        output_vocab: str,
    ):
        if n < len(vocab):
            raise ValueError(f"n must be >= {len(vocab)} to fit fixed one-hot heads, got {n}")
        if not output_vocab:
            raise ValueError("output_vocab must not be empty")
        if any(ch not in vocab for ch in output_vocab):
            raise ValueError("output_vocab must be a subset of vocab")
        self.n = n
        self.device = device
        self.vocab = vocab
        self.output_vocab = output_vocab
        self.vocab_size = len(self.vocab)
        self.output_vocab_size = len(self.output_vocab)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

        self.query = one_hot_vectors(1, n, device)[0]
        self.unembed_vectors = one_hot_vectors(self.vocab_size, n, device)

        self.base_mat = initialize_rotation_like((n, n), device, 0.0)
        self.token_mats = initialize_rotation_like((self.vocab_size, n, n), device, 0.0)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]

    def prefix_state_from_query(self, token_ids: Sequence[int], query: torch.Tensor) -> torch.Tensor:
        v = query
        for tid in reversed(token_ids):
            v = self.token_mats[tid] @ v
        return self.base_mat @ v

    def prefix_state_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        return self.prefix_state_from_query(token_ids, self.query)

    def predict_next_id(self, token_ids: Sequence[int]) -> int:
        state = self.prefix_state_ids(token_ids)
        scores = self.unembed_vectors[: self.output_vocab_size] @ normalize_columns(state.unsqueeze(1))
        return int(scores[:, 0].argmax().item())

    def predict_next(self, prefix: str) -> str:
        return self.itos[self.predict_next_id(self.encode(prefix))]

    @classmethod
    def from_checkpoint_dict(
        cls,
        ckpt: Dict[str, Any],
        device: torch.device,
    ) -> "MatrixNetwork":
        n = int(ckpt["n"])
        vocab = str(ckpt["vocab"])
        output_vocab = str(ckpt["output_vocab"])
        model = cls(
            n=n,
            device=device,
            vocab=vocab,
            output_vocab=output_vocab,
        )
        model.token_mats = ckpt["token_mats"].to(device)
        model.base_mat = ckpt["base_mat"].to(device)
        model.query = one_hot_vectors(1, n, device)[0]
        model.unembed_vectors = one_hot_vectors(model.vocab_size, n, device)
        return model

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device) -> "MatrixNetwork":
        ckpt = load_checkpoint_dict(path, device)
        return cls.from_checkpoint_dict(ckpt, device)


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
    model = MatrixNetwork.from_checkpoint_dict(ckpt, device)
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
        "output_vocab": model.output_vocab,
        "token_mats": model.token_mats,
        "base_mat": model.base_mat,
        "query": model.query,
        "unembed_vectors": model.unembed_vectors,
        "optimizer_state": None if optimizer_state is None else optimizer_state.state_dict(),
        "update_orthogonalize_steps": int(update_orthogonalize_steps),
        "completed_iters": None if completed_iters is None else int(completed_iters),
        "metadata": dict(metadata or {}),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            **payload,
        },
        tmp_path,
    )
    tmp_path.replace(path)


def prefix_operator_from_ids(model: MatrixNetwork, token_ids: Sequence[int]) -> torch.Tensor:
    eye = cached_eye(model.n, model.device, model.base_mat.dtype)
    prefix_op = eye
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
    scores = model.unembed_vectors[: model.output_vocab_size] @ normalize_columns(state.unsqueeze(1))
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
        return {"rank_eps": 0.0, "rank_99": 0.0, "pr": 0.0, "erank": 0.0}

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


@torch.no_grad()
def apply_batch_update(
    model: MatrixNetwork,
    optimizer_state: MatrixNetworkOptimizerState,
    workspace: BatchUpdateWorkspace,
    sequences: Sequence[Sequence[int]],
    prompt_lens: Sequence[int],
    token_learning_rate: float,
    base_learning_rate: float,
    state_exploration_scale: float,
    state_exploration_directions: torch.Tensor | None,
    state_exploration_samples: List[torch.Tensor] | None,
    update_orthogonalize_steps: int,
) -> Tuple[float, float]:
    n = model.n
    workspace.prepare(model)
    base_primary_delta = workspace.base_primary_delta
    token_primary_delta = workspace.token_primary_delta
    correct = 0
    total = 0
    primary_objective_weight = 0.0
    mean_target_score = 0.0

    base = model.base_mat
    base_t = base.transpose(-1, -2)
    eye = cached_eye(n, model.device, base.dtype)

    def extend_prefix_cache(
        parent_cache: PrefixOperatorCache | None,
        token_id: int,
        prefix_len: int,
        pre_token_buffer: torch.Tensor,
    ) -> PrefixOperatorCache:
        prefix_key = (token_id,) if parent_cache is None else (*parent_cache.prefix_ids, token_id)
        last_tid = token_id

        token_mat = model.token_mats[last_tid]
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
            prefix_ids=prefix_key,
            prefix_op=prefix_op,
            prefix_transpose_op=prefix_transpose_op,
            middle_op=middle_op,
            pre_token_ops=pre_token_buffer[:prefix_len],
        )

    def accumulate_primary_objectives(
        full_ids: Sequence[int],
        prompt_len: int,
        prefix_caches: Sequence[PrefixOperatorCache],
        middle_ops_buffer: torch.Tensor,
    ) -> Tuple[float, int, int, int]:
        total_prefixes = len(full_ids) - 1
        primary_start = prompt_len - 1
        total_count = total_prefixes - primary_start
        if total_count <= 0:
            return 0.0, 0, 0, 0

        target_ids_tensor = torch.tensor(full_ids[prompt_len:], device=model.device, dtype=torch.long)
        prefix_lens = torch.arange(prompt_len, len(full_ids), device=model.device, dtype=torch.long)
        query_mat = model.query.unsqueeze(1)
        middle_ops = middle_ops_buffer[primary_start:total_prefixes]
        middle_states = middle_ops @ query_mat
        state_vecs = normalize_last_dim(middle_states.squeeze(-1))
        if state_exploration_samples is not None:
            state_exploration_samples.append(state_vecs.detach())
        scores = model.unembed_vectors @ state_vecs.transpose(0, 1)
        prefix_indices = torch.arange(total_count, device=model.device)
        target_scores = scores[target_ids_tensor, prefix_indices]
        predicted_ids = scores[: model.output_vocab_size].argmax(dim=0)
        mistaken_prefix_mask = predicted_ids != target_ids_tensor
        mistake_count = int(mistaken_prefix_mask.sum().item())
        correct_count = total_count - mistake_count
        if mistake_count == 0:
            return float(target_scores.sum().item()), correct_count, total_count, 0

        target_mats = explored_targets(
            target_ids_tensor,
            vocab_size=model.vocab_size,
            dim=n,
            device=model.device,
            exploration_directions=state_exploration_directions,
            exploration_strength=state_exploration_scale,
        ).unsqueeze(-1)

        u_base = normalize_last_dim(middle_states[mistaken_prefix_mask].squeeze(-1)).unsqueeze(-1)
        v_base = normalize_last_dim(target_mats[mistaken_prefix_mask].squeeze(-1)).unsqueeze(-1)
        base_primary_delta.add_(matrix_rotation_generator(u_base, v_base, 1.0).sum(dim=0))

        primary_caches = prefix_caches[primary_start:total_prefixes]
        base_query_targets = base_t @ target_mats
        max_prefix_len = int(prefix_lens.max().item())
        for idx in range(max_prefix_len):
            active_start = max(0, idx - prompt_len + 1)
            if active_start >= total_count:
                continue
            active_mask = mistaken_prefix_mask[active_start:]
            if not active_mask.any():
                continue
            tid = full_ids[idx]
            v_ops = pre_token_buffer[idx].unsqueeze(0).expand(total_count - active_start, -1, -1)[active_mask]
            prefix_ops = torch.stack(
                [cache.prefix_op for cache in primary_caches[active_start:]],
                dim=0,
            )[active_mask]
            u_ops = v_ops @ prefix_ops
            active_count = int(active_mask.sum().item())
            u = normalize_last_dim((u_ops @ query_mat.unsqueeze(0).expand(active_count, -1, -1)).squeeze(-1)).unsqueeze(-1)
            v = normalize_last_dim((v_ops @ base_query_targets[active_start:][active_mask]).squeeze(-1)).unsqueeze(-1)
            token_primary_delta[tid].add_(matrix_rotation_generator(u, v, 1.0).sum(dim=0))

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

        score_sum, correct_count, total_count, primary_mistake_count = accumulate_primary_objectives(
            full_ids,
            prompt_len,
            prefix_caches,
            middle_ops_buffer,
        )
        primary_objective_weight += float(primary_mistake_count)
        mean_target_score += score_sum
        correct += correct_count
        total += total_count

    primary_scale = 1.0 / float(max(primary_objective_weight, 1.0))
    optimizer_state.ensure_initialized(model)
    decay = optimizer_state.momentum_decay
    momentum_inject = 1.0 - decay

    assert optimizer_state.base_momentum is not None
    assert optimizer_state.token_momentum is not None

    optimizer_state.base_momentum.mul_(decay).add_(base_primary_delta * (primary_scale * momentum_inject))
    if float(optimizer_state.base_momentum.abs().amax().item()) > EPS:
        model.base_mat = apply_matrix_rotation(
            model.base_mat,
            optimizer_state.base_momentum,
            base_learning_rate,
            update_orthogonalize_steps,
        )

    optimizer_state.token_momentum.mul_(decay).add_(token_primary_delta * (primary_scale * momentum_inject))
    token_hist_active = optimizer_state.token_momentum.abs().amax(dim=(1, 2)) > EPS
    if token_hist_active.any():
        model.token_mats[token_hist_active] = apply_matrix_rotation(
            model.token_mats[token_hist_active],
            optimizer_state.token_momentum[token_hist_active],
            token_learning_rate,
            update_orthogonalize_steps,
        )

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
    state_exploration_scale: float,
    state_exploration_rank: int,
    state_exploration_period: int,
    state_exploration_samples: int,
    log_every: int,
    eval_every: int,
    update_orthogonalize_steps: int,
    checkpoint_every: int = 0,
    checkpoint_callback: Callable[[int, MatrixNetwork, MatrixNetworkOptimizerState], None] | None = None,
) -> Tuple[MatrixNetwork, MatrixNetworkOptimizerState]:
    workspace = BatchUpdateWorkspace()
    state_exploration_cache = StateExplorationCache()

    for iter_idx in range(1, iters + 1):
        sequences, prompt_lens = sample_batch()
        collected_state_samples: List[torch.Tensor] | None = [] if state_exploration_scale > 0 else None
        mean_target_score, token_acc = apply_batch_update(
            model,
            optimizer_state,
            workspace,
            sequences=sequences,
            prompt_lens=prompt_lens,
            token_learning_rate=token_learning_rate,
            base_learning_rate=base_learning_rate,
            state_exploration_scale=state_exploration_scale,
            state_exploration_directions=state_exploration_cache.directions,
            state_exploration_samples=collected_state_samples,
            update_orthogonalize_steps=update_orthogonalize_steps,
        )
        if collected_state_samples:
            state_exploration_cache.add_samples(
                torch.cat(collected_state_samples, dim=0),
                max_samples=state_exploration_samples,
            )
            if iter_idx == 1 or iter_idx % state_exploration_period == 0:
                state_exploration_cache.refresh(rank=state_exploration_rank)

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
