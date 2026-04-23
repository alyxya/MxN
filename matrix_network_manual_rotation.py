#!/usr/bin/env python3
import argparse
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch


EOS_TOKEN = "~"
PLUS_TOKEN = "+"
EQUALS_TOKEN = "="
DIGIT_SYMBOLS = "0123456789ABCDEF"
EPS = 1e-12
Problem = Tuple[int, int, str, str]
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


def vector_rotation_generators(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...i,...j->...ij", target, current) - torch.einsum("...i,...j->...ij", current, target)


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


def apply_vector_rotation(current: torch.Tensor, momentum_generator: torch.Tensor, learning_rate: float) -> torch.Tensor:
    eye = cached_eye(current.shape[-1], current.device, current.dtype)
    if current.ndim == 1:
        updated = (eye + learning_rate * momentum_generator) @ current
    else:
        updated = ((eye.unsqueeze(0) + learning_rate * momentum_generator) @ current.unsqueeze(-1)).squeeze(-1)
    return normalize_last_dim(updated)


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


def random_unit_vectors(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    x = torch.randn(shape, device=device)
    return normalize_last_dim(x)


def _zeros_like_shape(shape: Sequence[int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(tuple(shape), device=device, dtype=dtype)


@dataclass
class ManualRotationOptimizerState:
    momentum_decay: float
    base_momentum: torch.Tensor | None = None
    left_token_momentum: torch.Tensor | None = None
    right_token_momentum: torch.Tensor | None = None

    def ensure_initialized(self, model: "ManualRotationMatrixNetwork") -> None:
        dtype = model.base_mat.dtype
        device = model.device
        n = model.n
        vocab_size = model.vocab_size

        if self.base_momentum is None or self.base_momentum.shape != (n, n):
            self.base_momentum = _zeros_like_shape((n, n), device=device, dtype=dtype)
        if self.left_token_momentum is None or self.left_token_momentum.shape != (vocab_size, n, n):
            self.left_token_momentum = _zeros_like_shape((vocab_size, n, n), device=device, dtype=dtype)
        if self.right_token_momentum is None or self.right_token_momentum.shape != (vocab_size, n, n):
            self.right_token_momentum = _zeros_like_shape((vocab_size, n, n), device=device, dtype=dtype)

        self.base_momentum = self.base_momentum.to(device=device, dtype=dtype)
        self.left_token_momentum = self.left_token_momentum.to(device=device, dtype=dtype)
        self.right_token_momentum = self.right_token_momentum.to(device=device, dtype=dtype)

    def state_dict(self) -> Dict[str, torch.Tensor | float]:
        result: Dict[str, torch.Tensor | float] = {"momentum_decay": float(self.momentum_decay)}
        if self.base_momentum is not None:
            result["base_momentum"] = self.base_momentum
        if self.left_token_momentum is not None:
            result["left_token_momentum"] = self.left_token_momentum
        if self.right_token_momentum is not None:
            result["right_token_momentum"] = self.right_token_momentum
        return result

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, Any] | None,
        *,
        momentum_decay: float,
        device: torch.device,
    ) -> "ManualRotationOptimizerState":
        state = cls(momentum_decay=momentum_decay)
        if not state_dict:
            return state
        state.momentum_decay = float(state_dict.get("momentum_decay", momentum_decay))
        for attr in ("base_momentum", "left_token_momentum", "right_token_momentum"):
            value = state_dict.get(attr)
            if isinstance(value, torch.Tensor):
                setattr(state, attr, value.to(device))
        return state


@dataclass
class BatchUpdateWorkspace:
    base_primary_delta: torch.Tensor | None = None
    left_primary_delta: torch.Tensor | None = None
    right_primary_delta: torch.Tensor | None = None
    left_secondary_delta: torch.Tensor | None = None
    right_secondary_delta: torch.Tensor | None = None
    query_positive_sum: torch.Tensor | None = None
    query_negative_sum: torch.Tensor | None = None
    past_query_positive_sum: torch.Tensor | None = None
    past_query_negative_sum: torch.Tensor | None = None
    unembed_positive_sum: torch.Tensor | None = None
    unembed_negative_sum: torch.Tensor | None = None
    past_unembed_positive_sum: torch.Tensor | None = None
    past_unembed_negative_sum: torch.Tensor | None = None
    past_query_delta: torch.Tensor | None = None
    unembed_delta: torch.Tensor | None = None
    past_unembed_delta: torch.Tensor | None = None

    def prepare(self, model: "ManualRotationMatrixNetwork") -> None:
        dtype = model.base_mat.dtype
        device = model.device
        n = model.n
        vocab_size = model.vocab_size
        past_count = model.past_queries.shape[0]

        def ensure_zero(name: str, shape: Tuple[int, ...]) -> torch.Tensor:
            tensor = getattr(self, name)
            if tensor is None or tensor.shape != shape or tensor.device != device or tensor.dtype != dtype:
                tensor = torch.zeros(shape, device=device, dtype=dtype)
                setattr(self, name, tensor)
            else:
                tensor.zero_()
            return tensor

        ensure_zero("base_primary_delta", (n, n))
        ensure_zero("left_primary_delta", (vocab_size, n, n))
        ensure_zero("right_primary_delta", (vocab_size, n, n))
        ensure_zero("left_secondary_delta", (vocab_size, n, n))
        ensure_zero("right_secondary_delta", (vocab_size, n, n))
        ensure_zero("query_positive_sum", (n,))
        ensure_zero("query_negative_sum", (n,))
        ensure_zero("past_query_positive_sum", (past_count, n))
        ensure_zero("past_query_negative_sum", (past_count, n))
        ensure_zero("unembed_positive_sum", (vocab_size, n))
        ensure_zero("unembed_negative_sum", (vocab_size, n))
        ensure_zero("past_unembed_positive_sum", (vocab_size, n))
        ensure_zero("past_unembed_negative_sum", (vocab_size, n))
        ensure_zero("past_query_delta", (past_count, n, n))
        ensure_zero("unembed_delta", (vocab_size, n, n))
        ensure_zero("past_unembed_delta", (vocab_size, n, n))


@dataclass
class PrefixOperatorCache:
    prefix_ids: Tuple[int, ...]
    right_total_op: torch.Tensor
    right_all_transpose_op: torch.Tensor
    middle_op: torch.Tensor
    left_total_op: torch.Tensor
    final_state_op: torch.Tensor
    left_u_ops: torch.Tensor
    left_target_ops: torch.Tensor
    left_all_transpose_op: torch.Tensor
    right_u_ops: torch.Tensor
    right_v_prefix_ops: torch.Tensor
    base_query_target_op: torch.Tensor
    primary_query_targets: torch.Tensor
    secondary_query_targets: torch.Tensor


def digit_alphabet(number_base: int) -> str:
    if not (2 <= number_base <= len(DIGIT_SYMBOLS)):
        raise ValueError(f"number_base must be in [2, {len(DIGIT_SYMBOLS)}], got {number_base}")
    return DIGIT_SYMBOLS[:number_base]


def format_in_base(value: int, number_base: int, min_digits: int = 1) -> str:
    digits = digit_alphabet(number_base)
    if value == 0:
        out = "0"
    else:
        chars: List[str] = []
        x = value
        while x > 0:
            x, rem = divmod(x, number_base)
            chars.append(digits[rem])
        out = "".join(reversed(chars))
    if len(out) < min_digits:
        out = ("0" * (min_digits - len(out))) + out
    return out


def random_problem(rng: random.Random, addend_digits: int, number_base: int) -> Problem:
    max_val = (number_base**addend_digits) - 1
    left_addend = rng.randint(0, max_val)
    right_addend = rng.randint(0, max_val)
    prompt_text = (
        f"{format_in_base(left_addend, number_base, min_digits=addend_digits)}"
        f"{PLUS_TOKEN}"
        f"{format_in_base(right_addend, number_base, min_digits=addend_digits)}"
        f"{EQUALS_TOKEN}"
    )
    target_text = format_in_base(left_addend + right_addend, number_base)
    return left_addend, right_addend, prompt_text, target_text


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ManualRotationMatrixNetwork:
    def __init__(
        self,
        *,
        n: int,
        device: torch.device,
        number_base: int,
        token_mat_mode: str,
        base_randomize: float,
        token_randomize: float,
    ):
        output_vocab = EOS_TOKEN + digit_alphabet(number_base)
        vocab = output_vocab + PLUS_TOKEN + EQUALS_TOKEN
        if n < len(vocab):
            raise ValueError(f"n must be >= {len(vocab)} to fit fixed one-hot heads, got {n}")
        if token_mat_mode not in {"left", "right", "both"}:
            raise ValueError(f"unsupported token_mat_mode={token_mat_mode}")
        self.n = n
        self.device = device
        self.number_base = number_base
        self.token_mat_mode = token_mat_mode
        self.base_randomize = base_randomize
        self.token_randomize = token_randomize
        self.vocab = vocab
        self.output_vocab = output_vocab
        self.vocab_size = len(self.vocab)
        self.output_vocab_size = len(self.output_vocab)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

        self.query = random_unit_vectors((1, n), device)[0]
        self.past_queries = torch.empty((0, n), device=device)
        self.unembed_vectors = random_unit_vectors((self.vocab_size, n), device)
        self.past_unembed_vectors = random_unit_vectors((self.vocab_size, n), device)

        self.base_mat = initialize_rotation_like((n, n), device, base_randomize)
        self.left_token_mats = initialize_rotation_like((self.vocab_size, n, n), device, token_randomize)
        self.right_token_mats = initialize_rotation_like((self.vocab_size, n, n), device, token_randomize)

    def uses_left_token_mats(self) -> bool:
        return self.token_mat_mode in {"left", "both"}

    def uses_right_token_mats(self) -> bool:
        return self.token_mat_mode in {"right", "both"}

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]

    def ensure_past_queries(self, count: int) -> None:
        current = self.past_queries.shape[0]
        if count <= current:
            return
        extra = random_unit_vectors((count - current, self.n), self.device)
        if current == 0:
            self.past_queries = extra
        else:
            self.past_queries = torch.cat([self.past_queries, extra], dim=0)

    def prefix_state_from_query(self, token_ids: Sequence[int], query: torch.Tensor) -> torch.Tensor:
        v = query
        if self.uses_right_token_mats():
            for tid in reversed(token_ids):
                v = self.right_token_mats[tid] @ v
        v = self.base_mat @ v
        if self.uses_left_token_mats():
            for tid in token_ids:
                v = self.left_token_mats[tid] @ v
        return v

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
    ) -> Tuple["ManualRotationMatrixNetwork", int | None]:
        n = int(ckpt["n"])
        number_base = int(ckpt["number_base"])
        token_mat_mode = str(ckpt["token_mat_mode"])
        model = cls(
            n=n,
            device=device,
            number_base=number_base,
            token_mat_mode=token_mat_mode,
            base_randomize=0.0,
            token_randomize=0.0,
        )
        model.left_token_mats = ckpt["left_token_mats"].to(device)
        model.right_token_mats = ckpt["right_token_mats"].to(device)
        model.base_mat = ckpt["base_mat"].to(device)
        if "query" in ckpt:
            model.query = ckpt["query"].to(device)
        if "past_queries" in ckpt:
            model.past_queries = ckpt["past_queries"].to(device)
        if "unembed_vectors" in ckpt:
            model.unembed_vectors = ckpt["unembed_vectors"].to(device)
        if "past_unembed_vectors" in ckpt:
            model.past_unembed_vectors = ckpt["past_unembed_vectors"].to(device)
        addend_digits = ckpt.get("addend_digits")
        if addend_digits is not None:
            addend_digits = int(addend_digits)
        return model, addend_digits

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device) -> Tuple["ManualRotationMatrixNetwork", int | None]:
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
) -> Tuple[ManualRotationMatrixNetwork, int | None, ManualRotationOptimizerState, int]:
    ckpt = load_checkpoint_dict(path, device)
    model, addend_digits = ManualRotationMatrixNetwork.from_checkpoint_dict(ckpt, device)
    optimizer_state = ManualRotationOptimizerState.from_state_dict(
        ckpt.get("optimizer_state"),
        momentum_decay=momentum_decay,
        device=device,
    )
    optimizer_state.ensure_initialized(model)
    completed_iters = int(ckpt.get("completed_iters") or 0)
    return model, addend_digits, optimizer_state, completed_iters


def save_checkpoint(
    model: ManualRotationMatrixNetwork,
    save_path: str,
    addend_digits: int,
    *,
    optimizer_state: ManualRotationOptimizerState | None = None,
    update_orthogonalize_steps: int = DEFAULT_UPDATE_ORTHOGONALIZE_STEPS,
    completed_iters: int | None = None,
) -> None:
    path = Path(save_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": model.n,
        "number_base": model.number_base,
        "vocab": model.vocab,
        "output_vocab": model.output_vocab,
        "token_mat_mode": model.token_mat_mode,
        "left_token_mats": model.left_token_mats,
        "right_token_mats": model.right_token_mats,
        "base_mat": model.base_mat,
        "query": model.query,
        "past_queries": model.past_queries,
        "unembed_vectors": model.unembed_vectors,
        "past_unembed_vectors": model.past_unembed_vectors,
        "addend_digits": addend_digits,
        "optimizer_state": None if optimizer_state is None else optimizer_state.state_dict(),
        "update_orthogonalize_steps": int(update_orthogonalize_steps),
        "completed_iters": None if completed_iters is None else int(completed_iters),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            **payload,
        },
        tmp_path,
    )
    tmp_path.replace(path)


def format_float_token(x: float) -> str:
    return format(x, ".6g").replace("-", "m").replace(".", "p")


def format_run_config(args: argparse.Namespace, *, addend_digits: int) -> str:
    items = [
        ("n", args.n),
        ("number_base", args.number_base),
        ("addend_digits", addend_digits),
        ("token_mat_mode", args.token_mat_mode),
        ("base_randomize", args.base_randomize),
        ("token_randomize", args.token_randomize),
        ("iters", args.iters),
        ("batch_size", args.batch_size),
        ("token_learning_rate", args.token_learning_rate),
        ("base_learning_rate", args.base_learning_rate),
        ("primary_query_learning_rate", args.primary_query_learning_rate),
        ("primary_unembed_learning_rate", args.primary_unembed_learning_rate),
        ("secondary_query_learning_rate", args.secondary_query_learning_rate),
        ("secondary_unembed_learning_rate", args.secondary_unembed_learning_rate),
        ("momentum_decay", args.momentum_decay),
        ("negative_scale", args.negative_scale),
        ("secondary_matrix_scale", args.secondary_matrix_scale),
        ("update_orthogonalize_steps", args.update_orthogonalize_steps),
        ("checkpoint_every", getattr(args, "checkpoint_every", 0)),
        ("seed", args.seed),
        ("device", args.device),
        ("load_path", args.load_path),
        ("save_path", args.save_path),
    ]
    return "\n".join(f"{k}={v}" for k, v in items)


def default_save_path(args: argparse.Namespace, addend_digits: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = (
        f"manual_rotation_n{args.n}"
        f"_d{addend_digits}"
        f"_base{args.number_base}"
        f"_mode{args.token_mat_mode}"
        f"_brand{format_float_token(args.base_randomize)}"
        f"_trand{format_float_token(args.token_randomize)}"
        f"_it{args.iters}"
        f"_bs{args.batch_size}"
        f"_tlr{format_float_token(args.token_learning_rate)}"
        f"_blr{format_float_token(args.base_learning_rate)}"
        f"_pqlr{format_float_token(args.primary_query_learning_rate)}"
        f"_pulr{format_float_token(args.primary_unembed_learning_rate)}"
        f"_sqlr{format_float_token(args.secondary_query_learning_rate)}"
        f"_sulr{format_float_token(args.secondary_unembed_learning_rate)}"
        f"_mom{format_float_token(args.momentum_decay)}"
        f"_neg{format_float_token(args.negative_scale)}"
        f"_sms{format_float_token(args.secondary_matrix_scale)}"
        f"_orth{args.update_orthogonalize_steps}"
        f"_seed{args.seed}"
        f"_{timestamp}.pt"
    )
    return str(Path("checkpoints") / name)


def prefix_operator_pair_from_ids(model: ManualRotationMatrixNetwork, token_ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    eye = cached_eye(model.n, model.device, model.base_mat.dtype)
    right_total_op = eye
    left_total_op = eye
    if model.uses_right_token_mats():
        for tid in token_ids:
            right_total_op = right_total_op @ model.right_token_mats[tid]
    if model.uses_left_token_mats():
        for tid in token_ids:
            left_total_op = model.left_token_mats[tid] @ left_total_op
    return left_total_op, right_total_op


def advance_prefix_operator_pair(
    model: ManualRotationMatrixNetwork,
    left_total_op: torch.Tensor,
    right_total_op: torch.Tensor,
    token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if model.uses_right_token_mats():
        right_total_op = right_total_op @ model.right_token_mats[token_id]
    if model.uses_left_token_mats():
        left_total_op = model.left_token_mats[token_id] @ left_total_op
    return left_total_op, right_total_op


def state_from_prefix_operators(
    model: ManualRotationMatrixNetwork,
    left_total_op: torch.Tensor,
    right_total_op: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    return left_total_op @ (model.base_mat @ (right_total_op @ query))


def predict_next_id_from_prefix_operators(
    model: ManualRotationMatrixNetwork,
    left_total_op: torch.Tensor,
    right_total_op: torch.Tensor,
) -> int:
    state = state_from_prefix_operators(model, left_total_op, right_total_op, model.query)
    scores = model.unembed_vectors[: model.output_vocab_size] @ normalize_columns(state.unsqueeze(1))
    return int(scores[:, 0].argmax().item())


def generate_until_eos(model: ManualRotationMatrixNetwork, prompt_text: str, max_len: int) -> Tuple[str, bool]:
    prefix_ids = model.encode(prompt_text)
    left_total_op, right_total_op = prefix_operator_pair_from_ids(model, prefix_ids)
    pred = ""
    for _ in range(max_len):
        next_id = predict_next_id_from_prefix_operators(model, left_total_op, right_total_op)
        next_ch = model.itos[next_id]
        if next_ch == EOS_TOKEN:
            return pred, True
        pred += next_ch
        left_total_op, right_total_op = advance_prefix_operator_pair(model, left_total_op, right_total_op, next_id)
    return pred, False


def evaluate(
    model: ManualRotationMatrixNetwork,
    eval_samples: int,
    seed: int,
    addend_digits: int,
    number_base: int,
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    exact = 0
    tf_correct = 0
    tf_total = 0
    stopped = 0

    for _ in range(eval_samples):
        _, _, prompt_text, target_text = random_problem(rng, addend_digits, number_base)
        pred, did_stop = generate_until_eos(model, prompt_text, len(target_text) + 2)
        stopped += int(did_stop)
        if did_stop and pred == target_text:
            exact += 1

        target_seq = target_text + EOS_TOKEN
        prompt_ids = model.encode(prompt_text)
        target_ids = model.encode(target_seq)
        left_total_op, right_total_op = prefix_operator_pair_from_ids(model, prompt_ids)
        for i, ch in enumerate(target_seq):
            pred_id = predict_next_id_from_prefix_operators(model, left_total_op, right_total_op)
            target_id = target_ids[i]
            tf_correct += int(pred_id == target_id)
            tf_total += 1
            left_total_op, right_total_op = advance_prefix_operator_pair(model, left_total_op, right_total_op, target_id)

    return (
        exact / max(eval_samples, 1),
        tf_correct / max(tf_total, 1),
        stopped / max(eval_samples, 1),
    )


def show_samples(
    model: ManualRotationMatrixNetwork,
    seed: int,
    addend_digits: int,
    number_base: int,
    count: int = 10,
) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        left_addend, right_addend, prompt_text, target_text = random_problem(rng, addend_digits, number_base)
        pred, did_stop = generate_until_eos(model, prompt_text, len(target_text) + 2)
        ok = "OK" if (did_stop and pred == target_text) else "XX"
        stop_txt = "eos" if did_stop else "max"
        print(f"{prompt_text}{target_text:>4s} | pred={pred:>4s} ({stop_txt}) [{ok}]   ({left_addend}+{right_addend})")


@torch.no_grad()
def apply_batch_update(
    model: ManualRotationMatrixNetwork,
    optimizer_state: ManualRotationOptimizerState,
    workspace: BatchUpdateWorkspace,
    prefixes: Sequence[Sequence[int]],
    target_ids: Sequence[int],
    token_learning_rate: float,
    base_learning_rate: float,
    primary_query_learning_rate: float,
    primary_unembed_learning_rate: float,
    secondary_query_learning_rate: float,
    secondary_unembed_learning_rate: float,
    negative_scale: float,
    secondary_matrix_scale: float,
    update_orthogonalize_steps: int,
) -> Tuple[float, float]:
    n = model.n
    max_lag = max((len(prefix_ids) for prefix_ids in prefixes), default=0)
    model.ensure_past_queries(max_lag)
    workspace.prepare(model)
    base_primary_delta = workspace.base_primary_delta
    left_primary_delta = workspace.left_primary_delta
    right_primary_delta = workspace.right_primary_delta
    left_secondary_delta = workspace.left_secondary_delta
    right_secondary_delta = workspace.right_secondary_delta
    query_positive_sum = workspace.query_positive_sum
    query_negative_sum = workspace.query_negative_sum
    past_query_positive_sum = workspace.past_query_positive_sum
    past_query_negative_sum = workspace.past_query_negative_sum
    unembed_positive_sum = workspace.unembed_positive_sum
    unembed_negative_sum = workspace.unembed_negative_sum
    past_unembed_positive_sum = workspace.past_unembed_positive_sum
    past_unembed_negative_sum = workspace.past_unembed_negative_sum
    correct = 0
    total = 0
    primary_objective_weight = 0.0
    secondary_objective_weight = 0.0
    mean_target_score = 0.0

    base = model.base_mat
    base_t = base.transpose(-1, -2)
    eye = cached_eye(n, model.device, base.dtype)
    prefix_cache_map: Dict[Tuple[int, ...], PrefixOperatorCache] = {}

    def build_prefix_cache(prefix_ids: Sequence[int]) -> PrefixOperatorCache:
        prefix_key = tuple(prefix_ids)
        cached = prefix_cache_map.get(prefix_key)
        if cached is not None:
            return cached

        prefix_len = len(prefix_key)
        last_tid = prefix_key[-1]
        parent_cache = build_prefix_cache(prefix_key[:-1]) if prefix_len > 1 else None

        if model.uses_right_token_mats():
            right_mat = model.right_token_mats[last_tid]
            right_mat_t = right_mat.transpose(-1, -2)
            if parent_cache is None:
                right_total_op = right_mat
                right_all_transpose_op = right_mat_t
                right_u_ops = right_total_op.unsqueeze(0)
                right_v_prefix_ops = eye.unsqueeze(0)
            else:
                right_total_op = parent_cache.right_total_op @ right_mat
                right_all_transpose_op = right_mat_t @ parent_cache.right_all_transpose_op
                right_u_ops = torch.cat([parent_cache.right_u_ops @ right_mat, right_mat.unsqueeze(0)], dim=0)
                right_v_prefix_ops = torch.cat(
                    [parent_cache.right_v_prefix_ops, parent_cache.right_all_transpose_op.unsqueeze(0)],
                    dim=0,
                )
        else:
            right_u_ops = torch.empty((0, n, n), device=model.device, dtype=base.dtype)
            right_v_prefix_ops = torch.empty((0, n, n), device=model.device, dtype=base.dtype)
            right_total_op = eye
            right_all_transpose_op = eye

        middle_op = base @ right_total_op

        if model.uses_left_token_mats():
            left_mat = model.left_token_mats[last_tid]
            left_mat_t = left_mat.transpose(-1, -2)
            if parent_cache is None:
                left_total_op = left_mat
                left_all_transpose_op = left_mat_t
                left_u_ops = left_total_op.unsqueeze(0)
                left_target_ops = eye.unsqueeze(0)
            else:
                left_total_op = left_mat @ parent_cache.left_total_op
                left_all_transpose_op = parent_cache.left_all_transpose_op @ left_mat_t
                left_u_ops = torch.cat([parent_cache.left_u_ops, left_total_op.unsqueeze(0)], dim=0)
                left_target_ops = torch.cat(
                    [parent_cache.left_target_ops @ left_mat_t, eye.unsqueeze(0)],
                    dim=0,
                )
        else:
            left_u_ops = torch.empty((0, n, n), device=model.device, dtype=base.dtype)
            left_target_ops = torch.empty((0, n, n), device=model.device, dtype=base.dtype)
            left_total_op = eye
            left_all_transpose_op = eye

        final_state_op = left_total_op @ middle_op
        base_query_target_op = base_t @ left_all_transpose_op
        query_target_op = right_all_transpose_op @ base_query_target_op
        primary_query_targets = normalize_last_dim(model.unembed_vectors @ query_target_op.transpose(-1, -2))
        secondary_query_targets = normalize_last_dim(model.past_unembed_vectors @ query_target_op.transpose(-1, -2))

        cached = PrefixOperatorCache(
            prefix_ids=prefix_key,
            right_total_op=right_total_op,
            right_all_transpose_op=right_all_transpose_op,
            middle_op=middle_op,
            left_total_op=left_total_op,
            final_state_op=final_state_op,
            left_u_ops=left_u_ops,
            left_target_ops=left_target_ops,
            left_all_transpose_op=left_all_transpose_op,
            right_u_ops=right_u_ops,
            right_v_prefix_ops=right_v_prefix_ops,
            base_query_target_op=base_query_target_op,
            primary_query_targets=primary_query_targets,
            secondary_query_targets=secondary_query_targets,
        )
        prefix_cache_map[prefix_key] = cached
        return cached

    def accumulate_query_vector_updates(
        cache: PrefixOperatorCache,
        query_index: int | None,
        state_normed: torch.Tensor,
        scores: torch.Tensor,
        target_id: int,
        vector_weight: float,
        query_target_bank: torch.Tensor,
        unembed_positive_acc: torch.Tensor,
        unembed_negative_acc: torch.Tensor,
        use_output_slice: bool,
    ) -> None:
        target_score = float(scores[target_id].item())
        unembed_positive_acc[target_id].add_(vector_weight * state_normed)
        target_query = query_target_bank[target_id]
        if query_index is None:
            query_positive_sum.add_(vector_weight * target_query)
        else:
            past_query_positive_sum[query_index].add_(vector_weight * target_query)

        candidate_size = model.output_vocab_size if use_output_slice else model.vocab_size
        hard_negative_mask = scores[:candidate_size] > target_score
        hard_negative_mask[target_id] = False
        hard_negative_ids = hard_negative_mask.nonzero(as_tuple=False).flatten()
        if hard_negative_ids.numel() > 0:
            wrong_state = state_normed.unsqueeze(0).expand(hard_negative_ids.shape[0], -1) * vector_weight
            unembed_negative_acc.index_add_(0, hard_negative_ids, wrong_state)
            wrong_queries = query_target_bank[hard_negative_ids].sum(dim=0) * vector_weight
            if query_index is None:
                query_negative_sum.add_(wrong_queries)
            else:
                past_query_negative_sum[query_index].add_(wrong_queries)

    def accumulate_objective(
        *,
        cache: PrefixOperatorCache,
        query_vec: torch.Tensor,
        unembed_bank: torch.Tensor,
        query_target_bank: torch.Tensor,
        objective_target_id: int,
        matrix_weight: float,
        vector_weight: float,
        query_index: int | None,
        unembed_positive_acc: torch.Tensor,
        unembed_negative_acc: torch.Tensor,
        use_output_slice: bool,
        min_update_index: int,
        allow_base_update: bool,
        left_delta_acc: torch.Tensor,
        right_delta_acc: torch.Tensor,
        base_delta_acc: torch.Tensor | None,
    ) -> Tuple[float, bool]:
        query_mat = query_vec.unsqueeze(1)
        middle_state = cache.middle_op @ query_mat
        state_mat = cache.final_state_op @ query_mat
        target_mat = unembed_bank[objective_target_id].unsqueeze(1)
        state_normed = normalize_columns(state_mat)[:, 0]
        scores = unembed_bank @ state_normed
        accumulate_query_vector_updates(
            cache,
            query_index,
            state_normed,
            scores,
            objective_target_id,
            vector_weight,
            query_target_bank,
            unembed_positive_acc,
            unembed_negative_acc,
            use_output_slice,
        )

        if model.uses_left_token_mats():
            for idx in range(len(cache.prefix_ids) - 1, -1, -1):
                tid = cache.prefix_ids[idx]
                u = normalize_columns(cache.left_u_ops[idx] @ middle_state)
                v = normalize_columns(cache.left_target_ops[idx] @ target_mat)
                if idx >= min_update_index:
                    left_delta_acc[tid].add_(matrix_rotation_generator(u, v, matrix_weight))

        if allow_base_update and base_delta_acc is not None:
            u_base = normalize_columns(middle_state)
            v_base = normalize_columns(cache.left_all_transpose_op @ target_mat)
            base_delta_acc.add_(matrix_rotation_generator(u_base, v_base, matrix_weight))

        if model.uses_right_token_mats():
            for idx, tid in enumerate(cache.prefix_ids):
                u = normalize_columns(cache.right_u_ops[idx] @ query_mat)
                v = normalize_columns(cache.right_v_prefix_ops[idx] @ (cache.base_query_target_op @ target_mat))
                if idx >= min_update_index:
                    right_delta_acc[tid].add_(matrix_rotation_generator(u, v, matrix_weight))

        if use_output_slice:
            predicted = int(scores[: model.output_vocab_size].argmax().item())
        else:
            predicted = int(scores.argmax().item())
        return float(scores[objective_target_id].item()), predicted == objective_target_id

    for prefix_ids, target_id in zip(prefixes, target_ids):
        cache = build_prefix_cache(prefix_ids)
        if target_id >= 0:
            score, is_correct = accumulate_objective(
                cache=cache,
                query_vec=model.query,
                unembed_bank=model.unembed_vectors,
                query_target_bank=cache.primary_query_targets,
                objective_target_id=target_id,
                matrix_weight=1.0,
                vector_weight=1.0,
                query_index=None,
                unembed_positive_acc=unembed_positive_sum,
                unembed_negative_acc=unembed_negative_sum,
                use_output_slice=True,
                min_update_index=0,
                allow_base_update=True,
                left_delta_acc=left_primary_delta,
                right_delta_acc=right_primary_delta,
                base_delta_acc=base_primary_delta,
            )
            primary_objective_weight += 1.0
            mean_target_score += score
            correct += int(is_correct)
            total += 1

        for lag in range(1, len(prefix_ids) + 1):
            query_index = lag - 1
            past_target_id = prefix_ids[-lag]
            accumulate_objective(
                cache=cache,
                query_vec=model.past_queries[query_index],
                unembed_bank=model.past_unembed_vectors,
                query_target_bank=cache.secondary_query_targets,
                objective_target_id=past_target_id,
                matrix_weight=1.0,
                vector_weight=1.0,
                query_index=query_index,
                unembed_positive_acc=past_unembed_positive_sum,
                unembed_negative_acc=past_unembed_negative_sum,
                use_output_slice=False,
                min_update_index=query_index,
                allow_base_update=False,
                left_delta_acc=left_secondary_delta,
                right_delta_acc=right_secondary_delta,
                base_delta_acc=None,
            )
            secondary_objective_weight += 1.0

    primary_scale = 1.0 / float(max(primary_objective_weight, 1.0))
    secondary_scale = 1.0 / float(max(secondary_objective_weight, 1.0))
    optimizer_state.ensure_initialized(model)
    decay = optimizer_state.momentum_decay
    momentum_inject = 1.0 - decay

    assert optimizer_state.base_momentum is not None
    assert optimizer_state.left_token_momentum is not None
    assert optimizer_state.right_token_momentum is not None

    optimizer_state.base_momentum.mul_(decay).add_(base_primary_delta * (primary_scale * momentum_inject))
    if float(optimizer_state.base_momentum.abs().amax().item()) > EPS:
        model.base_mat = apply_matrix_rotation(
            model.base_mat,
            optimizer_state.base_momentum,
            base_learning_rate,
            update_orthogonalize_steps,
        )

    optimizer_state.left_token_momentum.mul_(decay).add_(left_primary_delta * (primary_scale * momentum_inject))
    left_hist_active = optimizer_state.left_token_momentum.abs().amax(dim=(1, 2)) > EPS
    if left_hist_active.any():
        model.left_token_mats[left_hist_active] = apply_matrix_rotation(
            model.left_token_mats[left_hist_active],
            optimizer_state.left_token_momentum[left_hist_active],
            token_learning_rate,
            update_orthogonalize_steps,
        )

    optimizer_state.right_token_momentum.mul_(decay).add_(right_primary_delta * (primary_scale * momentum_inject))
    right_hist_active = optimizer_state.right_token_momentum.abs().amax(dim=(1, 2)) > EPS
    if right_hist_active.any():
        model.right_token_mats[right_hist_active] = apply_matrix_rotation(
            model.right_token_mats[right_hist_active],
            optimizer_state.right_token_momentum[right_hist_active],
            token_learning_rate,
            update_orthogonalize_steps,
        )

    left_secondary_active = left_secondary_delta.abs().amax(dim=(1, 2)) > 0
    if left_secondary_active.any():
        model.left_token_mats[left_secondary_active] = apply_matrix_rotation(
            model.left_token_mats[left_secondary_active],
            left_secondary_delta[left_secondary_active] * (secondary_scale * secondary_matrix_scale),
            token_learning_rate,
            update_orthogonalize_steps,
        )

    right_secondary_active = right_secondary_delta.abs().amax(dim=(1, 2)) > 0
    if right_secondary_active.any():
        model.right_token_mats[right_secondary_active] = apply_matrix_rotation(
            model.right_token_mats[right_secondary_active],
            right_secondary_delta[right_secondary_active] * (secondary_scale * secondary_matrix_scale),
            token_learning_rate,
            update_orthogonalize_steps,
        )

    query_net = query_positive_sum - negative_scale * query_negative_sum
    query_net_norm = float(query_net.norm().item())
    if query_net_norm > EPS:
        query_target = query_net / query_net_norm
        query_delta = vector_rotation_generators(model.query, query_target)
        model.query = apply_vector_rotation(model.query, query_delta, primary_query_learning_rate)

    past_query_net = past_query_positive_sum - negative_scale * past_query_negative_sum
    if model.past_queries.shape[0] > 0:
        past_query_delta = workspace.past_query_delta
        past_query_net_norm = past_query_net.norm(dim=1)
        past_query_active = past_query_net_norm > EPS
        if past_query_active.any():
            target_dirs = past_query_net[past_query_active] / past_query_net_norm[past_query_active].unsqueeze(1)
            past_query_delta[past_query_active] = vector_rotation_generators(model.past_queries[past_query_active], target_dirs)
            model.past_queries[past_query_active] = apply_vector_rotation(
                model.past_queries[past_query_active],
                past_query_delta[past_query_active],
                secondary_query_learning_rate,
            )

    unembed_net = unembed_positive_sum - negative_scale * unembed_negative_sum
    unembed_delta = workspace.unembed_delta
    unembed_net_norm = unembed_net.norm(dim=1)
    unembed_active = unembed_net_norm > EPS
    if unembed_active.any():
        target_dirs = unembed_net[unembed_active] / unembed_net_norm[unembed_active].unsqueeze(1)
        unembed_delta[unembed_active] = vector_rotation_generators(model.unembed_vectors[unembed_active], target_dirs)
        model.unembed_vectors[unembed_active] = apply_vector_rotation(
            model.unembed_vectors[unembed_active],
            unembed_delta[unembed_active],
            primary_unembed_learning_rate,
        )

    past_unembed_net = past_unembed_positive_sum - negative_scale * past_unembed_negative_sum
    past_unembed_delta = workspace.past_unembed_delta
    past_unembed_net_norm = past_unembed_net.norm(dim=1)
    past_unembed_active = past_unembed_net_norm > EPS
    if past_unembed_active.any():
        target_dirs = past_unembed_net[past_unembed_active] / past_unembed_net_norm[past_unembed_active].unsqueeze(1)
        past_unembed_delta[past_unembed_active] = vector_rotation_generators(
            model.past_unembed_vectors[past_unembed_active],
            target_dirs,
        )
        model.past_unembed_vectors[past_unembed_active] = apply_vector_rotation(
            model.past_unembed_vectors[past_unembed_active],
            past_unembed_delta[past_unembed_active],
            secondary_unembed_learning_rate,
        )

    return (
        mean_target_score / max(total, 1),
        correct / max(total, 1),
    )


def train(
    *,
    model: ManualRotationMatrixNetwork,
    optimizer_state: ManualRotationOptimizerState,
    iters: int,
    batch_size: int,
    token_learning_rate: float,
    base_learning_rate: float,
    primary_query_learning_rate: float,
    primary_unembed_learning_rate: float,
    secondary_query_learning_rate: float,
    secondary_unembed_learning_rate: float,
    addend_digits: int,
    number_base: int,
    seed: int,
    log_every: int,
    eval_every: int,
    eval_samples: int,
    negative_scale: float,
    secondary_matrix_scale: float,
    update_orthogonalize_steps: int,
    checkpoint_every: int = 0,
    checkpoint_callback: Callable[[int, ManualRotationMatrixNetwork, ManualRotationOptimizerState], None] | None = None,
) -> Tuple[ManualRotationMatrixNetwork, ManualRotationOptimizerState]:
    rng = random.Random(seed)
    workspace = BatchUpdateWorkspace()

    def sample_batch() -> Tuple[List[List[int]], List[int]]:
        prefixes: List[List[int]] = []
        target_ids: List[int] = []

        for _ in range(batch_size):
            _, _, prompt_text, target_text = random_problem(rng, addend_digits, number_base)
            full_seq = prompt_text + target_text + EOS_TOKEN
            prompt_len = len(prompt_text)
            full_ids = model.encode(full_seq)
            for prefix_len in range(1, len(full_seq)):
                prefixes.append(full_ids[:prefix_len])
                if prefix_len >= prompt_len:
                    target_ids.append(full_ids[prefix_len])
                else:
                    target_ids.append(-1)

        return prefixes, target_ids

    for iter_idx in range(1, iters + 1):
        prefixes, target_ids = sample_batch()
        mean_target_score, token_acc = apply_batch_update(
            model,
            optimizer_state,
            workspace,
            prefixes=prefixes,
            target_ids=target_ids,
            token_learning_rate=token_learning_rate,
            base_learning_rate=base_learning_rate,
            primary_query_learning_rate=primary_query_learning_rate,
            primary_unembed_learning_rate=primary_unembed_learning_rate,
            secondary_query_learning_rate=secondary_query_learning_rate,
            secondary_unembed_learning_rate=secondary_unembed_learning_rate,
            negative_scale=negative_scale,
            secondary_matrix_scale=secondary_matrix_scale,
            update_orthogonalize_steps=update_orthogonalize_steps,
        )

        if iter_idx % log_every == 0 or iter_idx == 1:
            print(
                f"iter={iter_idx:5d} mean_target_score={mean_target_score:.4f} "
                f"token_acc={token_acc:.3f}"
            )

        if iter_idx % eval_every == 0 or iter_idx == iters:
            exact, tf_acc, stop_rate = evaluate(
                model,
                eval_samples=eval_samples,
                seed=seed + iter_idx,
                addend_digits=addend_digits,
                number_base=number_base,
            )
            print(
                f"  eval exact_match={exact:.3f} teacher_forced_token_acc={tf_acc:.3f} "
                f"stop_rate={stop_rate:.3f}"
            )

        if checkpoint_callback is not None and checkpoint_every > 0 and iter_idx % checkpoint_every == 0:
            checkpoint_callback(iter_idx, model, optimizer_state)

    return model, optimizer_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense matrix network with manual rotation updates and learned query/unembedding vectors")
    p.add_argument("--n", type=int, default=32, help="Square matrix dimension; must be >= vocab size")
    p.add_argument("--number-base", type=int, default=10, help="Arithmetic base for generated addition problems (2-16)")
    p.add_argument("--token-mat-mode", type=str, default="right", choices=["left", "right", "both"], help="Apply learned token matrices on the left of base, right of base, or both")
    p.add_argument("--base-randomize", type=float, default=0.0, help="Base init randomization strength; 0 gives identity")
    p.add_argument("--token-randomize", type=float, default=0.0, help="Token-matrix init randomization strength; 0 gives identity")
    p.add_argument("--iters", type=int, default=5000, help="Training iterations")
    p.add_argument("--batch-size", type=int, default=32, help="Problems per iteration")
    p.add_argument("--token-learning-rate", type=float, default=1.0, help="Step size for token embedding matrices")
    p.add_argument("--base-learning-rate", type=float, default=0.1, help="Step size for the base matrix")
    p.add_argument("--primary-query-learning-rate", type=float, default=0.001, help="Step size for the primary next-token query vector")
    p.add_argument("--primary-unembed-learning-rate", type=float, default=0.001, help="Step size for primary next-token unembedding vectors")
    p.add_argument("--secondary-query-learning-rate", type=float, default=0.1, help="Step size for secondary past-token query vectors")
    p.add_argument("--secondary-unembed-learning-rate", type=float, default=0.1, help="Step size for secondary past-token unembedding vectors")
    p.add_argument("--momentum-decay", type=float, default=0.9, help="EMA decay for primary matrix momentum buffers")
    p.add_argument("--addend-digits", type=int, default=3, help="Digits for each addend in a+b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--negative-scale", type=float, default=2.0, help="Repulsion weight for wrong learned vectors that outscore the correct one")
    p.add_argument("--secondary-matrix-scale", type=float, default=0.1, help="Scale multiplier for matrix learning from past-token auxiliary objectives")
    p.add_argument("--update-orthogonalize-steps", type=int, default=1, help="Newton-Schulz orthogonalization steps after each matrix update")
    p.add_argument("--checkpoint-every", type=int, default=0, help="Save latest checkpoint every N iterations; 0 disables periodic saves")
    p.add_argument("--load-path", type=str, default=None, help="Optional checkpoint to continue training from")
    p.add_argument("--save-path", type=str, default=None, help="Optional checkpoint path override")
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.momentum_decay < 1.0):
        raise ValueError("--momentum-decay must be in [0, 1)")
    if args.update_orthogonalize_steps < 0:
        raise ValueError("--update-orthogonalize-steps must be >= 0")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)
    print(f"device={device}")

    optimizer_state = ManualRotationOptimizerState(momentum_decay=args.momentum_decay)
    previous_completed_iters = 0
    if args.load_path is None:
        model = ManualRotationMatrixNetwork(
            n=args.n,
            device=device,
            number_base=args.number_base,
            token_mat_mode=args.token_mat_mode,
            base_randomize=args.base_randomize,
            token_randomize=args.token_randomize,
        )
        addend_digits = args.addend_digits
    else:
        model, loaded_addend_digits, optimizer_state, previous_completed_iters = load_training_checkpoint(
            args.load_path,
            device,
            momentum_decay=args.momentum_decay,
        )
        if model.n != args.n:
            print(f"loaded_n={model.n}; overriding --n={args.n}")
        if model.number_base != args.number_base:
            print(f"loaded_number_base={model.number_base}; ignoring --number-base={args.number_base}")
        if model.token_mat_mode != args.token_mat_mode:
            print(f"loaded_token_mat_mode={model.token_mat_mode}; ignoring --token-mat-mode={args.token_mat_mode}")
        addend_digits = int(loaded_addend_digits or args.addend_digits)
        if loaded_addend_digits is not None and loaded_addend_digits != args.addend_digits:
            print(f"loaded_addend_digits={loaded_addend_digits}; overriding --addend-digits={args.addend_digits}")

    save_path = args.save_path or default_save_path(args, addend_digits)
    print(f"output_vocab={model.output_vocab}")
    print(f"save_path={save_path}")
    args.save_path = save_path
    print(format_run_config(args, addend_digits=addend_digits))

    def checkpoint_callback(iter_idx: int, checkpoint_model: ManualRotationMatrixNetwork, checkpoint_optimizer: ManualRotationOptimizerState) -> None:
        save_checkpoint(
            checkpoint_model,
            save_path,
            addend_digits=addend_digits,
            optimizer_state=checkpoint_optimizer,
            update_orthogonalize_steps=args.update_orthogonalize_steps,
            completed_iters=previous_completed_iters + iter_idx,
        )
        print(f"checkpoint_iter={iter_idx} save_path={save_path}")

    model, optimizer_state = train(
        model=model,
        optimizer_state=optimizer_state,
        iters=args.iters,
        batch_size=args.batch_size,
        token_learning_rate=args.token_learning_rate,
        base_learning_rate=args.base_learning_rate,
        primary_query_learning_rate=args.primary_query_learning_rate,
        primary_unembed_learning_rate=args.primary_unembed_learning_rate,
        secondary_query_learning_rate=args.secondary_query_learning_rate,
        secondary_unembed_learning_rate=args.secondary_unembed_learning_rate,
        addend_digits=addend_digits,
        number_base=model.number_base,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        negative_scale=args.negative_scale,
        secondary_matrix_scale=args.secondary_matrix_scale,
        update_orthogonalize_steps=args.update_orthogonalize_steps,
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=checkpoint_callback if args.checkpoint_every > 0 else None,
    )

    save_checkpoint(
        model,
        save_path,
        addend_digits=addend_digits,
        optimizer_state=optimizer_state,
        update_orthogonalize_steps=args.update_orthogonalize_steps,
        completed_iters=previous_completed_iters + args.iters,
    )
    print(f"saved_checkpoint={save_path}")
    print("\nSample predictions:")
    show_samples(model, seed=args.seed + 999, addend_digits=addend_digits, number_base=model.number_base)


if __name__ == "__main__":
    main()
