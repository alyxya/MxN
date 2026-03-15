#!/usr/bin/env python3
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


EOS_TOKEN = "~"
OUTPUT_VOCAB = f"{EOS_TOKEN}0123456789"
VOCAB = OUTPUT_VOCAB + "+="
EPS = 1e-12
Problem = Tuple[int, int, str, str]
TAU = 2.0 * torch.pi
INIT_ORTHOGONALIZE_STEPS = 4


def normalize_vector(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm() + eps)


def normalize_columns(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=0, keepdim=True) + eps)


def manual_rotation_delta(u: torch.Tensor, v: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # A skew-symmetric alternative would use outer(v, u) - outer(u, v), but the
    # default trainer keeps the simpler outer-diff update here.
    return ((v - u) * weights.unsqueeze(0)) @ u.transpose(-1, -2)


def orthogonalize_newton_schulz(w: torch.Tensor) -> torch.Tensor:
    return 1.5 * w - 0.5 * (w @ w.transpose(-1, -2) @ w)


def initialize_rotation_like(shape: Tuple[int, ...], device: torch.device, mode: str) -> torch.Tensor:
    n = shape[-1]
    eye = torch.eye(n, device=device)
    if len(shape) > 2:
        eye = eye.expand(*shape[:-2], n, n).clone()
    if mode == "identity":
        return eye
    noise = torch.randn(shape, device=device) / (n**0.5)
    if mode == "full_random":
        w = noise
    elif mode == "partial_random":
        w = (eye + noise) / (2.0**0.5)
    else:
        raise ValueError(f"unsupported init mode={mode}")
    for _ in range(INIT_ORTHOGONALIZE_STEPS):
        w = orthogonalize_newton_schulz(w)
    return w


def random_problem(rng: random.Random, addend_digits: int) -> Problem:
    max_val = (10**addend_digits) - 1
    left_addend = rng.randint(0, max_val)
    right_addend = rng.randint(0, max_val)
    prompt_text = f"{left_addend:0{addend_digits}d}+{right_addend:0{addend_digits}d}="
    target_text = str(left_addend + right_addend)
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
        token_mat_mode: str,
        base_init_mode: str,
        token_init_mode: str,
    ):
        if n < len(VOCAB):
            raise ValueError(f"n must be >= {len(VOCAB)} to fit fixed one-hot heads, got {n}")
        if token_mat_mode not in {"left", "right", "both"}:
            raise ValueError(f"unsupported token_mat_mode={token_mat_mode}")
        if base_init_mode not in {"identity", "full_random", "partial_random"}:
            raise ValueError(f"unsupported base_init_mode={base_init_mode}")
        if token_init_mode not in {"identity", "full_random", "partial_random"}:
            raise ValueError(f"unsupported token_init_mode={token_init_mode}")

        self.n = n
        self.device = device
        self.token_mat_mode = token_mat_mode
        self.base_init_mode = base_init_mode
        self.token_init_mode = token_init_mode
        self.vocab = VOCAB
        self.output_vocab = OUTPUT_VOCAB
        self.vocab_size = len(self.vocab)
        self.output_vocab_size = len(self.output_vocab)
        self.past_pairs = n // 2
        self.token_subspace_target_norm = (self.vocab_size / self.n) ** 0.5
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

        self.query = torch.zeros(n, device=device)
        self.query[0] = 1.0
        self.past_freqs = torch.rand(self.past_pairs, device=device) * TAU

        self.base_mat = initialize_rotation_like((n, n), device, base_init_mode)
        self.left_token_mats = initialize_rotation_like((self.vocab_size, n, n), device, token_init_mode)
        self.right_token_mats = initialize_rotation_like((self.vocab_size, n, n), device, token_init_mode)

    def eye(self) -> torch.Tensor:
        return torch.eye(self.n, device=self.device)

    def base_matrix(self) -> torch.Tensor:
        return self.base_mat

    def uses_left_token_mats(self) -> bool:
        return self.token_mat_mode in {"left", "both"}

    def uses_right_token_mats(self) -> bool:
        return self.token_mat_mode in {"right", "both"}

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]

    def prefix_state_from_query(self, token_ids: Sequence[int], query: torch.Tensor) -> torch.Tensor:
        v = query
        if self.uses_right_token_mats():
            for tid in reversed(token_ids):
                v = self.right_token_mats[tid] @ v
        v = self.base_matrix() @ v
        if self.uses_left_token_mats():
            for tid in token_ids:
                v = self.left_token_mats[tid] @ v
        return v

    def prefix_state_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        return self.prefix_state_from_query(token_ids, self.query)

    def predict_next_id(self, token_ids: Sequence[int]) -> int:
        state = self.prefix_state_ids(token_ids)
        return int(state[: self.output_vocab_size].argmax().item())

    def predict_next(self, prefix: str) -> str:
        return self.itos[self.predict_next_id(self.encode(prefix))]

    def _encode_past_angles(self, angles: torch.Tensor) -> torch.Tensor:
        out_shape = (self.n,) + tuple(angles.shape[1:])
        out = torch.zeros(out_shape, device=self.device)
        if self.past_pairs == 0:
            return out
        scale = 1.0 / (self.past_pairs ** 0.5)
        out[0 : 2 * self.past_pairs : 2] = torch.sin(angles) * scale
        out[1 : 2 * self.past_pairs : 2] = torch.cos(angles) * scale
        return out

    def past_query(self, lag: int | torch.Tensor) -> torch.Tensor:
        lag_t = torch.as_tensor(lag, device=self.device, dtype=self.past_freqs.dtype)
        if lag_t.ndim == 0:
            return self._encode_past_angles(self.past_freqs * lag_t)
        return self._encode_past_angles(self.past_freqs.unsqueeze(1) * lag_t.unsqueeze(0))

    def token_subspace_target(self, current: torch.Tensor, token_id: int | torch.Tensor) -> torch.Tensor:
        token_t = torch.as_tensor(token_id, device=self.device, dtype=torch.long)
        complement_target_norm = max(1.0 - self.token_subspace_target_norm**2, 0.0) ** 0.5
        if current.ndim == 1:
            target = current.clone()
            rest = current[self.vocab_size :]
            rest_norm = rest.norm()
            if float(rest_norm.item()) > EPS:
                target[self.vocab_size :] = rest * (complement_target_norm / float(rest_norm.item()))
            else:
                target[self.vocab_size :] = 0.0
            target[: self.vocab_size] = 0.0
            target[int(token_t.item())] = self.token_subspace_target_norm
            return target

        target = current.clone()
        rest = current[self.vocab_size :]
        rest_norm = rest.norm(dim=0)
        rest_scale = torch.where(
            rest_norm > EPS,
            torch.full_like(rest_norm, complement_target_norm) / rest_norm,
            torch.zeros_like(rest_norm),
        )
        target[self.vocab_size :] = rest * rest_scale.unsqueeze(0)
        target[: self.vocab_size] = 0.0
        cols = torch.arange(token_t.numel(), device=self.device)
        target[token_t, cols] = self.token_subspace_target_norm
        return target

    def state_dict(self) -> Dict[str, torch.Tensor | str | int]:
        out: Dict[str, torch.Tensor | str | int] = {
            "n": self.n,
            "vocab": self.vocab,
            "output_vocab": self.output_vocab,
            "token_mat_mode": self.token_mat_mode,
            "base_init_mode": self.base_init_mode,
            "token_init_mode": self.token_init_mode,
            "left_token_mats": self.left_token_mats,
            "right_token_mats": self.right_token_mats,
            "past_freqs": self.past_freqs,
            "base_mat": self.base_mat,
        }
        return out

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device) -> Tuple["ManualRotationMatrixNetwork", int | None]:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        n = int(ckpt["n"])
        token_mat_mode = str(ckpt.get("token_mat_mode", "right"))
        base_init_mode = str(ckpt.get("base_init_mode", "identity"))
        token_init_mode = str(ckpt.get("token_init_mode", "identity"))
        model = cls(
            n=n,
            device=device,
            token_mat_mode=token_mat_mode,
            base_init_mode=base_init_mode,
            token_init_mode=token_init_mode,
        )
        if "left_token_mats" in ckpt:
            model.left_token_mats = ckpt["left_token_mats"].to(device)
        if "right_token_mats" in ckpt:
            model.right_token_mats = ckpt["right_token_mats"].to(device)
        if "token_mats" in ckpt:
            model.right_token_mats = ckpt["token_mats"].to(device)
        model.past_freqs = ckpt["past_freqs"].to(device)
        model.base_mat = ckpt["base_mat"].to(device)
        addend_digits = ckpt.get("addend_digits")
        if addend_digits is not None:
            addend_digits = int(addend_digits)
        return model, addend_digits


def save_checkpoint(model: ManualRotationMatrixNetwork, save_path: str, addend_digits: int) -> None:
    path = Path(save_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **model.state_dict(),
            "addend_digits": addend_digits,
        },
        path,
    )


def format_float_token(x: float) -> str:
    return format(x, ".6g").replace("-", "m").replace(".", "p")


def default_save_path(args: argparse.Namespace, addend_digits: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = (
        f"manual_rotation_n{args.n}"
        f"_d{addend_digits}"
        f"_mode{args.token_mat_mode}"
        f"_binit{args.base_init_mode}"
        f"_tinit{args.token_init_mode}"
        f"_it{args.iters}"
        f"_bs{args.batch_size}"
        f"_lr{format_float_token(args.learning_rate)}"
        f"_ctx{format_float_token(args.context_length_power)}"
        f"_mem{format_float_token(args.memory_weight)}"
        f"_seed{args.seed}"
        f"_{timestamp}.pt"
    )
    return str(Path("checkpoints") / name)


def generate_until_eos(model: ManualRotationMatrixNetwork, prompt_text: str, max_len: int) -> Tuple[str, bool]:
    pred = ""
    for _ in range(max_len):
        next_ch = model.predict_next(prompt_text + pred)
        if next_ch == EOS_TOKEN:
            return pred, True
        pred += next_ch
    return pred, False


def evaluate(
    model: ManualRotationMatrixNetwork,
    eval_samples: int,
    seed: int,
    max_gen_len: int,
    addend_digits: int,
) -> Tuple[float, float, float, float, float]:
    rng = random.Random(seed)
    exact = 0
    tf_correct = 0
    tf_total = 0
    stopped = 0
    memory_correct = 0
    memory_total = 0
    memory_subspace_norm_sum = 0.0

    for _ in range(eval_samples):
        _, _, prompt_text, target_text = random_problem(rng, addend_digits)
        pred, did_stop = generate_until_eos(model, prompt_text, max_gen_len)
        stopped += int(did_stop)
        if did_stop and pred == target_text:
            exact += 1

        target_seq = target_text + EOS_TOKEN
        for i, ch in enumerate(target_seq):
            pred_id = model.predict_next_id(model.encode(prompt_text + target_seq[:i]))
            tf_correct += int(pred_id == model.stoi[ch])
            tf_total += 1

        full_seq = prompt_text + target_text + EOS_TOKEN
        for prefix_len in range(1, len(full_seq) + 1):
            prefix_ids = model.encode(full_seq[:prefix_len])
            for lag in range(1, prefix_len + 1):
                memory_vec = model.prefix_state_from_query(prefix_ids, model.past_query(lag))
                pred_id = int(memory_vec[: model.vocab_size].argmax().item())
                target_id = prefix_ids[-lag]
                memory_correct += int(pred_id == target_id)
                memory_total += 1
                memory_subspace_norm_sum += float(memory_vec[: model.vocab_size].norm().item())

    return (
        exact / max(eval_samples, 1),
        tf_correct / max(tf_total, 1),
        stopped / max(eval_samples, 1),
        memory_correct / max(memory_total, 1),
        memory_subspace_norm_sum / max(memory_total, 1),
    )


def show_samples(model: ManualRotationMatrixNetwork, seed: int, max_gen_len: int, addend_digits: int, count: int = 10) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        left_addend, right_addend, prompt_text, target_text = random_problem(rng, addend_digits)
        pred, did_stop = generate_until_eos(model, prompt_text, max_gen_len)
        ok = "OK" if (did_stop and pred == target_text) else "XX"
        stop_txt = "eos" if did_stop else "max"
        print(f"{prompt_text}{target_text:>4s} | pred={pred:>4s} ({stop_txt}) [{ok}]   ({left_addend}+{right_addend})")


@torch.no_grad()
def apply_batch_update(
    model: ManualRotationMatrixNetwork,
    prefixes: Sequence[Sequence[int]],
    target_ids: Sequence[Optional[int]],
    learning_rate: float,
    context_length_power: float,
    memory_weight: float,
) -> Tuple[float, float]:
    n = model.n
    base_delta = torch.zeros(n, n, device=model.device)
    left_token_delta = torch.zeros(model.vocab_size, n, n, device=model.device)
    right_token_delta = torch.zeros(model.vocab_size, n, n, device=model.device)
    correct = 0
    total = 0
    total_objective_columns = 0
    mean_target_score = 0.0
    mean_target_subspace_norm = 0.0
    target_subspace_count = 0

    base = model.base_matrix()
    for prefix_ids, target_id in zip(prefixes, target_ids):
        context_scale = float(len(prefix_ids) ** (-context_length_power)) if len(prefix_ids) > 0 else 1.0

        query_cols: List[torch.Tensor] = []
        target_token_cols: List[torch.Tensor] = []
        weight_cols: List[torch.Tensor] = []

        if target_id is not None:
            query_cols.append(model.query.unsqueeze(1))
            target_token_cols.append(torch.tensor([target_id], device=model.device, dtype=torch.long))
            weight_cols.append(torch.tensor([context_scale], device=model.device, dtype=base.dtype))

        if memory_weight > 0.0 and prefix_ids:
            memory_terms = len(prefix_ids)
            lags = torch.arange(1, memory_terms + 1, device=model.device, dtype=base.dtype)
            token_history = torch.tensor(prefix_ids[::-1], device=model.device, dtype=torch.long)
            memory_source_positions = torch.arange(memory_terms - 1, -1, -1, device=model.device, dtype=torch.long)
            memory_scale = context_scale * memory_weight
            query_cols.append(model.past_query(lags))
            target_token_cols.append(token_history)
            weight_cols.append(torch.full((memory_terms,), memory_scale, device=model.device, dtype=base.dtype))
        else:
            memory_source_positions = None

        if not query_cols:
            continue

        query_mat = torch.cat(query_cols, dim=1)
        weights = torch.cat(weight_cols, dim=0)
        target_token_ids = torch.cat(target_token_cols, dim=0)
        next_col_count = 1 if target_id is not None else 0
        total_objective_columns += int(query_mat.shape[1])

        total_col_count = int(query_mat.shape[1])

        def active_columns(idx: int) -> torch.Tensor:
            if memory_source_positions is None:
                return torch.ones(total_col_count, dtype=torch.bool, device=model.device)
            return torch.cat(
                [
                    torch.ones(next_col_count, dtype=torch.bool, device=model.device),
                    memory_source_positions <= idx,
                ]
            )

        if model.uses_right_token_mats():
            right_inputs: List[torch.Tensor] = [torch.empty(0, device=model.device) for _ in prefix_ids]
            right_state = query_mat
            for idx in range(len(prefix_ids) - 1, -1, -1):
                right_inputs[idx] = right_state
                right_state = model.right_token_mats[prefix_ids[idx]] @ right_state
        else:
            right_inputs = []
            right_state = query_mat

        middle_state = base @ right_state
        if model.uses_left_token_mats():
            left_inputs: List[torch.Tensor] = [torch.empty(0, device=model.device) for _ in prefix_ids]
            state_mat = middle_state
            for idx, tid in enumerate(prefix_ids):
                left_inputs[idx] = state_mat
                state_mat = model.left_token_mats[tid] @ state_mat
        else:
            left_inputs = []
            state_mat = middle_state

        subspace_norms = state_mat[: model.vocab_size].norm(dim=0)
        target_mat = model.token_subspace_target(state_mat, target_token_ids)
        mean_target_subspace_norm += float(subspace_norms.sum().item())
        target_subspace_count += int(subspace_norms.numel())
        if target_id is not None:
            state = state_mat[:, 0]
            correct += int(int(state[: model.output_vocab_size].argmax().item()) == target_id)
            total += 1
            mean_target_score += float(state[target_id].item())

        left_target = target_mat
        if model.uses_left_token_mats():
            for idx in range(len(prefix_ids) - 1, -1, -1):
                tid = prefix_ids[idx]
                active_cols = active_columns(idx)
                u = normalize_columns(model.left_token_mats[tid] @ left_inputs[idx][:, active_cols])
                v = normalize_columns(left_target[:, active_cols])
                left_token_delta[tid].add_(manual_rotation_delta(u, v, weights[active_cols]))
                left_target = model.left_token_mats[tid].transpose(-1, -2) @ left_target

        # Base sits before the whole prefix, so memory targets for tokens inside the
        # prefix should not push it to "remember" future tokens. Only next-token
        # prediction updates the base matrix.
        if next_col_count > 0:
            u_base = normalize_columns(middle_state[:, :next_col_count])
            v_base = normalize_columns(left_target[:, :next_col_count])
            base_delta.add_(manual_rotation_delta(u_base, v_base, weights[:next_col_count]))

        if model.uses_right_token_mats():
            right_target = base.transpose(-1, -2) @ left_target
            for idx, tid in enumerate(prefix_ids):
                active_cols = active_columns(idx)
                u = normalize_columns(model.right_token_mats[tid] @ right_inputs[idx][:, active_cols])
                v = normalize_columns(right_target[:, active_cols])
                right_token_delta[tid].add_(manual_rotation_delta(u, v, weights[active_cols]))
                right_target = model.right_token_mats[tid].transpose(-1, -2) @ right_target

    eye = model.eye()
    objective_scale = 1.0 / float(max(total_objective_columns, 1))
    if total > 0:
        updated = (eye + learning_rate * (base_delta * objective_scale)) @ model.base_mat
        model.base_mat = orthogonalize_newton_schulz(updated)

    left_active = left_token_delta.abs().amax(dim=(1, 2)) > 0
    if left_active.any():
        updated = (eye.unsqueeze(0) + learning_rate * (left_token_delta[left_active] * objective_scale)) @ model.left_token_mats[left_active]
        model.left_token_mats[left_active] = orthogonalize_newton_schulz(updated)

    right_active = right_token_delta.abs().amax(dim=(1, 2)) > 0
    if right_active.any():
        updated = (eye.unsqueeze(0) + learning_rate * (right_token_delta[right_active] * objective_scale)) @ model.right_token_mats[right_active]
        model.right_token_mats[right_active] = orthogonalize_newton_schulz(updated)

    return (
        mean_target_score / max(total, 1),
        correct / max(total, 1),
        mean_target_subspace_norm / max(target_subspace_count, 1),
    )


def train(
    *,
    model: ManualRotationMatrixNetwork,
    iters: int,
    batch_size: int,
    learning_rate: float,
    addend_digits: int,
    seed: int,
    log_every: int,
    eval_every: int,
    eval_samples: int,
    max_gen_len: int,
    context_length_power: float,
    memory_weight: float,
) -> ManualRotationMatrixNetwork:
    rng = random.Random(seed)

    def sample_batch() -> Tuple[List[List[int]], List[Optional[int]]]:
        prefixes: List[List[int]] = []
        target_ids: List[Optional[int]] = []

        for _ in range(batch_size):
            _, _, prompt_text, target_text = random_problem(rng, addend_digits)
            full_seq = prompt_text + target_text + EOS_TOKEN
            prompt_len = len(prompt_text)
            for prefix_len in range(1, len(full_seq) + 1):
                prefixes.append(model.encode(full_seq[:prefix_len]))
                if prefix_len >= prompt_len and prefix_len < len(full_seq):
                    target_ids.append(model.stoi[full_seq[prefix_len]])
                else:
                    target_ids.append(None)

        return prefixes, target_ids

    for iter_idx in range(1, iters + 1):
        prefixes, target_ids = sample_batch()
        mean_target_score, token_acc, mean_target_subspace_norm = apply_batch_update(
            model,
            prefixes=prefixes,
            target_ids=target_ids,
            learning_rate=learning_rate,
            context_length_power=context_length_power,
            memory_weight=memory_weight,
        )

        if iter_idx % log_every == 0 or iter_idx == 1:
            print(
                f"iter={iter_idx:5d} mean_target_score={mean_target_score:.4f} "
                f"token_acc={token_acc:.3f} mean_target_subspace_norm={mean_target_subspace_norm:.4f}"
            )

        if iter_idx % eval_every == 0 or iter_idx == iters:
            exact, tf_acc, stop_rate, memory_acc, memory_subspace_norm = evaluate(
                model,
                eval_samples=eval_samples,
                seed=seed + iter_idx,
                max_gen_len=max_gen_len,
                addend_digits=addend_digits,
            )
            print(
                f"  eval exact_match={exact:.3f} teacher_forced_token_acc={tf_acc:.3f} "
                f"stop_rate={stop_rate:.3f} memory_token_acc={memory_acc:.3f} "
                f"memory_subspace_norm={memory_subspace_norm:.3f}"
            )

    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense matrix network with manual rotation updates and fixed query/unembedding")
    p.add_argument("--n", type=int, required=True, help="Square matrix dimension; must be >= vocab size")
    p.add_argument("--token-mat-mode", type=str, default="right", choices=["left", "right", "both"], help="Apply learned token matrices on the left of base, right of base, or both")
    p.add_argument("--base-init-mode", type=str, default="identity", choices=["identity", "full_random", "partial_random"], help="Initialization for the base matrix")
    p.add_argument("--token-init-mode", type=str, default="identity", choices=["identity", "full_random", "partial_random"], help="Initialization for per-token matrices")
    p.add_argument("--iters", type=int, default=1500, help="Training iterations")
    p.add_argument("--batch-size", type=int, default=32, help="Problems per iteration")
    p.add_argument("--learning-rate", type=float, default=0.01, help="Manual rotation step size")
    p.add_argument("--addend-digits", type=int, default=3, help="Digits for each addend in a+b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--context-length-power", type=float, default=0.0, help="Scale each prediction context by len(prefix)**(-power); 1.0 gives each context a fixed total update budget")
    p.add_argument("--memory-weight", type=float, default=0.0, help="Weight of the auxiliary memory objective for each remembered token")
    p.add_argument("--load-path", type=str, default=None, help="Optional checkpoint to continue training from")
    p.add_argument("--save-path", type=str, default=None, help="Optional checkpoint path override")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be > 0")
    if args.context_length_power < 0.0:
        raise ValueError("--context-length-power must be >= 0")
    if args.memory_weight < 0.0:
        raise ValueError("--memory-weight must be >= 0")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)
    print(f"device={device}")
    print(f"iters={args.iters} learning_rate={args.learning_rate}")
    print(f"base_mode=learned token_mat_mode={args.token_mat_mode} addend_digits={args.addend_digits}")
    print(
        f"base_init_mode={args.base_init_mode} token_init_mode={args.token_init_mode} "
        f"orthogonalize=one_step_per_update"
    )
    print(f"context_length_power={args.context_length_power}")
    print(f"memory_scope=all_prefixes memory_weight={args.memory_weight}")

    if args.load_path is None:
        model = ManualRotationMatrixNetwork(
            n=args.n,
            device=device,
            token_mat_mode=args.token_mat_mode,
            base_init_mode=args.base_init_mode,
            token_init_mode=args.token_init_mode,
        )
        addend_digits = args.addend_digits
    else:
        model, loaded_addend_digits = ManualRotationMatrixNetwork.from_checkpoint(args.load_path, device)
        if model.n != args.n:
            print(f"loaded_n={model.n}; overriding --n={args.n}")
        if model.token_mat_mode != args.token_mat_mode:
            print(f"loaded_token_mat_mode={model.token_mat_mode}; ignoring --token-mat-mode={args.token_mat_mode}")
        if model.base_init_mode != args.base_init_mode:
            print(f"loaded_base_init_mode={model.base_init_mode}; ignoring --base-init-mode={args.base_init_mode}")
        if model.token_init_mode != args.token_init_mode:
            print(f"loaded_token_init_mode={model.token_init_mode}; ignoring --token-init-mode={args.token_init_mode}")
        addend_digits = int(loaded_addend_digits or args.addend_digits)
        if loaded_addend_digits is not None and loaded_addend_digits != args.addend_digits:
            print(f"loaded_addend_digits={loaded_addend_digits}; overriding --addend-digits={args.addend_digits}")

    max_gen_len = addend_digits + 2
    save_path = args.save_path or default_save_path(args, addend_digits)
    print(f"max_gen_len={max_gen_len}")
    print(f"output_vocab={OUTPUT_VOCAB}")
    print(f"save_path={save_path}")

    model = train(
        model=model,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        addend_digits=addend_digits,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        max_gen_len=max_gen_len,
        context_length_power=args.context_length_power,
        memory_weight=args.memory_weight,
    )

    save_checkpoint(model, save_path, addend_digits=addend_digits)
    print(f"saved_checkpoint={save_path}")
    print("\nSample predictions:")
    show_samples(model, seed=args.seed + 999, max_gen_len=max_gen_len, addend_digits=addend_digits)


if __name__ == "__main__":
    main()
