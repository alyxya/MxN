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


def random_problem(rng: random.Random, addend_digits: int) -> Problem:
    max_val = (10**addend_digits) - 1
    a = rng.randint(0, max_val)
    b = rng.randint(0, max_val)
    lhs = f"{a:0{addend_digits}d}+{b:0{addend_digits}d}="
    rhs = str(a + b)
    return a, b, lhs, rhs


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
    ):
        if n < len(VOCAB):
            raise ValueError(f"n must be >= {len(VOCAB)} to fit fixed one-hot heads, got {n}")

        self.n = n
        self.device = device
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

        self.base_mat = torch.eye(n, device=device)
        self.token_mats = torch.eye(n, device=device).expand(self.vocab_size, n, n).clone()

    def eye(self) -> torch.Tensor:
        return torch.eye(self.n, device=self.device)

    def base_matrix(self) -> torch.Tensor:
        return self.base_mat

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]

    def prefix_state_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        v = self.query
        for tid in reversed(token_ids):
            v = self.token_mats[tid] @ v
        return self.base_matrix() @ v

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
            "token_mats": self.token_mats,
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
        model = cls(n=n, device=device)
        model.token_mats = ckpt["token_mats"].to(device)
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
        f"_it{args.iters}"
        f"_bs{args.batch_size}"
        f"_lr{format_float_token(args.learning_rate)}"
        f"_ctx{format_float_token(args.context_length_power)}"
        f"_mem{format_float_token(args.memory_weight)}"
        f"_seed{args.seed}"
        f"_{timestamp}.pt"
    )
    return str(Path("checkpoints") / name)


def generate_until_eos(model: ManualRotationMatrixNetwork, lhs: str, max_len: int) -> Tuple[str, bool]:
    pred = ""
    for _ in range(max_len):
        next_ch = model.predict_next(lhs + pred)
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
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    exact = 0
    tf_correct = 0
    tf_total = 0
    stopped = 0

    for _ in range(eval_samples):
        _, _, lhs, rhs = random_problem(rng, addend_digits)
        pred, did_stop = generate_until_eos(model, lhs, max_gen_len)
        stopped += int(did_stop)
        if did_stop and pred == rhs:
            exact += 1

        target_seq = rhs + EOS_TOKEN
        for i, ch in enumerate(target_seq):
            pred_id = model.predict_next_id(model.encode(lhs + target_seq[:i]))
            tf_correct += int(pred_id == model.stoi[ch])
            tf_total += 1

    return exact / max(eval_samples, 1), tf_correct / max(tf_total, 1), stopped / max(eval_samples, 1)


def show_samples(model: ManualRotationMatrixNetwork, seed: int, max_gen_len: int, addend_digits: int, count: int = 10) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        a, b, lhs, rhs = random_problem(rng, addend_digits)
        pred, did_stop = generate_until_eos(model, lhs, max_gen_len)
        ok = "OK" if (did_stop and pred == rhs) else "XX"
        stop_txt = "eos" if did_stop else "max"
        print(f"{lhs}{rhs:>4s} | pred={pred:>4s} ({stop_txt}) [{ok}]   ({a}+{b})")


@torch.no_grad()
def apply_batch_update(
    model: ManualRotationMatrixNetwork,
    prefixes: Sequence[Sequence[int]],
    target_ids: Sequence[Optional[int]],
    objective_count: int,
    learning_rate: float,
    context_length_power: float,
    memory_weight: float,
    causal_memory: bool,
) -> Tuple[float, float]:
    n = model.n
    base_delta = torch.zeros(n, n, device=model.device)
    token_delta = torch.zeros(model.vocab_size, n, n, device=model.device)
    correct = 0
    total = 0
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

        suffix_inputs: List[torch.Tensor] = [torch.empty(0, device=model.device) for _ in prefix_ids]
        suffix = query_mat
        for idx in range(len(prefix_ids) - 1, -1, -1):
            suffix_inputs[idx] = suffix
            suffix = model.token_mats[prefix_ids[idx]] @ suffix

        state_mat = base @ suffix
        subspace_norms = state_mat[: model.vocab_size].norm(dim=0)
        target_mat = model.token_subspace_target(state_mat, target_token_ids)
        mean_target_subspace_norm += float(subspace_norms.sum().item())
        target_subspace_count += int(subspace_norms.numel())
        if target_id is not None:
            state = state_mat[:, 0]
            correct += int(int(state[: model.output_vocab_size].argmax().item()) == target_id)
            total += 1
            mean_target_score += float(state[target_id].item())

        if causal_memory:
            # Base sits before the whole prefix, so memory targets for tokens inside the
            # prefix should not push it to "remember" future tokens. Only next-token
            # prediction updates the base matrix.
            if next_col_count > 0:
                u_base = normalize_columns(state_mat[:, :next_col_count])
                base_delta.add_(manual_rotation_delta(u_base, target_mat[:, :next_col_count], weights[:next_col_count]))
        else:
            u_base = normalize_columns(state_mat)
            base_delta.add_(manual_rotation_delta(u_base, target_mat, weights))

        left_target = base.transpose(-1, -2) @ target_mat
        for idx, tid in enumerate(prefix_ids):
            x_in = suffix_inputs[idx]
            if not causal_memory:
                active_cols = slice(None)
            elif memory_source_positions is None:
                active_cols = slice(0, next_col_count)
            else:
                active_memory = memory_source_positions <= idx
                active_cols = torch.cat(
                    [
                        torch.ones(next_col_count, dtype=torch.bool, device=model.device),
                        active_memory,
                    ]
                )
            u = normalize_columns(model.token_mats[tid] @ x_in[:, active_cols])
            v = normalize_columns(left_target[:, active_cols])
            token_delta[tid].add_(manual_rotation_delta(u, v, weights[active_cols]))
            left_target = model.token_mats[tid].transpose(-1, -2) @ left_target

    eye = model.eye()
    objective_scale = 1.0 / float(max(objective_count, 1))
    if total > 0:
        updated = (eye + learning_rate * (base_delta * objective_scale)) @ model.base_mat
        model.base_mat = orthogonalize_newton_schulz(updated)

    active = token_delta.abs().amax(dim=(1, 2)) > 0
    if active.any():
        updated = (eye.unsqueeze(0) + learning_rate * (token_delta[active] * objective_scale)) @ model.token_mats[active]
        model.token_mats[active] = orthogonalize_newton_schulz(updated)

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
    causal_memory: bool,
) -> ManualRotationMatrixNetwork:
    rng = random.Random(seed)

    def sample_batch() -> Tuple[List[List[int]], List[Optional[int]]]:
        prefixes: List[List[int]] = []
        target_ids: List[Optional[int]] = []

        for _ in range(batch_size):
            _, _, lhs, rhs = random_problem(rng, addend_digits)
            target_seq = rhs + EOS_TOKEN
            for i, ch in enumerate(target_seq):
                prefixes.append(model.encode(lhs + target_seq[:i]))
                target_ids.append(model.stoi[ch])

        return prefixes, target_ids

    for iter_idx in range(1, iters + 1):
        prefixes, target_ids = sample_batch()
        mean_target_score, token_acc, mean_target_subspace_norm = apply_batch_update(
            model,
            prefixes=prefixes,
            target_ids=target_ids,
            objective_count=len(prefixes),
            learning_rate=learning_rate,
            context_length_power=context_length_power,
            memory_weight=memory_weight,
            causal_memory=causal_memory,
        )

        if iter_idx % log_every == 0 or iter_idx == 1:
            print(
                f"iter={iter_idx:5d} mean_target_score={mean_target_score:.4f} "
                f"token_acc={token_acc:.3f} mean_target_subspace_norm={mean_target_subspace_norm:.4f}"
            )

        if iter_idx % eval_every == 0 or iter_idx == iters:
            exact, tf_acc, stop_rate = evaluate(
                model,
                eval_samples=eval_samples,
                seed=seed + iter_idx,
                max_gen_len=max_gen_len,
                addend_digits=addend_digits,
            )
            print(f"  eval exact_match={exact:.3f} teacher_forced_token_acc={tf_acc:.3f} stop_rate={stop_rate:.3f}")

    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense matrix network with manual rotation updates and fixed query/unembedding")
    p.add_argument("--n", type=int, required=True, help="Square matrix dimension; must be >= vocab size")
    p.add_argument("--iters", type=int, default=1500, help="Training iterations")
    p.add_argument("--batch-size", type=int, default=32, help="Problems per iteration")
    p.add_argument("--learning-rate", type=float, default=0.01, help="Manual rotation step size")
    p.add_argument("--addend-digits", type=int, default=3, help="Digits for each addend in a+b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--context-length-power", type=float, default=0.0, help="Scale each prediction context by len(prefix)**(-power); 1.0 gives each context a fixed total update budget")
    p.add_argument("--memory-weight", type=float, default=0.0, help="Total weight of the auxiliary memory objective per prefix, distributed across active memory lags")
    p.add_argument("--causal-memory", action="store_true", help="Only let memory targets update the remembered token's matrix and later matrices")
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
    print(f"base_mode=learned addend_digits={args.addend_digits}")
    print("init_mode=identity orthogonalize=one_step_per_update")
    print(f"context_length_power={args.context_length_power}")
    print(f"memory_scope=output_prefixes memory_weight={args.memory_weight}")
    print(f"causal_memory={args.causal_memory}")

    if args.load_path is None:
        model = ManualRotationMatrixNetwork(
            n=args.n,
            device=device,
        )
        addend_digits = args.addend_digits
    else:
        model, loaded_addend_digits = ManualRotationMatrixNetwork.from_checkpoint(args.load_path, device)
        if model.n != args.n:
            print(f"loaded_n={model.n}; overriding --n={args.n}")
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
        causal_memory=args.causal_memory,
    )

    save_checkpoint(model, save_path, addend_digits=addend_digits)
    print(f"saved_checkpoint={save_path}")
    print("\nSample predictions:")
    show_samples(model, seed=args.seed + 999, max_gen_len=max_gen_len, addend_digits=addend_digits)


if __name__ == "__main__":
    main()
