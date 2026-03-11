#!/usr/bin/env python3
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


EOS_TOKEN = "~"
VOCAB = f"0123456789+={EOS_TOKEN}"
EPS = 1e-12
Problem = Tuple[int, int, str, str]
TokenMatrixTrace = List[Tuple[int, torch.Tensor]]
BASE_MODES = ("learned", "identity_fixed")
TOKEN_MODES = ("dense", "lowrank_ab", "subspace_rot")


def normalize_last_dim(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def normalize_dim(x: torch.Tensor, dim: int, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def infer_token_rank_from_state_dict(state_dict: Dict[str, torch.Tensor], token_mode: str, n: int) -> int | None:
    if token_mode == "dense":
        token_mats = state_dict.get("token_mats")
        if token_mats is None:
            raise ValueError("Dense checkpoint missing token_mats")
        if token_mats.ndim != 3 or token_mats.shape[1:] != (n, n):
            raise ValueError(f"token_mats shape mismatch for n={n}: got {tuple(token_mats.shape)}")
        return None

    if token_mode == "lowrank_ab":
        token_a = state_dict.get("token_a")
        token_b = state_dict.get("token_b")
        if token_a is None or token_b is None:
            raise ValueError("Low-rank AB checkpoint must include token_a and token_b")
        if token_a.ndim != 3 or token_b.ndim != 3:
            raise ValueError("token_a/token_b must be rank-3 tensors")
        if token_a.shape[2] != n or token_b.shape[2] != n:
            raise ValueError(f"token_a/token_b shape mismatch for n={n}: {tuple(token_a.shape)} vs {tuple(token_b.shape)}")
        if token_a.shape[0] != token_b.shape[0] or token_a.shape[1] != token_b.shape[1]:
            raise ValueError(f"token_a/token_b rank mismatch: {tuple(token_a.shape)} vs {tuple(token_b.shape)}")
        return int(token_a.shape[1])

    if token_mode == "subspace_rot":
        token_u = state_dict.get("token_u")
        token_r = state_dict.get("token_r")
        if token_u is None or token_r is None:
            raise ValueError("Subspace-rotation checkpoint must include token_u and token_r")
        if token_u.ndim != 3 or token_r.ndim != 3:
            raise ValueError("token_u/token_r must be rank-3 tensors")
        if token_u.shape[1] != n:
            raise ValueError(f"token_u shape mismatch for n={n}: got {tuple(token_u.shape)}")
        if token_r.shape[1] != token_r.shape[2] or token_u.shape[2] != token_r.shape[1]:
            raise ValueError(f"token_u/token_r rank mismatch: {tuple(token_u.shape)} vs {tuple(token_r.shape)}")
        if token_u.shape[0] != token_r.shape[0]:
            raise ValueError(f"token_u/token_r vocab mismatch: {tuple(token_u.shape)} vs {tuple(token_r.shape)}")
        return int(token_u.shape[2])

    raise ValueError(f"Unsupported token_mode={token_mode!r}")


class MatrixNetwork(torch.nn.Module):
    def __init__(
        self,
        n: int,
        device: torch.device,
        *,
        base_mode: str = "learned",
        token_mode: str = "dense",
        token_rank: int | None = None,
    ):
        super().__init__()
        if base_mode not in BASE_MODES:
            raise ValueError(f"Unsupported base_mode={base_mode!r}; expected one of {BASE_MODES}")
        if token_mode not in TOKEN_MODES:
            raise ValueError(f"Unsupported token_mode={token_mode!r}; expected one of {TOKEN_MODES}")

        self.n = n
        self.base_mode = base_mode
        self.token_mode = token_mode
        self.learn_base_mat = base_mode == "learned"
        if token_mode == "dense":
            if token_rank is not None:
                raise ValueError("Dense token mode does not use --token-rank")
        else:
            if token_rank is None:
                token_rank = max(1, n // 2)
            if token_rank < 1:
                raise ValueError(f"token_rank must be >= 1, got {token_rank}")
        self.token_rank = None if token_mode == "dense" else int(token_rank)
        self.vocab = VOCAB
        self.vocab_size = len(self.vocab)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

        std = (1.0 / n) ** 0.5
        if self.learn_base_mat:
            self.base_mat = torch.nn.Parameter(self.eye_n(device) + torch.randn(n, n, device=device) * 1e-4)
        else:
            self.base_mat = None

        if self.token_mode == "dense":
            self.token_mats = torch.nn.Parameter(self.eye_n(device).unsqueeze(0).repeat(self.vocab_size, 1, 1))
            self.token_a = None
            self.token_b = None
            self.token_u = None
            self.token_r = None
        elif self.token_mode == "lowrank_ab":
            k = int(self.token_rank)
            self.token_mats = None
            lowrank_std = 1.0 / ((n ** 0.5) * (k ** 0.25))
            self.token_a = torch.nn.Parameter(torch.randn(self.vocab_size, k, n, device=device) * lowrank_std)
            self.token_b = torch.nn.Parameter(torch.randn(self.vocab_size, k, n, device=device) * lowrank_std)
            self.token_u = None
            self.token_r = None
        else:
            k = int(self.token_rank)
            self.token_mats = None
            self.token_a = None
            self.token_b = None
            self.token_u = torch.nn.Parameter(torch.randn(self.vocab_size, n, k, device=device) * std)
            self.token_r = torch.nn.Parameter(
                self.eye_k(device).unsqueeze(0).repeat(self.vocab_size, 1, 1)
                + torch.randn(self.vocab_size, k, k, device=device) * 1e-4
            )

        self.decode_vecs = torch.nn.Parameter(torch.randn(self.vocab_size, n, device=device) * std)
        self.query = torch.nn.Parameter(torch.randn(n, device=device) * std)
        self.renormalize_in_place()

    def eye_n(self, device: torch.device | None = None) -> torch.Tensor:
        use_device = device if device is not None else (
            self.query.device if hasattr(self, "query") else torch.device("cpu")
        )
        use_dtype = self.query.dtype if hasattr(self, "query") else torch.get_default_dtype()
        return torch.eye(self.n, device=use_device, dtype=use_dtype)

    def eye_k(self, device: torch.device | None = None) -> torch.Tensor:
        if self.token_rank is None:
            raise ValueError("Dense token mode does not use eye_k")
        use_device = device if device is not None else (
            self.query.device if hasattr(self, "query") else torch.device("cpu")
        )
        use_dtype = self.query.dtype if hasattr(self, "query") else torch.get_default_dtype()
        return torch.eye(self.token_rank, device=use_device, dtype=use_dtype)

    def base_matrix(self) -> torch.Tensor:
        if self.base_mat is None:
            return self.eye_n()
        return self.base_mat

    @torch.no_grad()
    def renormalize_in_place(self) -> None:
        if self.base_mat is not None:
            self.base_mat.copy_(normalize_last_dim(self.base_mat))
        if self.token_mode == "dense":
            assert self.token_mats is not None
            self.token_mats.copy_(normalize_last_dim(self.token_mats))
        elif self.token_mode == "subspace_rot":
            assert self.token_u is not None and self.token_r is not None
            self.token_u.copy_(normalize_dim(self.token_u, dim=-2))
            self.token_r.copy_(normalize_last_dim(self.token_r))
        self.decode_vecs.copy_(normalize_last_dim(self.decode_vecs))
        self.query.copy_(normalize_last_dim(self.query))

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]

    def token_matrix(self, tid: int) -> torch.Tensor:
        if self.token_mode == "dense":
            assert self.token_mats is not None
            return self.token_mats[tid]
        if self.token_mode == "lowrank_ab":
            assert self.token_a is not None and self.token_b is not None
            return self.eye_n() + self.token_a[tid].transpose(-1, -2) @ self.token_b[tid]
        assert self.token_u is not None and self.token_r is not None
        return self.eye_n() + self.token_u[tid] @ (self.token_r[tid] - self.eye_k()) @ self.token_u[tid].transpose(-1, -2)

    @torch.no_grad()
    def rescale_lowrank_ab_in_place(self) -> None:
        if self.token_mode != "lowrank_ab":
            return
        assert self.token_a is not None and self.token_b is not None
        r = self.eye_n().unsqueeze(0) + self.token_a.transpose(-1, -2) @ self.token_b
        size = r.norm(dim=-1).mean(dim=-1, keepdim=True).clamp_min(EPS)
        scale = torch.sqrt(1.0 / size).unsqueeze(-1)
        self.token_a.mul_(scale)
        self.token_b.mul_(scale)

    def queried_vector_ids(self, token_ids: Sequence[int], matrix_trace: TokenMatrixTrace | None = None) -> torch.Tensor:
        p = self.base_matrix()
        for tid in token_ids:
            m = self.token_matrix(tid)
            if matrix_trace is not None:
                m.retain_grad()
                matrix_trace.append((tid, m))
            p = p @ m
        return p @ self.query

    def forward_prefix_ids(self, token_ids: Sequence[int], matrix_trace: TokenMatrixTrace | None = None) -> torch.Tensor:
        v = normalize_last_dim(self.queried_vector_ids(token_ids, matrix_trace=matrix_trace))
        return self.decode_vecs @ v

    @torch.no_grad()
    def predict_next(self, prefix: str) -> str:
        logits = self.forward_prefix_ids(self.encode(prefix))
        return self.itos[int(logits.argmax().item())]


@dataclass
class RotationalGD:
    learning_rate: float
    use_momentum: bool = False
    momentum_decay: float = 0.98
    momentum_blend_start: float = 0.0
    momentum_blend: float = 0.5
    momentum_blend_ramp_iters: int = 1000
    _iter_count: int = 0
    _hist_base: torch.Tensor | None = None
    _hist_token_1: torch.Tensor | None = None
    _hist_token_2: torch.Tensor | None = None
    _hist_d: torch.Tensor | None = None
    _hist_q: torch.Tensor | None = None

    def _current_blend(self) -> float:
        if self.momentum_blend_ramp_iters <= 0:
            return self.momentum_blend
        ramp = min(1.0, max(0.0, (self._iter_count - 1) / float(self.momentum_blend_ramp_iters)))
        return self.momentum_blend_start + (self.momentum_blend - self.momentum_blend_start) * ramp

    def _target_direction(
        self,
        grad: torch.Tensor,
        hist: torch.Tensor | None,
        blend: float,
        *,
        norm_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cur = -normalize_dim(grad, dim=norm_dim)
        if not self.use_momentum:
            return cur, hist
        if hist is None or hist.shape != cur.shape or hist.device != cur.device or hist.dtype != cur.dtype:
            hist = cur.clone()
        else:
            hist = self.momentum_decay * hist + (1.0 - self.momentum_decay) * cur
        hist_dir = normalize_dim(hist, dim=norm_dim)
        target = normalize_dim((1.0 - blend) * cur + blend * hist_dir, dim=norm_dim)
        return target, hist

    @torch.no_grad()
    def step(self, model: MatrixNetwork) -> None:
        self._iter_count += 1
        blend = self._current_blend()

        if model.base_mat is not None and model.base_mat.grad is not None:
            target_base, self._hist_base = self._target_direction(model.base_mat.grad, self._hist_base, blend, norm_dim=-1)
            model.base_mat.copy_(normalize_last_dim(model.base_mat + self.learning_rate * (target_base - model.base_mat)))

        if model.token_mode == "dense":
            assert model.token_mats is not None
            if model.token_mats.grad is not None:
                target, self._hist_token_1 = self._target_direction(model.token_mats.grad, self._hist_token_1, blend, norm_dim=-1)
                model.token_mats.copy_(normalize_last_dim(model.token_mats + self.learning_rate * (target - model.token_mats)))
        elif model.token_mode == "lowrank_ab":
            assert model.token_a is not None and model.token_b is not None
            if model.token_a.grad is not None:
                model.token_a.add_(-self.learning_rate * model.token_a.grad)
            if model.token_b.grad is not None:
                model.token_b.add_(-self.learning_rate * model.token_b.grad)
            model.rescale_lowrank_ab_in_place()
        else:
            assert model.token_u is not None and model.token_r is not None
            if model.token_u.grad is not None:
                target_u, self._hist_token_1 = self._target_direction(model.token_u.grad, self._hist_token_1, blend, norm_dim=-2)
                model.token_u.copy_(normalize_dim(model.token_u + self.learning_rate * (target_u - model.token_u), dim=-2))
            if model.token_r.grad is not None:
                target_r, self._hist_token_2 = self._target_direction(model.token_r.grad, self._hist_token_2, blend, norm_dim=-1)
                model.token_r.copy_(normalize_last_dim(model.token_r + self.learning_rate * (target_r - model.token_r)))

        if model.decode_vecs.grad is not None:
            target_d, self._hist_d = self._target_direction(model.decode_vecs.grad, self._hist_d, blend, norm_dim=-1)
            model.decode_vecs.copy_(normalize_last_dim(model.decode_vecs + self.learning_rate * (target_d - model.decode_vecs)))

        if model.query.grad is not None:
            target_q, self._hist_q = self._target_direction(model.query.grad, self._hist_q, blend, norm_dim=-1)
            model.query.copy_(normalize_last_dim(model.query + self.learning_rate * (target_q - model.query)))

        model.zero_grad(set_to_none=True)


def random_problem(rng: random.Random, addend_digits: int) -> Problem:
    max_val = (10**addend_digits) - 1
    a = rng.randint(0, max_val)
    b = rng.randint(0, max_val)
    lhs = f"{a:0{addend_digits}d}+{b:0{addend_digits}d}="
    rhs = str(a + b)
    return a, b, lhs, rhs


def make_fixed_dataset(seed: int, size: int, addend_digits: int) -> List[Problem]:
    rng = random.Random(seed)
    return [random_problem(rng, addend_digits) for _ in range(size)]


def generate_until_eos(model: MatrixNetwork, lhs: str, max_len: int) -> Tuple[str, bool]:
    pred = ""
    for _ in range(max_len):
        next_ch = model.predict_next(lhs + pred)
        if next_ch == EOS_TOKEN:
            return pred, True
        pred += next_ch
    return pred, False


def evaluate(
    model: MatrixNetwork,
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
        for i in range(len(target_seq)):
            next_ch = model.predict_next(lhs + target_seq[:i])
            tf_correct += int(next_ch == target_seq[i])
            tf_total += 1

    return exact / eval_samples, tf_correct / max(tf_total, 1), stopped / eval_samples


def evaluate_on_dataset(model: MatrixNetwork, dataset: Sequence[Problem], max_gen_len: int) -> Tuple[float, float, float]:
    exact = 0
    tf_correct = 0
    tf_total = 0
    stopped = 0

    for _, _, lhs, rhs in dataset:
        pred, did_stop = generate_until_eos(model, lhs, max_gen_len)
        stopped += int(did_stop)
        if did_stop and pred == rhs:
            exact += 1

        target_seq = rhs + EOS_TOKEN
        for i in range(len(target_seq)):
            next_ch = model.predict_next(lhs + target_seq[:i])
            tf_correct += int(next_ch == target_seq[i])
            tf_total += 1

    total = max(len(dataset), 1)
    return exact / total, tf_correct / max(tf_total, 1), stopped / total


def show_samples(model: MatrixNetwork, seed: int, max_gen_len: int, addend_digits: int, count: int = 10) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        a, b, lhs, rhs = random_problem(rng, addend_digits)
        pred, did_stop = generate_until_eos(model, lhs, max_gen_len)
        ok = "OK" if (did_stop and pred == rhs) else "XX"
        stop_txt = "eos" if did_stop else "max"
        print(f"{lhs}{rhs:>4s} | pred={pred:>4s} ({stop_txt}) [{ok}]   ({a}+{b})")


@torch.no_grad()
def overwrite_lowrank_ab_grads_from_matrix_trace(model: MatrixNetwork, matrix_trace: TokenMatrixTrace) -> None:
    assert model.token_mode == "lowrank_ab"
    assert model.token_a is not None and model.token_b is not None

    grad_r_by_tid: Dict[int, torch.Tensor] = {}
    for tid, mat in matrix_trace:
        if mat.grad is None:
            continue
        if tid not in grad_r_by_tid:
            grad_r_by_tid[tid] = torch.zeros_like(mat.grad)
        grad_r_by_tid[tid].add_(mat.grad)

    grad_a = torch.zeros_like(model.token_a)
    grad_b = torch.zeros_like(model.token_b)

    for tid, grad_r in grad_r_by_tid.items():
        grad_r = normalize_last_dim(grad_r)
        a = model.token_a[tid]
        b = model.token_b[tid]
        grad_a[tid].copy_(b @ grad_r.transpose(-1, -2))
        grad_b[tid].copy_(a @ grad_r)

    model.token_a.grad = grad_a
    model.token_b.grad = grad_b


def train(
    n: int,
    token_mode: str,
    token_rank: int | None,
    iters: int,
    batch_size: int,
    learning_rate: float,
    loss_temp: float,
    use_momentum: bool,
    momentum_decay: float,
    momentum_blend_start: float,
    momentum_blend: float,
    momentum_blend_ramp_iters: int,
    base_mode: str,
    addend_digits: int,
    seed: int,
    log_every: int,
    eval_every: int,
    eval_samples: int,
    max_gen_len: int,
    device: torch.device,
    model: MatrixNetwork | None = None,
    fixed_train_size: int = 0,
    fixed_train_seed: int | None = None,
    wandb_run: Any | None = None,
    wandb_log_every: int = 10,
) -> MatrixNetwork:
    rng = random.Random(seed)
    if model is None:
        model = MatrixNetwork(n=n, device=device, base_mode=base_mode, token_mode=token_mode, token_rank=token_rank)
    optim = RotationalGD(
        learning_rate=learning_rate,
        use_momentum=use_momentum,
        momentum_decay=momentum_decay,
        momentum_blend_start=momentum_blend_start,
        momentum_blend=momentum_blend,
        momentum_blend_ramp_iters=momentum_blend_ramp_iters,
    )
    fixed_train: List[Problem] | None = None
    if fixed_train_size > 0:
        fixed_seed = seed if fixed_train_seed is None else fixed_train_seed
        fixed_train = make_fixed_dataset(seed=fixed_seed, size=fixed_train_size, addend_digits=addend_digits)
        print(f"fixed_train_size={len(fixed_train)} fixed_train_seed={fixed_seed}")

    for iter_idx in range(1, iters + 1):
        model.zero_grad(set_to_none=True)
        losses: List[torch.Tensor] = []
        total_correct = 0
        total_targets = 0
        matrix_trace: TokenMatrixTrace | None = [] if model.token_mode == "lowrank_ab" else None

        for _ in range(batch_size):
            if fixed_train is None:
                _, _, lhs, rhs = random_problem(rng, addend_digits)
            else:
                _, _, lhs, rhs = fixed_train[rng.randrange(len(fixed_train))]
            target_seq = rhs + EOS_TOKEN
            for i in range(len(target_seq)):
                prefix = lhs + target_seq[:i]
                target_id = model.stoi[target_seq[i]]
                logits = model.forward_prefix_ids(model.encode(prefix), matrix_trace=matrix_trace)
                losses.append(F.cross_entropy((logits / loss_temp).unsqueeze(0), torch.tensor([target_id], device=device)))
                total_correct += int(int(logits.argmax().item()) == target_id)
                total_targets += 1

        loss = torch.stack(losses).mean()
        loss.backward()
        if matrix_trace is not None:
            overwrite_lowrank_ab_grads_from_matrix_trace(model, matrix_trace)
        optim.step(model)
        acc = total_correct / max(total_targets, 1)

        if wandb_run is not None and (iter_idx == 1 or iter_idx % wandb_log_every == 0 or iter_idx == iters):
            wandb_run.log({"train/loss": float(loss.item()), "train/token_acc": float(acc), "iter": iter_idx}, step=iter_idx)

        if iter_idx % log_every == 0 or iter_idx == 1:
            print(f"iter={iter_idx:5d} loss={loss.item():.4f} token_acc={acc:.3f}")

        if iter_idx % eval_every == 0 or iter_idx == iters:
            eval_log: Dict[str, float | int] = {"iter": iter_idx}
            if fixed_train is not None:
                tr_exact, tr_tf_acc, tr_stop_rate = evaluate_on_dataset(model, dataset=fixed_train, max_gen_len=max_gen_len)
                print(
                    f"  eval[fixed_train] exact_match={tr_exact:.3f} "
                    f"teacher_forced_token_acc={tr_tf_acc:.3f} stop_rate={tr_stop_rate:.3f}"
                )
                eval_log.update(
                    {
                        "eval_fixed/exact_match": float(tr_exact),
                        "eval_fixed/teacher_forced_token_acc": float(tr_tf_acc),
                        "eval_fixed/stop_rate": float(tr_stop_rate),
                    }
                )

            exact, tf_acc, stop_rate = evaluate(
                model,
                eval_samples=eval_samples,
                seed=seed + iter_idx,
                max_gen_len=max_gen_len,
                addend_digits=addend_digits,
            )
            print(f"  eval[random] exact_match={exact:.3f} teacher_forced_token_acc={tf_acc:.3f} stop_rate={stop_rate:.3f}")
            eval_log.update(
                {
                    "eval_random/exact_match": float(exact),
                    "eval_random/teacher_forced_token_acc": float(tf_acc),
                    "eval_random/stop_rate": float(stop_rate),
                }
            )
            if wandb_run is not None:
                wandb_run.log(eval_log, step=iter_idx)

    return model


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matrix-network toy model for addition")
    p.add_argument("--n", type=int, default=16, help="Embedding matrix/vector dimension")
    p.add_argument("--token-mode", type=str, default="dense", choices=list(TOKEN_MODES), help="Token parameterization")
    p.add_argument("--token-rank", type=int, default=None, help="Rank/subspace width for non-dense token modes")
    p.add_argument("--iters", type=int, default=1500, help="Training iterations")
    p.add_argument("--batch-size", type=int, default=64, help="Problems per iteration")
    p.add_argument("--learning-rate", type=float, default=0.2, help="Rotational interpolation rate toward normalized negative gradient target")
    p.add_argument("--loss-temp", type=float, default=1.0, help="Cross-entropy temperature (loss is computed on logits / loss_temp)")
    p.add_argument("--use-momentum", action="store_true", help="Enable EMA momentum on normalized gradient directions")
    p.add_argument("--momentum-decay", type=float, default=0.98, help="EMA decay for momentum history (used with --use-momentum)")
    p.add_argument("--momentum-blend-start", type=float, default=0.0, help="Initial blend weight for EMA history vs current gradient")
    p.add_argument("--momentum-blend", type=float, default=0.5, help="Final blend weight for EMA history vs current gradient after ramp")
    p.add_argument("--momentum-blend-ramp-iters", type=int, default=1000, help="Iterations for blend ramp from start to final blend")
    p.add_argument("--base-mode", type=str, default="learned", choices=list(BASE_MODES), help="Base matrix behavior")
    p.add_argument("--addend-digits", type=int, default=3, help="Digits for each addend in a+b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--fixed-train-size", type=int, default=0, help="If >0, sample this many training problems once and reuse them every iteration")
    p.add_argument("--fixed-train-seed", type=int, default=None, help="Seed used for fixed training set generation (defaults to --seed)")
    p.add_argument("--load-path", type=str, default=None, help="Optional checkpoint to continue training from")
    p.add_argument("--save-path", type=str, default="checkpoints/matrix_network.pt", help="Checkpoint path")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases experiment logging")
    p.add_argument("--wandb-project", type=str, default="matrix-networks", help="W&B project name")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team/user)")
    p.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    p.add_argument("--wandb-group", type=str, default=None, help="W&B group")
    p.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags")
    p.add_argument("--wandb-dir", type=str, default=None, help="Optional local directory for W&B files")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"], help="W&B mode")
    p.add_argument("--wandb-log-every", type=int, default=10, help="Log train metrics to W&B every N iterations")
    return p.parse_args()


def init_wandb_run(
    args: argparse.Namespace,
    *,
    resolved_n: int,
    resolved_token_mode: str,
    resolved_token_rank: int | None,
    resolved_base_mode: str,
    addend_digits: int,
    max_gen_len: int,
) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("W&B logging requested but `wandb` is not installed. Install via `pip install wandb`.") from exc

    tags: list[str] | None = None
    if args.wandb_tags:
        parsed_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        tags = parsed_tags if parsed_tags else None

    config = {
        "n": resolved_n,
        "token_mode": resolved_token_mode,
        "token_rank": resolved_token_rank,
        "base_mode": resolved_base_mode,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_temp": args.loss_temp,
        "use_momentum": args.use_momentum,
        "momentum_decay": args.momentum_decay,
        "momentum_blend_start": args.momentum_blend_start,
        "momentum_blend": args.momentum_blend,
        "momentum_blend_ramp_iters": args.momentum_blend_ramp_iters,
        "addend_digits": addend_digits,
        "seed": args.seed,
        "eval_every": args.eval_every,
        "eval_samples": args.eval_samples,
        "max_gen_len": max_gen_len,
        "fixed_train_size": args.fixed_train_size,
        "fixed_train_seed": args.fixed_train_seed,
        "load_path": args.load_path,
        "save_path": args.save_path,
        "device": str(args.device),
    }

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=tags,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config=config,
    )
    print(f"wandb=on project={args.wandb_project} mode={args.wandb_mode}")
    return run


def load_checkpoint(load_path: str, device: torch.device) -> tuple[MatrixNetwork, int | None]:
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    n = int(ckpt["n"])
    base_mode = str(ckpt["base_mode"])
    token_mode = str(ckpt["token_mode"])
    raw_state = ckpt["state_dict"]
    if not isinstance(raw_state, dict):
        raise ValueError("Checkpoint state_dict must be a dict")
    state_dict: Dict[str, torch.Tensor] = dict(raw_state)
    token_rank = infer_token_rank_from_state_dict(state_dict, token_mode=token_mode, n=n)
    model = MatrixNetwork(n=n, device=device, base_mode=base_mode, token_mode=token_mode, token_rank=token_rank)
    model.load_state_dict(state_dict, strict=True)
    addend_digits = ckpt.get("addend_digits")
    if addend_digits is not None:
        addend_digits = int(addend_digits)
    return model, addend_digits


def save_checkpoint(model: MatrixNetwork, save_path: str, addend_digits: int) -> None:
    path = Path(save_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "n": model.n,
            "vocab": model.vocab,
            "addend_digits": addend_digits,
            "base_mode": model.base_mode,
            "token_mode": model.token_mode,
            "token_rank": model.token_rank,
            "state_dict": model.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.loss_temp <= 0.0:
        raise ValueError("--loss-temp must be > 0")
    if args.token_rank is not None and args.token_rank <= 0:
        raise ValueError("--token-rank must be > 0")
    if args.token_mode == "dense" and args.token_rank is not None:
        raise ValueError("--token-rank is only valid for --token-mode lowrank_ab or subspace_rot")
    if not (0.0 <= args.momentum_decay < 1.0):
        raise ValueError("--momentum-decay must be in [0, 1)")
    if not (0.0 <= args.momentum_blend_start <= 1.0):
        raise ValueError("--momentum-blend-start must be in [0, 1]")
    if not (0.0 <= args.momentum_blend <= 1.0):
        raise ValueError("--momentum-blend must be in [0, 1]")
    if args.momentum_blend < args.momentum_blend_start:
        raise ValueError("--momentum-blend must be >= --momentum-blend-start")
    if args.momentum_blend_ramp_iters < 0:
        raise ValueError("--momentum-blend-ramp-iters must be >= 0")
    if args.wandb_log_every <= 0:
        raise ValueError("--wandb-log-every must be > 0")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)
    print(f"device={device}")
    print(f"iters={args.iters} learning_rate={args.learning_rate} loss_temp={args.loss_temp}")
    print("momentum=off" if not args.use_momentum else (
        f"momentum=on decay={args.momentum_decay} blend_start={args.momentum_blend_start} "
        f"blend={args.momentum_blend} blend_ramp_iters={args.momentum_blend_ramp_iters}"
    ))

    model = None
    resolved_n = args.n
    resolved_base_mode = args.base_mode
    resolved_token_mode = args.token_mode
    resolved_token_rank = args.token_rank
    addend_digits = args.addend_digits

    if args.load_path is not None:
        model, loaded_addend_digits = load_checkpoint(args.load_path, device)
        if model.base_mode != args.base_mode:
            print(f"loaded_base_mode={model.base_mode}; overriding --base-mode={args.base_mode}")
        if model.token_mode != args.token_mode:
            print(f"loaded_token_mode={model.token_mode}; overriding --token-mode={args.token_mode}")
        if args.token_rank != model.token_rank and args.token_rank is not None:
            print(f"loaded_token_rank={model.token_rank}; overriding --token-rank={args.token_rank}")
        if model.n != args.n:
            print(f"loaded_n={model.n}; overriding --n={args.n} with checkpoint dimension")
        if loaded_addend_digits is not None and loaded_addend_digits != args.addend_digits:
            print(f"loaded_addend_digits={loaded_addend_digits}; overriding --addend-digits={args.addend_digits}")
            addend_digits = loaded_addend_digits
        resolved_n = model.n
        resolved_base_mode = model.base_mode
        resolved_token_mode = model.token_mode
        resolved_token_rank = model.token_rank

    if resolved_token_mode != "dense" and resolved_token_rank is None:
        resolved_token_rank = max(1, resolved_n // 2)

    print(f"addend_digits={addend_digits}")
    print(f"base_mode={resolved_base_mode}")
    print(f"token_mode={resolved_token_mode}")
    if resolved_token_rank is not None:
        print(f"token_rank={resolved_token_rank}")
    max_gen_len = addend_digits + 2
    print(f"max_gen_len={max_gen_len}")

    wandb_run = init_wandb_run(
        args,
        resolved_n=resolved_n,
        resolved_token_mode=resolved_token_mode,
        resolved_token_rank=resolved_token_rank,
        resolved_base_mode=resolved_base_mode,
        addend_digits=addend_digits,
        max_gen_len=max_gen_len,
    )
    try:
        model = train(
            n=resolved_n,
            token_mode=resolved_token_mode,
            token_rank=resolved_token_rank,
            iters=args.iters,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            loss_temp=args.loss_temp,
            use_momentum=args.use_momentum,
            momentum_decay=args.momentum_decay,
            momentum_blend_start=args.momentum_blend_start,
            momentum_blend=args.momentum_blend,
            momentum_blend_ramp_iters=args.momentum_blend_ramp_iters,
            base_mode=resolved_base_mode,
            addend_digits=addend_digits,
            seed=args.seed,
            log_every=args.log_every,
            eval_every=args.eval_every,
            eval_samples=args.eval_samples,
            max_gen_len=max_gen_len,
            device=device,
            model=model,
            fixed_train_size=args.fixed_train_size,
            fixed_train_seed=args.fixed_train_seed,
            wandb_run=wandb_run,
            wandb_log_every=args.wandb_log_every,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    save_checkpoint(model, args.save_path, addend_digits=addend_digits)
    print(f"saved_checkpoint={args.save_path}")
    print("\nSample predictions:")
    show_samples(model, seed=args.seed + 999, max_gen_len=max_gen_len, addend_digits=addend_digits)


if __name__ == "__main__":
    main()
