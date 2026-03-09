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
BASE_MODES = ("learned", "identity_fixed")


def normalize_last_dim(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def normalize_dim(x: torch.Tensor, dim: int, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def infer_token_rank_from_state_dict(state_dict: Dict[str, torch.Tensor], n: int) -> int:
    token_a = state_dict.get("token_a")
    token_b = state_dict.get("token_b")
    if token_a is not None or token_b is not None:
        if token_a is None or token_b is None:
            raise ValueError("Checkpoint state_dict must include both token_a and token_b")
        if token_a.ndim != 3 or token_b.ndim != 3:
            raise ValueError("token_a/token_b must be rank-3 tensors")
        if token_a.shape[1] != n:
            raise ValueError(f"token_a shape mismatch for n={n}: got {tuple(token_a.shape)}")
        if token_b.shape[2] != n:
            raise ValueError(f"token_b shape mismatch for n={n}: got {tuple(token_b.shape)}")
        if token_a.shape[2] != token_b.shape[1]:
            raise ValueError(f"token_a/token_b rank mismatch: {tuple(token_a.shape)} vs {tuple(token_b.shape)}")
        if token_a.shape[0] != token_b.shape[0]:
            raise ValueError(f"token_a/token_b vocab mismatch: {tuple(token_a.shape)} vs {tuple(token_b.shape)}")
        return int(token_a.shape[2])
    token_mats = state_dict.get("token_mats")
    if token_mats is not None:
        if token_mats.ndim != 3 or token_mats.shape[1] != n or token_mats.shape[2] != n:
            raise ValueError(f"legacy token_mats shape mismatch for n={n}: got {tuple(token_mats.shape)}")
        return n
    raise ValueError("Checkpoint state_dict missing token representation keys (token_a/token_b or token_mats)")


def convert_legacy_token_state_if_needed(state_dict: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
    if "token_mats" not in state_dict or "token_a" in state_dict or "token_b" in state_dict:
        return state_dict
    token_mats = state_dict["token_mats"]
    vocab_size = token_mats.shape[0]
    eye = torch.eye(n, dtype=token_mats.dtype, device=token_mats.device).unsqueeze(0).expand(vocab_size, -1, -1)
    converted = dict(state_dict)
    converted.pop("token_mats")
    converted["token_a"] = token_mats - eye
    converted["token_b"] = eye.clone()
    return converted


class MatrixNetwork(torch.nn.Module):
    def __init__(self, n: int, device: torch.device, base_mode: str = "learned", token_rank: int | None = None):
        super().__init__()
        self.n = n
        if base_mode not in BASE_MODES:
            raise ValueError(f"Unsupported base_mode={base_mode!r}; expected one of {BASE_MODES}")
        self.base_mode = base_mode
        self.learn_base_mat = base_mode == "learned"
        if token_rank is None:
            token_rank = max(1, n // 2)
        if token_rank < 1:
            raise ValueError(f"token_rank must be >= 1, got {token_rank}")
        self.token_rank = int(token_rank)
        self.vocab = VOCAB
        self.vocab_size = len(self.vocab)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

        std = (1.0 / n) ** 0.5
        self.register_buffer("eye_n", torch.eye(n, device=device), persistent=False)
        if self.learn_base_mat:
            self.base_mat = torch.nn.Parameter(self.eye_n.clone() + torch.randn(n, n, device=device) * 1e-4)
        else:
            self.register_buffer("base_mat", self.eye_n.clone())
        self.token_a = torch.nn.Parameter(torch.randn(self.vocab_size, n, self.token_rank, device=device) * 1e-2)
        self.token_b = torch.nn.Parameter(torch.randn(self.vocab_size, self.token_rank, n, device=device) * 1e-2)
        self.decode_vecs = torch.nn.Parameter(torch.randn(self.vocab_size, n, device=device) * std)
        self.query = torch.nn.Parameter(torch.randn(n, device=device) * std)

        self.renormalize_in_place()

    @torch.no_grad()
    def renormalize_in_place(self) -> None:
        if self.learn_base_mat:
            self.base_mat.copy_(normalize_last_dim(self.base_mat))
        # Keep factor scales bounded in n-direction for stable low-rank products.
        self.token_a.copy_(normalize_dim(self.token_a, dim=-2))
        self.token_b.copy_(normalize_last_dim(self.token_b))
        self.decode_vecs.copy_(normalize_last_dim(self.decode_vecs))
        self.query.copy_(normalize_last_dim(self.query))

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]

    def token_matrix(self, tid: int) -> torch.Tensor:
        m = self.eye_n + self.token_a[tid] @ self.token_b[tid]
        return normalize_last_dim(m)

    def queried_vector_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        p = self.base_mat
        for tid in token_ids:
            p = p @ self.token_matrix(tid)
        v = p @ self.query
        return v

    def forward_prefix_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        v = self.queried_vector_ids(token_ids)
        v = normalize_last_dim(v)
        logits = self.decode_vecs @ v
        return logits

    @torch.no_grad()
    def predict_next(self, prefix: str) -> str:
        token_ids = self.encode(prefix)
        logits = self.forward_prefix_ids(token_ids)
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
    _hist_a: torch.Tensor | None = None
    _hist_b: torch.Tensor | None = None
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
        if (
            hist is None
            or hist.shape != cur.shape
            or hist.device != cur.device
            or hist.dtype != cur.dtype
        ):
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

        # Base matrix.
        if model.learn_base_mat:
            w_b = model.base_mat
            g_b = model.base_mat.grad
            if g_b is not None:
                target_b, self._hist_base = self._target_direction(g_b, self._hist_base, blend, norm_dim=-1)
                new_b = w_b + self.learning_rate * (target_b - w_b)
                w_b.copy_(normalize_last_dim(new_b))

        # Token low-rank factors.
        w_a = model.token_a
        g_a = model.token_a.grad
        if g_a is not None:
            target_a, self._hist_a = self._target_direction(g_a, self._hist_a, blend, norm_dim=-2)
            new_a = w_a + self.learning_rate * (target_a - w_a)
            w_a.copy_(normalize_dim(new_a, dim=-2))

        w_b = model.token_b
        g_b = model.token_b.grad
        if g_b is not None:
            target_b, self._hist_b = self._target_direction(g_b, self._hist_b, blend, norm_dim=-1)
            new_b = w_b + self.learning_rate * (target_b - w_b)
            w_b.copy_(normalize_last_dim(new_b))

        # Decoder vectors.
        w_d = model.decode_vecs
        g_d = model.decode_vecs.grad
        if g_d is not None:
            target_d, self._hist_d = self._target_direction(g_d, self._hist_d, blend, norm_dim=-1)
            new_d = w_d + self.learning_rate * (target_d - w_d)
            w_d.copy_(normalize_last_dim(new_d))

        # Query vector.
        w_q = model.query
        g_q = model.query.grad
        if g_q is not None:
            target_q, self._hist_q = self._target_direction(g_q, self._hist_q, blend, norm_dim=-1)
            new_q = w_q + self.learning_rate * (target_q - w_q)
            w_q.copy_(normalize_last_dim(new_q))

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


def evaluate_on_dataset(
    model: MatrixNetwork,
    dataset: Sequence[Problem],
    max_gen_len: int,
) -> Tuple[float, float, float]:
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


def show_samples(
    model: MatrixNetwork,
    seed: int,
    max_gen_len: int,
    addend_digits: int,
    count: int = 10,
) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        a, b, lhs, rhs = random_problem(rng, addend_digits)
        pred, did_stop = generate_until_eos(model, lhs, max_gen_len)
        ok = "OK" if (did_stop and pred == rhs) else "XX"
        stop_txt = "eos" if did_stop else "max"
        print(f"{lhs}{rhs:>4s} | pred={pred:>4s} ({stop_txt}) [{ok}]   ({a}+{b})")


def train(
    n: int,
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
        model = MatrixNetwork(n=n, device=device, base_mode=base_mode, token_rank=token_rank)
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
        fixed_train = make_fixed_dataset(
            seed=fixed_seed,
            size=fixed_train_size,
            addend_digits=addend_digits,
        )
        print(f"fixed_train_size={len(fixed_train)} fixed_train_seed={fixed_seed}")

    for iter_idx in range(1, iters + 1):
        model.zero_grad(set_to_none=True)
        losses: List[torch.Tensor] = []
        total_correct = 0
        total_targets = 0

        for _ in range(batch_size):
            if fixed_train is None:
                _, _, lhs, rhs = random_problem(rng, addend_digits)
            else:
                _, _, lhs, rhs = fixed_train[rng.randrange(len(fixed_train))]
            target_seq = rhs + EOS_TOKEN
            for i in range(len(target_seq)):
                prefix = lhs + target_seq[:i]
                target_id = model.stoi[target_seq[i]]
                token_ids = model.encode(prefix)
                logits = model.forward_prefix_ids(token_ids)
                losses.append(
                    F.cross_entropy(
                        (logits / loss_temp).unsqueeze(0),
                        torch.tensor([target_id], device=device),
                    )
                )
                total_correct += int(int(logits.argmax().item()) == target_id)
                total_targets += 1

        loss = torch.stack(losses).mean()
        loss.backward()
        optim.step(model)
        acc = total_correct / max(total_targets, 1)

        if wandb_run is not None and (iter_idx == 1 or iter_idx % wandb_log_every == 0 or iter_idx == iters):
            wandb_run.log(
                {
                    "train/loss": float(loss.item()),
                    "train/token_acc": float(acc),
                    "iter": iter_idx,
                },
                step=iter_idx,
            )

        if iter_idx % log_every == 0 or iter_idx == 1:
            print(f"iter={iter_idx:5d} loss={loss.item():.4f} token_acc={acc:.3f}")

        if iter_idx % eval_every == 0 or iter_idx == iters:
            eval_log: Dict[str, float | int] = {"iter": iter_idx}
            if fixed_train is not None:
                tr_exact, tr_tf_acc, tr_stop_rate = evaluate_on_dataset(
                    model,
                    dataset=fixed_train,
                    max_gen_len=max_gen_len,
                )
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
    p = argparse.ArgumentParser(description="Matrix-network toy model for 3-digit addition")
    p.add_argument("--n", type=int, default=16, help="Embedding matrix/vector dimension")
    p.add_argument(
        "--token-rank",
        type=int,
        default=None,
        help="Token matrix factor width k in I + A@B (defaults to n//2; may be > n)",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=1500,
        help="Training iterations",
    )
    p.add_argument("--batch-size", type=int, default=64, help="Problems per iteration")
    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.2,
        help="Rotational interpolation rate toward normalized negative gradient target",
    )
    p.add_argument(
        "--loss-temp",
        type=float,
        default=1.0,
        help="Cross-entropy temperature (loss is computed on logits / loss_temp)",
    )
    p.add_argument(
        "--use-momentum",
        action="store_true",
        help="Enable EMA momentum on normalized gradient directions",
    )
    p.add_argument(
        "--momentum-decay",
        type=float,
        default=0.98,
        help="EMA decay for momentum history (used with --use-momentum)",
    )
    p.add_argument(
        "--momentum-blend-start",
        type=float,
        default=0.0,
        help="Initial blend weight for EMA history vs current gradient (used with --use-momentum)",
    )
    p.add_argument(
        "--momentum-blend",
        type=float,
        default=0.5,
        help="Final blend weight for EMA history vs current gradient after ramp (used with --use-momentum)",
    )
    p.add_argument(
        "--momentum-blend-ramp-iters",
        type=int,
        default=1000,
        help="Iterations for blend ramp from --momentum-blend-start to --momentum-blend (used with --use-momentum)",
    )
    p.add_argument(
        "--base-mode",
        type=str,
        default="learned",
        choices=list(BASE_MODES),
        help="Base matrix behavior: learned trainable matrix or fixed identity (no base matrix learning)",
    )
    p.add_argument("--addend-digits", type=int, default=3, help="Digits for each addend in a+b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument(
        "--fixed-train-size",
        type=int,
        default=0,
        help="If >0, sample this many training problems once and reuse them every iteration",
    )
    p.add_argument(
        "--fixed-train-seed",
        type=int,
        default=None,
        help="Seed used for fixed training set generation (defaults to --seed)",
    )
    p.add_argument("--load-path", type=str, default=None, help="Optional checkpoint to continue training from")
    p.add_argument("--save-path", type=str, default="checkpoints/matrix_network_addition.pt", help="Checkpoint path")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases experiment logging")
    p.add_argument("--wandb-project", type=str, default="matrix-networks", help="W&B project name")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team/user)")
    p.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    p.add_argument("--wandb-group", type=str, default=None, help="W&B group")
    p.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags")
    p.add_argument("--wandb-dir", type=str, default=None, help="Optional local directory for W&B files")
    p.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode",
    )
    p.add_argument("--wandb-log-every", type=int, default=10, help="Log train metrics to W&B every N iterations")
    return p.parse_args()


def init_wandb_run(
    args: argparse.Namespace,
    *,
    resolved_n: int,
    resolved_token_rank: int,
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
    base_mode = str(ckpt.get("base_mode", "learned"))
    n = int(ckpt["n"])
    raw_state = ckpt["state_dict"]
    if not isinstance(raw_state, dict):
        raise ValueError("Checkpoint state_dict must be a dict")
    state_dict: Dict[str, torch.Tensor] = dict(raw_state)
    inferred_rank = infer_token_rank_from_state_dict(state_dict, n=n)
    token_rank = int(ckpt.get("token_rank", inferred_rank))
    state_dict = convert_legacy_token_state_if_needed(state_dict, n=n)
    model = MatrixNetwork(n=n, device=device, base_mode=base_mode, token_rank=token_rank)
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"base_mat", "eye_n"}
    allowed_unexpected = {"eye_n"}
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    disallowed_missing = missing - allowed_missing
    disallowed_unexpected = unexpected - allowed_unexpected
    if disallowed_missing:
        raise ValueError(f"Checkpoint missing unsupported keys: {sorted(disallowed_missing)}")
    if disallowed_unexpected:
        raise ValueError(f"Checkpoint has unexpected keys: {sorted(disallowed_unexpected)}")
    if "base_mat" in missing:
        print("checkpoint_missing_base_mat=true; using newly initialized base matrix")
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
    if args.use_momentum:
        print(
            f"momentum=on decay={args.momentum_decay} "
            f"blend_start={args.momentum_blend_start} blend={args.momentum_blend} "
            f"blend_ramp_iters={args.momentum_blend_ramp_iters}"
        )
    else:
        print("momentum=off")
    model = None
    resolved_base_mode = args.base_mode
    addend_digits = args.addend_digits
    resolved_token_rank = args.token_rank
    if args.load_path is not None:
        model, loaded_addend_digits = load_checkpoint(args.load_path, device)
        if model.base_mode != args.base_mode:
            print(f"loaded_base_mode={model.base_mode}; overriding --base-mode={args.base_mode}")
        resolved_base_mode = model.base_mode
        if args.token_rank is not None and args.token_rank != model.token_rank:
            print(f"loaded_token_rank={model.token_rank}; overriding --token-rank={args.token_rank}")
        resolved_token_rank = model.token_rank
        if model.n != args.n:
            print(f"loaded_n={model.n}; overriding --n={args.n} with checkpoint dimension")
        if loaded_addend_digits is not None and loaded_addend_digits != args.addend_digits:
            print(
                f"loaded_addend_digits={loaded_addend_digits}; overriding --addend-digits={args.addend_digits}"
            )
            addend_digits = loaded_addend_digits
    print(f"addend_digits={addend_digits}")
    print(f"base_mode={resolved_base_mode}")
    resolved_n = model.n if model is not None else args.n
    if resolved_token_rank is None:
        resolved_token_rank = max(1, resolved_n // 2)
    print(f"token_rank={resolved_token_rank}")
    max_gen_len = addend_digits + 2
    print(f"max_gen_len={max_gen_len}")
    wandb_run = init_wandb_run(
        args,
        resolved_n=resolved_n,
        resolved_token_rank=resolved_token_rank,
        resolved_base_mode=resolved_base_mode,
        addend_digits=addend_digits,
        max_gen_len=max_gen_len,
    )
    try:
        model = train(
            n=resolved_n,
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
    show_samples(
        model,
        seed=args.seed + 999,
        max_gen_len=max_gen_len,
        addend_digits=addend_digits,
    )


if __name__ == "__main__":
    main()
