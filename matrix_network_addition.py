#!/usr/bin/env python3
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


EOS_TOKEN = "~"
VOCAB = f"0123456789+={EOS_TOKEN}"
EPS = 1e-12
Problem = Tuple[int, int, str, str]


def normalize_last_dim(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class MatrixNetwork(torch.nn.Module):
    def __init__(self, n: int, device: torch.device):
        super().__init__()
        self.n = n
        self.vocab = VOCAB
        self.vocab_size = len(self.vocab)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

        std = (1.0 / n) ** 0.5
        self.base_mat = torch.nn.Parameter(torch.eye(n, device=device) + torch.randn(n, n, device=device) * 1e-4)
        eye = torch.eye(n, device=device).unsqueeze(0).repeat(self.vocab_size, 1, 1)
        self.token_mats = torch.nn.Parameter(eye + torch.randn(self.vocab_size, n, n, device=device) * 1e-4)
        self.decode_vecs = torch.nn.Parameter(torch.randn(self.vocab_size, n, device=device) * std)
        self.query = torch.nn.Parameter(torch.randn(n, device=device) * std)

        self.renormalize_in_place()

    @torch.no_grad()
    def renormalize_in_place(self) -> None:
        self.base_mat.copy_(normalize_last_dim(self.base_mat))
        self.token_mats.copy_(normalize_last_dim(self.token_mats))
        self.decode_vecs.copy_(normalize_last_dim(self.decode_vecs))
        self.query.copy_(normalize_last_dim(self.query))

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]

    def queried_vector_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        p = self.base_mat
        for tid in token_ids:
            p = p @ self.token_mats[tid]
        v = p @ self.query
        return v

    def forward_prefix_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        v = self.queried_vector_ids(token_ids)
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
    momentum_start: float = 0.2
    momentum_end: float = 0.98
    momentum_ramp_iters: int = 3000
    _iter_count: int = 0
    _hist_b: torch.Tensor | None = None
    _hist_m: torch.Tensor | None = None
    _hist_d: torch.Tensor | None = None
    _hist_q: torch.Tensor | None = None

    def _current_momentum(self) -> float:
        if self.momentum_ramp_iters <= 0:
            ramp = 1.0
        else:
            ramp = min(1.0, max(0.0, (self._iter_count - 1) / float(self.momentum_ramp_iters)))
        return self.momentum_start + (self.momentum_end - self.momentum_start) * ramp

    def _target_direction(
        self,
        grad: torch.Tensor,
        hist: torch.Tensor | None,
        momentum: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cur = -normalize_last_dim(grad)
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
            hist = normalize_last_dim(momentum * hist + (1.0 - momentum) * cur)
        return hist, hist

    @torch.no_grad()
    def step(self, model: MatrixNetwork) -> None:
        self._iter_count += 1
        momentum = self._current_momentum()

        # Base matrix.
        w_b = model.base_mat
        g_b = model.base_mat.grad
        if g_b is not None:
            target_b, self._hist_b = self._target_direction(g_b, self._hist_b, momentum)
            new_b = w_b + self.learning_rate * (target_b - w_b)
            w_b.copy_(normalize_last_dim(new_b))

        # Token matrices: move toward negative normalized gradient target, then renormalize.
        w_m = model.token_mats
        g_m = model.token_mats.grad
        if g_m is not None:
            target_m, self._hist_m = self._target_direction(g_m, self._hist_m, momentum)
            new_m = w_m + self.learning_rate * (target_m - w_m)
            w_m.copy_(normalize_last_dim(new_m))

        # Decoder vectors.
        w_d = model.decode_vecs
        g_d = model.decode_vecs.grad
        if g_d is not None:
            target_d, self._hist_d = self._target_direction(g_d, self._hist_d, momentum)
            new_d = w_d + self.learning_rate * (target_d - w_d)
            w_d.copy_(normalize_last_dim(new_d))

        # Query vector.
        w_q = model.query
        g_q = model.query.grad
        if g_q is not None:
            target_q, self._hist_q = self._target_direction(g_q, self._hist_q, momentum)
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
    iters: int,
    batch_size: int,
    learning_rate: float,
    loss_temp: float,
    use_momentum: bool,
    momentum_start: float,
    momentum_end: float,
    momentum_ramp_iters: int,
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
) -> MatrixNetwork:
    rng = random.Random(seed)
    if model is None:
        model = MatrixNetwork(n=n, device=device)
    optim = RotationalGD(
        learning_rate=learning_rate,
        use_momentum=use_momentum,
        momentum_start=momentum_start,
        momentum_end=momentum_end,
        momentum_ramp_iters=momentum_ramp_iters,
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

        if iter_idx % log_every == 0 or iter_idx == 1:
            acc = total_correct / max(total_targets, 1)
            print(f"iter={iter_idx:5d} loss={loss.item():.4f} token_acc={acc:.3f}")

        if iter_idx % eval_every == 0 or iter_idx == iters:
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

            exact, tf_acc, stop_rate = evaluate(
                model,
                eval_samples=eval_samples,
                seed=seed + iter_idx,
                max_gen_len=max_gen_len,
                addend_digits=addend_digits,
            )
            print(f"  eval[random] exact_match={exact:.3f} teacher_forced_token_acc={tf_acc:.3f} stop_rate={stop_rate:.3f}")

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
        help="Enable scheduled momentum on normalized gradient directions",
    )
    p.add_argument(
        "--momentum-start",
        type=float,
        default=0.2,
        help="Initial momentum weight for gradient history (used with --use-momentum)",
    )
    p.add_argument(
        "--momentum-end",
        type=float,
        default=0.98,
        help="Final momentum weight for gradient history after ramp (used with --use-momentum)",
    )
    p.add_argument(
        "--momentum-ramp-iters",
        type=int,
        default=3000,
        help="Iterations over which momentum increases from start to end (used with --use-momentum)",
    )
    p.add_argument("--addend-digits", type=int, default=3, help="Digits for each addend in a+b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--max-gen-len", type=int, default=8, help="Autoregressive decode cutoff when EOS is not produced")
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
    return p.parse_args()


def load_checkpoint(load_path: str, device: torch.device) -> tuple[MatrixNetwork, int | None]:
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model = MatrixNetwork(n=int(ckpt["n"]), device=device)
    incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
    allowed_missing = {"base_mat"}
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    disallowed_missing = missing - allowed_missing
    if disallowed_missing:
        raise ValueError(f"Checkpoint missing unsupported keys: {sorted(disallowed_missing)}")
    if unexpected:
        raise ValueError(f"Checkpoint has unexpected keys: {sorted(unexpected)}")
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
            "state_dict": model.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.loss_temp <= 0.0:
        raise ValueError("--loss-temp must be > 0")
    if not (0.0 <= args.momentum_start < 1.0):
        raise ValueError("--momentum-start must be in [0, 1)")
    if not (0.0 <= args.momentum_end < 1.0):
        raise ValueError("--momentum-end must be in [0, 1)")
    if args.momentum_end < args.momentum_start:
        raise ValueError("--momentum-end must be >= --momentum-start")
    if args.momentum_ramp_iters < 0:
        raise ValueError("--momentum-ramp-iters must be >= 0")

    device = pick_device(args.device)
    print(f"device={device}")
    print(f"iters={args.iters} learning_rate={args.learning_rate} loss_temp={args.loss_temp}")
    if args.use_momentum:
        print(
            f"momentum=on start={args.momentum_start} end={args.momentum_end} "
            f"ramp_iters={args.momentum_ramp_iters}"
        )
    else:
        print("momentum=off")
    model = None
    addend_digits = args.addend_digits
    if args.load_path is not None:
        model, loaded_addend_digits = load_checkpoint(args.load_path, device)
        if model.n != args.n:
            print(f"loaded_n={model.n}; overriding --n={args.n} with checkpoint dimension")
        if loaded_addend_digits is not None and loaded_addend_digits != args.addend_digits:
            print(
                f"loaded_addend_digits={loaded_addend_digits}; overriding --addend-digits={args.addend_digits}"
            )
            addend_digits = loaded_addend_digits
    print(f"addend_digits={addend_digits}")
    model = train(
        n=model.n if model is not None else args.n,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss_temp=args.loss_temp,
        use_momentum=args.use_momentum,
        momentum_start=args.momentum_start,
        momentum_end=args.momentum_end,
        momentum_ramp_iters=args.momentum_ramp_iters,
        addend_digits=addend_digits,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        max_gen_len=args.max_gen_len,
        device=device,
        model=model,
        fixed_train_size=args.fixed_train_size,
        fixed_train_seed=args.fixed_train_seed,
    )
    save_checkpoint(model, args.save_path, addend_digits=addend_digits)
    print(f"saved_checkpoint={args.save_path}")
    print("\nSample predictions:")
    show_samples(
        model,
        seed=args.seed + 999,
        max_gen_len=args.max_gen_len,
        addend_digits=addend_digits,
    )


if __name__ == "__main__":
    main()
