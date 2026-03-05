#!/usr/bin/env python3
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


VOCAB = "0123456789+="
EPS = 1e-12


def normalize_vectors(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def normalize_matrices_rowcol(x: torch.Tensor, iters: int = 2, eps: float = EPS) -> torch.Tensor:
    out = x
    for _ in range(iters):
        out = out / (out.norm(dim=-1, keepdim=True) + eps)  # row vectors
        out = out / (out.norm(dim=-2, keepdim=True) + eps)  # column vectors
    return out


class MatrixNetwork(torch.nn.Module):
    def __init__(self, n: int, device: torch.device):
        super().__init__()
        self.n = n
        self.vocab = VOCAB
        self.vocab_size = len(self.vocab)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

        std = (1.0 / n) ** 0.5
        self.token_mats = torch.nn.Parameter(torch.randn(self.vocab_size, n, n, device=device) * std)
        self.decode_vecs = torch.nn.Parameter(torch.randn(self.vocab_size, n, device=device) * std)
        self.query = torch.nn.Parameter(torch.randn(n, device=device) * std)

        self.renormalize_in_place()

    @torch.no_grad()
    def renormalize_in_place(self) -> None:
        self.token_mats.copy_(normalize_matrices_rowcol(self.token_mats))
        self.decode_vecs.copy_(normalize_vectors(self.decode_vecs, dim=-1))
        self.query.copy_(normalize_vectors(self.query, dim=0))

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]

    def forward_prefix_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        p = torch.eye(self.n, device=self.token_mats.device)
        for tid in token_ids:
            p = p @ self.token_mats[tid]
        v = p @ self.query
        logits = self.decode_vecs @ v
        return logits

    @torch.no_grad()
    def predict_next(self, prefix: str) -> str:
        token_ids = self.encode(prefix)
        logits = self.forward_prefix_ids(token_ids)
        return self.itos[int(logits.argmax().item())]


@dataclass
class RotationalGD:
    step_size: float

    @torch.no_grad()
    def step(self, model: MatrixNetwork) -> None:
        # Token matrices: normalize rows+cols, move toward normalized gradient update, renormalize.
        w_m = model.token_mats
        g_m = model.token_mats.grad
        if g_m is not None:
            w_m_norm = normalize_matrices_rowcol(w_m)
            target_m = normalize_matrices_rowcol(w_m_norm - g_m)
            new_m = w_m_norm + self.step_size * (target_m - w_m_norm)
            w_m.copy_(normalize_matrices_rowcol(new_m))

        # Decoder vectors.
        w_d = model.decode_vecs
        g_d = model.decode_vecs.grad
        if g_d is not None:
            w_d_norm = normalize_vectors(w_d, dim=-1)
            target_d = normalize_vectors(w_d_norm - g_d, dim=-1)
            new_d = w_d_norm + self.step_size * (target_d - w_d_norm)
            w_d.copy_(normalize_vectors(new_d, dim=-1))

        # Query vector.
        w_q = model.query
        g_q = model.query.grad
        if g_q is not None:
            w_q_norm = normalize_vectors(w_q, dim=0)
            target_q = normalize_vectors(w_q_norm - g_q, dim=0)
            new_q = w_q_norm + self.step_size * (target_q - w_q_norm)
            w_q.copy_(normalize_vectors(new_q, dim=0))

        model.zero_grad(set_to_none=True)


def random_problem(rng: random.Random) -> Tuple[int, int, str, str]:
    a = rng.randint(0, 999)
    b = rng.randint(0, 999)
    lhs = f"{a:03d}+{b:03d}="
    rhs = str(a + b)
    return a, b, lhs, rhs


def evaluate(model: MatrixNetwork, eval_samples: int, seed: int) -> Tuple[float, float]:
    rng = random.Random(seed)
    exact = 0
    tf_correct = 0
    tf_total = 0

    for _ in range(eval_samples):
        _, _, lhs, rhs = random_problem(rng)

        pred = ""
        for _ in range(len(rhs)):
            pred += model.predict_next(lhs + pred)
        if pred == rhs:
            exact += 1

        for i in range(len(rhs)):
            next_ch = model.predict_next(lhs + rhs[:i])
            tf_correct += int(next_ch == rhs[i])
            tf_total += 1

    return exact / eval_samples, tf_correct / max(tf_total, 1)


def show_samples(model: MatrixNetwork, seed: int, count: int = 10) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        a, b, lhs, rhs = random_problem(rng)
        pred = ""
        for _ in range(len(rhs)):
            pred += model.predict_next(lhs + pred)
        ok = "OK" if pred == rhs else "XX"
        print(f"{lhs}{rhs:>4s} | pred={pred:>4s}  [{ok}]   ({a}+{b})")


def train(
    n: int,
    steps: int,
    batch_size: int,
    step_size: float,
    seed: int,
    log_every: int,
    eval_every: int,
    eval_samples: int,
    device: torch.device,
) -> MatrixNetwork:
    rng = random.Random(seed)
    model = MatrixNetwork(n=n, device=device)
    optim = RotationalGD(step_size=step_size)

    for step in range(1, steps + 1):
        losses: List[torch.Tensor] = []
        total_correct = 0
        total_targets = 0

        for _ in range(batch_size):
            _, _, lhs, rhs = random_problem(rng)
            for i in range(len(rhs)):
                prefix = lhs + rhs[:i]
                target_id = model.stoi[rhs[i]]
                logits = model.forward_prefix_ids(model.encode(prefix))
                losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id], device=device)))
                total_correct += int(int(logits.argmax().item()) == target_id)
                total_targets += 1

        loss = torch.stack(losses).mean()
        loss.backward()
        optim.step(model)

        if step % log_every == 0 or step == 1:
            acc = total_correct / max(total_targets, 1)
            print(f"step={step:5d} loss={loss.item():.4f} digit_acc={acc:.3f}")

        if step % eval_every == 0 or step == steps:
            exact, tf_acc = evaluate(model, eval_samples=eval_samples, seed=seed + step)
            print(f"  eval exact_match={exact:.3f} teacher_forced_digit_acc={tf_acc:.3f}")

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
    p.add_argument("--steps", type=int, default=1500, help="Training steps")
    p.add_argument("--batch-size", type=int, default=64, help="Problems per step")
    p.add_argument("--step-size", type=float, default=0.2, help="Rotational interpolation step toward normalized update target")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device={device}")
    model = train(
        n=args.n,
        steps=args.steps,
        batch_size=args.batch_size,
        step_size=args.step_size,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        device=device,
    )
    print("\nSample predictions:")
    show_samples(model, seed=args.seed + 999)


if __name__ == "__main__":
    main()
