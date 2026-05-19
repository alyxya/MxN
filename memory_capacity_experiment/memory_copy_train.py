#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch

from memory_matrix_network import MemoryMatrixNetwork
from memory_optimizer import MemoryMatrixNetworkOptimizer
from memory_training import train
from memory_utils import subspace_summary


EOS = "~"
EQUALS = "="
DIGITS = "0123456789"


def random_copy_problem(rng: random.Random, digits: str, copy_digits: int) -> Tuple[str, str]:
    left = "".join(rng.choice(digits) for _ in range(copy_digits))
    return left + EQUALS, left + EOS


def make_sampler(
    model: MemoryMatrixNetwork,
    rng: random.Random,
    batch_size: int,
    digits: str,
    copy_digits: int,
):
    def sample_batch() -> Tuple[List[List[int]], List[int]]:
        sequences: List[List[int]] = []
        target_starts: List[int] = []
        for _ in range(batch_size):
            prompt, answer = random_copy_problem(rng, digits, copy_digits)
            prompt_ids = model.encode(prompt)
            target_ids = model.encode(answer)
            sequences.append(prompt_ids + target_ids)
            target_starts.append(len(prompt_ids))
        return sequences, target_starts

    return sample_batch


def generate(
    model: MemoryMatrixNetwork,
    prompt: str,
    stop: str,
    max_len: int,
) -> Tuple[str, bool]:
    stop_id = model.stoi[stop]
    model.reset_state()
    model.apply_context(model.encode(prompt))
    out: List[int] = []
    for _ in range(max_len):
        token_id = model.predict()
        if token_id == stop_id:
            return "".join(model.decode(t) for t in out), True
        out.append(token_id)
        model.apply_context([token_id])
    return "".join(model.decode(t) for t in out), False


@torch.no_grad()
def evaluate(
    model: MemoryMatrixNetwork,
    samples: int,
    seed: int,
    digits: str,
    copy_digits: int,
    it: int,
) -> None:
    rng = random.Random(seed)
    exact = stopped = tf_correct = tf_total = 0
    states: List[torch.Tensor] = []

    for _ in range(samples):
        prompt, answer = random_copy_problem(rng, digits, copy_digits)
        pred, did_stop = generate(model, prompt, EOS, len(answer) + 1)
        stopped += int(did_stop)
        exact += int(did_stop and pred == answer[:-1])

        prefix = model.encode(prompt)
        for token_id in model.encode(answer):
            model.reset_state()
            model.apply_context(prefix)
            state = model.query_state()
            states.append(state.detach().cpu().clone())
            tf_correct += int(model.predict() == token_id)
            tf_total += 1
            prefix.append(token_id)

    print(
        f"  eval_iter={it} exact_match={exact / max(samples, 1):.3f} "
        f"teacher_forced_token_acc={tf_correct / max(tf_total, 1):.3f} "
        f"stop_rate={stopped / max(samples, 1):.3f}"
    )
    if states:
        state_rows = torch.stack(states)
        decode_norms = state_rows[:, : model.vocab_size].norm(dim=1)
        print(
            "  "
            + subspace_summary("state", state_rows)
            + (
                f" decode_norm[mean={decode_norms.mean().item():.3f} "
                f"min={decode_norms.min().item():.3f} "
                f"max={decode_norms.max().item():.3f}]"
            )
        )


def show_samples(
    model: MemoryMatrixNetwork,
    seed: int,
    digits: str,
    copy_digits: int,
    count: int = 10,
) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        prompt, answer = random_copy_problem(rng, digits, copy_digits)
        target = answer[:-1]
        pred, did_stop = generate(model, prompt, EOS, len(answer) + 1)
        ok = "OK" if did_stop and pred == target else "XX"
        stop_txt = "eos" if did_stop else "max"
        print(f"{prompt}{target} | pred={pred} ({stop_txt}) [{ok}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matrix network memory-copy capacity trainer")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--number-base", type=int, default=10)
    parser.add_argument("--copy-digits", type=int, default=10)
    parser.add_argument(
        "--update-side",
        choices=["left", "right", "double-left", "double-right"],
        default="left",
    )
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-learning-rate", type=float, default=1.0)
    parser.add_argument("--base-learning-rate", type=float, default=0.1)
    parser.add_argument("--correct-margin", type=float, default=None)
    parser.add_argument("--recency-decay", type=float, default=1.0)
    parser.add_argument("--momentum-decay", type=float, default=0.0)
    parser.add_argument("--momentum-weight", type=float, default=0.0)
    parser.add_argument("--update-noise-scale", type=float, default=0.5)
    parser.add_argument("--update-orthogonalize-period", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=300)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 2 <= args.number_base <= len(DIGITS):
        raise ValueError(f"--number-base must be between 2 and {len(DIGITS)}")
    if args.correct_margin is not None and args.correct_margin < 0.0:
        raise ValueError("--correct-margin must be >= 0")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = None if args.device is None else torch.device(args.device)
    digits = DIGITS[: args.number_base]
    vocab = tuple(EOS + digits + EQUALS)
    model = MemoryMatrixNetwork(
        n=args.n,
        vocab=vocab,
        update_side=args.update_side,
        device=device,
    )
    optimizer = MemoryMatrixNetworkOptimizer(
        model,
        momentum_decay=args.momentum_decay,
        base_lr=args.base_learning_rate,
        token_lr=args.token_learning_rate,
        momentum_weight=args.momentum_weight,
        update_noise_scale=args.update_noise_scale,
        orthogonalize_period=args.update_orthogonalize_period,
    )

    print(f"device={device}")
    print(f"vocab={model.vocab}")
    for key, value in vars(args).items():
        print(f"{key}={value}")

    rng = random.Random(args.seed)
    sample_batch = make_sampler(model, rng, args.batch_size, digits, args.copy_digits)

    def evaluate_cb(eval_model: MemoryMatrixNetwork, it: int) -> None:
        evaluate(eval_model, args.eval_samples, args.seed + it, digits, args.copy_digits, it)

    train(
        model=model,
        optimizer=optimizer,
        sample_batch=sample_batch,
        iters=args.iters,
        recency_decay=args.recency_decay,
        correct_margin=args.correct_margin,
        eval_every=args.eval_every,
        evaluate=evaluate_cb,
    )

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n": model.n,
                "vocab": model.vocab,
                "update_side": model.update_side,
                "number_base": args.number_base,
                "copy_digits": args.copy_digits,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            save_path,
        )
        print(f"saved_checkpoint={save_path}")

    print("\nSample predictions:")
    show_samples(model, seed=args.seed + 999, digits=digits, copy_digits=args.copy_digits)


if __name__ == "__main__":
    main()
