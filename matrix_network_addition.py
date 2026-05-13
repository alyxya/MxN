#!/usr/bin/env python3
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch

from matrix_network import MatrixNetwork
from matrix_network_optimizer import MatrixNetworkOptimizer
from matrix_network_training import train
from matrix_network_utils import (
    load_checkpoint,
    save_checkpoint,
    subspace_summary,
)


EOS = "~"
PLUS = "+"
EQUALS = "="
DIGITS = "0123456789ABCDEF"


def vocab_for(base: int) -> Tuple[str, ...]:
    if not 2 <= base <= len(DIGITS):
        raise ValueError(f"base must be in [2, {len(DIGITS)}], got {base}")
    return tuple(EOS + DIGITS[:base] + PLUS + EQUALS)


def format_in_base(value: int, base: int, min_digits: int = 1) -> str:
    if value == 0:
        s = "0"
    else:
        chars: List[str] = []
        while value > 0:
            value, rem = divmod(value, base)
            chars.append(DIGITS[rem])
        s = "".join(reversed(chars))
    return s.rjust(min_digits, "0")


def random_problem(rng: random.Random, addend_digits: int, base: int) -> Tuple[str, str]:
    max_val = base ** addend_digits - 1
    a = rng.randint(0, max_val)
    b = rng.randint(0, max_val)
    prompt = f"{format_in_base(a, base, addend_digits)}{PLUS}{format_in_base(b, base, addend_digits)}{EQUALS}"
    answer = format_in_base(a + b, base) + EOS
    return prompt, answer


def make_sampler(
    model: MatrixNetwork,
    rng: random.Random,
    batch_size: int,
    addend_digits: int,
    base: int,
    train_full_sequence: bool,
):
    def sample_batch() -> Tuple[List[List[int]], List[int]]:
        sequences: List[List[int]] = []
        prompt_lens: List[int] = []
        for _ in range(batch_size):
            prompt, answer = random_problem(rng, addend_digits, base)
            prompt_ids = model.encode(prompt)
            target_ids = model.encode(answer)
            sequences.append(prompt_ids + target_ids)
            prompt_lens.append(0 if train_full_sequence else len(prompt_ids))
        return sequences, prompt_lens
    return sample_batch


def generate(model: MatrixNetwork, prompt: str, stop: str, max_len: int) -> Tuple[str, bool]:
    stop_id = model.stoi[stop]
    model.reset_state()
    model.apply_context(model.encode(prompt))
    out: List[int] = []
    for _ in range(max_len):
        tid = model.predict()
        if tid == stop_id:
            return "".join(model.decode(t) for t in out), True
        out.append(tid)
        model.apply_context([tid])
    return "".join(model.decode(t) for t in out), False


def evaluate(model: MatrixNetwork, samples: int, seed: int, addend_digits: int, base: int, it: int) -> None:
    rng = random.Random(seed)
    eye = torch.eye(model.n, device=model.base_mat.device, dtype=model.base_mat.dtype)
    exact = stopped = tf_correct = tf_total = 0
    states: List[torch.Tensor] = []

    for _ in range(samples):
        prompt, answer = random_problem(rng, addend_digits, base)
        prompt_ids = model.encode(prompt)
        target_ids = model.encode(answer)
        pred, did_stop = generate(model, prompt, EOS, len(answer) + 1)
        stopped += int(did_stop)
        if did_stop and pred == answer[:-1]:
            exact += 1

        prefix_op = eye
        for tid in prompt_ids:
            prefix_op = model.token_mats[tid] @ prefix_op
        for tid in target_ids:
            state = model.query @ (prefix_op @ model.base_mat)
            states.append(state.detach().cpu())
            pred_id = int((model.unembed_vectors @ state).argmax().item())
            tf_correct += int(pred_id == tid)
            tf_total += 1
            prefix_op = model.token_mats[tid] @ prefix_op

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


def show_samples(model: MatrixNetwork, seed: int, addend_digits: int, base: int, count: int = 10) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        prompt, answer = random_problem(rng, addend_digits, base)
        target = answer[:-1]
        pred, did_stop = generate(model, prompt, EOS, len(answer) + 1)
        ok = "OK" if did_stop and pred == target else "XX"
        stop_txt = "eos" if did_stop else "max"
        print(f"{prompt}{target:>4s} | pred={pred:>4s} ({stop_txt}) [{ok}]")


def default_save_filename(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"matrix_network_addition_n{args.n}_d{args.addend_digits}"
        f"_base{args.number_base}_seed{args.seed}_{timestamp}.pt"
    )


def run_training(
    args: argparse.Namespace,
    *,
    default_save_dir: Path = Path("checkpoints"),
    on_checkpoint_saved: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = None if args.device is None else torch.device(args.device)
    print(f"device={device}")

    if args.load_path:
        ckpt = load_checkpoint(args.load_path, device)
        model = MatrixNetwork(n=int(ckpt["n"]), vocab=ckpt["vocab"], device=device)
        model.load_state_dict(ckpt["model_state"])
        model.reset_state()
        optimizer = MatrixNetworkOptimizer(
            model,
            momentum_decay=args.momentum_decay,
            base_lr=args.base_learning_rate,
            token_lr=args.token_learning_rate,
            momentum_weight=args.momentum_weight,
            update_noise_scale=args.update_noise_scale,
            orthogonalize_period=args.update_orthogonalize_period,
        )
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if model.n != args.n:
            print(f"loaded_n={model.n}; overriding --n={args.n}")
            args.n = model.n
        loaded_meta = dict(ckpt["metadata"])
        loaded_base = int(loaded_meta.get("number_base", args.number_base))
        if loaded_base != args.number_base:
            print(f"loaded_number_base={loaded_base}; overriding --number-base={args.number_base}")
            args.number_base = loaded_base
        loaded_digits = int(loaded_meta.get("addend_digits", args.addend_digits))
        if loaded_digits != args.addend_digits:
            print(f"loaded_addend_digits={loaded_digits}; overriding --addend-digits={args.addend_digits}")
            args.addend_digits = loaded_digits
    else:
        model = MatrixNetwork(n=args.n, vocab=vocab_for(args.number_base), device=device)
        optimizer = MatrixNetworkOptimizer(
            model,
            momentum_decay=args.momentum_decay,
            base_lr=args.base_learning_rate,
            token_lr=args.token_learning_rate,
            momentum_weight=args.momentum_weight,
            update_noise_scale=args.update_noise_scale,
            orthogonalize_period=args.update_orthogonalize_period,
        )

    save_path = args.save_path or str(default_save_dir / default_save_filename(args))
    args.save_path = save_path
    print(f"vocab={model.vocab}")
    print(f"save_path={save_path}")
    for k, v in vars(args).items():
        print(f"{k}={v}")

    metadata = {
        "task": "addition",
        "number_base": int(args.number_base),
        "addend_digits": int(args.addend_digits),
    }

    def save(it: int) -> None:
        save_checkpoint(model, optimizer, save_path, metadata=metadata)
        if on_checkpoint_saved is not None:
            on_checkpoint_saved(save_path)

    rng = random.Random(args.seed)
    sample_batch = make_sampler(
        model,
        rng,
        args.batch_size,
        args.addend_digits,
        args.number_base,
        args.train_full_sequence,
    )

    def evaluate_cb(_: MatrixNetwork, it: int) -> None:
        evaluate(model, args.eval_samples, args.seed + it, args.addend_digits, args.number_base, it)

    def checkpoint_cb(_model: MatrixNetwork, _opt: MatrixNetworkOptimizer, it: int) -> None:
        save(it)
        print(f"checkpoint_iter={it} save_path={save_path}")

    train(
        model=model,
        optimizer=optimizer,
        sample_batch=sample_batch,
        iters=args.iters,
        recency_decay=args.recency_decay,
        eval_every=args.eval_every,
        evaluate=evaluate_cb,
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=checkpoint_cb if args.checkpoint_every > 0 else None,
    )

    save(args.iters)
    print(f"saved_checkpoint={save_path}")
    print("\nSample predictions:")
    show_samples(model, seed=args.seed + 999, addend_digits=args.addend_digits, base=args.number_base)
    return {
        "saved_checkpoint": save_path,
        "device": str(device),
        "n": model.n,
        "number_base": args.number_base,
        "addend_digits": args.addend_digits,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matrix network addition trainer")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--number-base", type=int, default=10)
    p.add_argument("--addend-digits", type=int, default=3)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--token-learning-rate", type=float, default=1.0)
    p.add_argument("--base-learning-rate", type=float, default=1.0)
    p.add_argument("--train-full-sequence", action="store_true")
    p.add_argument("--recency-decay", type=float, default=1.0)
    p.add_argument("--momentum-decay", type=float, default=0.9)
    p.add_argument("--momentum-weight", type=float, default=1.0)
    p.add_argument("--update-noise-scale", type=float, default=1.0)
    p.add_argument("--update-orthogonalize-period", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--load-path", type=str, default=None)
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    return p.parse_args()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
