#!/usr/bin/env python3
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from matrix_network import (
    DEFAULT_UPDATE_ORTHOGONALIZE_STEPS,
    MatrixNetwork,
    MatrixNetworkOptimizerState,
    advance_prefix_operator,
    format_float_token,
    format_subspace_summary,
    generate_until_token,
    generate_until_token_id,
    load_training_checkpoint,
    normalize_columns,
    pick_device,
    predict_next_id_from_prefix_operators,
    prefix_operator_from_ids,
    save_checkpoint,
    state_from_prefix_operators,
    subspace_summary,
    train,
)


EOS_TOKEN = "~"
PLUS_TOKEN = "+"
EQUALS_TOKEN = "="
DIGIT_SYMBOLS = "0123456789ABCDEF"
Problem = Tuple[int, int, str, str]


def digit_alphabet(number_base: int) -> str:
    if not (2 <= number_base <= len(DIGIT_SYMBOLS)):
        raise ValueError(f"number_base must be in [2, {len(DIGIT_SYMBOLS)}], got {number_base}")
    return DIGIT_SYMBOLS[:number_base]


def addition_vocabs(number_base: int) -> Tuple[str, str]:
    output_vocab = EOS_TOKEN + digit_alphabet(number_base)
    vocab = output_vocab + PLUS_TOKEN + EQUALS_TOKEN
    return vocab, output_vocab


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


def encode_number_ids(
    value: int,
    number_base: int,
    digit_token_ids: Sequence[int],
    *,
    min_digits: int = 1,
) -> List[int]:
    if value == 0:
        out = [digit_token_ids[0]]
    else:
        digits: List[int] = []
        x = value
        while x > 0:
            x, rem = divmod(x, number_base)
            digits.append(digit_token_ids[rem])
        out = list(reversed(digits))
    if len(out) < min_digits:
        out = ([digit_token_ids[0]] * (min_digits - len(out))) + out
    return out


def random_problem_ids(
    rng: random.Random,
    *,
    addend_digits: int,
    number_base: int,
    digit_token_ids: Sequence[int],
    plus_id: int,
    equals_id: int,
    eos_id: int,
) -> Tuple[List[int], List[int]]:
    max_val = (number_base**addend_digits) - 1
    left_addend = rng.randint(0, max_val)
    right_addend = rng.randint(0, max_val)
    prompt_ids = (
        encode_number_ids(left_addend, number_base, digit_token_ids, min_digits=addend_digits)
        + [plus_id]
        + encode_number_ids(right_addend, number_base, digit_token_ids, min_digits=addend_digits)
        + [equals_id]
    )
    target_ids = encode_number_ids(left_addend + right_addend, number_base, digit_token_ids, min_digits=1) + [eos_id]
    return prompt_ids, target_ids


def make_addition_model(*, n: int, device: torch.device, number_base: int) -> MatrixNetwork:
    vocab, output_vocab = addition_vocabs(number_base)
    return MatrixNetwork(n=n, device=device, vocab=vocab, output_vocab=output_vocab)


def make_addition_batch_sampler(
    *,
    model: MatrixNetwork,
    rng: random.Random,
    batch_size: int,
    addend_digits: int,
    number_base: int,
):
    digit_token_ids = [model.stoi[ch] for ch in digit_alphabet(number_base)]
    plus_id = model.stoi[PLUS_TOKEN]
    equals_id = model.stoi[EQUALS_TOKEN]
    eos_id = model.stoi[EOS_TOKEN]

    def sample_batch() -> Tuple[List[List[int]], List[int]]:
        sequences: List[List[int]] = []
        prompt_lens: List[int] = []
        for _ in range(batch_size):
            prompt_ids, target_seq_ids = random_problem_ids(
                rng,
                addend_digits=addend_digits,
                number_base=number_base,
                digit_token_ids=digit_token_ids,
                plus_id=plus_id,
                equals_id=equals_id,
                eos_id=eos_id,
            )
            sequences.append(prompt_ids + target_seq_ids)
            prompt_lens.append(len(prompt_ids))
        return sequences, prompt_lens

    return sample_batch


def evaluate_addition(
    model: MatrixNetwork,
    eval_samples: int,
    seed: int,
    addend_digits: int,
    number_base: int,
) -> Tuple[float, float, float, Dict[str, Dict[str, float]]]:
    rng = random.Random(seed)
    exact = 0
    tf_correct = 0
    tf_total = 0
    stopped = 0
    digit_token_ids = [model.stoi[ch] for ch in digit_alphabet(number_base)]
    plus_id = model.stoi[PLUS_TOKEN]
    equals_id = model.stoi[EQUALS_TOKEN]
    eos_id = model.stoi[EOS_TOKEN]
    primary_states: List[torch.Tensor] = []
    primary_query_targets: List[torch.Tensor] = []

    for _ in range(eval_samples):
        prompt_ids, target_ids = random_problem_ids(
            rng,
            addend_digits=addend_digits,
            number_base=number_base,
            digit_token_ids=digit_token_ids,
            plus_id=plus_id,
            equals_id=equals_id,
            eos_id=eos_id,
        )
        pred_ids, did_stop = generate_until_token_id(model, prompt_ids, eos_id, len(target_ids) + 1)
        stopped += int(did_stop)
        if did_stop and pred_ids == target_ids[:-1]:
            exact += 1

        right_total_op = prefix_operator_from_ids(model, prompt_ids)
        for target_id in target_ids:
            state = state_from_prefix_operators(model, right_total_op, model.query)
            state_normed = normalize_columns(state.unsqueeze(1))[:, 0]
            primary_states.append(state_normed.detach().cpu())
            query_target = normalize_columns(
                (right_total_op.transpose(-1, -2) @ model.base_mat.transpose(-1, -2) @ model.unembed_vectors[target_id].unsqueeze(1))
            )[:, 0]
            primary_query_targets.append(query_target.detach().cpu())
            pred_id = predict_next_id_from_prefix_operators(model, right_total_op)
            tf_correct += int(pred_id == target_id)
            tf_total += 1
            right_total_op = advance_prefix_operator(model, right_total_op, target_id)

    primary_state_summary = subspace_summary(torch.stack(primary_states, dim=0))
    primary_query_summary = subspace_summary(torch.stack(primary_query_targets, dim=0))
    return (
        exact / max(eval_samples, 1),
        tf_correct / max(tf_total, 1),
        stopped / max(eval_samples, 1),
        {
            "state": primary_state_summary,
            "query_target": primary_query_summary,
        },
    )


def show_addition_samples(
    model: MatrixNetwork,
    seed: int,
    addend_digits: int,
    number_base: int,
    count: int = 10,
) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        left_addend, right_addend, prompt_text, target_text = random_problem(rng, addend_digits, number_base)
        pred, did_stop = generate_until_token(model, prompt_text, EOS_TOKEN, len(target_text) + 2)
        ok = "OK" if (did_stop and pred == target_text) else "XX"
        stop_txt = "eos" if did_stop else "max"
        print(f"{prompt_text}{target_text:>4s} | pred={pred:>4s} ({stop_txt}) [{ok}]   ({left_addend}+{right_addend})")


def format_run_config(args: argparse.Namespace, *, addend_digits: int) -> str:
    items = [
        ("task", "addition"),
        ("n", args.n),
        ("number_base", args.number_base),
        ("addend_digits", addend_digits),
        ("iters", args.iters),
        ("batch_size", args.batch_size),
        ("token_learning_rate", args.token_learning_rate),
        ("base_learning_rate", args.base_learning_rate),
        ("primary_target_randomize", args.primary_target_randomize),
        ("state_exploration_scale", args.state_exploration_scale),
        ("state_exploration_rank", args.state_exploration_rank),
        ("state_exploration_period", args.state_exploration_period),
        ("state_exploration_samples", args.state_exploration_samples),
        ("momentum_decay", args.momentum_decay),
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
        f"matrix_network_addition_n{args.n}"
        f"_d{addend_digits}"
        f"_base{args.number_base}"
        f"_it{args.iters}"
        f"_bs{args.batch_size}"
        f"_tlr{format_float_token(args.token_learning_rate)}"
        f"_blr{format_float_token(args.base_learning_rate)}"
        f"_ptr{format_float_token(args.primary_target_randomize)}"
        f"_ses{format_float_token(args.state_exploration_scale)}"
        f"_ser{args.state_exploration_rank}"
        f"_sep{args.state_exploration_period}"
        f"_mom{format_float_token(args.momentum_decay)}"
        f"_orth{args.update_orthogonalize_steps}"
        f"_seed{args.seed}"
        f"_{timestamp}.pt"
    )
    return str(Path("checkpoints") / name)


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.momentum_decay < 1.0):
        raise ValueError("--momentum-decay must be in [0, 1)")
    if args.update_orthogonalize_steps < 0:
        raise ValueError("--update-orthogonalize-steps must be >= 0")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    if args.primary_target_randomize < 0:
        raise ValueError("--primary-target-randomize must be >= 0")
    if args.state_exploration_scale < 0:
        raise ValueError("--state-exploration-scale must be >= 0")
    if args.state_exploration_rank < 0:
        raise ValueError("--state-exploration-rank must be >= 0")
    if args.state_exploration_period < 1:
        raise ValueError("--state-exploration-period must be >= 1")
    if args.state_exploration_samples < 0:
        raise ValueError("--state-exploration-samples must be >= 0")

def run_addition_training(
    args: argparse.Namespace,
    *,
    save_path_override: str | None = None,
    save_path_transform: Any | None = None,
    checkpoint_saved_callback: Any | None = None,
) -> Dict[str, Any]:
    validate_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)
    print(f"device={device}")

    optimizer_state = MatrixNetworkOptimizerState(momentum_decay=args.momentum_decay)
    previous_completed_iters = 0
    if args.load_path is None:
        model = make_addition_model(
            n=args.n,
            device=device,
            number_base=args.number_base,
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
        loaded_number_base = len(model.output_vocab) - 1
        if loaded_number_base != args.number_base:
            print(f"loaded_number_base={loaded_number_base}; overriding --number-base={args.number_base}")
        addend_digits = int(loaded_addend_digits or args.addend_digits)
        if loaded_addend_digits is not None and loaded_addend_digits != args.addend_digits:
            print(f"loaded_addend_digits={loaded_addend_digits}; overriding --addend-digits={args.addend_digits}")
        args.n = model.n
        args.number_base = loaded_number_base
        args.addend_digits = addend_digits

    save_path = save_path_override or args.save_path or default_save_path(args, addend_digits)
    if save_path_transform is not None:
        save_path = save_path_transform(save_path)
    print(f"output_vocab={model.output_vocab}")
    print(f"save_path={save_path}")
    args.save_path = save_path
    print(format_run_config(args, addend_digits=addend_digits))

    metadata = {
        "task": "addition",
        "number_base": int(args.number_base),
        "addend_digits": int(addend_digits),
    }

    def checkpoint_callback(iter_idx: int, checkpoint_model: MatrixNetwork, checkpoint_optimizer: MatrixNetworkOptimizerState) -> None:
        save_checkpoint(
            checkpoint_model,
            save_path,
            optimizer_state=checkpoint_optimizer,
            update_orthogonalize_steps=args.update_orthogonalize_steps,
            completed_iters=previous_completed_iters + iter_idx,
            metadata=metadata,
        )
        if checkpoint_saved_callback is not None:
            checkpoint_saved_callback(save_path)
        print(f"checkpoint_iter={iter_idx} save_path={save_path}")

    rng = random.Random(args.seed)
    sample_batch = make_addition_batch_sampler(
        model=model,
        rng=rng,
        batch_size=args.batch_size,
        addend_digits=addend_digits,
        number_base=args.number_base,
    )

    def evaluate_callback(eval_model: MatrixNetwork, iter_idx: int) -> None:
        exact, tf_acc, stop_rate, subspace_stats = evaluate_addition(
            eval_model,
            eval_samples=args.eval_samples,
            seed=args.seed + iter_idx,
            addend_digits=addend_digits,
            number_base=args.number_base,
        )
        print(
            f"  eval exact_match={exact:.3f} teacher_forced_token_acc={tf_acc:.3f} "
            f"stop_rate={stop_rate:.3f}"
        )
        print(
            "  "
            + " ".join(
                [
                    format_subspace_summary("state", subspace_stats["state"]),
                    format_subspace_summary("target", subspace_stats["query_target"]),
                ]
            )
        )

    model, optimizer_state = train(
        model=model,
        optimizer_state=optimizer_state,
        sample_batch=sample_batch,
        evaluate_callback=evaluate_callback,
        iters=args.iters,
        token_learning_rate=args.token_learning_rate,
        base_learning_rate=args.base_learning_rate,
        primary_target_randomize=args.primary_target_randomize,
        state_exploration_scale=args.state_exploration_scale,
        state_exploration_rank=args.state_exploration_rank,
        state_exploration_period=args.state_exploration_period,
        state_exploration_samples=args.state_exploration_samples,
        log_every=args.log_every,
        eval_every=args.eval_every,
        update_orthogonalize_steps=args.update_orthogonalize_steps,
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=checkpoint_callback if args.checkpoint_every > 0 else None,
    )

    save_checkpoint(
        model,
        save_path,
        optimizer_state=optimizer_state,
        update_orthogonalize_steps=args.update_orthogonalize_steps,
        completed_iters=previous_completed_iters + args.iters,
        metadata=metadata,
    )
    if checkpoint_saved_callback is not None:
        checkpoint_saved_callback(save_path)
    print(f"saved_checkpoint={save_path}")
    print("\nSample predictions:")
    show_addition_samples(model, seed=args.seed + 999, addend_digits=addend_digits, number_base=args.number_base)
    return {
        "saved_checkpoint": save_path,
        "device": str(device),
        "n": model.n,
        "number_base": args.number_base,
        "addend_digits": addend_digits,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matrix network addition trainer")
    p.add_argument("--n", type=int, default=32, help="Square matrix dimension; must be >= vocab size")
    p.add_argument("--number-base", type=int, default=10, help="Arithmetic base for generated addition problems (2-16)")
    p.add_argument("--iters", type=int, default=5000, help="Training iterations")
    p.add_argument("--batch-size", type=int, default=32, help="Problems per iteration")
    p.add_argument("--token-learning-rate", type=float, default=1.0, help="Step size for token matrices")
    p.add_argument("--base-learning-rate", type=float, default=1.0, help="Step size for the base matrix")
    p.add_argument("--primary-target-randomize", type=float, default=0.0, help="Deprecated isotropic Gaussian noise strength added to primary one-hot target directions during training")
    p.add_argument("--state-exploration-scale", type=float, default=0.0, help="Training-only target noise strength sampled from low-usage state SVD directions")
    p.add_argument("--state-exploration-rank", type=int, default=8, help="Number of least-used SVD directions to use for state exploration target noise")
    p.add_argument("--state-exploration-period", type=int, default=100, help="Refresh low-usage state directions every N training iterations")
    p.add_argument("--state-exploration-samples", type=int, default=1024, help="Rolling state sample count used when computing exploration directions")
    p.add_argument("--momentum-decay", type=float, default=0.9, help="EMA decay for primary matrix momentum buffers")
    p.add_argument("--addend-digits", type=int, default=3, help="Digits for each addend in a+b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--update-orthogonalize-steps", type=int, default=DEFAULT_UPDATE_ORTHOGONALIZE_STEPS, help="Newton-Schulz orthogonalization steps after each matrix update")
    p.add_argument("--checkpoint-every", type=int, default=0, help="Save latest checkpoint every N iterations; 0 disables periodic saves")
    p.add_argument("--load-path", type=str, default=None, help="Optional checkpoint to continue training from")
    p.add_argument("--save-path", type=str, default=None, help="Optional checkpoint path override")
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def main() -> None:
    run_addition_training(parse_args())


if __name__ == "__main__":
    main()
