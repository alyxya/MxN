#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Tuple

import torch

from matrix_network_addition import generate_until_eos, load_checkpoint, pick_device


def required_max_gen_len(addend_digits: int) -> int:
    return addend_digits + 2


def parse_expression(expr: str) -> Tuple[int, int]:
    text = expr.strip().replace(" ", "")
    m = re.fullmatch(r"(\d+)\+(\d+)=?", text)
    if not m:
        raise ValueError("Expected format like 123+456")
    return int(m.group(1)), int(m.group(2))


def parse_prompt_line(line: str) -> Tuple[int, int]:
    text = line.strip()
    if "+" in text:
        return parse_expression(text)
    parts = text.split()
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return int(parts[0]), int(parts[1])
    raise ValueError("Enter either '123+456' or '123 456'")


def format_lhs(a: int, b: int, addend_digits: int) -> str:
    return f"{a:0{addend_digits}d}+{b:0{addend_digits}d}="


@torch.no_grad()
def run_prediction(model, a: int, b: int, addend_digits: int) -> None:
    lhs = format_lhs(a, b, addend_digits)
    pred, did_stop = generate_until_eos(model, lhs, required_max_gen_len(addend_digits))
    expected = str(a + b)
    status = "eos" if did_stop else "max_len"
    correct = did_stop and pred == expected
    print(f"input={lhs} pred={pred} expected={expected} stop={status} correct={correct}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive tester for trained matrix-network addition model")
    p.add_argument("--checkpoint", type=str, default="models/dense_identity_n30_d3.pt", help="Path to saved checkpoint")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--addend-digits", type=int, default=None, help="Override addend width used for prompt formatting")
    p.add_argument("--expr", type=str, default=None, help="Single expression, e.g. '123+456'")
    p.add_argument("--a", type=int, default=None, help="First addend for one-shot run")
    p.add_argument("--b", type=int, default=None, help="Second addend for one-shot run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    ckpt_path = Path(args.checkpoint)
    model, ckpt_addend_digits = load_checkpoint(str(ckpt_path), device)
    model.eval()
    addend_digits = args.addend_digits if args.addend_digits is not None else int(ckpt_addend_digits or 3)
    print(
        f"loaded={ckpt_path} device={device} n={model.n} base_mode={model.base_mode} "
        f"token_mode={model.token_mode} addend_digits={addend_digits} "
        f"max_gen_len={required_max_gen_len(addend_digits)}"
    )

    if args.expr is not None:
        a, b = parse_expression(args.expr)
        run_prediction(model, a, b, addend_digits)
        return

    if args.a is not None or args.b is not None:
        if args.a is None or args.b is None:
            raise ValueError("If using --a/--b, both must be set")
        run_prediction(model, args.a, args.b, addend_digits)
        return

    print("Enter additions as '123+456' or '123 456'. Type 'quit' to exit.")
    while True:
        try:
            line = input("add> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line.lower() in {"q", "quit", "exit"}:
            break
        try:
            a, b = parse_prompt_line(line)
            run_prediction(model, a, b, addend_digits)
        except Exception as exc:
            print(f"error: {exc}")


if __name__ == "__main__":
    main()
