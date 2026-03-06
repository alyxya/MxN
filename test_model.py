#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Tuple

import torch

from matrix_network_addition import MatrixNetwork, generate_until_eos, pick_device


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


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[MatrixNetwork, int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if "n" not in ckpt or "state_dict" not in ckpt:
        raise ValueError("Checkpoint missing required keys: 'n' and 'state_dict'")

    model = MatrixNetwork(n=int(ckpt["n"]), device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    addend_digits = int(ckpt.get("addend_digits", 3))
    return model, addend_digits


@torch.no_grad()
def run_prediction(model: MatrixNetwork, a: int, b: int, max_gen_len: int, addend_digits: int) -> None:
    lhs = format_lhs(a, b, addend_digits)
    pred, did_stop = generate_until_eos(model, lhs, max_gen_len)
    expected = str(a + b)
    status = "eos" if did_stop else "max_len"
    correct = did_stop and pred == expected
    print(
        f"input={lhs} pred={pred} expected={expected} stop={status} correct={correct}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive tester for trained matrix-network addition model")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="models/matrix_network_n30_step0005_resume10k.pt",
        help="Path to saved checkpoint",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--max-gen-len", type=int, default=8)
    p.add_argument("--addend-digits", type=int, default=None, help="Override addend width used for prompt formatting")
    p.add_argument("--expr", type=str, default=None, help="Single expression, e.g. '123+456'")
    p.add_argument("--a", type=int, default=None, help="First addend for one-shot run")
    p.add_argument("--b", type=int, default=None, help="Second addend for one-shot run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    ckpt_path = Path(args.checkpoint)
    model, ckpt_addend_digits = load_model(ckpt_path, device)
    addend_digits = args.addend_digits if args.addend_digits is not None else ckpt_addend_digits
    print(f"loaded={ckpt_path} device={device} n={model.n} addend_digits={addend_digits}")

    if args.expr is not None:
        a, b = parse_expression(args.expr)
        run_prediction(model, a, b, args.max_gen_len, addend_digits)
        return

    if args.a is not None or args.b is not None:
        if args.a is None or args.b is None:
            raise ValueError("If using --a/--b, both must be set")
        run_prediction(model, args.a, args.b, args.max_gen_len, addend_digits)
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
            run_prediction(model, a, b, args.max_gen_len, addend_digits)
        except Exception as exc:
            print(f"error: {exc}")


if __name__ == "__main__":
    main()
