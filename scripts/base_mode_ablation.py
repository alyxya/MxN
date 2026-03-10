#!/usr/bin/env python3
import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matrix_network_addition import evaluate, load_checkpoint, pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run controlled base-matrix ablation: learned vs fixed identity")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--token-mode", type=str, default="dense", choices=["dense", "lowrank_ab", "subspace_rot"])
    p.add_argument("--token-rank", type=int, default=None)
    p.add_argument("--addend-digits", type=int, default=10)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--loss-temp", type=float, default=1.0)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--fixed-train-size", type=int, default=0)
    p.add_argument("--fixed-train-seed", type=int, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--out-dir", type=str, default="checkpoints/base_mode_ablation")
    p.add_argument("--summary-csv", type=str, default=None, help="Defaults to <out-dir>/summary.csv")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable to run training")
    p.add_argument("--use-momentum", action="store_true")
    p.add_argument("--momentum-decay", type=float, default=0.98)
    p.add_argument("--momentum-blend-start", type=float, default=0.0)
    p.add_argument("--momentum-blend", type=float, default=0.5)
    p.add_argument("--momentum-blend-ramp-iters", type=int, default=1000)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="matrix-networks")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default="base-mode-ablation")
    p.add_argument("--wandb-tags", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-dir", type=str, default=None)
    p.add_argument("--wandb-log-every", type=int, default=10)
    return p.parse_args()


def resolved_rank(args: argparse.Namespace) -> int | None:
    if args.token_mode == "dense":
        return None
    return args.token_rank if args.token_rank is not None else max(1, args.n // 2)


def build_train_cmd(args: argparse.Namespace, repo_root: Path, *, base_mode: str, seed: int, save_path: Path) -> List[str]:
    cmd = [
        args.python,
        str(repo_root / "matrix_network_addition.py"),
        "--n",
        str(args.n),
        "--token-mode",
        args.token_mode,
        "--base-mode",
        base_mode,
        "--addend-digits",
        str(args.addend_digits),
        "--iters",
        str(args.iters),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--loss-temp",
        str(args.loss_temp),
        "--seed",
        str(seed),
        "--eval-every",
        str(args.eval_every),
        "--eval-samples",
        str(args.eval_samples),
        "--save-path",
        str(save_path),
        "--device",
        args.device,
    ]
    rank = resolved_rank(args)
    if rank is not None:
        cmd.extend(["--token-rank", str(rank)])
    if args.fixed_train_size > 0:
        cmd.extend(["--fixed-train-size", str(args.fixed_train_size)])
    if args.fixed_train_seed is not None:
        cmd.extend(["--fixed-train-seed", str(args.fixed_train_seed)])
    if args.use_momentum:
        cmd.extend(
            [
                "--use-momentum",
                "--momentum-decay",
                str(args.momentum_decay),
                "--momentum-blend-start",
                str(args.momentum_blend_start),
                "--momentum-blend",
                str(args.momentum_blend),
                "--momentum-blend-ramp-iters",
                str(args.momentum_blend_ramp_iters),
            ]
        )
    if args.wandb:
        cmd.extend(
            [
                "--wandb",
                "--wandb-project",
                args.wandb_project,
                "--wandb-group",
                args.wandb_group,
                "--wandb-mode",
                args.wandb_mode,
                "--wandb-log-every",
                str(args.wandb_log_every),
                "--wandb-run-name",
                f"base-{base_mode}-seed{seed}",
            ]
        )
        if args.wandb_entity is not None:
            cmd.extend(["--wandb-entity", args.wandb_entity])
        if args.wandb_tags is not None:
            cmd.extend(["--wandb-tags", f"{args.wandb_tags},base_mode={base_mode},seed={seed}"])
        if args.wandb_dir is not None:
            cmd.extend(["--wandb-dir", args.wandb_dir])
    return cmd


def evaluate_checkpoint(path: Path, eval_samples: int, eval_seed: int, device_name: str) -> Dict[str, float]:
    device = pick_device(device_name)
    model, addend_digits = load_checkpoint(str(path), device)
    if addend_digits is None:
        raise ValueError(f"Checkpoint missing addend_digits: {path}")
    exact, tf_acc, stop_rate = evaluate(
        model,
        eval_samples=eval_samples,
        seed=eval_seed,
        max_gen_len=addend_digits + 2,
        addend_digits=addend_digits,
    )
    return {
        "eval_exact_match": exact,
        "eval_teacher_forced_token_acc": tf_acc,
        "eval_stop_rate": stop_rate,
    }


def summarize(results: List[Dict[str, float | int | str]]) -> None:
    by_mode: Dict[str, List[Dict[str, float | int | str]]] = {}
    for row in results:
        by_mode.setdefault(str(row["base_mode"]), []).append(row)
    print("\nAggregate by base_mode:")
    for mode in sorted(by_mode):
        rows = by_mode[mode]
        exact = [float(r["eval_exact_match"]) for r in rows]
        tf = [float(r["eval_teacher_forced_token_acc"]) for r in rows]
        print(
            f"  {mode}: exact_mean={statistics.mean(exact):.4f} exact_std={statistics.pstdev(exact):.4f} "
            f"tf_mean={statistics.mean(tf):.4f} tf_std={statistics.pstdev(tf):.4f} n={len(rows)}"
        )


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv) if args.summary_csv is not None else out_dir / "summary.csv"

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    modes = ["learned", "identity_fixed"]
    results: List[Dict[str, float | int | str]] = []
    rank = resolved_rank(args)
    rank_tag = "dense" if rank is None else f"k{rank}"

    for seed in seeds:
        for base_mode in modes:
            save_path = out_dir / (
                f"n{args.n}_{args.token_mode}_{rank_tag}_d{args.addend_digits}_it{args.iters}_"
                f"lr{args.learning_rate:g}_{base_mode}_seed{seed}.pt"
            )
            cmd = build_train_cmd(args, repo_root, base_mode=base_mode, seed=seed, save_path=save_path)
            print("\n>>>", " ".join(cmd))
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True, cwd=repo_root)
            metrics = evaluate_checkpoint(save_path, eval_samples=max(1000, args.eval_samples), eval_seed=2026, device_name=args.device)
            row: Dict[str, float | int | str] = {"seed": seed, "base_mode": base_mode, "checkpoint": str(save_path)}
            row.update(metrics)
            results.append(row)
            print(
                f"result seed={seed} mode={base_mode} exact={metrics['eval_exact_match']:.4f} "
                f"tf={metrics['eval_teacher_forced_token_acc']:.4f}"
            )

    if args.dry_run:
        return
    if results:
        fields = [
            "seed",
            "base_mode",
            "checkpoint",
            "eval_exact_match",
            "eval_teacher_forced_token_acc",
            "eval_stop_rate",
        ]
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nsummary_csv={summary_csv}")
        summarize(results)


if __name__ == "__main__":
    main()
