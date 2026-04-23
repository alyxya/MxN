#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any

import modal
import torch

from matrix_network_manual_rotation import (
    ManualRotationMatrixNetwork,
    default_save_path,
    format_run_config,
    load_training_checkpoint,
    pick_device,
    save_checkpoint,
    show_samples,
    train,
    ManualRotationOptimizerState,
)

APP_NAME = "mxn-matrix-network"
VOLUME_NAME = "mxn-matrix-network-checkpoints"
REMOTE_CHECKPOINT_ROOT = Path("/checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch")
    .add_local_python_source("matrix_network_manual_rotation")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _strip_checkpoints_prefix(path: Path) -> Path:
    if path.parts and path.parts[0] == "checkpoints":
        return Path(*path.parts[1:])
    return path


def _remote_checkpoint_path(save_path: str) -> str:
    path = Path(save_path)
    if path.is_absolute():
        return str(path)
    return str(REMOTE_CHECKPOINT_ROOT / _strip_checkpoints_prefix(path))


def _remote_load_path(load_path: str) -> str:
    path = Path(load_path)
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == REMOTE_CHECKPOINT_ROOT.name:
        return str(REMOTE_CHECKPOINT_ROOT / Path(*path.parts[1:]))
    return str(REMOTE_CHECKPOINT_ROOT / _strip_checkpoints_prefix(path))


def parse_modal_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["train", *argv]

    p = argparse.ArgumentParser(description="Run matrix network training and checkpoint management on Modal")
    sub = p.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train", help="Launch or resume a remote training job")
    list_p = sub.add_parser("list", help="List checkpoints stored on the Modal volume")
    download_p = sub.add_parser("download", help="Download a checkpoint from the Modal volume")
    upload_p = sub.add_parser("upload", help="Upload a local checkpoint to the Modal volume")

    p = train_p
    p.add_argument("--n", type=int, default=32, help="Square matrix dimension; must be >= vocab size")
    p.add_argument("--number-base", type=int, default=10, help="Arithmetic base for generated addition problems (2-16)")
    p.add_argument(
        "--token-mat-mode",
        type=str,
        default="right",
        choices=["left", "right", "both"],
        help="Apply learned token matrices on the left of base, right of base, or both",
    )
    p.add_argument(
        "--base-randomize",
        type=float,
        default=0.0,
        help="Base init randomization strength; 0 gives identity",
    )
    p.add_argument(
        "--token-randomize",
        type=float,
        default=0.0,
        help="Token-matrix init randomization strength; 0 gives identity",
    )
    p.add_argument("--iters", type=int, default=5000, help="Training iterations")
    p.add_argument("--batch-size", type=int, default=32, help="Problems per iteration")
    p.add_argument("--token-learning-rate", type=float, default=1.0, help="Step size for token embedding matrices")
    p.add_argument("--base-learning-rate", type=float, default=0.1, help="Step size for the base matrix")
    p.add_argument("--primary-query-learning-rate", type=float, default=0.001, help="Step size for the primary next-token query vector")
    p.add_argument("--primary-unembed-learning-rate", type=float, default=0.001, help="Step size for primary next-token unembedding vectors")
    p.add_argument("--secondary-query-learning-rate", type=float, default=0.1, help="Step size for secondary past-token query vectors")
    p.add_argument("--secondary-unembed-learning-rate", type=float, default=0.1, help="Step size for secondary past-token unembedding vectors")
    p.add_argument("--momentum-decay", type=float, default=0.9, help="EMA decay for primary matrix momentum buffers")
    p.add_argument("--addend-digits", type=int, default=3, help="Digits for each addend in a+b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-samples", type=int, default=300)
    p.add_argument("--negative-scale", type=float, default=2.0, help="Repulsion weight for wrong learned vectors that outscore the correct one")
    p.add_argument("--secondary-matrix-scale", type=float, default=0.1, help="Scale multiplier for matrix learning from past-token auxiliary objectives")
    p.add_argument("--update-orthogonalize-steps", type=int, default=1, help="Newton-Schulz orthogonalization steps after each matrix update")
    p.add_argument("--checkpoint-every", type=int, default=250, help="Save latest checkpoint to the Modal volume every N iterations; 0 disables periodic saves")
    p.add_argument("--load-path", type=str, default=None, help="Optional checkpoint path relative to Modal volume")
    p.add_argument("--save-path", type=str, default=None, help="Optional checkpoint path relative to Modal volume")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Remote device choice")
    p.add_argument("--gpu", type=str, default="T4", help="Modal GPU type, e.g. T4 or A10G; set to empty string for CPU")
    p.add_argument("--timeout", type=int, default=24 * 60 * 60, help="Modal function timeout in seconds")
    list_p.add_argument("--prefix", type=str, default="", help="Optional checkpoints/ prefix filter")
    download_p.add_argument("remote_path", type=str, help="Remote checkpoint path, absolute or relative to checkpoints/")
    download_p.add_argument("local_path", type=str, nargs="?", default=None, help="Optional local destination path")
    upload_p.add_argument("local_path", type=str, help="Local checkpoint file to upload")
    upload_p.add_argument("remote_path", type=str, nargs="?", default=None, help="Optional remote path under checkpoints/")
    return p.parse_args(argv)



def _train_impl(args_dict: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if not (0.0 <= args.momentum_decay < 1.0):
        raise ValueError("--momentum-decay must be in [0, 1)")
    if args.update_orthogonalize_steps < 0:
        raise ValueError("--update-orthogonalize-steps must be >= 0")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)
    print(f"device={device}")

    optimizer_state = ManualRotationOptimizerState(momentum_decay=args.momentum_decay)
    previous_completed_iters = 0
    if args.load_path is None:
        model = ManualRotationMatrixNetwork(
            n=args.n,
            device=device,
            number_base=args.number_base,
            token_mat_mode=args.token_mat_mode,
            base_randomize=args.base_randomize,
            token_randomize=args.token_randomize,
        )
        addend_digits = args.addend_digits
    else:
        model, loaded_addend_digits, optimizer_state, previous_completed_iters = load_training_checkpoint(
            _remote_load_path(args.load_path),
            device,
            momentum_decay=args.momentum_decay,
        )
        if model.n != args.n:
            print(f"loaded_n={model.n}; overriding --n={args.n}")
        if model.number_base != args.number_base:
            print(f"loaded_number_base={model.number_base}; ignoring --number-base={args.number_base}")
        if model.token_mat_mode != args.token_mat_mode:
            print(f"loaded_token_mat_mode={model.token_mat_mode}; ignoring --token-mat-mode={args.token_mat_mode}")
        addend_digits = int(loaded_addend_digits or args.addend_digits)
        if loaded_addend_digits is not None and loaded_addend_digits != args.addend_digits:
            print(f"loaded_addend_digits={loaded_addend_digits}; overriding --addend-digits={args.addend_digits}")

    default_path = default_save_path(args, addend_digits)
    save_path = _remote_checkpoint_path(args.save_path or default_path)
    print(f"output_vocab={model.output_vocab}")
    print(f"save_path={save_path}")
    args.save_path = save_path
    print(format_run_config(args, addend_digits=addend_digits))

    def checkpoint_callback(iter_idx: int, checkpoint_model: ManualRotationMatrixNetwork, checkpoint_optimizer: ManualRotationOptimizerState) -> None:
        save_checkpoint(
            checkpoint_model,
            save_path,
            addend_digits=addend_digits,
            optimizer_state=checkpoint_optimizer,
            update_orthogonalize_steps=args.update_orthogonalize_steps,
            completed_iters=previous_completed_iters + iter_idx,
        )
        volume.commit()
        print(f"checkpoint_iter={iter_idx} save_path={save_path}")

    model, optimizer_state = train(
        model=model,
        optimizer_state=optimizer_state,
        iters=args.iters,
        batch_size=args.batch_size,
        token_learning_rate=args.token_learning_rate,
        base_learning_rate=args.base_learning_rate,
        primary_query_learning_rate=args.primary_query_learning_rate,
        primary_unembed_learning_rate=args.primary_unembed_learning_rate,
        secondary_query_learning_rate=args.secondary_query_learning_rate,
        secondary_unembed_learning_rate=args.secondary_unembed_learning_rate,
        addend_digits=addend_digits,
        number_base=model.number_base,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        negative_scale=args.negative_scale,
        secondary_matrix_scale=args.secondary_matrix_scale,
        update_orthogonalize_steps=args.update_orthogonalize_steps,
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=checkpoint_callback if args.checkpoint_every > 0 else None,
    )

    save_checkpoint(
        model,
        save_path,
        addend_digits=addend_digits,
        optimizer_state=optimizer_state,
        update_orthogonalize_steps=args.update_orthogonalize_steps,
        completed_iters=previous_completed_iters + args.iters,
    )
    volume.commit()
    print(f"saved_checkpoint={save_path}")
    print("\nSample predictions:")
    show_samples(model, seed=args.seed + 999, addend_digits=addend_digits, number_base=model.number_base)
    return {
        "saved_checkpoint": save_path,
        "n": model.n,
        "number_base": model.number_base,
        "token_mat_mode": model.token_mat_mode,
        "addend_digits": addend_digits,
    }


@app.function(
    image=image,
    volumes={str(REMOTE_CHECKPOINT_ROOT): volume},
    gpu=None,
    timeout=24 * 60 * 60,
)
def train_remote(args_dict: dict[str, Any]) -> dict[str, Any]:
    return _train_impl(args_dict)


@app.function(
    image=image,
    volumes={str(REMOTE_CHECKPOINT_ROOT): volume},
    gpu=None,
    timeout=10 * 60,
)
def list_checkpoints_remote(prefix: str = "") -> list[str]:
    root = REMOTE_CHECKPOINT_ROOT
    prefix_path = str(_strip_checkpoints_prefix(Path(prefix))) if prefix else ""
    results: list[str] = []
    if not root.exists():
        return results
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        rel_str = str(rel)
        if prefix_path and not rel_str.startswith(prefix_path):
            continue
        results.append(str(Path("checkpoints") / rel))
    return results


@app.function(
    image=image,
    volumes={str(REMOTE_CHECKPOINT_ROOT): volume},
    gpu=None,
    timeout=10 * 60,
)
def download_checkpoint_remote(remote_path: str) -> bytes:
    path = Path(_remote_load_path(remote_path))
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.read_bytes()


@app.function(
    image=image,
    volumes={str(REMOTE_CHECKPOINT_ROOT): volume},
    gpu=None,
    timeout=10 * 60,
)
def upload_checkpoint_remote(remote_path: str, data: bytes) -> str:
    path = Path(_remote_checkpoint_path(remote_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)
    volume.commit()
    return str(path)


@app.local_entrypoint()
def main() -> None:
    args = parse_modal_args()

    if args.command == "list":
        checkpoints = list_checkpoints_remote.remote(args.prefix)
        for path in checkpoints:
            print(path)
        return

    if args.command == "download":
        local_path = Path(args.local_path) if args.local_path else Path("checkpoints") / Path(_strip_checkpoints_prefix(Path(args.remote_path)))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(download_checkpoint_remote.remote(args.remote_path))
        print(f"downloaded_checkpoint={local_path}")
        return

    if args.command == "upload":
        local_path = Path(args.local_path)
        remote_path = args.remote_path or str(Path("checkpoints") / local_path.name)
        remote_saved_path = upload_checkpoint_remote.remote(remote_path, local_path.read_bytes())
        print(f"uploaded_checkpoint={remote_saved_path}")
        return

    gpu = args.gpu
    timeout = args.timeout
    args_dict = vars(args).copy()
    args_dict.pop("command")
    args_dict.pop("gpu")
    args_dict.pop("timeout")

    remote_fn = train_remote
    if gpu == "":
        gpu = None
    if gpu is not None or timeout != 24 * 60 * 60:
        remote_fn = train_remote.with_options(gpu=gpu, timeout=timeout)

    result = remote_fn.remote(args_dict)
    print("\nModal run result:")
    for k, v in result.items():
        print(f"{k}={v}")
