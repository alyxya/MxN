#!/usr/bin/env python3
import argparse
import time
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
    .pip_install("torch", "numpy")
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


def _train_impl(args_dict: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if not (0.0 <= args.momentum_decay < 1.0):
        raise ValueError("--momentum-decay must be in [0, 1)")
    if args.update_orthogonalize_steps < 0:
        raise ValueError("--update-orthogonalize-steps must be >= 0")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    if args.primary_target_randomize < 0:
        raise ValueError("--primary-target-randomize must be >= 0")
    if args.secondary_matrix_period < 1:
        raise ValueError("--secondary-matrix-period must be >= 1")
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
        args.n = model.n
        args.number_base = model.number_base
        args.token_mat_mode = model.token_mat_mode
        args.addend_digits = addend_digits

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

    start_time = time.perf_counter()
    model, optimizer_state = train(
        model=model,
        optimizer_state=optimizer_state,
        iters=args.iters,
        batch_size=args.batch_size,
        token_learning_rate=args.token_learning_rate,
        base_learning_rate=args.base_learning_rate,
        primary_target_randomize=args.primary_target_randomize,
        addend_digits=addend_digits,
        number_base=model.number_base,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        secondary_matrix_scale=args.secondary_matrix_scale,
        secondary_matrix_period=args.secondary_matrix_period,
        update_orthogonalize_steps=args.update_orthogonalize_steps,
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=checkpoint_callback if args.checkpoint_every > 0 else None,
    )
    elapsed_seconds = time.perf_counter() - start_time

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
        "elapsed_seconds": elapsed_seconds,
        "iters_per_second": (args.iters / elapsed_seconds) if elapsed_seconds > 0 else None,
        "device": str(device),
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
def train_remote_cpu(args_dict: dict[str, Any]) -> dict[str, Any]:
    return _train_impl(args_dict)


@app.function(
    image=image,
    volumes={str(REMOTE_CHECKPOINT_ROOT): volume},
    gpu="T4",
    timeout=24 * 60 * 60,
)
def train_remote_t4(args_dict: dict[str, Any]) -> dict[str, Any]:
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
def main(
    command: str = "train",
    n: int = 32,
    number_base: int = 10,
    token_mat_mode: str = "right",
    iters: int = 5000,
    batch_size: int = 32,
    token_learning_rate: float = 1.0,
    base_learning_rate: float = 0.1,
    primary_target_randomize: float = 1.0,
    momentum_decay: float = 0.9,
    addend_digits: int = 3,
    seed: int = 0,
    log_every: int = 50,
    eval_every: int = 250,
    eval_samples: int = 300,
    secondary_matrix_scale: float = 0.0,
    secondary_matrix_period: int = 1,
    update_orthogonalize_steps: int = 1,
    checkpoint_every: int = 250,
    load_path: str = "",
    save_path: str = "",
    device: str = "auto",
    gpu: str = "T4",
    prefix: str = "",
    remote_path: str = "",
    local_path: str = "",
) -> None:
    if command == "list":
        checkpoints = list_checkpoints_remote.remote(prefix)
        for path in checkpoints:
            print(path)
        return

    if command == "download":
        if not remote_path:
            raise ValueError("--remote-path is required for --command download")
        local_destination = Path(local_path) if local_path else Path("checkpoints") / Path(_strip_checkpoints_prefix(Path(remote_path)))
        local_destination.parent.mkdir(parents=True, exist_ok=True)
        local_destination.write_bytes(download_checkpoint_remote.remote(remote_path))
        print(f"downloaded_checkpoint={local_destination}")
        return

    if command == "upload":
        if not local_path:
            raise ValueError("--local-path is required for --command upload")
        local_source = Path(local_path)
        remote_destination = remote_path or str(Path("checkpoints") / local_source.name)
        remote_saved_path = upload_checkpoint_remote.remote(remote_destination, local_source.read_bytes())
        print(f"uploaded_checkpoint={remote_saved_path}")
        return

    if command != "train":
        raise ValueError(f"unsupported command={command!r}; expected train, list, download, or upload")

    args_dict = {
        "n": n,
        "number_base": number_base,
        "token_mat_mode": token_mat_mode,
        "iters": iters,
        "batch_size": batch_size,
        "token_learning_rate": token_learning_rate,
        "base_learning_rate": base_learning_rate,
        "primary_target_randomize": primary_target_randomize,
        "momentum_decay": momentum_decay,
        "addend_digits": addend_digits,
        "seed": seed,
        "log_every": log_every,
        "eval_every": eval_every,
        "eval_samples": eval_samples,
        "secondary_matrix_scale": secondary_matrix_scale,
        "secondary_matrix_period": secondary_matrix_period,
        "update_orthogonalize_steps": update_orthogonalize_steps,
        "checkpoint_every": checkpoint_every,
        "load_path": load_path or None,
        "save_path": save_path or None,
        "device": device,
    }

    gpu_key = gpu.lower()
    if gpu_key in {"", "none", "cpu"}:
        remote_fn = train_remote_cpu
    elif gpu_key == "t4":
        remote_fn = train_remote_t4
    else:
        raise ValueError(f"unsupported --gpu {gpu!r}; supported values are 'none' and 'T4'")

    result = remote_fn.remote(args_dict)
    print("\nModal run result:")
    for k, v in result.items():
        print(f"{k}={v}")
