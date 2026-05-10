#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
from typing import Any

import modal

from matrix_network_addition import run_training

APP_NAME = "mxn-matrix-network"
VOLUME_NAME = "mxn-matrix-network-checkpoints"
REMOTE_ROOT = Path("/checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
    .add_local_python_source(
        "matrix_network",
        "matrix_network_optimizer",
        "matrix_network_training",
        "matrix_network_addition",
        "matrix_network_utils",
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def to_remote(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if p.parts and p.parts[0] == REMOTE_ROOT.name:
        p = Path(*p.parts[1:])
    return str(REMOTE_ROOT / p)


def _train_impl(args_dict: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if args.load_path:
        args.load_path = to_remote(args.load_path)
    if args.save_path:
        args.save_path = to_remote(args.save_path)

    start = time.perf_counter()
    result = run_training(
        args,
        default_save_dir=REMOTE_ROOT,
        on_checkpoint_saved=lambda _: volume.commit(),
    )
    elapsed = time.perf_counter() - start
    return {
        **result,
        "elapsed_seconds": elapsed,
        "iters_per_second": (args.iters / elapsed) if elapsed > 0 else None,
    }


def train_remote(gpu: str | None):
    @app.function(image=image, volumes={str(REMOTE_ROOT): volume}, gpu=gpu, timeout=24 * 3600)
    def run(args_dict: dict[str, Any]) -> dict[str, Any]:
        return _train_impl(args_dict)

    return run


@app.function(image=image, volumes={str(REMOTE_ROOT): volume}, gpu=None, timeout=600)
def list_checkpoints_remote(prefix: str = "") -> list[str]:
    if not REMOTE_ROOT.exists():
        return []
    prefix_path = Path(prefix)
    if prefix_path.parts and prefix_path.parts[0] == REMOTE_ROOT.name:
        prefix_path = Path(*prefix_path.parts[1:])
    prefix_str = str(prefix_path) if prefix else ""
    out: list[str] = []
    for path in sorted(REMOTE_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(REMOTE_ROOT))
        if prefix_str and not rel.startswith(prefix_str):
            continue
        out.append(str(Path(REMOTE_ROOT.name) / rel))
    return out


@app.function(image=image, volumes={str(REMOTE_ROOT): volume}, gpu=None, timeout=600)
def download_checkpoint_remote(remote_path: str) -> bytes:
    p = Path(to_remote(remote_path))
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p.read_bytes()


@app.function(image=image, volumes={str(REMOTE_ROOT): volume}, gpu=None, timeout=600)
def upload_checkpoint_remote(remote_path: str, data: bytes) -> str:
    p = Path(to_remote(remote_path))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(p)
    volume.commit()
    return str(p)


@app.local_entrypoint()
def main(
    command: str = "train",
    n: int = 32,
    number_base: int = 10,
    iters: int = 5000,
    batch_size: int = 32,
    token_learning_rate: float = 1.0,
    base_learning_rate: float = 1.0,
    target_randomize_scale: float = 0.0,
    recency_decay: float = 1.0,
    momentum_decay: float = 0.9,
    momentum_weight: float = 1.0,
    update_orthogonalize_period: int = 100,
    addend_digits: int = 3,
    seed: int = 0,
    eval_every: int = 250,
    eval_samples: int = 300,
    checkpoint_every: int = 250,
    load_path: str = "",
    save_path: str = "",
    gpu: str = "T4",
    prefix: str = "",
    remote_path: str = "",
    local_path: str = "",
) -> None:
    if command == "list":
        for path in list_checkpoints_remote.remote(prefix):
            print(path)
        return

    if command == "download":
        if not remote_path:
            raise ValueError("--remote-path is required for download")
        local_dest = Path(local_path) if local_path else Path("checkpoints") / Path(remote_path).name
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        local_dest.write_bytes(download_checkpoint_remote.remote(remote_path))
        print(f"downloaded_checkpoint={local_dest}")
        return

    if command == "upload":
        if not local_path:
            raise ValueError("--local-path is required for upload")
        src = Path(local_path)
        dest = remote_path or str(Path("checkpoints") / src.name)
        saved = upload_checkpoint_remote.remote(dest, src.read_bytes())
        print(f"uploaded_checkpoint={saved}")
        return

    if command != "train":
        raise ValueError(f"unsupported command={command!r}; expected train, list, download, or upload")

    gpu_key = gpu.lower()
    if gpu_key in {"", "none", "cpu"}:
        gpu_type = None
        device = "cpu"
    elif gpu_key == "t4":
        gpu_type = "T4"
        device = "cuda"
    else:
        raise ValueError(f"unsupported --gpu {gpu!r}; supported values are 'none' and 'T4'")

    args_dict = {
        "n": n,
        "number_base": number_base,
        "iters": iters,
        "batch_size": batch_size,
        "token_learning_rate": token_learning_rate,
        "base_learning_rate": base_learning_rate,
        "target_randomize_scale": target_randomize_scale,
        "recency_decay": recency_decay,
        "momentum_decay": momentum_decay,
        "momentum_weight": momentum_weight,
        "update_orthogonalize_period": update_orthogonalize_period,
        "addend_digits": addend_digits,
        "seed": seed,
        "eval_every": eval_every,
        "eval_samples": eval_samples,
        "checkpoint_every": checkpoint_every,
        "load_path": load_path or None,
        "save_path": save_path or None,
        "device": device,
    }

    result = train_remote(gpu_type).remote(args_dict)
    print("\nModal run result:")
    for k, v in result.items():
        print(f"{k}={v}")
