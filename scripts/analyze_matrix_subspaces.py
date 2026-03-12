#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matrix_network_addition import load_checkpoint, pick_device


EPS = 1e-12
DEFAULT_ENERGY_LEVELS = (0.9, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze how much of a subspace each matrix operator acts on. "
            "Reports raw update rank (M - I), skew/symmetric decompositions of "
            "that update, nearest-orthogonal rotation-support rank from the "
            "polar factor (Q - I), and active-subspace closure seeded by query "
            "and decode vectors."
        )
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Checkpoint files or directories containing .pt checkpoints",
    )
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument(
        "--component",
        type=str,
        default="all",
        choices=["all", "base", "tokens"],
        help="Which checkpoint operators to analyze",
    )
    p.add_argument(
        "--energy-levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_ENERGY_LEVELS),
        help="Energy thresholds used for low-rank summaries, e.g. 0.9 0.95 0.99",
    )
    p.add_argument(
        "--rel-rank-threshold",
        type=float,
        default=1e-2,
        help="Relative singular-value threshold for numerical rank (relative to the top singular value)",
    )
    p.add_argument(
        "--abs-rank-threshold",
        type=float,
        default=1e-6,
        help="Absolute singular-value threshold for numerical rank",
    )
    p.add_argument(
        "--angle-threshold-deg",
        type=float,
        default=1.0,
        help="Eigen-angle threshold, in degrees, for counting nontrivial polar-factor rotations",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many token operators to show in the human-readable summary",
    )
    p.add_argument(
        "--show-all-tokens",
        action="store_true",
        help="Print every token row in the human-readable summary",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save full analysis as JSON",
    )
    p.add_argument(
        "--closure-rel-threshold",
        type=float,
        default=1e-2,
        help="Relative singular-value threshold used when estimating closure subspace dimension",
    )
    p.add_argument(
        "--closure-abs-threshold",
        type=float,
        default=1e-6,
        help="Absolute singular-value threshold used when estimating closure subspace dimension",
    )
    p.add_argument(
        "--closure-max-iters",
        type=int,
        default=8,
        help="Maximum closure-expansion iterations",
    )
    p.add_argument(
        "--state-samples",
        type=int,
        default=256,
        help="Number of random addition problems to sample for empirical state-span analysis",
    )
    p.add_argument(
        "--state-seed",
        type=int,
        default=2026,
        help="Random seed for empirical state-span analysis",
    )
    return p.parse_args()


def resolve_checkpoint_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        if path.is_dir():
            for found in sorted(path.rglob("*.pt")):
                resolved = found.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(found)
        else:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    if not paths:
        raise ValueError("No checkpoint files found")
    return paths


def to_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def to_int(value: torch.Tensor | float | int) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def identity_like(matrix: torch.Tensor) -> torch.Tensor:
    return torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)


def stable_rank(svals: torch.Tensor) -> float:
    if svals.numel() == 0:
        return 0.0
    top = to_float(svals.max())
    if top <= EPS:
        return 0.0
    return to_float((svals.square().sum()) / (top * top))


def entropy_effective_rank(svals: torch.Tensor) -> float:
    if svals.numel() == 0:
        return 0.0
    energy = svals.square()
    total = to_float(energy.sum())
    if total <= EPS:
        return 0.0
    probs = (energy / total).clamp_min(EPS)
    return float(torch.exp(-(probs * probs.log()).sum()).item())


def energy_rank(svals: torch.Tensor, level: float) -> int:
    if svals.numel() == 0:
        return 0
    energy = svals.square()
    total = to_float(energy.sum())
    if total <= EPS:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    return int(torch.searchsorted(cumulative, torch.tensor(level, dtype=cumulative.dtype)).item()) + 1


def singular_value_summary(
    svals: torch.Tensor,
    *,
    rel_rank_threshold: float,
    abs_rank_threshold: float,
    energy_levels: Sequence[float],
    prefix: str,
) -> Dict[str, float | int | List[float]]:
    svals = svals.detach().to(dtype=torch.float64, device="cpu")
    result: Dict[str, float | int | List[float]] = {
        f"{prefix}_top_singular_values": [float(x) for x in svals[: min(6, svals.numel())].tolist()],
        f"{prefix}_stable_rank": stable_rank(svals),
        f"{prefix}_effective_rank": entropy_effective_rank(svals),
    }
    if svals.numel() == 0:
        result[f"{prefix}_rank_rel"] = 0
        result[f"{prefix}_rank_abs"] = 0
        for level in energy_levels:
            result[f"{prefix}_energy_rank_{int(round(level * 100))}"] = 0
        return result

    smax = to_float(svals.max())
    result[f"{prefix}_rank_rel"] = int((svals > (rel_rank_threshold * smax)).sum().item()) if smax > EPS else 0
    result[f"{prefix}_rank_abs"] = int((svals > abs_rank_threshold).sum().item())
    for level in energy_levels:
        result[f"{prefix}_energy_rank_{int(round(level * 100))}"] = energy_rank(svals, level)
    return result


def orthonormal_basis(
    vectors: torch.Tensor,
    *,
    rel_threshold: float,
    abs_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vectors.ndim != 2:
        raise ValueError(f"Expected a rank-2 matrix of column vectors, got shape {tuple(vectors.shape)}")
    if vectors.shape[1] == 0:
        empty = torch.empty((vectors.shape[0], 0), dtype=vectors.dtype, device=vectors.device)
        return empty, torch.empty((0,), dtype=vectors.dtype, device=vectors.device)
    u, svals, _ = torch.linalg.svd(vectors, full_matrices=False)
    if svals.numel() == 0:
        empty = torch.empty((vectors.shape[0], 0), dtype=vectors.dtype, device=vectors.device)
        return empty, svals
    smax = to_float(svals.max())
    keep = svals > max(abs_threshold, rel_threshold * smax)
    rank = int(keep.sum().item()) if smax > EPS else 0
    return u[:, :rank], svals[:rank]


def collect_model_operators(model: Any) -> List[torch.Tensor]:
    ops: List[torch.Tensor] = [model.base_matrix().detach().to(dtype=torch.float64, device="cpu")]
    ops.extend(model.token_matrix(token_id).detach().to(dtype=torch.float64, device="cpu") for token_id in range(model.vocab_size))
    return ops


def closure_seed_vectors(model: Any) -> torch.Tensor:
    query = model.query.detach().to(dtype=torch.float64, device="cpu").unsqueeze(1)
    decode = model.decode_vecs.detach().to(dtype=torch.float64, device="cpu").transpose(0, 1)
    return torch.cat([query, decode], dim=1)


def closure_step(
    basis: torch.Tensor,
    operators: Sequence[torch.Tensor],
    *,
    include_transpose: bool,
) -> torch.Tensor:
    blocks: List[torch.Tensor] = [basis]
    for op in operators:
        blocks.append(op @ basis)
        if include_transpose:
            blocks.append(op.transpose(-1, -2) @ basis)
    return torch.cat(blocks, dim=1)


def active_subspace_closure(
    model: Any,
    *,
    rel_threshold: float,
    abs_threshold: float,
    max_iters: int,
) -> Dict[str, Any]:
    operators = collect_model_operators(model)
    seed = closure_seed_vectors(model)
    seed_basis, seed_svals = orthonormal_basis(seed, rel_threshold=rel_threshold, abs_threshold=abs_threshold)
    seed_dim = int(seed_basis.shape[1])
    result: Dict[str, Any] = {
        "seed_dim": seed_dim,
        "seed_fraction_of_n": seed_dim / max(model.n, 1),
        "num_operators": len(operators),
    }
    result.update(
        singular_value_summary(
            seed_svals,
            rel_rank_threshold=rel_threshold,
            abs_rank_threshold=abs_threshold,
            energy_levels=DEFAULT_ENERGY_LEVELS,
            prefix="seed_span",
        )
    )

    for name, include_transpose in (("forward", False), ("two_sided", True)):
        basis = seed_basis
        dims_by_iter = [seed_dim]
        growth_by_iter: List[int] = []
        converged = False
        for _ in range(max_iters):
            expanded = closure_step(basis, operators, include_transpose=include_transpose)
            new_basis, _ = orthonormal_basis(expanded, rel_threshold=rel_threshold, abs_threshold=abs_threshold)
            new_dim = int(new_basis.shape[1])
            growth = new_dim - int(basis.shape[1])
            dims_by_iter.append(new_dim)
            growth_by_iter.append(growth)
            basis = new_basis
            if growth <= 0:
                converged = True
                break
        result[f"{name}_closure_dim"] = int(basis.shape[1])
        result[f"{name}_closure_fraction_of_n"] = int(basis.shape[1]) / max(model.n, 1)
        result[f"{name}_closure_iters"] = len(growth_by_iter)
        result[f"{name}_closure_converged"] = converged
        result[f"{name}_closure_dims_by_iter"] = dims_by_iter
        result[f"{name}_closure_growth_by_iter"] = growth_by_iter
    return result


def random_problem(addend_digits: int, rng: random.Random) -> tuple[str, str]:
    max_val = (10**addend_digits) - 1
    a = rng.randint(0, max_val)
    b = rng.randint(0, max_val)
    lhs = f"{a:0{addend_digits}d}+{b:0{addend_digits}d}="
    rhs = str(a + b)
    return lhs, rhs


def collect_teacher_forced_states(
    model: Any,
    *,
    addend_digits: int,
    num_samples: int,
    seed: int,
) -> Dict[str, torch.Tensor | int]:
    rng = random.Random(seed)
    prefix_vecs: List[torch.Tensor] = []
    lhs_vecs: List[torch.Tensor] = []
    rhs_vecs: List[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(num_samples):
            lhs, rhs = random_problem(addend_digits, rng)
            target_seq = rhs + "~"

            lhs_ids: List[int] = []
            for ch in lhs:
                lhs_ids.append(model.stoi[ch])
                lhs_vecs.append(model.queried_vector_ids(lhs_ids).detach().to(dtype=torch.float64, device="cpu"))

            full_prefix = lhs
            for i in range(len(target_seq)):
                prefix = full_prefix + target_seq[:i]
                prefix_vecs.append(
                    model.queried_vector_ids(model.encode(prefix)).detach().to(dtype=torch.float64, device="cpu")
                )
                if i > 0:
                    rhs_vecs.append(
                        model.queried_vector_ids(model.encode(prefix)).detach().to(dtype=torch.float64, device="cpu")
                    )

    def stack_or_empty(vectors: Sequence[torch.Tensor], n: int) -> torch.Tensor:
        if not vectors:
            return torch.empty((n, 0), dtype=torch.float64)
        return torch.stack(list(vectors), dim=1)

    return {
        "teacher_forced": stack_or_empty(prefix_vecs, model.n),
        "lhs_only": stack_or_empty(lhs_vecs, model.n),
        "rhs_only": stack_or_empty(rhs_vecs, model.n),
        "num_problems": num_samples,
        "num_teacher_forced_prefixes": len(prefix_vecs),
        "num_lhs_prefixes": len(lhs_vecs),
        "num_rhs_prefixes": len(rhs_vecs),
    }


def summarize_state_span_matrix(
    state_matrix: torch.Tensor,
    *,
    rel_threshold: float,
    abs_threshold: float,
    energy_levels: Sequence[float],
    prefix: str,
) -> Dict[str, Any]:
    svals = torch.linalg.svdvals(state_matrix) if state_matrix.numel() else torch.empty((0,), dtype=torch.float64)
    result: Dict[str, Any] = {
        f"{prefix}_num_states": int(state_matrix.shape[1]),
    }
    result.update(
        singular_value_summary(
            svals,
            rel_rank_threshold=rel_threshold,
            abs_rank_threshold=abs_threshold,
            energy_levels=energy_levels,
            prefix=prefix,
        )
    )
    return result


def empirical_state_span(
    model: Any,
    *,
    addend_digits: int | None,
    num_samples: int,
    seed: int,
    rel_threshold: float,
    abs_threshold: float,
    energy_levels: Sequence[float],
) -> Dict[str, Any]:
    if addend_digits is None:
        raise ValueError("Checkpoint is missing addend_digits; empirical state-span analysis requires it")
    collected = collect_teacher_forced_states(
        model,
        addend_digits=addend_digits,
        num_samples=num_samples,
        seed=seed,
    )
    teacher_forced = collected["teacher_forced"]
    lhs_only = collected["lhs_only"]
    rhs_only = collected["rhs_only"]
    if not isinstance(teacher_forced, torch.Tensor) or not isinstance(lhs_only, torch.Tensor) or not isinstance(rhs_only, torch.Tensor):
        raise TypeError("Collected state matrices must be tensors")

    centered_teacher = teacher_forced - teacher_forced.mean(dim=1, keepdim=True) if teacher_forced.shape[1] else teacher_forced
    centered_lhs = lhs_only - lhs_only.mean(dim=1, keepdim=True) if lhs_only.shape[1] else lhs_only
    centered_rhs = rhs_only - rhs_only.mean(dim=1, keepdim=True) if rhs_only.shape[1] else rhs_only

    result: Dict[str, Any] = {
        "num_problems": int(collected["num_problems"]),
        "num_teacher_forced_prefixes": int(collected["num_teacher_forced_prefixes"]),
        "num_lhs_prefixes": int(collected["num_lhs_prefixes"]),
        "num_rhs_prefixes": int(collected["num_rhs_prefixes"]),
    }
    result.update(
        summarize_state_span_matrix(
            teacher_forced,
            rel_threshold=rel_threshold,
            abs_threshold=abs_threshold,
            energy_levels=energy_levels,
            prefix="teacher_forced_state_span",
        )
    )
    result.update(
        summarize_state_span_matrix(
            centered_teacher,
            rel_threshold=rel_threshold,
            abs_threshold=abs_threshold,
            energy_levels=energy_levels,
            prefix="teacher_forced_centered_state_span",
        )
    )
    result.update(
        summarize_state_span_matrix(
            lhs_only,
            rel_threshold=rel_threshold,
            abs_threshold=abs_threshold,
            energy_levels=energy_levels,
            prefix="lhs_state_span",
        )
    )
    result.update(
        summarize_state_span_matrix(
            centered_lhs,
            rel_threshold=rel_threshold,
            abs_threshold=abs_threshold,
            energy_levels=energy_levels,
            prefix="lhs_centered_state_span",
        )
    )
    result.update(
        summarize_state_span_matrix(
            rhs_only,
            rel_threshold=rel_threshold,
            abs_threshold=abs_threshold,
            energy_levels=energy_levels,
            prefix="rhs_state_span",
        )
    )
    result.update(
        summarize_state_span_matrix(
            centered_rhs,
            rel_threshold=rel_threshold,
            abs_threshold=abs_threshold,
            energy_levels=energy_levels,
            prefix="rhs_centered_state_span",
        )
    )
    return result


def analyze_polar_factor(
    matrix: torch.Tensor,
    *,
    rel_rank_threshold: float,
    abs_rank_threshold: float,
    energy_levels: Sequence[float],
    angle_threshold_deg: float,
) -> Dict[str, float | int | List[float]]:
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    q = u @ vh
    eye = identity_like(q)
    q_delta = q - eye
    q_delta_svals = torch.linalg.svdvals(q_delta)
    mtm = matrix.transpose(-1, -2) @ matrix
    orth_error = torch.linalg.matrix_norm(mtm - eye, ord="fro") / math.sqrt(matrix.shape[-1])
    polar_residual = torch.linalg.matrix_norm(matrix - q, ord="fro") / max(
        to_float(torch.linalg.matrix_norm(matrix, ord="fro")),
        EPS,
    )
    sign, logabsdet = torch.linalg.slogdet(q)
    eigvals = torch.linalg.eigvals(q)
    angles = torch.angle(eigvals).abs()
    angle_threshold_rad = angle_threshold_deg * math.pi / 180.0
    nontrivial_angle_count = int((angles > angle_threshold_rad).sum().item())
    nontrivial_angles = angles[angles > angle_threshold_rad]
    max_angle_deg = float((angles.max().item() * 180.0 / math.pi) if angles.numel() else 0.0)
    mean_nontrivial_angle_deg = (
        float(nontrivial_angles.mean().item() * 180.0 / math.pi) if nontrivial_angles.numel() else 0.0
    )

    result: Dict[str, float | int | List[float]] = {
        "polar_det_sign": to_float(sign),
        "polar_logabsdet": to_float(logabsdet),
        "orthogonality_error_fro_per_sqrt_n": to_float(orth_error),
        "polar_relative_residual": to_float(polar_residual),
        "polar_max_angle_deg": max_angle_deg,
        "polar_mean_nontrivial_angle_deg": mean_nontrivial_angle_deg,
        "polar_nontrivial_eigenvalue_count": nontrivial_angle_count,
    }
    result.update(
        singular_value_summary(
            q_delta_svals,
            rel_rank_threshold=rel_rank_threshold,
            abs_rank_threshold=abs_rank_threshold,
            energy_levels=energy_levels,
            prefix="polar_rotation_support",
        )
    )
    return result


def expand_matrix(
    model: Any,
    *,
    component: str,
    token_id: int | None = None,
) -> torch.Tensor:
    if component == "base":
        return model.base_matrix().detach().to(dtype=torch.float64, device="cpu")
    if token_id is None:
        raise ValueError("token_id is required for token components")
    return model.token_matrix(token_id).detach().to(dtype=torch.float64, device="cpu")


def parameterization_details(model: Any, token_id: int | None = None) -> Dict[str, float | int | None]:
    details: Dict[str, float | int | None] = {
        "parameter_rank_budget": model.token_rank if model.token_mode != "dense" else None,
    }
    if token_id is None:
        return details

    if model.token_mode == "dense":
        return details

    if model.token_mode == "lowrank_ab":
        a = model.token_a[token_id].detach().to(dtype=torch.float64, device="cpu")
        b = model.token_b[token_id].detach().to(dtype=torch.float64, device="cpu")
        a_svals = torch.linalg.svdvals(a)
        b_svals = torch.linalg.svdvals(b)
        details.update(
            {
                "factor_a_rank_abs": int((a_svals > 1e-6).sum().item()),
                "factor_b_rank_abs": int((b_svals > 1e-6).sum().item()),
                "factor_a_stable_rank": stable_rank(a_svals),
                "factor_b_stable_rank": stable_rank(b_svals),
            }
        )
        return details

    u = model.token_u[token_id].detach().to(dtype=torch.float64, device="cpu")
    r = model.token_r[token_id].detach().to(dtype=torch.float64, device="cpu")
    u_svals = torch.linalg.svdvals(u)
    r_delta_svals = torch.linalg.svdvals(r - torch.eye(r.shape[-1], dtype=r.dtype))
    details.update(
        {
            "factor_u_rank_abs": int((u_svals > 1e-6).sum().item()),
            "factor_u_stable_rank": stable_rank(u_svals),
            "core_update_rank_abs": int((r_delta_svals > 1e-6).sum().item()),
            "core_update_stable_rank": stable_rank(r_delta_svals),
        }
    )
    return details


def analyze_matrix(
    matrix: torch.Tensor,
    *,
    rel_rank_threshold: float,
    abs_rank_threshold: float,
    energy_levels: Sequence[float],
    angle_threshold_deg: float,
) -> Dict[str, float | int | List[float]]:
    matrix = matrix.detach().to(dtype=torch.float64, device="cpu")
    eye = identity_like(matrix)
    delta = matrix - eye
    delta_svals = torch.linalg.svdvals(delta)
    skew_delta = 0.5 * (delta - delta.transpose(-1, -2))
    sym_delta = 0.5 * (delta + delta.transpose(-1, -2))
    result: Dict[str, float | int | List[float]] = {
        "matrix_dim": int(matrix.shape[-1]),
        "update_fro_norm": to_float(torch.linalg.matrix_norm(delta, ord="fro")),
        "skew_update_fro_norm": to_float(torch.linalg.matrix_norm(skew_delta, ord="fro")),
        "symmetric_update_fro_norm": to_float(torch.linalg.matrix_norm(sym_delta, ord="fro")),
        "matrix_fro_norm": to_float(torch.linalg.matrix_norm(matrix, ord="fro")),
        "matrix_spectral_norm": to_float(torch.linalg.matrix_norm(matrix, ord=2)),
        "trace": to_float(torch.trace(matrix)),
    }
    result.update(
        singular_value_summary(
            delta_svals,
            rel_rank_threshold=rel_rank_threshold,
            abs_rank_threshold=abs_rank_threshold,
            energy_levels=energy_levels,
            prefix="update",
        )
    )
    result.update(
        singular_value_summary(
            torch.linalg.svdvals(skew_delta),
            rel_rank_threshold=rel_rank_threshold,
            abs_rank_threshold=abs_rank_threshold,
            energy_levels=energy_levels,
            prefix="skew_update",
        )
    )
    result.update(
        singular_value_summary(
            torch.linalg.svdvals(sym_delta),
            rel_rank_threshold=rel_rank_threshold,
            abs_rank_threshold=abs_rank_threshold,
            energy_levels=energy_levels,
            prefix="symmetric_update",
        )
    )
    result.update(
        analyze_polar_factor(
            matrix,
            rel_rank_threshold=rel_rank_threshold,
            abs_rank_threshold=abs_rank_threshold,
            energy_levels=energy_levels,
            angle_threshold_deg=angle_threshold_deg,
        )
    )
    return result


def summarize_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not rows:
        return summary
    for key in keys:
        values = [float(row[key]) for row in rows]
        summary[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "median": sorted(values)[len(values) // 2] if len(values) % 2 == 1 else (
                sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]
            ) / 2.0,
        }
    return summary


def token_label(model: Any, token_id: int) -> str:
    raw = model.itos[token_id]
    if raw == "~":
        return "<eos>"
    return raw


def analyze_checkpoint(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    device = pick_device(args.device)
    model, addend_digits = load_checkpoint(str(path), device)
    model.eval()

    checkpoint_result: Dict[str, Any] = {
        "path": str(path),
        "metadata": {
            "n": model.n,
            "base_mode": model.base_mode,
            "token_mode": model.token_mode,
            "token_rank": model.token_rank,
            "addend_digits": addend_digits,
            "vocab_size": model.vocab_size,
        },
        "active_subspace": active_subspace_closure(
            model,
            rel_threshold=args.closure_rel_threshold,
            abs_threshold=args.closure_abs_threshold,
            max_iters=args.closure_max_iters,
        ),
        "empirical_state_span": empirical_state_span(
            model,
            addend_digits=addend_digits,
            num_samples=args.state_samples,
            seed=args.state_seed,
            rel_threshold=args.rel_rank_threshold,
            abs_threshold=args.abs_rank_threshold,
            energy_levels=args.energy_levels,
        ),
    }

    if args.component in {"all", "base"}:
        base_matrix = expand_matrix(model, component="base")
        checkpoint_result["base"] = analyze_matrix(
            base_matrix,
            rel_rank_threshold=args.rel_rank_threshold,
            abs_rank_threshold=args.abs_rank_threshold,
            energy_levels=args.energy_levels,
            angle_threshold_deg=args.angle_threshold_deg,
        )

    if args.component in {"all", "tokens"}:
        token_rows: List[Dict[str, Any]] = []
        for token_id in range(model.vocab_size):
            matrix = expand_matrix(model, component="token", token_id=token_id)
            row: Dict[str, Any] = {
                "token_id": token_id,
                "token": token_label(model, token_id),
            }
            row.update(parameterization_details(model, token_id=token_id))
            row.update(
                analyze_matrix(
                    matrix,
                    rel_rank_threshold=args.rel_rank_threshold,
                    abs_rank_threshold=args.abs_rank_threshold,
                    energy_levels=args.energy_levels,
                    angle_threshold_deg=args.angle_threshold_deg,
                )
            )
            token_rows.append(row)

        summary_keys = [
            "update_fro_norm",
            "update_stable_rank",
            "update_effective_rank",
            "update_rank_rel",
            "skew_update_fro_norm",
            "skew_update_stable_rank",
            "skew_update_effective_rank",
            "skew_update_rank_rel",
            "polar_rotation_support_stable_rank",
            "polar_rotation_support_effective_rank",
            "polar_rotation_support_rank_rel",
            "orthogonality_error_fro_per_sqrt_n",
            "polar_relative_residual",
            "polar_max_angle_deg",
        ]
        checkpoint_result["token_summary"] = summarize_rows(token_rows, summary_keys)
        checkpoint_result["tokens"] = token_rows

    return checkpoint_result


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_base_summary(base: Dict[str, Any]) -> None:
    print("base:")
    print(
        "  "
        f"update_eff_rank={fmt(base['update_effective_rank'])} "
        f"update_rank_rel={fmt(base['update_rank_rel'])} "
        f"skew_eff_rank={fmt(base['skew_update_effective_rank'])} "
        f"skew_rank_rel={fmt(base['skew_update_rank_rel'])} "
        f"polar_rot_eff={fmt(base['polar_rotation_support_effective_rank'])} "
        f"update_norm={fmt(base['update_fro_norm'])} "
        f"ortho_err={fmt(base['orthogonality_error_fro_per_sqrt_n'])}"
    )
    print(
        "  "
        f"polar_max_angle_deg={fmt(base['polar_max_angle_deg'])} "
        f"polar_nontrivial_eigs={fmt(base['polar_nontrivial_eigenvalue_count'])} "
        f"polar_residual={fmt(base['polar_relative_residual'])} "
        f"det_sign={fmt(base['polar_det_sign'])}"
    )


def print_active_subspace_summary(active_subspace: Dict[str, Any]) -> None:
    print("active_subspace:")
    print(
        "  "
        f"seed_dim={fmt(active_subspace['seed_dim'])} "
        f"seed_frac={fmt(active_subspace['seed_fraction_of_n'])} "
        f"forward_dim={fmt(active_subspace['forward_closure_dim'])} "
        f"forward_frac={fmt(active_subspace['forward_closure_fraction_of_n'])}"
    )
    print(
        "  "
        f"two_sided_dim={fmt(active_subspace['two_sided_closure_dim'])} "
        f"two_sided_frac={fmt(active_subspace['two_sided_closure_fraction_of_n'])} "
        f"seed_eff_rank={fmt(active_subspace['seed_span_effective_rank'])}"
    )
    print(
        "  "
        f"forward_dims={active_subspace['forward_closure_dims_by_iter']} "
        f"two_sided_dims={active_subspace['two_sided_closure_dims_by_iter']}"
    )


def print_empirical_state_span_summary(empirical_state_span: Dict[str, Any], n: int) -> None:
    print("empirical_state_span:")
    print(
        "  "
        f"problems={empirical_state_span['num_problems']} "
        f"tf_prefixes={empirical_state_span['num_teacher_forced_prefixes']} "
        f"lhs_prefixes={empirical_state_span['num_lhs_prefixes']} "
        f"rhs_prefixes={empirical_state_span['num_rhs_prefixes']}"
    )
    print(
        "  "
        f"tf_eff={fmt(empirical_state_span['teacher_forced_state_span_effective_rank'])} "
        f"tf_rel={fmt(empirical_state_span['teacher_forced_state_span_rank_rel'])}/{n} "
        f"tf_centered_eff={fmt(empirical_state_span['teacher_forced_centered_state_span_effective_rank'])} "
        f"tf_centered_rel={fmt(empirical_state_span['teacher_forced_centered_state_span_rank_rel'])}/{n}"
    )
    print(
        "  "
        f"lhs_centered_eff={fmt(empirical_state_span['lhs_centered_state_span_effective_rank'])} "
        f"lhs_centered_rel={fmt(empirical_state_span['lhs_centered_state_span_rank_rel'])}/{n} "
        f"rhs_centered_eff={fmt(empirical_state_span['rhs_centered_state_span_effective_rank'])} "
        f"rhs_centered_rel={fmt(empirical_state_span['rhs_centered_state_span_rank_rel'])}/{n}"
    )


def print_token_summary(token_rows: Sequence[Dict[str, Any]], token_summary: Dict[str, Dict[str, float]], top_k: int, show_all: bool) -> None:
    print("tokens:")
    print(
        "  "
        f"skew_eff_rank mean={fmt(token_summary['skew_update_effective_rank']['mean'])} "
        f"median={fmt(token_summary['skew_update_effective_rank']['median'])} "
        f"max={fmt(token_summary['skew_update_effective_rank']['max'])}"
    )
    print(
        "  "
        f"update_eff_rank mean={fmt(token_summary['update_effective_rank']['mean'])} "
        f"median={fmt(token_summary['update_effective_rank']['median'])} "
        f"max={fmt(token_summary['update_effective_rank']['max'])}"
    )
    print(
        "  "
        f"polar_rot_eff mean={fmt(token_summary['polar_rotation_support_effective_rank']['mean'])} "
        f"median={fmt(token_summary['polar_rotation_support_effective_rank']['median'])} "
        f"max={fmt(token_summary['polar_rotation_support_effective_rank']['max'])}"
    )
    print(
        "  "
        f"ortho_err mean={fmt(token_summary['orthogonality_error_fro_per_sqrt_n']['mean'])} "
        f"polar_residual mean={fmt(token_summary['polar_relative_residual']['mean'])} "
        f"max_angle mean={fmt(token_summary['polar_max_angle_deg']['mean'])}"
    )

    ranked = sorted(
        token_rows,
        key=lambda row: (
            float(row["skew_update_effective_rank"]),
            float(row["update_effective_rank"]),
            float(row["update_fro_norm"]),
        ),
        reverse=True,
    )
    selected = ranked if show_all else ranked[:top_k]
    label = "all token rows" if show_all else f"top {min(top_k, len(ranked))} tokens by skew-update rank"
    print(f"  {label}:")
    for row in selected:
        budget = row.get("parameter_rank_budget")
        budget_txt = "-" if budget is None else str(budget)
        print(
            "    "
            f"token={row['token']!r} id={row['token_id']:>2d} "
            f"budget={budget_txt} "
            f"update_eff={fmt(row['update_effective_rank'])} "
            f"update_rel={fmt(row['update_rank_rel'])} "
            f"skew_eff={fmt(row['skew_update_effective_rank'])} "
            f"skew_rel={fmt(row['skew_update_rank_rel'])} "
            f"polar_rot_eff={fmt(row['polar_rotation_support_effective_rank'])} "
            f"norm={fmt(row['update_fro_norm'])} "
            f"ortho_err={fmt(row['orthogonality_error_fro_per_sqrt_n'])}"
        )


def main() -> None:
    args = parse_args()
    if any(level <= 0.0 or level > 1.0 for level in args.energy_levels):
        raise ValueError("--energy-levels must lie in (0, 1]")
    paths = resolve_checkpoint_paths(args.inputs)
    analyses = [analyze_checkpoint(path, args) for path in paths]

    for analysis in analyses:
        meta = analysis["metadata"]
        print(f"\n== {analysis['path']} ==")
        print(
            f"n={meta['n']} base_mode={meta['base_mode']} token_mode={meta['token_mode']} "
            f"token_rank={meta['token_rank']} addend_digits={meta['addend_digits']} vocab_size={meta['vocab_size']}"
        )
        print_active_subspace_summary(analysis["active_subspace"])
        print_empirical_state_span_summary(analysis["empirical_state_span"], n=int(meta["n"]))
        if "base" in analysis:
            print_base_summary(analysis["base"])
        if "tokens" in analysis:
            print_token_summary(
                analysis["tokens"],
                analysis["token_summary"],
                top_k=args.top_k,
                show_all=args.show_all_tokens,
            )

    if args.json_out is not None:
        out_path = Path(args.json_out)
        if out_path.parent:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"checkpoints": analyses}, indent=2))
        print(f"\njson_out={out_path}")


if __name__ == "__main__":
    main()
