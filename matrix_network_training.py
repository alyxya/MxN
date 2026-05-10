#!/usr/bin/env python3
from typing import Callable, List, Sequence, Tuple

import torch

from matrix_network import MatrixNetwork
from matrix_network_optimizer import MatrixNetworkOptimizer


@torch.no_grad()
def _query_triangle_rows(
    model: MatrixNetwork,
    context_ids: Sequence[int],
) -> torch.Tensor:
    context_len = len(context_ids)
    rows = torch.empty(
        (context_len + 1, context_len + 1, model.n),
        device=model.base_mat.device,
        dtype=model.base_mat.dtype,
    )
    diagonal = torch.arange(context_len + 1, device=model.base_mat.device)
    rows[diagonal, diagonal] = model.query
    active_rows = torch.empty(
        (context_len, model.n),
        device=model.base_mat.device,
        dtype=model.base_mat.dtype,
    )
    ends = torch.arange(context_len, 0, -1, device=model.base_mat.device)
    active_count = 0

    for token_pos in range(context_len - 1, -1, -1):
        active_rows[active_count].copy_(model.query)
        active_count += 1
        active_rows[:active_count] = active_rows[:active_count] @ model.token_mats[
            context_ids[token_pos]
        ]
        rows[ends[:active_count], token_pos] = active_rows[:active_count]
    return rows


@torch.no_grad()
def _target_triangle_rows(
    model: MatrixNetwork,
    context_ids: Sequence[int],
    target_positions: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    context_len = len(context_ids)
    base_rows = targets @ model.base_mat.T
    rows = torch.empty(
        (len(target_positions), context_len, model.n),
        device=model.base_mat.device,
        dtype=model.base_mat.dtype,
    )
    if context_len == 0 or len(target_positions) == 0:
        return base_rows, rows

    active_indices = torch.arange(len(target_positions), device=model.base_mat.device)
    active_positions = target_positions
    active_rows = base_rows
    for token_pos, token_id in enumerate(context_ids):
        keep = active_positions > token_pos
        if not bool(keep.any().item()):
            break
        active_indices = active_indices[keep]
        active_positions = active_positions[keep]
        active_rows = active_rows[keep] @ model.token_mats[token_id].T
        rows[active_indices, token_pos] = active_rows
    return base_rows, rows


@torch.no_grad()
def sequence_update_terms(
    model: MatrixNetwork,
    token_ids: Sequence[int],
    prompt_len: int,
    target_noise: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
    base_update_terms = torch.zeros_like(model.base_mat)
    token_update_terms = torch.zeros_like(model.token_mats)
    total = 0
    target_score_sum = 0.0

    prediction_positions = list(range(prompt_len, len(token_ids)))
    context_ids = token_ids[:-1]
    query_rows = _query_triangle_rows(model, context_ids)
    prefix_rows = query_rows[prediction_positions, 0]
    states = prefix_rows @ model.base_mat
    score_rows = states @ model.unembed_vectors.T

    for row, pos in zip(score_rows, prediction_positions):
        target_id = token_ids[pos]
        total += 1
        target_score_sum += float(row[target_id].item())

    if not prediction_positions:
        return base_update_terms, token_update_terms, target_score_sum, total

    target_ids = torch.tensor(
        [token_ids[pos] for pos in prediction_positions],
        device=model.base_mat.device,
    )
    targets = model.unembed_vectors[target_ids]
    if target_noise > 0.0:
        targets = targets + torch.randn_like(targets) * target_noise
        targets = targets / targets.norm(dim=1, keepdim=True).clamp_min(1e-12)
    target_positions = torch.tensor(prediction_positions, device=model.base_mat.device)
    base_target_rows, token_target_rows = _target_triangle_rows(
        model,
        context_ids,
        target_positions,
        targets,
    )
    # Base learns target @ base.T -> q @ prefix.
    base_update_terms.add_(prefix_rows.T @ base_target_rows)

    for token_pos, token_id in enumerate(context_ids):
        keep = target_positions > token_pos
        if not bool(keep.any().item()):
            break
        later_rows_for_token = query_rows[target_positions[keep], token_pos + 1]
        target_rows_for_token = token_target_rows[keep, token_pos]
        # Each token learns target @ base.T @ earlier.T -> q @ later.
        token_update_terms[token_id].add_(
            later_rows_for_token.T @ target_rows_for_token
        )

    return base_update_terms, token_update_terms, target_score_sum, total


@torch.no_grad()
def apply_sequence_update(
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    token_ids: Sequence[int],
    prompt_len: int,
    target_noise: float,
) -> float:
    base_update_terms, token_update_terms, target_score_sum, total = (
        sequence_update_terms(model, token_ids, prompt_len, target_noise)
    )
    if total > 0:
        base_update_terms = base_update_terms / total
        token_update_terms = token_update_terms / total
    optimizer.step(base_update_terms, token_update_terms)

    return target_score_sum / max(total, 1)


@torch.no_grad()
def apply_batch_update(
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    sequences: Sequence[Sequence[int]],
    prompt_lens: Sequence[int],
    target_noise: float,
) -> float:
    base_update_terms = torch.zeros_like(model.base_mat)
    token_update_terms = torch.zeros_like(model.token_mats)
    total = 0
    target_score_sum = 0.0

    for token_ids, prompt_len in zip(sequences, prompt_lens):
        (
            sequence_base_terms,
            sequence_token_terms,
            sequence_target_score_sum,
            sequence_total,
        ) = sequence_update_terms(model, token_ids, prompt_len, target_noise)
        base_update_terms.add_(sequence_base_terms)
        token_update_terms.add_(sequence_token_terms)
        target_score_sum += sequence_target_score_sum
        total += sequence_total

    if total > 0:
        base_update_terms = base_update_terms / total
        token_update_terms = token_update_terms / total
    optimizer.step(base_update_terms, token_update_terms)

    return target_score_sum / max(total, 1)


def train(
    *,
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    sample_batch: Callable[[], Tuple[List[List[int]], List[int]]],
    iters: int,
    target_noise: float,
    log_every: int,
    eval_every: int = 0,
    evaluate: Callable[[MatrixNetwork, int], None] | None = None,
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[int, MatrixNetwork, MatrixNetworkOptimizer], None] | None = None,
) -> None:
    for it in range(1, iters + 1):
        sequences, prompt_lens = sample_batch()
        score = apply_batch_update(
            model, optimizer, sequences, prompt_lens,
            target_noise=target_noise,
        )
        if it == 1 or it % log_every == 0:
            print(f"iter={it:5d} mean_target_score={score:.4f}")
        if evaluate is not None and eval_every > 0 and (it % eval_every == 0 or it == iters):
            evaluate(model, it)
        if on_checkpoint is not None and checkpoint_every > 0 and it % checkpoint_every == 0:
            on_checkpoint(it, model, optimizer)
