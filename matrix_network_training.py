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
    targets: torch.Tensor,
) -> torch.Tensor:
    context_len = len(context_ids)
    rows = torch.empty(
        (len(targets), context_len + 1, model.n),
        device=model.base_mat.device,
        dtype=model.base_mat.dtype,
    )
    rows[:, 0] = targets @ model.base_mat.T
    active_rows = rows[:, 0].clone()
    for token_pos, token_id in enumerate(context_ids):
        if token_pos + 1 >= len(targets):
            break
        active_rows[token_pos + 1 :] = (
            active_rows[token_pos + 1 :] @ model.token_mats[token_id].T
        )
        rows[token_pos + 1 :, token_pos + 1] = active_rows[token_pos + 1 :]
    return rows


@torch.no_grad()
def sequence_update_terms(
    model: MatrixNetwork,
    token_ids: Sequence[int],
    target_noise: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    base_update_terms = torch.zeros_like(model.base_mat)
    token_update_terms = torch.zeros_like(model.token_mats)

    context_ids = token_ids[:-1]
    query_triangle_rows = _query_triangle_rows(model, context_ids)

    target_ids = torch.tensor(token_ids, device=model.base_mat.device, dtype=torch.long)
    targets = model.unembed_vectors[target_ids]
    if target_noise > 0.0:
        noise = torch.randn_like(targets) / (model.n ** 0.5)
        targets = targets + noise * target_noise
        targets = targets / targets.norm(dim=1, keepdim=True).clamp_min(1e-12)
    target_triangle_rows = _target_triangle_rows(model, context_ids, targets)

    # Base learns target @ base.T -> q @ prefix.
    base_update_terms.add_(query_triangle_rows[:, 0].T @ target_triangle_rows[:, 0])

    for token_pos, token_id in enumerate(context_ids):
        later_rows_for_token = query_triangle_rows[token_pos + 1 :, token_pos + 1]
        target_rows_for_token = target_triangle_rows[token_pos + 1 :, token_pos + 1]
        # Each token learns target @ base.T @ earlier.T -> q @ later.
        token_update_terms[token_id].add_(
            later_rows_for_token.T @ target_rows_for_token
        )

    return base_update_terms, token_update_terms


@torch.no_grad()
def apply_sequence_update(
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    token_ids: Sequence[int],
    target_noise: float,
) -> None:
    base_update_terms, token_update_terms = sequence_update_terms(
        model, token_ids, target_noise
    )
    optimizer.step(base_update_terms, token_update_terms)


@torch.no_grad()
def apply_batch_update(
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    sequences: Sequence[Sequence[int]],
    target_noise: float,
) -> None:
    base_update_terms = torch.zeros_like(model.base_mat)
    token_update_terms = torch.zeros_like(model.token_mats)

    for token_ids in sequences:
        sequence_base_terms, sequence_token_terms = sequence_update_terms(
            model, token_ids, target_noise
        )
        base_update_terms.add_(sequence_base_terms)
        token_update_terms.add_(sequence_token_terms)

    optimizer.step(base_update_terms, token_update_terms)


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
        sequences, _prompt_lens = sample_batch()
        apply_batch_update(
            model, optimizer, sequences,
            target_noise=target_noise,
        )
        if it == 1 or it % log_every == 0:
            print(f"iter={it:5d}")
        if evaluate is not None and eval_every > 0 and (it % eval_every == 0 or it == iters):
            evaluate(model, it)
        if on_checkpoint is not None and checkpoint_every > 0 and it % checkpoint_every == 0:
            on_checkpoint(it, model, optimizer)
