#!/usr/bin/env python3
from typing import Callable, List, Sequence, Tuple

import torch

from matrix_network import MatrixNetwork
from matrix_network_optimizer import MatrixNetworkOptimizer


def _normalize_rows(rows: torch.Tensor) -> torch.Tensor:
    return rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def _query_triangle_rows(
    model: MatrixNetwork,
    context_ids: Sequence[int],
) -> torch.Tensor:
    context_len = len(context_ids)
    rows = torch.zeros(
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
    rows = torch.zeros(
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
def _target_states(
    model: MatrixNetwork,
    states: torch.Tensor,
    token_ids: torch.Tensor,
    correct_margin: float | None,
    decode_norm_pressure: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_mask = torch.ones(len(token_ids), device=states.device, dtype=torch.bool)
    targets = states.clone()
    decode_scale = (model.vocab_size / model.n) ** 0.5
    decode_norms = targets[:, : model.vocab_size].norm(dim=1, keepdim=True)
    target_decode_norms = decode_norms * (1.0 - decode_norm_pressure) + decode_scale * decode_norm_pressure

    if correct_margin is None:
        targets[:, : model.vocab_size] = 0.0
        targets[
            torch.arange(len(token_ids), device=states.device),
            token_ids,
        ] = target_decode_norms[:, 0]
        if decode_norm_pressure == 0.0:
            return targets, train_mask
        return _normalize_rows(targets), train_mask

    scores = states @ model.unembed_vectors.T
    correct_scores = scores.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    wrong_scores = scores.clone()
    wrong_scores.scatter_(1, token_ids.unsqueeze(1), -torch.inf)
    missing_margin = (wrong_scores.max(dim=1).values + correct_margin - correct_scores).clamp_min(0.0)
    train_mask = missing_margin > 0.0

    decode_dims = targets[:, : model.vocab_size].clone()
    decode_dims[
        torch.arange(len(token_ids), device=states.device),
        token_ids,
    ] += missing_margin
    targets[:, : model.vocab_size] = _normalize_rows(decode_dims) * target_decode_norms
    if decode_norm_pressure == 0.0:
        return targets, train_mask
    return _normalize_rows(targets), train_mask


@torch.no_grad()
def apply_batch_update(
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    sequences: Sequence[Sequence[int]],
    target_starts: Sequence[int],
    recency_decay: float,
    correct_margin: float | None = None,
    decode_norm_pressure: float = 1.0,
) -> None:
    base_update_terms = torch.zeros_like(model.base_mat)
    token_update_terms = torch.zeros_like(model.token_mats)
    trained_output_count = 0
    max_contribution_mass = 0.0
    has_update = False

    for token_ids, target_start in zip(sequences, target_starts):
        target_count = len(token_ids) - target_start
        if target_count <= 0:
            continue
        trained_output_count += target_count
        terms = len(token_ids)
        if recency_decay == 1.0:
            contribution_mass = float(terms)
        else:
            contribution_mass = (1.0 - recency_decay ** terms) / (1.0 - recency_decay)
        max_contribution_mass = max(max_contribution_mass, contribution_mass)
        context_ids = token_ids[:-1]
        query_triangle_rows = _query_triangle_rows(model, context_ids)

        token_id_tensor = torch.tensor(
            token_ids,
            device=model.base_mat.device,
            dtype=torch.long,
        )
        states = query_triangle_rows[:, 0] @ model.base_mat
        train_mask = torch.arange(len(token_ids), device=model.base_mat.device) >= target_start
        targets, target_mask = _target_states(
            model,
            states,
            token_id_tensor,
            correct_margin,
            decode_norm_pressure,
        )
        train_mask &= target_mask
        if not bool(train_mask.any().item()):
            continue
        has_update = True

        target_triangle_rows = _target_triangle_rows(model, context_ids, targets)
        positions = torch.arange(len(token_ids), device=model.base_mat.device)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        decay = torch.tensor(recency_decay, device=model.base_mat.device, dtype=model.base_mat.dtype)
        update_weights = torch.tril(torch.pow(decay, distances.clamp_min(0)))
        update_weights[~train_mask] = 0.0

        position_updates = torch.bmm(
            query_triangle_rows.permute(1, 2, 0),
            (target_triangle_rows * update_weights.unsqueeze(2)).permute(1, 0, 2),
        )
        base_update_terms.add_(position_updates[0])
        token_update_terms.index_add_(0, token_id_tensor[:-1], position_updates[1:])

    if trained_output_count == 0 or not has_update:
        return

    batch_update_scale = 1.0 / (trained_output_count * max_contribution_mass)
    base_update_terms.mul_(batch_update_scale)
    token_update_terms.mul_(batch_update_scale)
    optimizer.step(base_update_terms, token_update_terms)


def train(
    *,
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    sample_batch: Callable[[], Tuple[List[List[int]], List[int]]],
    iters: int,
    recency_decay: float,
    correct_margin: float | None = None,
    decode_norm_pressure: float = 1.0,
    eval_every: int = 0,
    evaluate: Callable[[MatrixNetwork, int], None] | None = None,
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[MatrixNetwork, MatrixNetworkOptimizer, int], None] | None = None,
) -> None:
    for it in range(1, iters + 1):
        sequences, prompt_lens = sample_batch()
        apply_batch_update(
            model, optimizer, sequences, prompt_lens,
            recency_decay=recency_decay,
            correct_margin=correct_margin,
            decode_norm_pressure=decode_norm_pressure,
        )
        if evaluate is not None and eval_every > 0 and (it % eval_every == 0 or it == iters):
            evaluate(model, it)
        if on_checkpoint is not None and checkpoint_every > 0 and it % checkpoint_every == 0:
            on_checkpoint(model, optimizer, it)
