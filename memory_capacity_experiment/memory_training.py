#!/usr/bin/env python3
from typing import Callable, List, Sequence, Tuple

import torch

from memory_matrix_network import MemoryMatrixNetwork
from memory_optimizer import MemoryMatrixNetworkOptimizer


def _row_outer(input_row: torch.Tensor, target_row: torch.Tensor) -> torch.Tensor:
    return input_row.unsqueeze(1) @ target_row.unsqueeze(0)


@torch.no_grad()
def _left_state_matrix(model: MemoryMatrixNetwork, context_ids: Sequence[int]) -> torch.Tensor:
    state_mat = model.base_mat
    for token_id in context_ids:
        state_mat = model.token_mats[token_id] @ state_mat
    return state_mat


@torch.no_grad()
def _right_state_matrix(model: MemoryMatrixNetwork, context_ids: Sequence[int]) -> torch.Tensor:
    state_mat = model.base_mat
    for token_id in context_ids:
        state_mat = state_mat @ model.token_mats[token_id]
    return state_mat


@torch.no_grad()
def _left_update_sequence(
    model: MemoryMatrixNetwork,
    token_ids: Sequence[int],
    target_start: int,
    recency_decay: float,
    correct_margin: float | None,
) -> Tuple[torch.Tensor, torch.Tensor, int, float, bool]:
    return _left_update_sequence_from_query(
        model,
        token_ids,
        target_start,
        recency_decay,
        correct_margin,
        use_double_query=False,
    )


@torch.no_grad()
def _double_query_update_sequence(
    model: MemoryMatrixNetwork,
    token_ids: Sequence[int],
    target_start: int,
    recency_decay: float,
    correct_margin: float | None,
) -> Tuple[torch.Tensor, torch.Tensor, int, float, bool]:
    return _left_update_sequence_from_query(
        model,
        token_ids,
        target_start,
        recency_decay,
        correct_margin,
        use_double_query=True,
    )


@torch.no_grad()
def _left_update_sequence_from_query(
    model: MemoryMatrixNetwork,
    token_ids: Sequence[int],
    target_start: int,
    recency_decay: float,
    correct_margin: float | None,
    *,
    use_double_query: bool,
) -> Tuple[torch.Tensor, torch.Tensor, int, float, bool]:
    base_update_terms = torch.zeros_like(model.base_mat)
    token_update_terms = torch.zeros_like(model.token_mats)
    trained_output_count = max(len(token_ids) - target_start, 0)
    max_contribution_mass = 0.0
    has_update = False

    for target_pos in range(target_start, len(token_ids)):
        context_ids = token_ids[:target_pos]
        target_id = token_ids[target_pos]
        contribution_mass = _contribution_mass(len(token_ids), recency_decay)
        max_contribution_mass = max(max_contribution_mass, contribution_mass)

        state_mat = _left_state_matrix(model, context_ids)
        query_row = state_mat[0] if use_double_query else model.query
        state = query_row @ state_mat
        row_scale = _row_scale(model, state, target_id, correct_margin)
        if row_scale <= 0.0:
            continue
        has_update = True
        target = model.unembed_vectors[target_id]

        # In double-query mode, the first query output becomes the row query for
        # this second pass. This update intentionally learns only the second pass.
        later_input = query_row
        later_rows: List[torch.Tensor] = []
        for token_id in reversed(context_ids):
            later_rows.append(later_input)
            later_input = later_input @ model.token_mats[token_id]
        base_input = later_input
        base_desired = target @ model.base_mat.T
        base_weight = row_scale * (recency_decay ** target_pos)
        base_update_terms.add_(base_weight * _row_outer(base_input, base_desired))

        prefix_mat = model.base_mat
        desired_rows: List[torch.Tensor] = []
        for token_id in context_ids:
            prefix_mat = model.token_mats[token_id] @ prefix_mat
            desired_rows.append(target @ prefix_mat.T)

        for token_pos, token_id in enumerate(context_ids):
            distance = target_pos - token_pos
            weight = row_scale * (recency_decay ** distance)
            input_row = later_rows[len(context_ids) - token_pos - 1]
            token_update_terms[token_id].add_(weight * _row_outer(input_row, desired_rows[token_pos]))

    return base_update_terms, token_update_terms, trained_output_count, max_contribution_mass, has_update


@torch.no_grad()
def _right_update_sequence(
    model: MemoryMatrixNetwork,
    token_ids: Sequence[int],
    target_start: int,
    recency_decay: float,
    correct_margin: float | None,
) -> Tuple[torch.Tensor, torch.Tensor, int, float, bool]:
    base_update_terms = torch.zeros_like(model.base_mat)
    token_update_terms = torch.zeros_like(model.token_mats)
    trained_output_count = max(len(token_ids) - target_start, 0)
    max_contribution_mass = 0.0
    has_update = False

    for target_pos in range(target_start, len(token_ids)):
        context_ids = token_ids[:target_pos]
        target_id = token_ids[target_pos]
        contribution_mass = _contribution_mass(len(token_ids), recency_decay)
        max_contribution_mass = max(max_contribution_mass, contribution_mass)

        state = _right_state_matrix(model, context_ids)[0]
        row_scale = _row_scale(model, state, target_id, correct_margin)
        if row_scale <= 0.0:
            continue
        has_update = True
        target = model.unembed_vectors[target_id]

        full_mat = _right_state_matrix(model, context_ids)
        base_weight = row_scale * (recency_decay ** target_pos)
        base_update_terms.add_(base_weight * _row_outer(model.query, target @ full_mat.T))

        left_row = model.query @ model.base_mat
        for token_pos, token_id in enumerate(context_ids):
            suffix_mat = torch.eye(model.n, device=model.base_mat.device, dtype=model.base_mat.dtype)
            suffix_mat = model.token_mats[token_id] @ suffix_mat
            for later_token_id in context_ids[token_pos + 1 :]:
                suffix_mat = suffix_mat @ model.token_mats[later_token_id]
            desired_row = target @ suffix_mat.T
            distance = target_pos - token_pos
            weight = row_scale * (recency_decay ** distance)
            token_update_terms[token_id].add_(weight * _row_outer(left_row, desired_row))
            left_row = left_row @ model.token_mats[token_id]

    return base_update_terms, token_update_terms, trained_output_count, max_contribution_mass, has_update


def _row_scale(
    model: MemoryMatrixNetwork,
    state: torch.Tensor,
    target_id: int,
    correct_margin: float | None,
) -> float:
    if correct_margin is None:
        return 1.0
    scores = state @ model.unembed_vectors.T
    correct_score = scores[target_id]
    wrong_scores = scores.clone()
    wrong_scores[target_id] = -torch.inf
    missing_margin = wrong_scores.max() + correct_margin - correct_score
    return float(missing_margin.clamp_min(0.0).item())


def _contribution_mass(terms: int, recency_decay: float) -> float:
    if recency_decay == 1.0:
        return float(terms)
    return (1.0 - recency_decay ** terms) / (1.0 - recency_decay)


@torch.no_grad()
def apply_batch_update(
    model: MemoryMatrixNetwork,
    optimizer: MemoryMatrixNetworkOptimizer,
    sequences: Sequence[Sequence[int]],
    target_starts: Sequence[int],
    recency_decay: float,
    correct_margin: float | None = None,
) -> None:
    base_update_terms = torch.zeros_like(model.base_mat)
    token_update_terms = torch.zeros_like(model.token_mats)
    trained_output_count = 0
    max_contribution_mass = 0.0
    has_update = False

    if model.update_side == "right":
        update_sequence = _right_update_sequence
    elif model.update_side == "double-query":
        update_sequence = _double_query_update_sequence
    else:
        update_sequence = _left_update_sequence
    for token_ids, target_start in zip(sequences, target_starts):
        base_terms, token_terms, count, mass, did_update = update_sequence(
            model,
            token_ids,
            target_start,
            recency_decay,
            correct_margin,
        )
        trained_output_count += count
        max_contribution_mass = max(max_contribution_mass, mass)
        has_update = has_update or did_update
        base_update_terms.add_(base_terms)
        token_update_terms.add_(token_terms)

    if trained_output_count == 0 or not has_update:
        return

    batch_update_scale = 1.0 / (trained_output_count * max_contribution_mass)
    optimizer.step(base_update_terms * batch_update_scale, token_update_terms * batch_update_scale)


def train(
    *,
    model: MemoryMatrixNetwork,
    optimizer: MemoryMatrixNetworkOptimizer,
    sample_batch: Callable[[], Tuple[List[List[int]], List[int]]],
    iters: int,
    recency_decay: float,
    correct_margin: float | None = None,
    eval_every: int = 0,
    evaluate: Callable[[MemoryMatrixNetwork, int], None] | None = None,
) -> None:
    for it in range(1, iters + 1):
        sequences, target_starts = sample_batch()
        apply_batch_update(
            model,
            optimizer,
            sequences,
            target_starts,
            recency_decay=recency_decay,
            correct_margin=correct_margin,
        )
        if evaluate is not None and eval_every > 0 and (it % eval_every == 0 or it == iters):
            evaluate(model, it)
