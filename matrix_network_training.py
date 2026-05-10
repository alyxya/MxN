#!/usr/bin/env python3
from typing import Callable, List, Sequence, Tuple

import torch

from matrix_network import MatrixNetwork
from matrix_network_optimizer import MatrixNetworkOptimizer


@torch.no_grad()
def _prefix_query_rows(
    model: MatrixNetwork,
    context_ids: Sequence[int],
    positions: Sequence[int] | None = None,
) -> torch.Tensor:
    if positions is None:
        positions = range(len(context_ids) + 1)

    rows = torch.empty(
        (len(positions), model.n),
        device=model.base_mat.device,
        dtype=model.base_mat.dtype,
    )
    if not positions:
        return rows

    position_to_row = {pos: i for i, pos in enumerate(positions)}
    active_rows = torch.empty(
        (len(positions), model.n),
        device=model.base_mat.device,
        dtype=model.base_mat.dtype,
    )
    active_positions: List[int] = []
    active_count = 0

    if 0 in position_to_row:
        rows[position_to_row[0]].copy_(model.query)

    for token_pos in range(len(context_ids) - 1, -1, -1):
        position = token_pos + 1
        if position in position_to_row:
            active_rows[active_count].copy_(model.query)
            active_positions.append(position)
            active_count += 1
        if active_count > 0:
            active_rows[:active_count] = active_rows[:active_count] @ model.token_mats[
                context_ids[token_pos]
            ]

    for row, position in zip(active_rows[:active_count], active_positions):
        rows[position_to_row[position]].copy_(row)
    return rows


@torch.no_grad()
def sequence_update_terms(
    model: MatrixNetwork,
    token_ids: Sequence[int],
    prompt_len: int,
    target_noise: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, int, int, int]:
    base_update_terms = torch.zeros_like(model.base_mat)
    token_update_terms = torch.zeros_like(model.token_mats)
    correct = total = mistakes = 0
    target_score_sum = 0.0

    prediction_positions = list(range(prompt_len, len(token_ids)))
    context_ids = token_ids[:-1]
    prefix_rows = _prefix_query_rows(model, context_ids, prediction_positions)
    states = prefix_rows @ model.base_mat
    score_rows = states @ model.unembed_vectors.T

    mistake_positions: List[int] = []
    target_rows: List[torch.Tensor] = []
    for row, pos in zip(score_rows, prediction_positions):
        target_id = token_ids[pos]
        total += 1
        target_score_sum += float(row[target_id].item())
        if int(row.argmax().item()) == target_id:
            correct += 1
            continue

        mistakes += 1
        mistake_positions.append(pos)
        target = model.unembed_vectors[target_id]
        if target_noise > 0.0:
            target = target + torch.randn_like(target) * target_noise
            target = target / (target.norm() + 1e-12)
        target_rows.append(target)

    if not mistake_positions:
        return base_update_terms, token_update_terms, target_score_sum, correct, total, mistakes

    mistake_prefix_rows = _prefix_query_rows(model, context_ids, mistake_positions)
    targets = torch.stack(target_rows)
    base_target_rows = targets @ model.base_mat.T
    # Base learns target @ base.T -> q @ prefix.
    base_update_terms.add_(mistake_prefix_rows.T @ base_target_rows)

    target_positions = torch.tensor(mistake_positions, device=model.base_mat.device)
    active_target_positions = target_positions
    active_target_rows = base_target_rows
    target_rows_by_token: List[Tuple[torch.Tensor, torch.Tensor] | None] = [
        None for _ in context_ids
    ]
    for token_pos, token_id in enumerate(context_ids):
        keep = active_target_positions > token_pos
        if not bool(keep.any().item()):
            break
        active_target_positions = active_target_positions[keep]
        active_target_rows = active_target_rows[keep] @ model.token_mats[token_id].T
        target_rows_by_token[token_pos] = (active_target_positions, active_target_rows)

    active_later_positions = torch.empty(
        len(mistake_positions),
        device=model.base_mat.device,
        dtype=torch.long,
    )
    active_later_rows = torch.empty(
        (len(mistake_positions), model.n),
        device=model.base_mat.device,
        dtype=model.base_mat.dtype,
    )
    active_later_count = 0
    mistake_position_set = set(mistake_positions)
    for token_pos in range(len(context_ids) - 1, -1, -1):
        position = token_pos + 1
        if position in mistake_position_set:
            active_later_positions[active_later_count] = position
            active_later_rows[active_later_count].copy_(model.query)
            active_later_count += 1

        token_targets = target_rows_by_token[token_pos]
        if token_targets is not None:
            target_positions_for_token, target_rows_for_token = token_targets
            order = torch.argsort(active_later_positions[:active_later_count])
            later_positions_for_token = active_later_positions[:active_later_count][order]
            later_rows_for_token = active_later_rows[:active_later_count][order]
            if not torch.equal(later_positions_for_token, target_positions_for_token):
                raise RuntimeError("internal token update position mismatch")
            # Each token learns target @ base.T @ earlier.T -> q @ later.
            token_update_terms[context_ids[token_pos]].add_(
                later_rows_for_token.T @ target_rows_for_token
            )

        if active_later_count > 0:
            active_later_rows[:active_later_count] = (
                active_later_rows[:active_later_count]
                @ model.token_mats[context_ids[token_pos]]
            )

    return base_update_terms, token_update_terms, target_score_sum, correct, total, mistakes


@torch.no_grad()
def apply_sequence_update(
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    token_ids: Sequence[int],
    prompt_len: int,
    target_noise: float,
) -> Tuple[float, float]:
    base_update_terms, token_update_terms, target_score_sum, correct, total, mistakes = (
        sequence_update_terms(model, token_ids, prompt_len, target_noise)
    )
    if mistakes > 0:
        base_update_terms = base_update_terms / mistakes
        token_update_terms = token_update_terms / mistakes
    optimizer.step(base_update_terms, token_update_terms)

    return target_score_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def apply_batch_update(
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    sequences: Sequence[Sequence[int]],
    prompt_lens: Sequence[int],
    target_noise: float,
) -> Tuple[float, float]:
    base_update_terms = torch.zeros_like(model.base_mat)
    token_update_terms = torch.zeros_like(model.token_mats)
    correct = total = mistakes = 0
    target_score_sum = 0.0

    for token_ids, prompt_len in zip(sequences, prompt_lens):
        (
            sequence_base_terms,
            sequence_token_terms,
            sequence_target_score_sum,
            sequence_correct,
            sequence_total,
            sequence_mistakes,
        ) = sequence_update_terms(model, token_ids, prompt_len, target_noise)
        base_update_terms.add_(sequence_base_terms)
        token_update_terms.add_(sequence_token_terms)
        target_score_sum += sequence_target_score_sum
        correct += sequence_correct
        total += sequence_total
        mistakes += sequence_mistakes

    if mistakes > 0:
        base_update_terms = base_update_terms / mistakes
        token_update_terms = token_update_terms / mistakes
    optimizer.step(base_update_terms, token_update_terms)

    return target_score_sum / max(total, 1), correct / max(total, 1)


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
        score, acc = apply_batch_update(
            model, optimizer, sequences, prompt_lens,
            target_noise=target_noise,
        )
        if it == 1 or it % log_every == 0:
            print(f"iter={it:5d} mean_target_score={score:.4f} token_acc={acc:.3f}")
        if evaluate is not None and eval_every > 0 and (it % eval_every == 0 or it == iters):
            evaluate(model, it)
        if on_checkpoint is not None and checkpoint_every > 0 and it % checkpoint_every == 0:
            on_checkpoint(it, model, optimizer)
