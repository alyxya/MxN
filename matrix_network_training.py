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
def apply_batch_update(
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    sequences: Sequence[Sequence[int]],
    target_starts: Sequence[int],
    target_noise: float,
    update_decay: float,
) -> None:
    base_update_terms = torch.zeros_like(model.base_mat)
    token_update_terms = torch.zeros_like(model.token_mats)
    base_update_weight = torch.zeros((), device=model.base_mat.device, dtype=model.base_mat.dtype)
    token_update_weights = torch.zeros(
        model.vocab_size,
        device=model.base_mat.device,
        dtype=model.base_mat.dtype,
    )

    for token_ids, target_start in zip(sequences, target_starts):
        context_ids = token_ids[:-1]
        query_triangle_rows = _query_triangle_rows(model, context_ids)

        token_id_tensor = torch.tensor(
            token_ids,
            device=model.base_mat.device,
            dtype=torch.long,
        )
        targets = model.unembed_vectors[token_id_tensor]
        if target_noise > 0.0:
            noise = torch.randn_like(targets) / (model.n ** 0.5)
            targets = targets + noise * target_noise
            targets = targets / targets.norm(dim=1, keepdim=True).clamp_min(1e-12)
        target_triangle_rows = _target_triangle_rows(model, context_ids, targets)
        positions = torch.arange(len(token_ids), device=model.base_mat.device)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        decay = torch.tensor(update_decay, device=model.base_mat.device, dtype=model.base_mat.dtype)
        update_weights = torch.tril(torch.pow(decay, distances.clamp_min(0)))
        update_weights[:target_start] = 0.0

        position_updates = torch.bmm(
            query_triangle_rows.permute(1, 2, 0),
            (target_triangle_rows * update_weights.unsqueeze(2)).permute(1, 0, 2),
        )
        base_update_terms.add_(position_updates[0])
        token_update_terms.index_add_(0, token_id_tensor[:-1], position_updates[1:])
        base_update_weight.add_(update_weights[:, 0].sum())
        token_update_weights.index_add_(0, token_id_tensor[:-1], update_weights[:, 1:].sum(dim=0))

    base_update_terms.div_(base_update_weight.clamp_min(1e-12))
    used_tokens = token_update_weights > 0.0
    token_update_terms[used_tokens] = (
        token_update_terms[used_tokens] / token_update_weights[used_tokens, None, None]
    )
    optimizer.step(base_update_terms, token_update_terms)


def train(
    *,
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    sample_batch: Callable[[], Tuple[List[List[int]], List[int]]],
    iters: int,
    target_noise: float,
    update_decay: float,
    eval_every: int = 0,
    evaluate: Callable[[MatrixNetwork, int], None] | None = None,
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[MatrixNetwork, MatrixNetworkOptimizer, int], None] | None = None,
) -> None:
    for it in range(1, iters + 1):
        sequences, prompt_lens = sample_batch()
        apply_batch_update(
            model, optimizer, sequences, prompt_lens,
            target_noise=target_noise,
            update_decay=update_decay,
        )
        if evaluate is not None and eval_every > 0 and (it % eval_every == 0 or it == iters):
            evaluate(model, it)
        if on_checkpoint is not None and checkpoint_every > 0 and it % checkpoint_every == 0:
            on_checkpoint(model, optimizer, it)
