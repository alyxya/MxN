#!/usr/bin/env python3
"""Frozen comparisons against the historical setup, not the active objective."""
from typing import Sequence

import torch

from .historical_97fe187.matrix_network import MatrixNetwork
from .historical_97fe187.matrix_network_optimizer import MatrixNetworkOptimizer


from .historical_97fe187.matrix_network_training import (
    _query_triangle_rows,
    _target_triangle_rows,
)


@torch.no_grad()
def objective_only_update(
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    sequences: Sequence[Sequence[int]],
    target_starts: Sequence[int],
    recency_decay: float,
) -> None:
    """Historical single-left training with the former margin-one-hot objective."""
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
        targets = model.unembed_vectors[token_id_tensor]
        scores = states @ model.unembed_vectors.T
        correct = scores.gather(1, token_id_tensor[:, None]).squeeze(1)
        scores.scatter_(1, token_id_tensor[:, None], -torch.inf)
        scales = (scores.max(dim=1).values + 0.1 - correct).clamp_min(0)
        has_update = has_update or bool(scales[target_start:].any().item())
        target_triangle_rows = _target_triangle_rows(model, context_ids, targets)
        positions = torch.arange(len(token_ids), device=model.base_mat.device)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        decay = torch.tensor(recency_decay, device=model.base_mat.device, dtype=model.base_mat.dtype)
        update_weights = torch.tril(torch.pow(decay, distances.clamp_min(0)))
        update_weights *= scales[:, None]
        update_weights[:target_start] = 0.0

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

@torch.no_grad()
def historical_targets(states, token_ids, vocab_size):
    """Exact 97fe187 target construction, including the decode-norm floor."""
    decode = states[..., :vocab_size].clone()
    norm = decode.norm(dim=-1, keepdim=True).clamp_min((vocab_size / states.shape[-1]) ** 0.5)
    delta = -torch.softmax(decode, dim=-1)
    delta.scatter_add_(-1, token_ids.unsqueeze(-1), torch.ones_like(token_ids.unsqueeze(-1), dtype=states.dtype))
    decode.add_(delta)
    targets = states.clone()
    targets[..., :vocab_size] = decode * norm / decode.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return targets


@torch.no_grad()
def read_only_update(model, optimizer, sequences, target_starts, recency_decay):
    """Change only composition/readout to double-right; retain the old target."""
    from matrix_network_training import _right_prefix_mats, _double_right_update_terms
    base_terms = torch.zeros_like(model.base_mat)
    token_terms = torch.zeros_like(model.token_mats)
    count = 0
    mass = 0.0
    for seq, start in zip(sequences, target_starts):
        if start >= len(seq):
            continue
        ids = torch.tensor([seq], device=model.base_mat.device, dtype=torch.long)
        prefixes = _right_prefix_mats(model, ids)
        states = torch.matmul(prefixes[:, :, 0].unsqueeze(-2), prefixes).squeeze(-2)
        targets = historical_targets(states, ids, model.vocab_size)
        scales = (torch.arange(len(seq), device=ids.device) >= start).to(states.dtype)[None]
        base, tokens = _double_right_update_terms(model, ids, prefixes, targets, scales, recency_decay)
        base_terms.add_(base)
        token_terms.add_(tokens)
        count += len(seq) - start
        mass = max(mass, sum(recency_decay ** k for k in range(len(seq))))
    if count:
        optimizer.step(base_terms / (count * mass), token_terms / (count * mass))
