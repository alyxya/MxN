#!/usr/bin/env python3
from typing import Callable, List, Sequence, Tuple

import torch

from matrix_network import MatrixNetwork
from matrix_network_optimizer import MatrixNetworkOptimizer


@torch.no_grad()
def _right_prefix_mats(model: MatrixNetwork, token_ids: torch.Tensor) -> torch.Tensor:
    """Return S_t before target t, batching the sequential products over examples."""
    batch_size, target_count = token_ids.shape
    prefixes = model.base_mat.new_empty((batch_size, target_count, model.n, model.n))
    prefixes[:, 0] = model.base_mat
    for pos in range(target_count - 1):
        prefixes[:, pos + 1] = torch.bmm(
            prefixes[:, pos], model.token_mats[token_ids[:, pos]]
        )
    return prefixes


@torch.no_grad()
def _double_right_update_terms(
    model: MatrixNetwork,
    token_ids: torch.Tensor,
    prefixes: torch.Tensor,
    targets: torch.Tensor,
    row_scales: torch.Tensor,
    recency_decay: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate both occurrences of S in q S S for left-applied rotations.

    For a factor M in S = L M R, its update is outer(q L, target R.T M.T).
    The two reads use (q S, target) and (q, target S.T), respectively.
    Reduce both reads and all targets with bmm, without materializing outer
    products or a [batch, target, matrix-position, n] triangle.
    """
    batch_size, target_count = token_ids.shape
    pulled_targets = torch.matmul(targets.unsqueeze(-2), prefixes.transpose(-1, -2)).squeeze(-2)
    queries = torch.stack((prefixes[:, :, 0], model.query.expand_as(targets)), dim=1)
    desired = torch.stack((targets, pulled_targets), dim=1)
    positions = torch.arange(target_count, device=token_ids.device)
    decay = model.base_mat.new_tensor(recency_decay)

    # The base factor has L = I and M R = S_t for each target.
    base_desired = torch.matmul(desired.unsqueeze(-2), prefixes[:, None].transpose(-1, -2)).squeeze(-2)
    base_weights = row_scales * decay.pow(positions)
    base_terms = torch.bmm(
        queries.reshape(batch_size, -1, model.n).transpose(1, 2),
        (base_desired * base_weights[:, None, :, None]).reshape(batch_size, -1, model.n),
    ).sum(dim=0)
    token_terms = torch.zeros_like(model.token_mats)

    # Sweep suffixes backwards. Introduce target t only when reaching token t-1,
    # so each desired row traverses exactly its own suffix (including M).
    suffix_rows = torch.zeros_like(desired)
    for pos in range(target_count - 2, -1, -1):
        suffix_rows[:, :, pos + 1] = desired[:, :, pos + 1]
        active_count = target_count - pos - 1
        suffix_rows[:, :, pos + 1:] = torch.bmm(
            suffix_rows[:, :, pos + 1:].reshape(batch_size, 2 * active_count, model.n),
            model.token_mats[token_ids[:, pos]].transpose(1, 2),
        ).reshape(batch_size, 2, active_count, model.n)
        inputs = torch.bmm(
            queries[:, :, pos + 1:].reshape(batch_size, 2 * active_count, model.n),
            prefixes[:, pos],
        )
        weights = row_scales[:, pos + 1:] * decay.pow(positions[pos + 1:] - pos)
        updates = torch.bmm(
            inputs.transpose(1, 2),
            (suffix_rows[:, :, pos + 1:] * weights[:, None, :, None]).reshape(
                batch_size, 2 * active_count, model.n
            ),
        )
        token_terms.index_add_(0, token_ids[:, pos], updates)
    return base_terms * 0.5, token_terms * 0.5


@torch.no_grad()
def apply_batch_update(
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    sequences: Sequence[Sequence[int]],
    target_starts: Sequence[int],
    recency_decay: float,
    correct_margin: float | None = None,
) -> None:
    if len(sequences) != len(target_starts):
        raise ValueError("sequences and target_starts must have the same length")
    if any(start < 0 or start > len(seq) for seq, start in zip(sequences, target_starts)):
        raise ValueError("target_starts must be between zero and the sequence length")
    examples = [(seq, start) for seq, start in zip(sequences, target_starts) if start < len(seq)]
    if not examples:
        return

    # Padding affects only unused future prefixes; masked targets contribute zero.
    lengths = [len(seq) for seq, _ in examples]
    target_count = max(lengths)
    token_ids = torch.tensor(
        [list(seq) + [0] * (target_count - len(seq)) for seq, _ in examples],
        device=model.base_mat.device, dtype=torch.long,
    )
    positions = torch.arange(target_count, device=token_ids.device)
    starts = torch.tensor([start for _, start in examples], device=token_ids.device)
    ends = torch.tensor(lengths, device=token_ids.device)
    train_mask = (positions >= starts[:, None]) & (positions < ends[:, None])
    row_scales = train_mask.to(model.base_mat.dtype)
    prefixes = _right_prefix_mats(model, token_ids)
    targets = model.unembed_vectors[token_ids]
    if correct_margin is not None:
        states = torch.matmul(prefixes[:, :, 0].unsqueeze(-2), prefixes).squeeze(-2)
        scores = states @ model.unembed_vectors.T
        correct_scores = scores.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        scores.scatter_(-1, token_ids.unsqueeze(-1), -torch.inf)
        row_scales *= (scores.max(dim=-1).values + correct_margin - correct_scores).clamp_min(0.0)
        if not bool(row_scales.any().item()):
            return

    base_terms, token_terms = _double_right_update_terms(
        model, token_ids, prefixes, targets, row_scales, recency_decay,
    )
    trained_output_count = sum(len(seq) - start for seq, start in examples)
    if recency_decay == 1.0:
        max_contribution_mass = float(target_count)
    else:
        max_contribution_mass = max(
            (1.0 - recency_decay ** length) / (1.0 - recency_decay) for length in lengths
        )
    scale = 1.0 / (trained_output_count * max_contribution_mass)
    optimizer.step(base_terms * scale, token_terms * scale)


def train(
    *,
    model: MatrixNetwork,
    optimizer: MatrixNetworkOptimizer,
    sample_batch: Callable[[], Tuple[List[List[int]], List[int]]],
    iters: int,
    recency_decay: float,
    correct_margin: float | None = None,
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
        )
        if evaluate is not None and eval_every > 0 and (it % eval_every == 0 or it == iters):
            evaluate(model, it)
        if on_checkpoint is not None and checkpoint_every > 0 and it % checkpoint_every == 0:
            on_checkpoint(model, optimizer, it)
