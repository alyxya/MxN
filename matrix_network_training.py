#!/usr/bin/env python3
from typing import Callable, List, Sequence, Tuple

import torch

from matrix_network import MatrixNetwork
from matrix_network_optimizer import MatrixNetworkOptimizer


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
    eye = torch.eye(model.n, device=model.base_mat.device, dtype=model.base_mat.dtype)

    for full_ids, prompt_len in zip(sequences, prompt_lens):
        prefix_op = eye
        for tid in full_ids[:prompt_len]:
            prefix_op = model.token_mats[tid] @ prefix_op

        for pos in range(prompt_len, len(full_ids)):
            target_id = full_ids[pos]
            state = model.query @ (prefix_op @ model.base_mat)
            scores = model.unembed_vectors @ state

            total += 1
            target_score_sum += float(scores[target_id].item())
            if int(scores.argmax().item()) == target_id:
                correct += 1
            else:
                mistakes += 1
                target = model.unembed_vectors[target_id]
                if target_noise > 0.0:
                    target = target + torch.randn_like(target) * target_noise
                    target = target / (target.norm() + 1e-12)

                base_query = (model.query @ prefix_op).unsqueeze(1)
                base_target = (model.base_mat @ target).unsqueeze(1)
                base_update_terms.add_(base_query @ base_target.T)

                prior_ids = full_ids[:pos]
                later_ops: List[torch.Tensor] = []
                later_op = eye
                for tid in reversed(prior_ids):
                    later_ops.append(later_op)
                    later_op = later_op @ model.token_mats[tid]
                later_ops.reverse()

                earlier_op = eye
                target_base = model.base_mat @ target
                for tid, later_op in zip(prior_ids, later_ops):
                    token_query = (model.query @ later_op).unsqueeze(1)
                    token_target = (model.token_mats[tid] @ (earlier_op @ target_base)).unsqueeze(1)
                    token_update_terms[tid].add_(token_query @ token_target.T)
                    earlier_op = model.token_mats[tid] @ earlier_op

            prefix_op = model.token_mats[target_id] @ prefix_op

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
