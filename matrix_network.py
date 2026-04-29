#!/usr/bin/env python3
from typing import Dict, List, Sequence

import torch

from matrix_network_ops import initialize_rotation_like, normalize_columns, one_hot_vectors


class MatrixNetwork:
    def __init__(
        self,
        *,
        n: int,
        vocab: str,
        device: torch.device | str | None = None,
    ):
        if n < len(vocab):
            raise ValueError(f"n must be >= {len(vocab)} to fit fixed one-hot heads, got {n}")
        if not vocab:
            raise ValueError("vocab must not be empty")

        self.n = n
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(vocab)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

        self.query = one_hot_vectors(1, n, device)[0]
        self.unembed_vectors = one_hot_vectors(self.vocab_size, n, device)
        self.base_mat = initialize_rotation_like((n, n), device, 0.0)
        self.token_mats = initialize_rotation_like((self.vocab_size, n, n), device, 0.0)
        self.device = self.base_mat.device

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def prefix_state_from_query(self, token_ids: Sequence[int], query: torch.Tensor) -> torch.Tensor:
        v = query
        for token_id in reversed(token_ids):
            v = self.token_mats[token_id] @ v
        return self.base_mat @ v

    def prefix_state_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        return self.prefix_state_from_query(token_ids, self.query)

    def predict_next_id(self, token_ids: Sequence[int]) -> int:
        state = self.prefix_state_ids(token_ids)
        scores = self.unembed_vectors @ normalize_columns(state.unsqueeze(1))
        return int(scores[:, 0].argmax().item())

    def predict_next(self, prefix: str) -> str:
        return self.itos[self.predict_next_id(self.encode(prefix))]
