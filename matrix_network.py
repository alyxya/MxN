#!/usr/bin/env python3
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from matrix_network_ops import initialize_rotation_like, normalize_columns, one_hot_vectors


class MatrixNetwork:
    def __init__(
        self,
        *,
        n: int,
        vocab: Iterable[str],
        device: torch.device | str | None = None,
    ):
        vocab_tokens = tuple(vocab)
        if not vocab_tokens:
            raise ValueError("vocab must not be empty")
        if any(not isinstance(token, str) or not token for token in vocab_tokens):
            raise ValueError("vocab tokens must be non-empty strings")
        if len(set(vocab_tokens)) != len(vocab_tokens):
            raise ValueError("vocab tokens must be unique")
        if n < len(vocab_tokens):
            raise ValueError(f"n must be >= {len(vocab_tokens)} to fit fixed one-hot heads, got {n}")

        self.n = n
        self.vocab: Tuple[str, ...] = vocab_tokens
        self.vocab_size = len(vocab_tokens)
        self.stoi: Dict[str, int] = {token: i for i, token in enumerate(vocab_tokens)}
        self.itos: Dict[int, str] = {i: token for token, i in self.stoi.items()}

        self.query = one_hot_vectors(1, n, device)[0]
        self.unembed_vectors = one_hot_vectors(self.vocab_size, n, device)
        self.base_mat = initialize_rotation_like((n, n), device, 0.0)
        self.token_mats = initialize_rotation_like((self.vocab_size, n, n), device, 0.0)
        self.device = self.base_mat.device

    def encode(self, text: str) -> List[int]:
        return self.encode_tokens(text)

    def encode_tokens(self, tokens: Iterable[str]) -> List[int]:
        return [self.stoi[token] for token in tokens]

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
