#!/usr/bin/env python3
from typing import Iterable, List, Sequence

import torch


class MatrixNetwork:
    def __init__(
        self,
        *,
        n: int,
        vocab: Iterable[str],
        device: torch.device | str | None = None,
    ):
        self.vocab = tuple(vocab)
        self.vocab_size = len(self.vocab)
        if n < self.vocab_size:
            raise ValueError(f"n must be >= len(vocab); got n={n}, vocab_size={self.vocab_size}")

        self.n = n
        self.stoi = {token: i for i, token in enumerate(self.vocab)}

        eye = torch.eye(n, device=device)
        self.query = eye[0].clone()
        self.unembed_vectors = eye[: self.vocab_size].clone()
        self.base_mat = eye.clone()
        self.token_mats = eye.expand(self.vocab_size, n, n).clone()
        self.state_mat = self.base_mat.clone()
        self.device = self.base_mat.device

    def encode(self, tokens: Iterable[str]) -> List[int]:
        return [self.stoi[token] for token in tokens]

    def decode(self, token_id: int) -> str:
        return self.vocab[token_id]

    def reset_state(self) -> None:
        self.state_mat = self.base_mat.clone()

    def apply_context(self, token_ids: Sequence[int]) -> None:
        for token_id in token_ids:
            self.state_mat = self.state_mat @ self.token_mats[token_id]

    def predict(self) -> int:
        state = self.state_mat @ self.query
        return int((self.unembed_vectors @ state).argmax().item())
