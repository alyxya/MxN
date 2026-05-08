#!/usr/bin/env python3
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


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
            raise ValueError(
                f"n must be >= {len(vocab_tokens)} to fit fixed one-hot heads, got {n}"
            )

        self.n = n
        self.vocab: Tuple[str, ...] = vocab_tokens
        self.vocab_size = len(vocab_tokens)
        self.stoi: Dict[str, int] = {token: i for i, token in enumerate(vocab_tokens)}
        self.itos: Dict[int, str] = {i: token for token, i in self.stoi.items()}

        eye = torch.eye(n, device=device)
        self.query = eye[0].clone()
        self.unembed_vectors = eye[: self.vocab_size].clone()
        self.base_mat = eye.clone()
        self.token_mats = (
            eye.expand(self.vocab_size, n, n).clone()
        )
        self.state_mat = self.base_mat.clone()
        self.device = self.base_mat.device

    def encode(self, tokens: Iterable[str]) -> List[int]:
        return [self.stoi[token] for token in tokens]

    def decode(self, token_id: int) -> str:
        return self.itos[token_id]

    def reset_state(self) -> None:
        self.state_mat = self.base_mat.clone()

    def apply_context(self, token_ids: Sequence[int]) -> None:
        for token_id in token_ids:
            self.state_mat = self.state_mat @ self.token_mats[token_id]

    def predict(self) -> int:
        state = self.state_mat @ self.query
        scores = self.unembed_vectors @ state
        return int(scores.argmax().item())
