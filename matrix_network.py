#!/usr/bin/env python3
from typing import Iterable, List, Sequence, Tuple

import torch


class MatrixNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        n: int,
        vocab: Iterable[str],
        eos_token: str,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.vocab = tuple(vocab)
        self.vocab_size = len(self.vocab)
        if n < self.vocab_size:
            raise ValueError(
                f"n must be >= len(vocab); got n={n}, vocab_size={self.vocab_size}"
            )

        self.n = n
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        if eos_token not in self.stoi:
            raise ValueError(f"eos_token {eos_token!r} is not in vocab")
        self.eos_token = eos_token
        self.eos_id = self.stoi[eos_token]

        eye = torch.eye(n, device=device)
        self.register_buffer("query", eye[0].clone())
        self.register_buffer("unembed_vectors", eye[: self.vocab_size].clone())
        self.register_buffer("base_mat", eye.clone())
        self.register_buffer(
            "token_mats",
            eye.expand(self.vocab_size, n, n).clone(),
        )
        self.register_buffer("state_mat", self.base_mat.clone(), persistent=False)

    def encode(self, tokens: Iterable[str]) -> List[int]:
        return [self.stoi[token] for token in tokens]

    def decode(self, token_id: int) -> str:
        return self.vocab[token_id]

    def reset_state(self) -> None:
        with torch.no_grad():
            self.state_mat.copy_(self.base_mat)

    def apply_context(self, token_ids: Sequence[int]) -> None:
        with torch.no_grad():
            for token_id in token_ids:
                self.state_mat.copy_(self.state_mat @ self.token_mats[token_id])

    def query_state(self) -> torch.Tensor:
        with torch.no_grad():
            # Equivalent to: (self.query @ self.state_mat) @ self.state_mat
            state = self.state_mat[0]
            return state @ self.state_mat

    def predict(self) -> int:
        with torch.no_grad():
            state = self.query_state()
            # Equivalent to scoring against the one-hot unembedding vectors.
            return int(state[: self.vocab_size].argmax().item())

    def generate(self, prefix: Iterable[str], max_len: int) -> Tuple[str, bool]:
        with torch.no_grad():
            self.reset_state()
            self.apply_context(self.encode(prefix))
            out: List[int] = []
            for _ in range(max_len):
                token_id = self.predict()
                if token_id == self.eos_id:
                    return "".join(self.decode(token) for token in out), True
                out.append(token_id)
                self.apply_context([token_id])
            return "".join(self.decode(token) for token in out), False
