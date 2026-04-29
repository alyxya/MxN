#!/usr/bin/env python3
from typing import Dict, List, Sequence, Tuple

import torch


EPS = 1e-12
INIT_ORTHOGONALIZE_STEPS = 4
_EYE_CACHE: Dict[Tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}


def normalize_columns(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=0, keepdim=True) + eps)


def normalize_last_dim(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cached_eye(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n, device.type, device.index, dtype)
    eye = _EYE_CACHE.get(key)
    if eye is None:
        eye = torch.eye(n, device=device, dtype=dtype)
        _EYE_CACHE[key] = eye
    return eye


def orthogonalize_newton_schulz(w: torch.Tensor) -> torch.Tensor:
    return 1.5 * w - 0.5 * (w @ w.transpose(-1, -2) @ w)


def orthogonalize_steps(w: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(max(steps, 0)):
        w = orthogonalize_newton_schulz(w)
    return w


def initialize_rotation_like(shape: Tuple[int, ...], device: torch.device, strength: float) -> torch.Tensor:
    n = shape[-1]
    eye = cached_eye(n, device, torch.float32)
    is_batched = len(shape) > 2
    if is_batched:
        eye = eye.expand(*shape[:-2], n, n).clone()
    if strength == 0.0:
        return eye if is_batched else eye.clone()
    noise = torch.randn(shape, device=device) / (n**0.5)
    w = (eye + strength * noise) / ((1.0 + strength**2) ** 0.5)
    return orthogonalize_steps(w, INIT_ORTHOGONALIZE_STEPS)


def one_hot_vectors(count: int, dim: int, device: torch.device) -> torch.Tensor:
    vectors = torch.zeros((count, dim), device=device)
    vectors[:, :count] = cached_eye(count, device, vectors.dtype)
    return vectors


class MatrixNetwork:
    def __init__(
        self,
        *,
        n: int,
        device: torch.device,
        vocab: str,
        output_vocab: str,
    ):
        if n < len(vocab):
            raise ValueError(f"n must be >= {len(vocab)} to fit fixed one-hot heads, got {n}")
        if not output_vocab:
            raise ValueError("output_vocab must not be empty")
        if any(ch not in vocab for ch in output_vocab):
            raise ValueError("output_vocab must be a subset of vocab")

        self.n = n
        self.device = device
        self.vocab = vocab
        self.output_vocab = output_vocab
        self.vocab_size = len(vocab)
        self.output_vocab_size = len(output_vocab)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(vocab)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

        self.query = one_hot_vectors(1, n, device)[0]
        self.unembed_vectors = one_hot_vectors(self.vocab_size, n, device)
        self.base_mat = initialize_rotation_like((n, n), device, 0.0)
        self.token_mats = initialize_rotation_like((self.vocab_size, n, n), device, 0.0)

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
        scores = self.unembed_vectors[: self.output_vocab_size] @ normalize_columns(state.unsqueeze(1))
        return int(scores[:, 0].argmax().item())

    def predict_next(self, prefix: str) -> str:
        return self.itos[self.predict_next_id(self.encode(prefix))]
