import unittest

import torch

from matrix_network import MatrixNetwork
from matrix_network_training import apply_batch_update


class CaptureOptimizer:
    def __init__(self):
        self.calls = []

    def step(self, base, tokens):
        self.calls.append((base.clone(), tokens.clone()))


def reference_updates(model, sequences, starts, decay, margin):
    """Differentiate independent left perturbations of every factor in q S S.

    Autograd is used only as a test oracle; production training has no autograd.
    Per-occurrence perturbations let us apply the existing recency weights.
    """
    base = torch.zeros_like(model.base_mat)
    tokens = torch.zeros_like(model.token_mats)
    count = sum(len(seq) - start for seq, start in zip(sequences, starts))
    masses = []
    for seq, start in zip(sequences, starts):
        if start == len(seq):
            continue
        masses.append(sum(decay ** k for k in range(len(seq))))
        for t in range(start, len(seq)):
            factors = [model.base_mat] + [model.token_mats[i] for i in seq[:t]]
            perturbations = [torch.zeros_like(model.base_mat, requires_grad=True) for _ in factors]
            state = torch.eye(model.n, dtype=model.base_mat.dtype)
            for factor, perturbation in zip(factors, perturbations):
                state = state @ ((torch.eye(model.n, dtype=state.dtype) + perturbation) @ factor)
            output = model.query @ state @ state
            scores = (output @ model.unembed_vectors.T).detach()
            scale = 1.0
            if margin is not None:
                correct = scores[seq[t]].clone()
                scores[seq[t]] = -torch.inf
                scale = max(float(scores.max() + margin - correct), 0.0)
            grads = torch.autograd.grad(output @ model.unembed_vectors[seq[t]], perturbations)
            base += grads[0] * (0.5 * scale * decay ** t)
            for pos, token in enumerate(seq[:t]):
                tokens[token] += grads[pos + 1] * (0.5 * scale * decay ** (t - pos))
    return base / (count * max(masses)), tokens / (count * max(masses))


class DoubleRightTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = MatrixNetwork(n=5, vocab='~01+=', eos_token='~').double()
        # Nonorthogonal matrices catch accidental use of inverse = transpose.
        self.model.base_mat.copy_(torch.eye(5) + torch.randn(5, 5) * 0.15)
        self.model.token_mats.copy_(torch.eye(5) + torch.randn(5, 5, 5) * 0.15)

    def test_updates_match_independent_derivatives(self):
        for sequences, starts in [([[1]], [0]), ([[1, 1, 2, 0], [2, 0], [], [3]], [2, 0, 0, 1])]:
            for decay in (0.0, 0.7, 1.0):
                for margin in (None, 0.0, 0.3):
                    with self.subTest(sequences=sequences, decay=decay, margin=margin):
                        optimizer = CaptureOptimizer()
                        apply_batch_update(self.model, optimizer, sequences, starts, decay, margin)
                        expected = reference_updates(self.model, sequences, starts, decay, margin)
                        actual = optimizer.calls[0] if optimizer.calls else (torch.zeros_like(expected[0]), torch.zeros_like(expected[1]))
                        for got, want in zip(actual, expected):
                            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-8)
                            self.assertFalse(got.requires_grad)

    def test_no_targets_or_satisfied_margin_skips_optimizer(self):
        optimizer = CaptureOptimizer()
        apply_batch_update(self.model, optimizer, [[], [1]], [0, 1], 1.0)
        model = MatrixNetwork(n=5, vocab='~01+=', eos_token='~')
        apply_batch_update(model, optimizer, [[0]], [0], 1.0, 0.5)
        self.assertEqual(optimizer.calls, [])

    def test_invalid_target_starts(self):
        for starts in ([], [-1], [2]):
            with self.assertRaises(ValueError):
                apply_batch_update(self.model, CaptureOptimizer(), [[1]], starts, 1.0)


if __name__ == '__main__':
    unittest.main()
