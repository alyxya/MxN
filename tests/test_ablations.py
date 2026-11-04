import unittest
import torch
from matrix_network import MatrixNetwork
from experiments.historical_97fe187.adapter import MatrixNetwork as LeftModel
from experiments.ablations import objective_only_update, read_only_update, historical_targets
from experiments.historical_97fe187.matrix_network_optimizer import MatrixNetworkOptimizer as OldOptimizer
from matrix_network_optimizer import MatrixNetworkOptimizer as NewOptimizer

class Capture:
    def step(self, base, tokens):
        self.base, self.tokens = base, tokens

class AblationTests(unittest.TestCase):
    def test_update_math(self):
        for double in (False, True):
            torch.manual_seed(42)
            model = (MatrixNetwork if double else LeftModel)(n=5, vocab='~01+=', eos_token='~').double()
            model.base_mat.copy_(torch.eye(5) + torch.randn(5, 5) * .1)
            model.token_mats.copy_(torch.eye(5) + torch.randn(5, 5, 5) * .1)
            seq = [1, 2, 1, 0]
            expected_base = torch.zeros_like(model.base_mat)
            expected_tokens = torch.zeros_like(model.token_mats)
            for t in range(1, len(seq)):
                ids = seq[:t] if double else list(reversed(seq[:t]))
                factors = [model.base_mat] + [model.token_mats[i] for i in ids] if double else [model.token_mats[i] for i in ids] + [model.base_mat]
                changes = [torch.zeros_like(model.base_mat, requires_grad=True) for _ in factors]
                state = torch.eye(5, dtype=torch.double)
                for factor, change in zip(factors, changes):
                    state = state @ ((torch.eye(5, dtype=torch.double) + change) @ factor)
                output = model.query @ state
                if double:
                    output = output @ state
                    target = historical_targets(output.detach()[None], torch.tensor([seq[t]]), model.vocab_size)[0]
                    scale = .5
                else:
                    target = model.unembed_vectors[seq[t]]
                    scores = output.detach().clone()
                    correct = scores[seq[t]].clone()
                    scores[seq[t]] = -torch.inf
                    scale = float((scores.max() + .1 - correct).clamp_min(0))
                gradients = torch.autograd.grad(output @ target, changes)
                base_index = 0 if double else -1
                expected_base += scale * gradients[base_index]
                token_gradients = gradients[1:] if double else gradients[:-1]
                for token, grad in zip(ids, token_gradients):
                    expected_tokens[token] += scale * grad
            capture = Capture()
            (read_only_update if double else objective_only_update)(model, capture, [seq], [1], 1.0)
            torch.testing.assert_close(capture.base, expected_base / 12)
            torch.testing.assert_close(capture.tokens, expected_tokens / 12)

    def test_noise_only_changes_base_at_these_learning_rates(self):
        a = LeftModel(n=5, vocab='~01+=', eos_token='~')
        b = LeftModel(n=5, vocab='~01+=', eos_token='~')
        old = OldOptimizer(a, momentum_decay=.9, momentum_weight=0, base_lr=.1, token_lr=1, update_noise_scale=1)
        new = NewOptimizer(b, base_lr=.1, token_lr=1, update_noise_scale=1)
        torch.manual_seed(3)
        base, tokens = torch.randn_like(a.base_mat), torch.randn_like(a.token_mats)
        torch.manual_seed(5); old.step(base, tokens)
        torch.manual_seed(5); new.step(base, tokens)
        torch.testing.assert_close(a.token_mats, b.token_mats)
        self.assertFalse(torch.allclose(a.base_mat, b.base_mat))
