import unittest
import torch
from matrix_network import MatrixNetwork
from experiments.historical_97fe187.adapter import MatrixNetwork as LeftModel
from experiments.ablations import objective_only_update, read_only_update, historical_targets, plain_one_hot_update
from experiments.historical_97fe187.matrix_network_optimizer import MatrixNetworkOptimizer as OldOptimizer
from matrix_network_optimizer import MatrixNetworkOptimizer as NewOptimizer

class Capture:
    def step(self, base, tokens):
        self.base, self.tokens = base, tokens

class AblationTests(unittest.TestCase):
    def test_update_math(self):
        for variant in ('objective', 'read', 'plain-one-hot'):
            double = variant == 'read'
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
                    scale = 1.0
                    scores = output.detach().clone()
                    correct = scores[seq[t]].clone()
                    scores[seq[t]] = -torch.inf
                    if variant == 'objective':
                        scale = float((scores.max() + .1 - correct).clamp_min(0))
                gradients = torch.autograd.grad(output @ target, changes)
                base_index = 0 if double else -1
                expected_base += scale * gradients[base_index]
                token_gradients = gradients[1:] if double else gradients[:-1]
                for token, grad in zip(ids, token_gradients):
                    expected_tokens[token] += scale * grad
            capture = Capture()
            update = {'objective': objective_only_update, 'read': read_only_update,
                      'plain-one-hot': plain_one_hot_update}[variant]
            update(model, capture, [seq], [1], 1.0)
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

    def test_target_component_removals(self):
        states = torch.tensor([[.2, -.3, .4, .7, -.2]], dtype=torch.double)
        ids = torch.tensor([1])
        one_hot = torch.tensor([[0., 1., 0.]], dtype=torch.double)
        decode = states[:, :3]
        corrected = decode + one_hot - decode.softmax(-1)
        norm = decode.norm(dim=-1, keepdim=True).clamp_min((3 / 5) ** .5)
        for removal in (None, 'no-softmax', 'no-hidden', 'no-normalization'):
            raw = decode + one_hot if removal == 'no-softmax' else corrected
            expected = states.clone()
            expected[:, :3] = raw if removal == 'no-normalization' else raw * norm / raw.norm(dim=-1, keepdim=True)
            if removal == 'no-hidden':
                expected[:, 3:] = 0
            actual = historical_targets(states, ids, 3, remove=removal)
            torch.testing.assert_close(actual, expected)

    def test_component_trainer_matches_frozen_baseline_when_nothing_removed(self):
        from experiments.ablations import target_component_update
        from experiments.historical_97fe187.matrix_network_training import apply_batch_update
        torch.manual_seed(17)
        model = LeftModel(n=8, vocab='~01+=', eos_token='~').double()
        model.base_mat.copy_(torch.eye(8) + torch.randn(8, 8) * .1)
        model.token_mats.copy_(torch.eye(8) + torch.randn(5, 8, 8) * .1)
        for decay in (.7, 1.0):
            baseline, variant = Capture(), Capture()
            args = ([[1, 2, 1, 0], [2, 0]], [1, 1], decay)
            apply_batch_update(model, baseline, *args)
            target_component_update(model, variant, *args, remove=None)
            torch.testing.assert_close(variant.base, baseline.base)
            torch.testing.assert_close(variant.tokens, baseline.tokens)
