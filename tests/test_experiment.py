import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

from matrix_network_experiment import run_experiment


class ExperimentTests(unittest.TestCase):
    def run_small(self, path, **options):
        config = dict(n=16, batch_size=1, max_iters=4, eval_every=2,
                      eval_samples=2, confirmation_samples=2, stop_on_target=False)
        config.update(options)
        with contextlib.redirect_stdout(io.StringIO()):
            return run_experiment(path, **config)

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as root:
            full, split = Path(root) / 'full', Path(root) / 'split'
            self.run_small(full)
            self.run_small(split, max_iters=2)
            result = self.run_small(split, resume_from=split)
            self.assertEqual(result['iteration'], 4)
            self.assertEqual(result['config']['objective'], 'one-hot')
            a = torch.load(full / 'latest.pt', weights_only=False)
            b = torch.load(split / 'latest.pt', weights_only=False)
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key], b['model_state'][key],
                                           rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'different batch_size'):
                self.run_small(split, resume_from=split, batch_size=2)

    def test_invalid_limits(self):
        with tempfile.TemporaryDirectory() as root:
            for key in ('max_iters', 'batch_size', 'eval_every', 'eval_samples',
                        'confirmation_samples', 'max_seconds'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.run_small(root, **{key: 0})
