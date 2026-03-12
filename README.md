# Matrix Networks

This repo trains matrix-network toy models on integer addition.

Supported architecture switches:
- base mode: `learned` or `identity_fixed`
- token mode: `dense`, `lowrank_ab`, or `subspace_rot`

Token modes:
- `dense`: one dense `n x n` matrix per token
- `lowrank_ab`: `I + A^T @ B`, with `A` shape `k x n` and `B` shape `k x n`
- `subspace_rot`: `I + U (R - I) U^T`, with `U` shape `n x k` and `R` shape `k x k`

Checkpoints use one explicit format only. There is no legacy checkpoint compatibility layer in the code.

## Kept Models

- `models/dense_identity_n30_d3.pt`
- `models/dense_learned_n30_d3.pt`
- `models/lowrank_ab_learned_n30_k20_d3.pt`
- `models/dense_identity_n50_d10.pt`

## Test A Model

Run one prediction with the default demo model:

```bash
python test_model.py --expr "123+456"
```

Test a specific model:

```bash
python test_model.py --checkpoint models/dense_learned_n30_d3.pt --expr "817+662"
```

Interactive mode:

```bash
python test_model.py
```

## Train

Dense token matrices with a learned base matrix:

```bash
python matrix_network_addition.py \
  --n 30 \
  --token-mode dense \
  --base-mode learned \
  --addend-digits 3 \
  --learning-rate 0.01 \
  --iters 3000 \
  --batch-size 64 \
  --eval-every 1000 \
  --save-path checkpoints/dense_learned_n30_d3.pt
```

Low-rank `I + A^T @ B` tokens:

```bash
python matrix_network_addition.py \
  --n 30 \
  --token-mode lowrank_ab \
  --token-rank 20 \
  --base-mode learned \
  --addend-digits 3 \
  --learning-rate 0.01 \
  --iters 3000 \
  --batch-size 64 \
  --eval-every 1000 \
  --save-path checkpoints/lowrank_ab_learned_n30_k20_d3.pt
```

Subspace-rotation tokens:

```bash
python matrix_network_addition.py \
  --n 30 \
  --token-mode subspace_rot \
  --token-rank 20 \
  --base-mode learned \
  --addend-digits 3 \
  --learning-rate 0.01 \
  --iters 3000 \
  --batch-size 64 \
  --eval-every 1000 \
  --save-path checkpoints/subspace_rot_learned_n30_k20_d3.pt
```

Fixed identity base matrix:

```bash
python matrix_network_addition.py \
  --n 30 \
  --token-mode dense \
  --base-mode identity_fixed \
  --addend-digits 3 \
  --learning-rate 0.01 \
  --iters 3000 \
  --batch-size 64 \
  --eval-every 1000 \
  --save-path checkpoints/dense_identity_n30_d3.pt
```

## Continue Training

Resume from any checkpoint. The architecture is taken from the checkpoint itself.

```bash
python matrix_network_addition.py \
  --load-path models/dense_learned_n30_d3.pt \
  --learning-rate 0.001 \
  --iters 5000 \
  --batch-size 64 \
  --save-path checkpoints/dense_learned_n30_d3_resume.pt
```

## Optional Momentum

```bash
python matrix_network_addition.py \
  --n 50 \
  --token-mode dense \
  --base-mode learned \
  --addend-digits 10 \
  --learning-rate 0.0001 \
  --iters 10000 \
  --batch-size 32 \
  --use-momentum \
  --momentum-decay 0.98 \
  --momentum-blend-start 0.0 \
  --momentum-blend 0.5 \
  --momentum-blend-ramp-iters 1000
```

## W&B Logging

```bash
pip install wandb
```

```bash
python matrix_network_addition.py \
  --n 30 \
  --token-mode dense \
  --base-mode learned \
  --addend-digits 3 \
  --learning-rate 0.01 \
  --iters 3000 \
  --wandb \
  --wandb-project matrix-networks \
  --wandb-run-name dense-learned-n30-d3
```

## Base-Mode Ablation Script

```bash
python scripts/base_mode_ablation.py \
  --n 50 \
  --token-mode dense \
  --addend-digits 10 \
  --learning-rate 0.0001 \
  --iters 10000 \
  --num-seeds 3 \
  --out-dir checkpoints/base_mode_ablation
```

## Matrix/Subspace Analysis

Analyze saved checkpoints to estimate how much of the ambient space each base or token matrix acts on.

```bash
python scripts/analyze_matrix_subspaces.py models/dense_learned_n30_d3.pt
```

Analyze multiple checkpoints and save the full results as JSON:

```bash
python scripts/analyze_matrix_subspaces.py models checkpoints --json-out analysis/matrix_subspaces.json
```

The script reports several complementary views:
- raw update rank from `M - I`
- skew-update rank from the antisymmetric part of `M - I`, which is the cleanest proxy for rotation-like action
- polar-factor rotation rank from the nearest orthogonal matrix `Q`
- active-subspace closure from the learned query and decoder vectors
- empirical state-span rank over sampled teacher-forced prefixes

For low-rank `A^T @ B` checkpoints, `update_*` is the tightest measure of rank collapse; the skew part can spread across a somewhat larger support even when the raw update is low-rank.
