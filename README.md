# Matrix Networks

Token representations use low-rank matrices: `M_token = I + A @ B`, with `A` shape `n x k` and `B` shape `k x n`.
Use `--token-rank k` to choose `k` (default is `n//2`).

## Test A Trained Model

Run one prediction:

```bash
python test_model.py --expr "123+456"
```

Interactive mode:

```bash
python test_model.py
```

The default model path is:

`models/matrix_network_n30_step0005_resume10k.pt`

You can test a specific checkpoint:

```bash
python test_model.py --checkpoint checkpoints/matrix_network_n30_step0005_resume10k.pt --expr "817+662"
```

## Train

Example fresh training run:

```bash
python matrix_network_addition.py \
  --n 30 \
  --token-rank 15 \
  --learning-rate 0.01 \
  --iters 10000 \
  --batch-size 64 \
  --eval-every 1000 \
  --save-path checkpoints/matrix_network_n30_step001_10k.pt
```

## Continue Training

Resume from a checkpoint (same architecture, smaller step size):

```bash
python matrix_network_addition.py \
  --load-path checkpoints/matrix_network_n30_step001_10k.pt \
  --n 30 \
  --token-rank 15 \
  --learning-rate 0.005 \
  --iters 10000 \
  --batch-size 64 \
  --eval-every 1000 \
  --save-path checkpoints/matrix_network_n30_step0005_resume10k.pt
```

## Optional Momentum

Enable EMA momentum on normalized gradient directions:

```bash
python matrix_network_addition.py \
  --n 30 \
  --token-rank 15 \
  --learning-rate 0.01 \
  --iters 10000 \
  --use-momentum \
  --momentum-decay 0.98 \
  --momentum-blend-start 0.0 \
  --momentum-blend 0.5 \
  --momentum-blend-ramp-iters 1000
```

## W&B Logging

Install W&B first:

```bash
pip install wandb
```

Then enable logging in training:

```bash
python matrix_network_addition.py \
  --n 30 \
  --token-rank 15 \
  --learning-rate 0.01 \
  --iters 10000 \
  --wandb \
  --wandb-project matrix-networks \
  --wandb-run-name n30-lr1e-2 \
  --wandb-tags addition,baseline
```

## Base Matrix A/B

Train with a learned base matrix (default):

```bash
python matrix_network_addition.py \
  --base-mode learned \
  --addend-digits 10 \
  --n 50 \
  --token-rank 25 \
  --learning-rate 0.0001 \
  --iters 10000
```

Train with no learned base matrix (fixed identity):

```bash
python matrix_network_addition.py \
  --base-mode identity_fixed \
  --addend-digits 10 \
  --n 50 \
  --token-rank 25 \
  --learning-rate 0.0001 \
  --iters 10000
```

Run a controlled multi-seed A/B sweep and get a CSV summary:

```bash
python scripts/base_mode_ablation.py \
  --addend-digits 10 \
  --n 50 \
  --token-rank 25 \
  --learning-rate 0.0001 \
  --iters 10000 \
  --num-seeds 3 \
  --out-dir checkpoints/base_mode_ablation
```
