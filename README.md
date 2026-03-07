# Matrix Networks

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
  --learning-rate 0.01 \
  --iters 10000 \
  --wandb \
  --wandb-project matrix-networks \
  --wandb-run-name n30-lr1e-2 \
  --wandb-tags addition,baseline
```
