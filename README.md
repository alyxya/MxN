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
  --step-size 0.01 \
  --steps 10000 \
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
  --step-size 0.005 \
  --steps 10000 \
  --batch-size 64 \
  --eval-every 1000 \
  --save-path checkpoints/matrix_network_n30_step0005_resume10k.pt
```
