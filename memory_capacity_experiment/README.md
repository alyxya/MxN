# Memory Capacity Experiment

This directory is a self-contained copy/memory experiment for matrix networks.
It is intentionally separate from the addition code.

Task format:

```text
12345=12345~
```

The prompt is the left side plus `=`, and training/evaluation predicts the
right-hand copy plus EOS.

The main comparison is the state update side:

```text
left:  S <- U @ S
right: S <- S @ U
```

Example runs:

```bash
python3 memory_capacity_experiment/memory_copy_train.py \
  --n 64 \
  --copy-digits 10 \
  --update-side left \
  --iters 10000 \
  --eval-every 1000
```

```bash
python3 memory_capacity_experiment/memory_copy_train.py \
  --n 64 \
  --copy-digits 10 \
  --update-side right \
  --iters 10000 \
  --eval-every 1000
```
