# Matrix Network

This repo explores a matrix-based sequence model where inference is done by
mutating a state matrix instead of passing tokens through layers of activations.

The core idea is:

1. Start with a learned `base_mat`.
2. Maintain a current `state_mat`, initially a copy of `base_mat`.
3. Each token owns a learned matrix.
4. Applying context means multiplying token matrices into `state_mat`.
5. Prediction reads a fixed query vector through the current state matrix and
   chooses the output token with the largest score.

In code, inference is intentionally small:

```python
model.reset_state()
model.apply_context(model.encode("12+34="))
next_id = model.predict()
next_token = model.decode(next_id)
```

## Architecture

`MatrixNetwork` has four important tensors:

- `base_mat`: the learned starting matrix for a sequence.
- `state_mat`: the mutable inference matrix, reset from `base_mat`.
- `token_mats`: one learned matrix per vocabulary token.
- `query` and `unembed_vectors`: fixed one-hot vectors used to read predictions.

Applying context is just matrix multiplication:

```text
state_mat = base_mat
state_mat = state_mat @ token_mat[token_0]
state_mat = state_mat @ token_mat[token_1]
...
```

Prediction is:

```text
state = state_mat @ query
scores = unembed_vectors @ state
prediction = argmax(scores)
```

The matrices are kept near-orthogonal, so the state acts like a sequence of
rotations/reflections in a shared vector space.

## Training Idea

Training does not use backpropagation. For each target token position:

1. Build the matrix state for the prefix.
2. Predict the next token.
3. If the prediction is wrong, choose the target vector.
4. Optionally add noise to the target vector.
5. Build skew-symmetric rotation updates that rotate the current state toward
   that target.
6. Accumulate updates for both `base_mat` and all token matrices that appear in
   the prefix.
7. Apply momentum to those rotation updates.
8. Apply the resulting rotations to the matrices, then orthogonalize.

The rotation generator is:

```text
v @ u.T - u @ v.T
```

where `u` is the current vector and `v` is the target vector. This produces a
skew-symmetric matrix that nudges `u` toward `v` while preserving the
rotation-like structure.

Momentum is an exponential moving average of these learned rotation updates:

```text
momentum = momentum_decay * previous_momentum
         + (1 - momentum_decay) * current_update
```

The applied update is:

```text
applied_update = momentum + current_update_weight * current_update
```

`current_update_weight` is an optional direct contribution from the current
batch update on top of the momentum term.

## Files

- `matrix_network.py`: core inference model.
- `matrix_network_training.py`: checkpointing, rotation updates, momentum, the
  generic training loop, and small tensor math helpers.
- `matrix_network_addition.py`: addition task data generation, evaluation, and
  the CLI training entrypoint.
- `matrix_network_modal.py`: Modal remote training/checkpoint utilities.
- `matrix_network_utils.py`: miscellaneous diagnostics such as subspace stats.

## Example

Run a short local addition experiment:

```bash
python3 matrix_network_addition.py \
  --n 32 \
  --addend-digits 3 \
  --iters 5000 \
  --batch-size 32
```

Useful training knobs:

- `--target-randomize-scale`: adds noise to target vectors during rotation
  update construction.
- `--momentum-decay`: EMA decay for base/token matrix update momentum.
- `--current-update-weight`: extra direct weight for the current batch update.
