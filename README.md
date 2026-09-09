# Matrix Network

This repo explores a matrix-based sequence model where inference is done by
mutating a state matrix instead of passing tokens through layers of activations.

The core idea is:

1. Start with a learned `base_mat`.
2. Maintain a current `state_mat`, initially a copy of `base_mat`.
3. Each token owns a learned matrix.
4. Applying context means right-multiplying token matrices into `state_mat`.
5. Prediction reads a fixed query row through the current state matrix twice and
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
state = (query @ state_mat) @ state_mat
scores = unembed_vectors @ state
prediction = argmax(scores)
```

This is the **double-right** mode: token matrices multiply on the right, and
the query passes through the same state twice. Since `query` is the first
one-hot row, the implementation reads `state_mat[0] @ state_mat`.

The matrices are kept near-orthogonal, so the state acts like a sequence of
rotations/reflections in a shared vector space.

## Training Idea

Training does not use backpropagation. For each target token position:

1. Build the matrix state for the prefix.
2. Predict the next token.
3. Choose the target vector.
4. Build learned rotation update terms that rotate the current state toward
   that target.
5. Accumulate updates for both `base_mat` and all token matrices that appear in
   the prefix.
6. Apply momentum to those learned update terms.
7. Convert the resulting terms to a skew-symmetric generator, exponentiate it,
   and apply the resulting rotation to the matrices.

For each factor in the state product, each learned update term is:

```text
input_row[:, None] @ desired_row[None, :]
```

The input row is propagated through the factors before that matrix; the desired
row is pulled back through the transposed suffix, including that matrix.
Both occurrences of the state are trained: one uses `(query @ state, target)`
and the other uses `(query, target @ state.T)`. Their updates are averaged.
These terms can be
averaged and mixed linearly. When applying the update, the optimizer turns the
terms into the skew-symmetric generator:

```text
A = update_terms - update_terms.T
R = exp(A * learning_rate)
matrix = R @ matrix
```

The exponential is approximated by scaling `A * learning_rate`, using
`I + scaled_A` as the small-step approximation, then repeatedly squaring back up
to the full rotation.

Momentum is an exponential moving average of these learned update terms:

```text
momentum = momentum_decay * previous_momentum
         + (1 - momentum_decay) * current_update
```

The applied update is:

```text
applied_update = (1 - momentum_weight) * current_update
               + momentum_weight * momentum
```

`momentum_weight` controls the fraction of the applied update that comes from
the momentum update instead of directly from the current batch update.

Training batches variable-length sequences with masked padding. Prefix products
are sequential over token positions but batched over examples. A backward suffix
sweep combines both reads and all applicable targets using batched matrix
multiplications, without storing a target-by-position triangle of vectors.
Temporary storage scales as `O(batch * length * n² + batch * length * n)`.

## Files

- `matrix_network.py`: core inference model.
- `matrix_network_optimizer.py`: custom non-autograd optimizer that turns
  rotation deltas into matrix updates, with momentum.
- `matrix_network_training.py`: batched double-right update construction and the
  generic training loop.
- `matrix_network_addition.py`: addition task data generation, evaluation, and
  the CLI training entrypoint.
- `matrix_network_modal.py`: Modal remote training/checkpoint utilities.
- `matrix_network_utils.py`: miscellaneous tensor helpers for rotations and
  diagnostics such as subspace stats.

## Example

Run a short local addition experiment:

```bash
uv run --offline --python .venv/bin/python python matrix_network_addition.py \
  --n 32 \
  --addend-digits 3 \
  --iters 5000
```

Useful training knobs:

- `--momentum-decay`: EMA decay for base/token matrix update momentum.
- `--momentum-weight`: fraction of the applied update from momentum.
- `--update-noise-scale`: skew-symmetric optimizer noise scaled relative to the
  learned skew update RMS.
- `--correct-margin`: train only targets below this decode-score margin, scaling
  each one-hot target update by the missing margin. Omit it to train every target.

Run the training math checks with the local environment:

```bash
uv run --offline --python .venv/bin/python python -m unittest discover -s tests -v
```

These compare batched updates against independent autograd derivatives of
`query @ state @ state`; autograd is used only in tests.
