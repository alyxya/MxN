# Addition experiments

`matrix_network_experiment.run_experiment` trains 3-digit decimal addition on one
CPU thread. Validation and confirmation prompts are fixed, unique, disjoint, and
excluded from training. `best.pt` is selected by validation exact-match accuracy;
`latest.pt` also stores sampler/torch RNG state and optimizer update count for
resuming. Confirmation evaluates the best checkpoint, not necessarily the last.
The confirmation set is reused across comparisons, so it is not a fresh final test.

Modes:

| Arguments | Read | Target | Noise scaling |
| --- | --- | --- | --- |
| defaults | double-right | plain one-hot | current |
| `historical=True` | single-left | historical softmax correction | historical |
| plus `ablation='objective'` | single-left | former margin-weighted one-hot (margin 0.1) | historical |
| plus `ablation='read'` | double-right | historical softmax correction | historical |
| plus `ablation='noise'` | single-left | historical softmax correction | current |

The objective ablation deliberately retains the former margin objective for
reproducibility. It does not represent today's plain one-hot trainer. The read
ablation changes both composition direction and the number of reads.

Historical noise is multiplied by the learning rate together with the signal;
current noise is added after scaling the signal. At token LR 1 these agree, but
at base LR 0.1 current noise is ten times larger relative to the signal.
The historical source snapshot remains frozen; momentum weight is zero.

## Launching on Modal without a persistent client

Deploy with the local uv environment:

```bash
uv run --offline --python .venv/bin/python modal deploy matrix_network_modal.py
```

Then spawn a deployed function (use a fresh run name for each experiment):

```python
import modal

train = modal.Function.from_name('mxn-matrix-network', 'addition_experiment_remote')
call = train.spawn('addition-historical-example', dict(
    n=32, batch_size=1, update_noise_scale=1.0,
    max_iters=500000, max_seconds=3000,
    historical=True, stop_on_target=False,
))
print(call.object_id)
```

Run launch snippets with `uv run --offline --python .venv/bin/python python`.
Store the call ID to retrieve results later with
`modal.FunctionCall.from_id(call_id).get(timeout=0)`. Container logs show periodic
metrics without starting a separate status-reader container. Checkpoints live on
the `mxn-matrix-network-checkpoints` volume under the run name.

`max_iters` and `max_seconds` are cumulative limits when resuming. Pass
`resume_from='/checkpoints/<previous-run>'` to continue. The objective and recorded
training/evaluation settings must match. Legacy checkpoints without RNG state
can only approximate continuation of the random stream.

## Completed comparison

Single seed (0), n=32, batch=1, noise=1, base LR=0.1, token LR=1,
orthogonalization every 100 steps, 500,000 iterations:

| Setup | Final validation exact sum | Best-checkpoint confirmation exact sum |
| --- | ---: | ---: |
| Historical | 52.2% | 51.6% |
| Objective only | 7.0% | 7.05% |
| Read only | 50.2% | 50.55% |
| Noise only | 45.8% | 46.6% |

Validation uses 1,000 examples; confirmation uses 2,000. The former objective
caused the largest regression in this comparison. These runs do not isolate
margin weighting from target direction, establish variability across seeds, or
measure interactions among changes. Plain one-hot performance remains unmeasured
at this training budget.
