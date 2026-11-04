# Historical addition reproduction

Source: Git commit `97fe187` (May 12, 2026), immediately before the user's
reported >90% teacher-forced accuracy at approximately 500,000 steps.

The four matrix_network*.py files are snapshots with only imports changed to
package-relative imports. Keep them frozen. The adapter supplies the current
generation API without changing prediction or update math. Historical momentum
code is retained for fidelity but its weight is explicitly zero in this experiment.

Configuration: n=32, batch=1, token LR=1, base LR=0.1, noise=1,
recency decay=1, orthogonalization every 100 steps. Single-left read,
softmax decode correction with decode-norm floor, no margin gating,
and learning rate applied to both rotation signal and noise.

The current experiment runner provides fixed held-out validation/confirmation
sets and checkpointing. Excluding these examples and using today's PyTorch runtime
mean this is a reproduction of the algorithm/configuration, not a bitwise replay
of the original training stream.
