# Reproducibility

Use fixed dataset folds and record checksums. Set `experiment.seed`; for new code, call `seed_everything(seed, deterministic=True)`. Deterministic CUDA algorithms may change runtime and some operators remain platform-dependent.

Archive the exact YAML, package versions (`python -m pip freeze`), Git commit, GPU/CUDA versions, checkpoint SHA-256, and JSON/CSV metrics. Keep generated runs below `outputs/`, which is ignored by Git.

