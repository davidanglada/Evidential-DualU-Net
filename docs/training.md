# Training

Run `python scripts/train.py --config CONFIG`. Required sections are `dataset`, `loader`, `model`, `optimizer`, `training`, `transforms`, and `experiment`. `--set section.key=value` forwards overrides to the original trainer.

The segmentation loss combines evidential MSE, Dice, and an annealed Dirichlet KL term according to the config. The centroid branch and uncertainty-weighted variants remain available in the sanitized historical configs. A run should archive its resolved YAML, console log, environment, metrics, and best checkpoint. Disable Weights & Biases for offline or public-safe execution.

