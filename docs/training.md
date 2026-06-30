# Training

Run `python scripts/train.py --config CONFIG`. Required sections are `dataset`, `loader`, `model`, `optimizer`, `training`, `transforms`, and `experiment`. `--set section.key=value` forwards overrides to the original trainer.

The segmentation loss combines evidential MSE, Dice, and an annealed Dirichlet KL term. The paper uses a ResNeXt-50 32×4d encoder, Gaussian centroid targets with `sigma=5` scaled by 100, and 200 epochs at constant learning rate. PanNuke uses LR `2e-4` and batch size 64; Ki-67 uses LR `1e-4` and batch size 8. The reported unweighted-Dice variant uses `lambda_seg=1`, `lambda_dice=0.4`, `lambda_cent=0.7`, `lambda_KL=0.4`; the weighted-Dice variant uses `lambda_KL=0.2`. Both warm up KL over 40 epochs.

The centroid branch and experimental variants remain available in the sanitized historical configs. Archive the resolved YAML, log, environment, metrics, and best checkpoint. Disable Weights & Biases for offline execution.
