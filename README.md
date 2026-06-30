# Evidential DualU-Net

Official research code for **“Evidential DualU-Net: Single-Pass Uncertainty for Cell Instance Segmentation.”** The method augments a dual-decoder U-Net with evidential outputs, producing cell segmentation, centroid cues, and calibrated uncertainty in one forward pass.

## Highlights

- Single-pass uncertainty estimation—no ensembles or repeated stochastic inference.
- Dirichlet evidential semantic segmentation outputs.
- Pixel-level aleatoric, epistemic, vacuity, entropy, and mutual-information maps.
- Instance-level uncertainty scores obtained by pooling evidence within each nucleus.
- A centroid decoder providing geometric cues for watershed instance reconstruction.
- PanNuke and Ki-67 configurations, COCO-style dataset loaders, reproducible training, and quantitative evaluation.

## Method overview

An ImageNet encoder feeds two U-Net decoders. The segmentation head predicts non-negative evidence $e_k$ and Dirichlet parameters $\alpha_k=e_k+1$; class probabilities are the Dirichlet mean $\alpha_k/\sum_j\alpha_j$. Its total evidence yields vacuity and supports closed-form decomposition into data (aleatoric) and distributional (epistemic) uncertainty. The second decoder predicts centroid geometry. Semantic foreground and centroid seeds are combined through watershed to recover nucleus instances. Pixel evidence can then be pooled per instance to report a class, confidence, and uncertainty score for every nucleus.

The original checkpoint-compatible implementation remains in `dual_unet/`. Stable reusable interfaces live in `src/evidential_dualunet/`; executable workflows live in `scripts/`. Historical experiment configs are retained under `configs/` with public-safe paths.

## Installation

Python 3.10+ is recommended. Install PyTorch for your CUDA platform first if needed, then:

```bash
git clone https://github.com/davidanglada/Evidential-DualU-Net.git
cd Evidential-DualU-Net
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install wandb  # optional experiment tracking
```

Conda users can run `conda env create -f environment.yml`. Development tests require `pip install -e '.[test]'`.

## Dataset preparation

The retained loaders expect COCO-style image/annotation layouts generated for the original experiments. Set every `dataset.<split>.root` in the YAML to your local dataset root. PanNuke must use disjoint folds for train/validation/test; Ki-67 split names depend on the prepared release. No data, patient images, or private annotations are included.

See [docs/datasets.md](docs/datasets.md) for the expected records and privacy guidance.

## Training

Copy an example config, change dataset/output paths, and train from the repository root:

```bash
python scripts/train.py --config configs/train_pannuke.yaml
python scripts/train.py --config configs/train_ki67.yaml --set experiment.seed=7
```

Outputs use `experiment.output_dir`; checkpoints preserve the legacy model state-dict format. Set `experiment.wandb: false` for fully local runs. See [docs/training.md](docs/training.md).

## Inference

Input is an RGB image readable by Pillow. The output is a compressed NumPy archive containing `alpha [K,H,W]`, `probabilities [K,H,W]`, uncertainty maps `[H,W]`, and (when available) `centroid [1,H,W]`.

```bash
python scripts/infer.py \
  --config configs/inference.yaml \
  --checkpoint /path/to/checkpoints/model.pth \
  --image /path/to/image.png \
  --output outputs/inference/prediction.npz
```

The image is normalized with the config mean/std and is not resized; dimensions should be compatible with the encoder (typically multiples of 32). See [docs/inference.md](docs/inference.md).

## Evaluation

Set the test dataset and `experiment.ckpt_path` in `configs/evaluation.yaml`:

```bash
python scripts/evaluate.py --config configs/evaluation.yaml
python scripts/evaluate.py --config configs/evaluation.yaml --uncertainty
```

Evaluation reports semantic, centroid/detection, cell typing, and panoptic-quality measures supported by the selected legacy evaluator. Save terminal output with your run artifacts; new programmatic utilities can save JSON/CSV alongside predictions.

## Uncertainty visualization and export

```bash
python scripts/visualize_uncertainty.py \
  --image /path/to/image.png \
  --predictions outputs/inference/prediction.npz \
  --output outputs/figures/uncertainty.png

python scripts/export_predictions.py \
  --input outputs/inference/prediction.npz \
  --output-dir outputs/inference/arrays
```

Use `evidential_dualunet.uncertainty.pool_instance_uncertainty` with a `[K,H,W]` alpha tensor and `[H,W]` labeled instance mask to obtain per-nucleus scores. Equations and interpretation are in [docs/uncertainty.md](docs/uncertainty.md).

## Reproducibility

Each config records dataset folds, architecture, losses, optimizer, seed, and output location. For exact comparisons, record the package environment, GPU model, dataset checksum, and checkpoint checksum. Deterministic kernels can reduce speed and are not guaranteed for every CUDA operation. See [docs/reproducibility.md](docs/reproducibility.md) and [docs/checkpoints.md](docs/checkpoints.md).

## Repository map

```text
configs/                   public examples + sanitized historical experiments
dual_unet/                 original checkpoint-compatible research implementation
scripts/                   train, evaluate, infer, visualize, export CLIs
src/evidential_dualunet/   reusable public package
tests/                     lightweight CPU tests
docs/                      practical workflow notes
assets/                    publication-safe asset guidance
```

## Citation

Publication metadata is not yet known; update the placeholders in `CITATION.cff` and below before archival release.

```bibtex
@article{angladarotger2026evidential,
  title   = {Evidential DualU-Net: Single-Pass Uncertainty for Cell Instance Segmentation},
  author  = {Anglada-Rotger, David and Marques, Ferran and Pardàs, Montse},
  journal = {TODO},
  year    = {2026},
  doi     = {TODO}
}
```

## License and contact

Code is released under the [MIT License](LICENSE). Dataset licenses remain with their respective owners. For questions, open a GitHub issue; add the corresponding author’s public institutional email here before release.
