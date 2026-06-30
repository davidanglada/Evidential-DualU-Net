# Evidential DualU-Net

Official research code for **“Evidential DualU-Net: Single-Pass Uncertainty for Cell Instance Segmentation,”** accepted as a full paper/poster at **MIDL 2026** and published in *Proceedings of Machine Learning Research*, volume 285, pages 1–28. Read the [paper and discussion on OpenReview](https://openreview.net/forum?id=GALXBAr9WX) or [download the PDF](https://openreview.net/pdf?id=GALXBAr9WX).

The method augments a dual-decoder U-Net with evidential outputs, producing cell segmentation, centroid cues, and interpretable uncertainty proxies in one forward pass.

## Highlights

- Single-pass uncertainty estimation—no ensembles or repeated stochastic inference.
- Dirichlet evidential semantic segmentation outputs.
- Closed-form pixel-level aleatoric, epistemic, and vacuity maps; entropy and mutual information are also exposed by the code.
- Size-invariant instance-level uncertainty from mean pooling of foreground Dirichlet parameters.
- Peak and mass-ratio geometric uncertainty from the centroid decoder.
- PanNuke and proprietary Ki-67 configurations, COCO-style loaders, reproducible training, and evaluation.

## Method overview

An ImageNet-pretrained ResNeXt-50 32×4d encoder feeds two U-Net decoders. The segmentation head predicts non-negative evidence $e_k$ and Dirichlet parameters $\alpha_k=e_k+1$; class probabilities are the Dirichlet mean $\alpha_k/\sum_j\alpha_j$. Its total evidence yields vacuity and closed-form aleatoric and epistemic uncertainty proxies. The unchanged centroid-regression decoder predicts a Gaussian map. Local maxima seed marker-controlled watershed over the semantic foreground to recover nucleus instances.

For each nucleus, Dirichlet parameters are averaged spatially—not summed—to avoid uncertainty depending artificially on nucleus area. The background component is excluded and the remaining foreground classes are renormalized. Detection reliability is described by peak uncertainty $1-p_{max}$ and mass-ratio uncertainty relative to the expected Gaussian mass $2\pi\sigma^2$; the paper uses $\sigma=5$, $\lambda_{peak}=0.3$, and $\lambda_{mass}=0.6$.

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

The paper evaluates PanNuke (7,904 H&E patches, 19 tissues, approximately 189k nuclei and five foreground classes) with three-fold cross-validation. It also evaluates a proprietary breast Ki-67 IHC dataset with 52 tiles from four patients and three foreground classes using leave-one-patient-out validation. The Ki-67 data cannot be distributed by this repository. No data, patient images, or private annotations are included.

The retained loaders expect COCO-style image/annotation layouts generated for the experiments. Set every `dataset.<split>.root` in the YAML to your local dataset root.

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

Use `evidential_dualunet.uncertainty.pool_instance_uncertainty` with a `[K,H,W]` alpha tensor and `[H,W]` labeled instance mask to obtain the paper's mean-pooled, foreground-only per-nucleus scores. Equations and interpretation are in [docs/uncertainty.md](docs/uncertainty.md).

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

If this repository is useful, cite the MIDL 2026 paper ([OpenReview](https://openreview.net/forum?id=GALXBAr9WX)):

```bibtex
@inproceedings{anglada_rotger2026evidential,
  title     = {Evidential DualU-Net: Single-Pass Uncertainty for Cell Instance Segmentation},
  author    = {Anglada-Rotger, David and Marques, Ferran and Pard\`as, Montse},
  booktitle = {Medical Imaging with Deep Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {285},
  pages     = {1--28},
  year      = {2026},
  url       = {https://openreview.net/forum?id=GALXBAr9WX}
}
```

## License and contact

Code is released under the [MIT License](LICENSE); the paper is available under CC BY 4.0. Dataset licenses remain with their respective owners. For questions, open a GitHub issue or contact David Anglada-Rotger at `david.anglada@upc.edu`.
