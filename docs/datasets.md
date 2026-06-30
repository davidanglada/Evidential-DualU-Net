# Datasets

Dataset roots are never committed. Point each YAML split at a local COCO-style dataset directory containing images and an annotation JSON. Existing loaders consume image records plus instance polygons/masks, category IDs, and centroid targets; inspect the matching class in `dual_unet/datasets/` if adapting a converter.

PanNuke configs use `foldfold_1`-style historical fold labels; the public examples use shorter placeholders and may need adjustment to match your prepared data. Ki-67 experiments use a precomputed train/validation/test split. Keep patients or source slides disjoint across splits.

Before publishing any example, verify its dataset license, remove embedded metadata, and confirm it contains no patient or hospital identifier. This repository intentionally ships no biomedical image.

