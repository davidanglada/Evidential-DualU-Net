# Datasets

Dataset roots are never committed. Point each YAML split at a local COCO-style dataset directory containing images and an annotation JSON. Existing loaders consume image records plus instance polygons/masks, category IDs, and centroid targets; inspect the matching class in `dual_unet/datasets/` if adapting a converter.

The paper uses PanNuke's 7,904 H&E patches (256×256), spanning 19 tissues and approximately 189k nuclei in five foreground classes, with three-fold cross-validation. Historical configs use `foldfold_1`-style labels; adjust the public example if your converter uses different names.

The breast Ki-67 IHC dataset is proprietary and is **not publicly distributed**: it contains 52 tiles (1024×1024) from four patients with positive, negative, and non-epithelial nuclei. Paper results use leave-one-patient-out cross-validation. The Ki-67 config documents the experiment but is not runnable without authorized data access. Both datasets were evaluated at 40× magnification (approximately 0.25 µm/pixel).

Before publishing any example, verify its dataset license, remove embedded metadata, and confirm it contains no patient or hospital identifier. This repository intentionally ships no biomedical image.
