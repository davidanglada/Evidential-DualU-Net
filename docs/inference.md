# Inference

`scripts/infer.py` accepts one RGB image, YAML model config, checkpoint, output archive, and device. It performs exactly one network forward pass. Images are normalized by config mean/std without resizing.

The `.npz` archive stores Dirichlet parameters and mean probabilities in channel-first layout, scalar uncertainty maps, and a centroid prediction when present. Use `scripts/export_predictions.py` to split fields into `.npy` files. Model inputs should normally have height and width divisible by 32.

