# Checkpoints

Evaluation and inference accept either a raw PyTorch state dictionary or a dictionary with a `model` key. Released checkpoints must document model config, dataset split, class ordering, normalization, epoch/selection metric, license, and SHA-256 checksum.

Checkpoint files are intentionally ignored because they are large and may carry redistribution restrictions. Publish them through an archival release and place the URL/checksum in this document. Loading a checkpoint executes PyTorch deserialization; only load files from trusted sources.

