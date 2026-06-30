"""Compatibility installer for environments with pre-PEP 621 setuptools."""

from pathlib import Path
from setuptools import setup

PUBLIC = [
    "evidential_dualunet", "evidential_dualunet.models", "evidential_dualunet.losses",
    "evidential_dualunet.datasets", "evidential_dualunet.training", "evidential_dualunet.evaluation",
    "evidential_dualunet.inference", "evidential_dualunet.uncertainty",
    "evidential_dualunet.postprocessing", "evidential_dualunet.visualization",
]
LEGACY = [
    "dual_unet", "dual_unet.datasets", "dual_unet.eval", "dual_unet.models",
    "dual_unet.models.base", "dual_unet.models.encoders", "dual_unet.models.losses",
    "dual_unet.models.mtunet", "dual_unet.utils", "dual_unet.utils.config",
]

setup(
    name="evidential-dualunet",
    version="0.1.0",
    description="Single-pass evidential uncertainty for cell instance segmentation",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23", "scipy>=1.9", "torch>=2.0", "torchvision>=0.15",
        "scikit-image>=0.20", "scikit-learn>=1.2", "opencv-python>=4.7",
        "matplotlib>=3.7", "PyYAML>=6.0", "tqdm>=4.65", "Pillow>=9.5",
        "h5py>=3.8", "pycocotools>=2.0.6", "torchmetrics>=1.0", "timm>=0.9",
    ],
    extras_require={"experiment": ["wandb>=0.15"], "test": ["pytest>=7.4"]},
    packages=PUBLIC + LEGACY,
    package_dir={"evidential_dualunet": "src/evidential_dualunet", "dual_unet": "dual_unet"},
)
