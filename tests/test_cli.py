import subprocess
import sys
from pathlib import Path
import pytest


@pytest.mark.parametrize("script", ["train.py", "evaluate.py", "infer.py", "visualize_uncertainty.py", "export_predictions.py"])
def test_cli_help(script):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts" / script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()

