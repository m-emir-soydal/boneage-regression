"""Path helpers for the UQ package.

Puts the project root on ``sys.path`` so UQ scripts can import the unchanged
core modules (``model``, ``data_loader``, ``config``), and resolves the
per-method results directories under ``UQ/results/``.
"""
import sys
from pathlib import Path

# UQ/common/paths.py -> UQ/common -> UQ -> project root
UQ_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = UQ_DIR.parent
RESULTS_DIR = UQ_DIR / "results"


def ensure_project_on_path():
    """Add the project root to ``sys.path`` (idempotent).

    Lets UQ scripts do ``from model import ...`` / ``from data_loader import ...``
    the same way ``tests/test_smoke.py`` does.
    """
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


def method_results_dir(method_name):
    """Return ``UQ/results/<method_name>/``, creating it if needed."""
    out = RESULTS_DIR / method_name
    out.mkdir(parents=True, exist_ok=True)
    return out
