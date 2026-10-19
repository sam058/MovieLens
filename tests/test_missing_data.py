"""Modules must fail with a clear message when the processed CSVs are missing."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


@pytest.mark.parametrize("module", ["tfidf_search", "bm25_search", "semantic_search"])
def test_missing_data_raises_clear_error(tmp_path, module):
    # A copy of src/ in an empty project has no data/ml-25m folder
    shutil.copytree(SRC_DIR, tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=tmp_path / "src", capture_output=True, text=True, timeout=300,
    )
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert "not found. Run src/preprocessing.py" in result.stderr
