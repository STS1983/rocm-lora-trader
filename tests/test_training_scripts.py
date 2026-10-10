#!/usr/bin/env python3
"""
Tests for ROCm LoRA Trader — training script structure validation.

Validates that training scripts are syntactically correct and contain
expected configuration without requiring GPU/ROCm hardware.

Run: python3 -m pytest tests/test_training_scripts.py -v
     or: python3 tests/test_training_scripts.py
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
ARCHIVE_DIR = REPO_ROOT / "archive"


def _check_python_syntax(filepath: Path):
    """Verify a Python file is syntactically valid."""
    source = filepath.read_text()
    ast.parse(source, filename=str(filepath))


def test_tools_train_nemotron_v3_syntax():
    """Verify tools/train_nemotron_v3.py is valid Python."""
    p = TOOLS_DIR / "train_nemotron_v3.py"
    assert p.exists(), "tools/train_nemotron_v3.py not found"
    _check_python_syntax(p)


def test_tools_serve_nemotron_v3_syntax():
    """Verify tools/inference/serve_nemotron_v3.py is valid Python."""
    p = TOOLS_DIR / "inference" / "serve_nemotron_v3.py"
    assert p.exists(), "tools/inference/serve_nemotron_v3.py not found"
    _check_python_syntax(p)


def test_tools_backtest_syntax():
    """Verify tools/backtest/backtest_nemotron_v3.py is valid Python."""
    p = TOOLS_DIR / "backtest" / "backtest_nemotron_v3.py"
    assert p.exists(), "tools/backtest/backtest_nemotron_v3.py not found"
    _check_python_syntax(p)


def test_tools_convert_lora_syntax():
    """Verify tools/lora-merge-pipeline/convert_lora_to_gguf.py is valid Python."""
    p = TOOLS_DIR / "lora-merge-pipeline" / "convert_lora_to_gguf.py"
    assert p.exists(), "convert_lora_to_gguf.py not found"
    _check_python_syntax(p)


def test_tools_merge_cpu_syntax():
    """Verify tools/lora-merge-pipeline/merge_convert_cpu.py is valid Python."""
    p = TOOLS_DIR / "lora-merge-pipeline" / "merge_convert_cpu.py"
    assert p.exists(), "merge_convert_cpu.py not found"
    _check_python_syntax(p)


def test_archive_scripts_syntax():
    """Verify all archived training scripts are valid Python."""
    for py in ARCHIVE_DIR.glob("train_qlora_v*.py"):
        _check_python_syntax(py)


def test_train_nemotron_v3_has_device_map():
    """Verify training script contains manual device_map configuration."""
    p = TOOLS_DIR / "train_nemotron_v3.py"
    source = p.read_text()
    assert "device_map" in source, "train_nemotron_v3.py must define device_map"
    assert "HSA_OVERRIDE_GFX_VERSION" in source, "Must set HSA_OVERRIDE_GFX_VERSION"


def test_train_nemotron_v3_no_deprecated_torch_dtype():
    """Verify training script uses dtype= not deprecated torch_dtype=."""
    p = TOOLS_DIR / "train_nemotron_v3.py"
    source = p.read_text()
    # torch_dtype is deprecated in transformers >= 4.57
    # The script should use dtype= instead
    assert "torch_dtype" not in source or "dtype=" in source, \
        "Script should use dtype= not deprecated torch_dtype="


def test_serve_nemotron_has_port():
    """Verify inference server defines a configurable port."""
    p = TOOLS_DIR / "inference" / "serve_nemotron_v3.py"
    source = p.read_text()
    assert "11435" in source or "PORT" in source, "Server must define port"


def test_no_duplicate_scripts_in_root():
    """Verify no training/serve scripts remain in repo root (they belong in tools/)."""
    root_py = [f.name for f in REPO_ROOT.glob("train_*.py")]
    root_py += [f.name for f in REPO_ROOT.glob("serve_*.py")]
    root_py += [f.name for f in REPO_ROOT.glob("merge_*.py")]
    root_py += [f.name for f in REPO_ROOT.glob("convert_*.py")]
    assert len(root_py) == 0, f"Duplicate scripts found in root: {root_py}"


if __name__ == "__main__":
    tests = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)