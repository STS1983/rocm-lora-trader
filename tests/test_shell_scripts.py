#!/usr/bin/env python3
"""
Tests for ROCm LoRA Trader — shell script validation.

Verifies shell scripts are syntactically valid and contain expected patterns.

Run: python3 -m pytest tests/test_shell_scripts.py -v
     or: python3 tests/test_shell_scripts.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _check_bash_syntax(filepath: Path):
    """Verify a bash script is syntactically valid."""
    result = subprocess.run(
        ["bash", "-n", str(filepath)],
        capture_output=True, text=True
    )
    assert result.returncode == 0, \
        f"bash syntax error in {filepath}: {result.stderr}"


def test_auto_retrain_syntax():
    """Verify auto-retrain.sh is valid bash."""
    p = REPO_ROOT / "auto-retrain.sh"
    assert p.exists(), "auto-retrain.sh not found"
    _check_bash_syntax(p)


def test_monitor_training_syntax():
    """Verify monitor_training.sh is valid bash."""
    p = REPO_ROOT / "monitor_training.sh"
    assert p.exists(), "monitor_training.sh not found"
    _check_bash_syntax(p)


def test_auto_retrain_has_error_handling():
    """Verify auto-retrain.sh uses set -euo pipefail."""
    p = REPO_ROOT / "auto-retrain.sh"
    content = p.read_text()
    assert "set -euo pipefail" in content or "set -e" in content, \
        "auto-retrain.sh should have error handling (set -e)"


def test_gitignore_excludes_weights():
    """Verify .gitignore excludes model weight files."""
    p = REPO_ROOT / ".gitignore"
    assert p.exists(), ".gitignore not found"
    content = p.read_text()
    assert "*.gguf" in content, ".gitignore must exclude *.gguf"
    assert "*.safetensors" in content, ".gitignore must exclude *.safetensors"


def test_requirements_exists():
    """Verify requirements.txt exists and lists core dependencies."""
    p = REPO_ROOT / "requirements.txt"
    assert p.exists(), "requirements.txt not found"
    content = p.read_text()
    assert "torch" in content, "requirements.txt must list torch"
    assert "transformers" in content, "requirements.txt must list transformers"
    assert "peft" in content, "requirements.txt must list peft"


def test_license_exists():
    """Verify LICENSE file exists."""
    p = REPO_ROOT / "LICENSE"
    assert p.exists(), "LICENSE not found"
    content = p.read_text()
    assert "MIT" in content, "LICENSE should be MIT license"


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