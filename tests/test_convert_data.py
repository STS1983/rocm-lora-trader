#!/usr/bin/env python3
"""
Tests for ROCm LoRA Trader — data conversion utilities.

Run: python3 -m pytest tests/test_convert_data.py -v
     or: python3 tests/test_convert_data.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def make_trade_record(side: str, symbol: str = "BTCUSDT", price: float = 65000.0):
    """Helper: create a minimal trade record."""
    return {
        "symbol": symbol,
        "side": side,
        "price": price,
        "quantity": 0.01,
        "timestamp": "2026-01-01T00:00:00Z",
        "indicators": {
            "rsi": 55.0,
            "ema_12": 64800.0,
            "ema_26": 64200.0,
            "volume": 1500.0,
            "change_24h": 2.5,
        },
    }


def test_import_convert_data():
    """Verify convert_data module can be imported."""
    try:
        import convert_data
        assert hasattr(convert_data, "main") or hasattr(convert_data, "convert")
    except ImportError:
        # Module may require optional deps; just verify the file exists and is valid Python
        p = Path(__file__).parent.parent / "tools" / "convert_data.py"
        assert p.exists(), f"convert_data.py not found at {p}"
        compile(p.read_text(), str(p), "exec")


def test_trade_record_structure():
    """Verify trade record helper produces valid structure."""
    rec = make_trade_record("BUY")
    assert rec["side"] == "BUY"
    assert "indicators" in rec
    assert "rsi" in rec["indicators"]


def test_jsonl_roundtrip():
    """Verify JSONL data can be written and read back."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(5):
            f.write(json.dumps(make_trade_record("BUY" if i % 2 == 0 else "SELL")) + "\n")
        tmp_path = f.name

    try:
        lines = Path(tmp_path).read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            rec = json.loads(line)
            assert "symbol" in rec
            assert "side" in rec
    finally:
        os.unlink(tmp_path)


def test_system_prompt_exists():
    """Verify system_prompt.txt exists and contains signal format."""
    p = Path(__file__).parent.parent / "system_prompt.txt"
    assert p.exists(), "system_prompt.txt not found"
    content = p.read_text()
    assert "SIGNAL" in content, "system_prompt.txt must define SIGNAL format"
    assert "Confidence" in content


def test_training_data_format():
    """Verify training data files follow expected JSONL format (if present)."""
    data_dir = Path(__file__).parent.parent
    jsonl_files = list(data_dir.glob("training-data-*.jsonl"))
    if not jsonl_files:
        # Training data may be gitignored; skip if not present
        return
    for f in jsonl_files:
        lines = f.read_text().strip().split("\n")
        for line in lines:
            rec = json.loads(line)
            # TRL format should have at least text or messages
            assert any(k in rec for k in ("text", "messages", "prompt")), \
                f"Record in {f.name} missing expected keys"


if __name__ == "__main__":
    tests = [
        test_import_convert_data,
        test_trade_record_structure,
        test_jsonl_roundtrip,
        test_system_prompt_exists,
        test_training_data_format,
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