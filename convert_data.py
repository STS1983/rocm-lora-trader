#!/usr/bin/env python3
"""
Convert training data to TRL SFT-compatible format for QLoRA fine-tuning.
Reads balanced-training-data.jsonl + trade-history.json
Outputs train/validation splits as JSONL files.
"""

import json
import random
import os
from pathlib import Path

# System prompt for the trading model
SYSTEM_PROMPT = """You are an expert cryptocurrency trading signal generator. Analyze the given market indicators and provide a clear trading signal.

Your response must follow this exact format:
SIGNAL | Confidence: X% | Pattern: PATTERN_NAME | Reasoning: explanation

Available patterns: UPTREND_BUY, DOWNTREND_SELL, OVERSOLD_REVERSAL, OVERBOUGHT_REVERSAL, CONSOLIDATION, BULLISH_BREAKOUT, DOWNTREND_CONTINUATION, TREND_EXHAUSTION, OVERSOLD_NO_REVERSAL

Consider: RSI levels, EMA crossovers, volume, trend direction, and 24h change.
Be decisive - always provide a signal. Never say HOLD unless truly consolidating."""

def convert_to_chat_format(sample: dict) -> dict:
    """Convert a {prompt, completion} sample to TRL chat format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["completion"]}
        ]
    }


def main():
    base_dir = Path("/home/nodeadmin/trading-llm/training")
    
    # Load training data
    samples = []
    with open(base_dir / "balanced-training-data.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples from balanced-training-data.jsonl")
    
    # Load trade history for reference
    trade_data = []
    trade_file = base_dir / "trade-history.json"
    if trade_file.exists():
        with open(trade_file) as f:
            trade_data = json.load(f)
        print(f"Loaded {len(trade_data)} trades from trade-history.json")
    
    # Convert to chat format
    chat_samples = [convert_to_chat_format(s) for s in samples]
    
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(chat_samples)
    
    # Split: 80% train, 20% validation
    split_idx = int(len(chat_samples) * 0.8)
    train_data = chat_samples[:split_idx]
    val_data = chat_samples[split_idx:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Save as JSONL
    train_file = base_dir / "train.jsonl"
    val_file = base_dir / "val.jsonl"
    
    with open(train_file, "w") as f:
        for sample in train_data:
            f.write(json.dumps(sample) + "\n")
    
    with open(val_file, "w") as f:
        for sample in val_data:
            f.write(json.dumps(sample) + "\n")
    
    print(f"\nSaved train data to {train_file}")
    print(f"Saved validation data to {val_file}")
    
    # Save system prompt for Ollama Modelfile
    with open(base_dir / "system_prompt.txt", "w") as f:
        f.write(SYSTEM_PROMPT)
    print(f"Saved system prompt to {base_dir / 'system_prompt.txt'}")
    
    # Print sample
    print("\n--- Sample train entry ---")
    print(json.dumps(train_data[0], indent=2))
    
    # Validate
    print("\n--- Validation check ---")
    for i, s in enumerate(train_data[:5]):
        assert "messages" in s, f"Sample {i} missing 'messages'"
        assert len(s["messages"]) == 3, f"Sample {i} has {len(s['messages'])} messages, expected 3"
        assert s["messages"][0]["role"] == "system"
        assert s["messages"][1]["role"] == "user"
        assert s["messages"][2]["role"] == "assistant"
    print("✅ All samples validated successfully!")


if __name__ == "__main__":
    main()