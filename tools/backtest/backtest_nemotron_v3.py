#!/usr/bin/env python3
"""
Backtest Nemotron-Mini-4B Trader v3 against validation data.
Compares model signals with expected (ground truth) signals.
"""
import json, time, requests, sys
from pathlib import Path
from datetime import datetime

API_URL = 'http://localhost:11435/api/generate'
VAL_FILE = '/home/nodeadmin/trading-llm/training-v11/val_balanced.jsonl'
MAX_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 250  # Default: all 2502
BATCH_DELAY = 0.5  # seconds between requests

def load_val_data():
    data = []
    with open(VAL_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if 'messages' in item and len(item['messages']) >= 2:
                    data.append(item)
    print(f'[DATA] Loaded {len(data)} validation samples')
    return data

def parse_signal(text):
    """Parse signal from model response: SIGNAL | Confidence: X% | Pattern: Y | Reasoning: Z"""
    text = text.strip()
    for keyword in ['BUY', 'SELL', 'HOLD']:
        if keyword in text.upper():
            # Try to extract confidence
            conf = 50
            if 'Confidence:' in text:
                try:
                    conf_str = text.split('Confidence:')[1].split('%')[0].strip()
                    conf = int(conf_str)
                except:
                    pass
            return keyword, conf
    return 'UNKNOWN', 0

def get_expected(item):
    """Extract expected signal from validation data"""
    if 'correct' in item:
        # The assistant message contains the expected signal
        assistant_msg = item['messages'][-1]['content']
        signal, conf = parse_signal(assistant_msg)
        return signal, conf
    return 'UNKNOWN', 0

def query_model(system_prompt, user_prompt):
    """Query the Nemotron model via HF Transformers API"""
    try:
        resp = requests.post(API_URL, json={
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            'stream': False,
            'options': {'temperature': 0.3, 'top_p': 0.85, 'top_k': 40, 'num_predict': 128},
        }, timeout=120)
        if resp.status_code == 200:
            return resp.json().get('response', '')
        else:
            return f'ERROR: HTTP {resp.status_code}'
    except Exception as e:
        return f'ERROR: {e}'

def run_backtest():
    print('=' * 70, flush=True)
    print('Nemotron-Mini-4B Trader v3 — Backtest', flush=True)
    print(f'Samples: {MAX_SAMPLES} | API: {API_URL}', flush=True)
    print('=' * 70, flush=True)

    data = load_val_data()
    if MAX_SAMPLES < len(data):
        data = data[:MAX_SAMPLES]
        print(f'[DATA] Using first {MAX_SAMPLES} samples')

    results = {
        'total': 0, 'correct': 0, 'wrong': 0, 'unknown': 0,
        'buy_correct': 0, 'buy_total': 0,
        'sell_correct': 0, 'sell_total': 0,
        'hold_correct': 0, 'hold_total': 0,
        'errors': 0, 'latencies': [],
        'details': [],
    }

    for i, item in enumerate(data):
        messages = item['messages']
        system_prompt = messages[0]['content']
        user_prompt = messages[1]['content']
        expected_signal, expected_conf = get_expected(item)
        pnl_pct = item.get('pnl_pct', 0)
        symbol = item.get('symbol', '?')

        t0 = time.time()
        response = query_model(system_prompt, user_prompt)
        latency = time.time() - t0
        results['latencies'].append(latency)

        if response.startswith('ERROR'):
            results['errors'] += 1
            results['details'].append({
                'i': i, 'symbol': symbol, 'expected': expected_signal,
                'got': 'ERROR', 'correct': False, 'latency': latency,
                'pnl_pct': pnl_pct, 'response': response[:100],
            })
            results['total'] += 1
            print(f'  [{i+1}/{len(data)}] ❌ ERROR ({latency:.1f}s)', flush=True)
            time.sleep(BATCH_DELAY)
            continue

        got_signal, got_conf = parse_signal(response)
        is_correct = got_signal == expected_signal

        results['total'] += 1
        if is_correct:
            results['correct'] += 1
        elif got_signal == 'UNKNOWN':
            results['unknown'] += 1
        else:
            results['wrong'] += 1

        # Track per-signal accuracy
        if expected_signal == 'BUY':
            results['buy_total'] += 1
            if is_correct: results['buy_correct'] += 1
        elif expected_signal == 'SELL':
            results['sell_total'] += 1
            if is_correct: results['sell_correct'] += 1
        elif expected_signal == 'HOLD':
            results['hold_total'] += 1
            if is_correct: results['hold_correct'] += 1

        emoji = '✅' if is_correct else '❌'
        print(f'  [{i+1}/{len(data)}] {emoji} {symbol} expected={expected_signal} got={got_signal} ({latency:.1f}s, conf={got_conf}%)', flush=True)

        results['details'].append({
            'i': i, 'symbol': symbol, 'expected': expected_signal, 'expected_conf': expected_conf,
            'got': got_signal, 'got_conf': got_conf, 'correct': is_correct,
            'latency': latency, 'pnl_pct': pnl_pct,
            'response': response[:200],
        })

        time.sleep(BATCH_DELAY)

    # Calculate metrics
    accuracy = results['correct'] / max(1, results['total']) * 100
    avg_latency = sum(results['latencies']) / max(1, len(results['latencies']))
    buy_acc = results['buy_correct'] / max(1, results['buy_total']) * 100
    sell_acc = results['sell_correct'] / max(1, results['sell_total']) * 100
    hold_acc = results['hold_correct'] / max(1, results['hold_total']) * 100

    # PnL simulation: if correct signal, take pnl_pct; if wrong, take -pnl_pct
    simulated_pnl = sum(d['pnl_pct'] if d['correct'] else -d['pnl_pct'] for d in results['details'] if d['got'] != 'ERROR')

    print('\n' + '=' * 70, flush=True)
    print('BACKTEST RESULTS', flush=True)
    print('=' * 70, flush=True)
    print(f'Total samples:    {results["total"]}', flush=True)
    print(f'Correct:          {results["correct"]} ({accuracy:.1f}%)', flush=True)
    print(f'Wrong:            {results["wrong"]}', flush=True)
    print(f'Unknown:          {results["unknown"]}', flush=True)
    print(f'Errors:           {results["errors"]}', flush=True)
    print(f'', flush=True)
    print(f'BUY accuracy:     {results["buy_correct"]}/{results["buy_total"]} ({buy_acc:.1f}%)', flush=True)
    print(f'SELL accuracy:    {results["sell_correct"]}/{results["sell_total"]} ({sell_acc:.1f}%)', flush=True)
    print(f'HOLD accuracy:    {results["hold_correct"]}/{results["hold_total"]} ({hold_acc:.1f}%)', flush=True)
    print(f'', flush=True)
    print(f'Avg latency:      {avg_latency:.1f}s', flush=True)
    print(f'Simulated PnL:    {simulated_pnl:+.2f}%', flush=True)
    print('=' * 70, flush=True)

    # Save results
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': 'nemotron-trader-v3',
        'samples': results['total'],
        'accuracy': accuracy,
        'correct': results['correct'],
        'wrong': results['wrong'],
        'unknown': results['unknown'],
        'errors': results['errors'],
        'buy_accuracy': buy_acc,
        'sell_accuracy': sell_acc,
        'hold_accuracy': hold_acc,
        'avg_latency_s': avg_latency,
        'simulated_pnl_pct': simulated_pnl,
        'details': results['details'],
    }
    report_path = '/home/nodeadmin/trading-llm/output/nemotron-trader-v3/backtest_results.json'
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n[SAVED] Report: {report_path}', flush=True)

if __name__ == '__main__':
    run_backtest()
