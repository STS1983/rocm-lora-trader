#!/usr/bin/env python3
"""
Nemotron-Mini-4B Trader v3 — HF Transformers Inference Server
Serves the merged model on port 11435 (separate from Ollama on 11434).
API-compatible with Ollama for easy integration.
"""
import os, json, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Single GPU for inference

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = '/home/nodeadmin/trading-llm/output/nemotron-trader-v3/merged'
PORT = 11435

print(f'[START] Loading model from {MODEL_PATH}...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
)
print(f'[START] Model loaded on {model.device}', flush=True)
print(f'[START] Serving on port {PORT}', flush=True)

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length))
        
        prompt = body.get('prompt', '')
        stream = body.get('stream', False)
        options = body.get('options', {})
        
        temp = options.get('temperature', 0.3)
        top_p = options.get('top_p', 0.85)
        top_k = options.get('top_k', 40)
        max_tokens = options.get('num_predict', 128)
        
        messages = body.get('messages', [])
        if messages:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        
        inputs = tokenizer(text, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp, top_p=top_p, top_k=top_k,
                do_sample=temp > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            'model': 'nemotron-trader-v3',
            'response': response,
            'done': True,
        }).encode())
    
    def do_GET(self):
        if self.path == '/api/tags':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'models': [{'name': 'nemotron-trader-v3', 'size': '8381 MB'}]
            }).encode())
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok'}).encode())
    
    def log_message(self, format, *args):
        print(f'[API] {args[0]}', flush=True)

server = HTTPServer(('0.0.0.0', PORT), Handler)
print(f'[READY] nemotron-trader-v3 API on http://0.0.0.0:{PORT}', flush=True)
server.serve_forever()
