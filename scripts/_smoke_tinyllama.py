"""Smoke test: load GPT-2 from ~/models/hf and harvest one hidden state.

Goal: confirm we can load a DVC-tracked pretrained causal LM, run forward,
and pull a mid-layer hidden state. Nothing more.
"""
import os
from pathlib import Path

os.environ["HF_HOME"] = str(Path.home() / "models" / "hf")
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # never reach out to the hub

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = "gpt2"

print(f"HF_HOME = {os.environ['HF_HOME']}")
print(f"Loading tokenizer: {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"  vocab_size = {tok.vocab_size}")

print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
model.eval()
print(f"  n_params = {sum(p.numel() for p in model.parameters()):,}")
print(f"  hidden_size = {model.config.hidden_size}")
print(f"  n_layers = {model.config.num_hidden_layers}")

import torch  # noqa: E402

text = "def add(x, y): return x + y"
enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
print(f"\nText: {text!r}")
print(f"  n_tokens = {enc['input_ids'].shape[1]}")

with torch.no_grad():
    out = model(input_ids=enc["input_ids"], output_hidden_states=True)
hs = out.hidden_states  # tuple of (n_layers + 1) tensors, each (1, seq, d)
mid = hs[len(hs) // 2]
print(f"  mid-layer hidden_states[{len(hs)//2}] shape = {tuple(mid.shape)}")
print(f"  mid[0, -1, :8] = {mid[0, -1, :8].tolist()}")

print("\nOK: smoke test passed.")
