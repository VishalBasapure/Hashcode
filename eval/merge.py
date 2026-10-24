"""Rebuild a merged model from a saved LoRA adapter (no retraining needed).

  python eval/merge.py --adapter /kaggle/working/runs/B1_s0_1024-1327/adapter --out /tmp/merged/B1_s0
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/tmp/llama")
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.adapter)  # includes the added <|predictor_k|> tokens
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16)
model.resize_token_embeddings(len(tok))  # same vocab as training; trained embedding rows load from the adapter
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
print("merged ->", a.out)