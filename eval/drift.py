"""Per-layer representation drift on English probe prompts (the CFT paper's mechanism metric).

For each model and each layer: drift = ||Cov_model - Cov_P1||_F / ||Cov_P1||_F over the last-token
hidden states of the 200 held-out English probe prompts. Higher = the model's English internals moved more.

  python eval/drift.py --models P1=/tmp/llama B1_s0=/tmp/merged/B1_s0 J1_s0=/tmp/merged/J1_s0
"""
import argparse, json, os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATE = "26 Jul 2024"
COLORS = {"P1": "#888780", "B1": "#D85A30", "B2": "#BA7517", "B4": "#378ADD", "J1": "#1D9E75", "J2": "#7F77DD"}


@torch.no_grad()
def layer_states(path, prompts, bs=16):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).to("cuda").eval()
    per_layer = None
    for i in range(0, len(prompts), bs):
        seqs = []
        for p in prompts[i:i + bs]:
            out = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                          add_generation_prompt=True, date_string=DATE)
            seqs.append(list(out["input_ids"] if hasattr(out, "keys") else out))
        L = max(len(s) for s in seqs)
        ids = torch.tensor([s + [tok.pad_token_id] * (L - len(s)) for s in seqs]).cuda()
        att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs]).cuda()
        hs = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states
        idx = att.sum(1) - 1
        rows = [h[torch.arange(h.size(0)), idx].float().cpu() for h in hs]
        per_layer = rows if per_layer is None else [torch.cat([a, b]) for a, b in zip(per_layer, rows)]
    del model
    torch.cuda.empty_cache()
    return per_layer


def cov(x):
    x = x - x.mean(0, keepdim=True)
    return x.T @ x / (x.size(0) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path; must include P1=...")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="/kaggle/working")
    a = ap.parse_args()
    models = dict(m.split("=", 1) for m in a.models)
    prompts = [json.loads(l)["en_prompt"] for l in open(os.path.join(a.data, "probe_test.jsonl"), encoding="utf-8") if l.strip()]

    ref = [cov(x) for x in layer_states(models["P1"], prompts)]
    result = {}
    for name, path in models.items():
        if name == "P1" or not os.path.exists(path):
            continue
        states = layer_states(path, prompts)
        result[name] = [((cov(x) - r).norm() / r.norm()).item() for x, r in zip(states, ref)]
        print(name, "mean drift", round(sum(result[name]) / len(result[name]), 4))

    with open(os.path.join(a.out, "drift.json"), "w") as f:
        json.dump(result, f, indent=1)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, vals in result.items():
        ax.plot(range(len(vals)), vals, marker="o", ms=3, lw=1.8, color=COLORS.get(name[:2]), label=name)
    ax.set_title("How much each layer's English representations moved vs the original model (lower = less drift)",
                 fontsize=11, loc="left")
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("relative covariance shift")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "figures", "drift.png"), dpi=160)
    print("wrote drift.png")


if __name__ == "__main__":
    main()