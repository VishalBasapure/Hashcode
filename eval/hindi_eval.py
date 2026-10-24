"""Hindi ability (MA) metrics on the 300 held-out Hindi validation rows.

  val_nll  : mean per-token loss on the Hindi responses (lower = better Hindi)
  script   : share of greedy replies (first --n_gen prompts) that are >= 80% Devanagari letters

Writes <out>/hindi.json with per-item values (for bootstrap CIs) and 5 sample replies.
  python eval/hindi_eval.py --model /tmp/merged/J1_s0 --out /kaggle/working/evals/J1_s0
"""
import argparse, json, os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DATE = "26 Jul 2024"


def chat_ids(tok, user, assistant=None, gen=False):
    msgs = [{"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen, date_string=DATE)
    return list(out["input_ids"] if hasattr(out, "keys") else out)


def deva_ratio(text):
    deva = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
    other = sum(1 for c in text if c.isalpha() and not (0x0900 <= ord(c) <= 0x097F))
    return deva / max(deva + other, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_gen", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda").eval()
    rows = [json.loads(l) for l in open(os.path.join(a.data, "val_hi.jsonl"), encoding="utf-8") if l.strip()]

    # 1) per-token loss on held-out Hindi responses
    nll = []
    with torch.no_grad():
        for i in range(0, len(rows), a.bs):
            ids_b, lab_b = [], []
            for r in rows[i:i + a.bs]:
                p = chat_ids(tok, r["hi_prompt"], gen=True)
                f = chat_ids(tok, r["hi_prompt"], r["hi_response"])
                ids_b.append(f)
                lab_b.append([-100] * len(p) + f[len(p):])
            L = max(len(x) for x in ids_b)
            ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x in ids_b]).cuda()
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids_b]).cuda()
            lab = torch.tensor([x + [-100] * (L - len(x)) for x in lab_b]).cuda()
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            tgt = lab[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                 ignore_index=-100, reduction="none").view(tgt.shape)
            n = (tgt != -100).sum(1).clamp(min=1)
            nll += (ce.sum(1) / n).tolist()

    # 2) greedy replies -> Devanagari script rate
    tok.padding_side = "left"
    script, samples = [], []
    gen_rows = rows[:a.n_gen]
    with torch.no_grad():
        for i in range(0, len(gen_rows), a.bs):
            prompts = [chat_ids(tok, r["hi_prompt"], gen=True) for r in gen_rows[i:i + a.bs]]
            L = max(len(x) for x in prompts)
            ids = torch.tensor([[tok.pad_token_id] * (L - len(x)) + x for x in prompts]).cuda()
            att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in prompts]).cuda()
            out = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for r, o in zip(gen_rows[i:i + a.bs], out[:, L:]):
                text = tok.decode(o, skip_special_tokens=True)
                ok = deva_ratio(text) >= 0.8
                script.append(int(ok))
                if len(samples) < 5:
                    samples.append({"prompt": r["hi_prompt"][:200], "reply": text[:400], "devanagari_ok": ok})

    res = {"model": a.model, "val_nll": sum(nll) / len(nll), "script_rate": sum(script) / len(script),
           "n_nll": len(nll), "n_gen": len(script), "per_item_nll": nll, "per_item_script": script,
           "samples": samples}
    with open(os.path.join(a.out, "hindi.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"val_nll={res['val_nll']:.4f}  script_rate={res['script_rate']:.3f}")


if __name__ == "__main__":
    main()