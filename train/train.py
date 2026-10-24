"""BhashaJEPA trainer: one script for every method (B1, B2, B4, J1, J2).

  B1  naive CFT            lambda=0
  B2  English replay       lambda=0, + replay_en.jsonl
  B4  plain alignment      lambda=1, k=0, frozen cached anchor
  J1  BhashaJEPA           lambda=1, k=1, frozen cached anchor
  J2  shared target        lambda=1, k=1, live target with gradients (LLM-JEPA default)

Smoke test first (checks + 20 steps), then the real run:
  CUDA_VISIBLE_DEVICES=0 python train/train.py --method B1 --seed 0 --model /tmp/llama --smoke
  CUDA_VISIBLE_DEVICES=0 python train/train.py --method B1 --seed 0 --model /tmp/llama --merge_to /tmp/merged/B1_s0

Uses ideas from galilai-group/llm-jepa (Apache-2.0): predictor-token naming and last-token indexing.
Tested-against versions: transformers==4.55.2, peft==0.17.0.
"""
import argparse, hashlib, json, os, random, subprocess, time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, TaskType, get_peft_model

DATE = "26 Jul 2024"  # must match data/prepare.py
PRED = [f"<|predictor_{i}|>" for i in range(1, 11)]
PRESETS = {
    "B1": dict(lbd=0.0, k=0, anchor="frozen", replay=False),
    "B2": dict(lbd=0.0, k=0, anchor="frozen", replay=True),
    "B4": dict(lbd=1.0, k=0, anchor="frozen", replay=False),
    "J1": dict(lbd=1.0, k=1, anchor="frozen", replay=False),
    "J2": dict(lbd=1.0, k=1, anchor="live", replay=False),
}


# ---------------------------------------------------------------- data
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def chat(tok, user, assistant=None, gen=False):
    msgs = [{"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen, date_string=DATE))


def find_offset(tok):
    """Position of the last content token relative to sequence end (Llama: -2, before <|eot_id|>)."""
    ids = chat(tok, "hello" + PRED[0])
    pid = tok.convert_tokens_to_ids(PRED[0])
    pos = len(ids) - 1 - ids[::-1].index(pid)
    return pos - len(ids)


def lm_example(tok, user, resp, max_len):
    prompt = chat(tok, user, gen=True)
    full = chat(tok, user, resp)
    assert full[:len(prompt)] == prompt, "chat template prefix mismatch: label masking would be wrong"
    full = full[:max_len]
    labels = ([-100] * len(prompt) + full[len(prompt):])[:len(full)]
    return full, labels


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def build_rows(tok, data_dir, k, use_jepa, replay, max_len):
    rows = []
    for r in read_jsonl(os.path.join(data_dir, "train_hi.jsonl")):
        ids, labels = lm_example(tok, r["hi_prompt"], r["hi_response"], max_len)
        row = {"input_ids": ids, "labels": labels}
        if use_jepa:
            suffix = "".join(PRED[j] for j in range(k - 1, -1, -1))  # ..., <|predictor_2|>, <|predictor_1|>
            row["pred_ids"] = chat(tok, r["hi_prompt"] + suffix)
            row["en_ids"] = chat(tok, r["en_prompt"])
            row["anchor_idx"] = r["anchor_idx"]
        rows.append(row)
    if replay:
        for r in read_jsonl(os.path.join(data_dir, "replay_en.jsonl")):
            ids, labels = lm_example(tok, r["en_prompt"], r["en_response"], max_len)
            rows.append({"input_ids": ids, "labels": labels})
    return rows


def pad(seqs, val):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [val] * (L - len(s)) for s in seqs])


def mask(seqs):
    L = max(len(s) for s in seqs)
    return torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs])


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        b = {"input_ids": pad([f["input_ids"] for f in feats], self.pad_id),
             "labels": pad([f["labels"] for f in feats], -100),
             "attention_mask": mask([f["input_ids"] for f in feats])}
        if all("pred_ids" in f for f in feats):
            for key in ("pred", "en"):
                seqs = [f[f"{key}_ids"] for f in feats]
                b[f"{key}_ids"], b[f"{key}_mask"] = pad(seqs, self.pad_id), mask(seqs)
            b["anchor_idx"] = torch.tensor([f["anchor_idx"] for f in feats])
        return b


# ---------------------------------------------------------------- model helpers
def backbone(model):
    model = getattr(model, "module", model)  # unwrap DataParallel/DDP if present
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model  # decoder stack; LoRA layers are injected inside it


def last_state(model, ids, att, offset):
    h = backbone(model)(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state  # after final norm
    idx = att.sum(1) + offset
    return h[torch.arange(h.size(0), device=h.device), idx]


@torch.no_grad()
def encode(model, seqs, pad_id, offset, device, bs=32):
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        out.append(last_state(model, pad(chunk, pad_id).to(device), mask(chunk).to(device), offset).float().cpu())
    return torch.cat(out)


# ---------------------------------------------------------------- trainer
class JepaTrainer(Trainer):
    def __init__(self, *a, lbd=0.0, anchor_mode="frozen", anchors=None, offset=-2, jepa_ratio=1.0,
                 log_path=None, **kw):
        super().__init__(*a, **kw)
        self.lbd, self.anchor_mode, self.anchors = lbd, anchor_mode, anchors
        self.offset, self.jepa_ratio, self.log_path = offset, jepa_ratio, log_path
        self.model_accepts_loss_kwargs = False  # we return a mean loss; let Trainer divide by grad accumulation
        self._acc = {"lm": 0.0, "n": 0, "jepa": 0.0, "nj": 0}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"], use_cache=False)
        loss = out.loss
        self._acc["lm"] += out.loss.detach().float().item()
        self._acc["n"] += 1
        if self.lbd > 0 and "pred_ids" in inputs and random.random() < self.jepa_ratio:
            z = last_state(model, inputs["pred_ids"], inputs["pred_mask"], self.offset)
            if self.anchor_mode == "frozen":  # cached from the base model: no gradient can reach it
                if self.anchors.device != z.device:
                    self.anchors = self.anchors.to(z.device)
                tgt = self.anchors[inputs["anchor_idx"]]
            else:  # J2: live target through the same adapter, with gradients (LLM-JEPA default)
                tgt = last_state(model, inputs["en_ids"], inputs["en_mask"], self.offset)
            jepa = 1 - F.cosine_similarity(z.float(), tgt.float(), dim=-1).mean()
            loss = loss + self.lbd * jepa
            self._acc["jepa"] += jepa.detach().item()
            self._acc["nj"] += 1
        return (loss, out) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._acc["n"]:
            logs["lm_loss"] = round(self._acc["lm"] / self._acc["n"], 5)
        if self._acc["nj"]:
            logs["jepa_loss"] = round(self._acc["jepa"] / self._acc["nj"], 5)
        self._acc = {"lm": 0.0, "n": 0, "jepa": 0.0, "nj": 0}
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"step": self.state.global_step, **logs}) + "\n")
        super().log(logs, *args, **kwargs)


# ---------------------------------------------------------------- smoke tests
def smoke_checks(model, tok, ds, collate, anchors, offset, a, device):
    print("\n===== SMOKE CHECKS =====")
    f = ds[0]
    lab = tok.decode([t for t in f["labels"] if t != -100])
    print("[1] supervised tokens (should be the Hindi response + end token only):\n", lab[:300])
    n_masked = next(i for i, t in enumerate(f["labels"]) if t != -100)
    assert 0 < n_masked < len(f["labels"]), "prompt masking is wrong"
    print(f"    {n_masked} prompt tokens masked; prompt ends with: {tok.decode(f['input_ids'][n_masked - 8:n_masked])!r}")

    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names), "something other than LoRA is trainable"
    print(f"[2] trainable tensors: {len(names)} (all LoRA)")

    if a.lbd > 0:
        p_tok = tok.convert_ids_to_tokens(f["pred_ids"][len(f["pred_ids"]) + offset])
        e_tok = tok.convert_ids_to_tokens(f["en_ids"][len(f["en_ids"]) + offset])
        print(f"[4] offset={offset}; pred view reads '{p_tok}', English view reads '{e_tok}'")
        if a.k >= 1:
            assert p_tok == PRED[0], "pred index is not on <|predictor_1|>"
        b = {k: v.to(device) for k, v in collate([ds[i] for i in range(4)]).items()}
        model.eval()
        with torch.no_grad():
            with model.disable_adapter():
                live = last_state(model, b["en_ids"], b["en_mask"], offset).float()
            if anchors is not None:
                c = F.cosine_similarity(live, anchors[b["anchor_idx"].cpu()].to(device), dim=-1)
                print(f"[3] cached vs live adapter-off anchor cosine: {c.tolist()}")
                assert c.min() > 0.999, "anchor cache does not match the frozen model"
            z = last_state(model, b["pred_ids"], b["pred_mask"], offset).float()
            c0 = F.cosine_similarity(z, live, dim=-1).mean().item()
        print(f"[5] step-0 cosine pred vs English anchor: {c0:.4f} (record this; ~1.0 means a trivial target)")
        if c0 > 0.98:
            print("WARNING: step-0 cosine is near 1; check which token is indexed")
        model.train()
        z = last_state(model, b["pred_ids"], b["pred_mask"], offset)
        (1 - F.cosine_similarity(z.float(), live, dim=-1).mean()).backward()
        g = sum(p.grad.abs().sum().item() for p in model.parameters() if p.requires_grad and p.grad is not None)
        print(f"[6] LoRA grad norm-sum from JEPA term alone: {g:.4e}")
        assert g > 0, "JEPA term sends no gradient to LoRA"
        model.zero_grad(set_to_none=True)
    print("===== checks passed; running 20 steps =====\n")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="/tmp/llama")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="/kaggle/working/runs")
    ap.add_argument("--lbd", type=float)
    ap.add_argument("--k", type=int)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--micro_bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)  # 4 x 8 = effective batch 32
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--jepa_ratio", type=float, default=1.0)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])  # T4 has no bf16
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--merge_to", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    p = PRESETS[a.method]
    a.lbd = p["lbd"] if a.lbd is None else a.lbd
    a.k = p["k"] if a.k is None else a.k
    a.anchor, a.replay = p["anchor"], p["replay"]
    assert not (a.replay and a.lbd > 0), "replay rows have no JEPA views"
    set_seed(a.seed)
    random.seed(a.seed)
    dtype = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    device = "cuda"

    name = f"{a.method}{a.tag}_s{a.seed}_{time.strftime('%m%d-%H%M')}" + ("_smoke" if a.smoke else "")
    run_dir = os.path.join(a.out, name)
    os.makedirs(run_dir, exist_ok=False)  # never overwrite a run

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.add_special_tokens({"additional_special_tokens": PRED})  # same vocab for every method
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
    tok.padding_side = "right"
    offset = find_offset(tok)

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(device)
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False

    use_jepa = a.lbd > 0
    rows = build_rows(tok, a.data, a.k, use_jepa, a.replay, a.max_len)
    ds = Rows(rows)
    collate = Collator(tok.pad_token_id)

    anchors = None
    if use_jepa and a.anchor == "frozen":
        data_hash = open(os.path.join(a.data, "DATA_HASH")).read().strip()
        tag = hashlib.sha1(f"{a.model}|{data_hash}|{offset}|{a.dtype}".encode()).hexdigest()[:10]
        cache = os.path.join(a.out, f"anchors_{tag}.pt")
        if os.path.exists(cache):
            anchors = torch.load(cache)
        else:
            en = [None] * len(ds)
            for r in rows:
                if "anchor_idx" in r:
                    en[r["anchor_idx"]] = r["en_ids"]
            model.eval()
            t0 = time.time()
            anchors = encode(model, en, tok.pad_token_id, offset, device)
            torch.save(anchors, cache)
            print(f"anchor cache {tuple(anchors.shape)} built in {time.time() - t0:.0f}s -> {cache}")

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.1,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    for n, prm in model.named_parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()  # fp32 LoRA weights: needed for fp16 grad scaling
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    cfg = vars(a) | {"offset": offset, "n_rows": len(ds), "git_commit": commit,
                     "data_hash": open(os.path.join(a.data, "DATA_HASH")).read().strip(), "run_dir": run_dir}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(json.dumps(cfg, indent=2))

    if a.smoke:
        model.to(device)
        with torch.autocast("cuda", dtype=dtype):
            smoke_checks(model, tok, ds, collate, anchors, offset, a, device)

    steps_per_epoch = max(len(ds) // (a.micro_bsz * a.grad_accum), 1)
    args = TrainingArguments(
        output_dir=run_dir, per_device_train_batch_size=a.micro_bsz, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, num_train_epochs=a.epochs, max_steps=20 if a.smoke else -1,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=1 if a.smoke else 5,
        save_strategy="steps", save_steps=max(steps_per_epoch // 2, 1), save_total_limit=2,
        fp16=a.dtype == "fp16", bf16=a.dtype == "bf16",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, dataloader_num_workers=0, report_to="none",
        seed=a.seed, data_seed=a.seed, optim="adamw_torch")
    trainer = JepaTrainer(model=model, args=args, train_dataset=ds, data_collator=collate, processing_class=tok,
                          lbd=a.lbd, anchor_mode=a.anchor, anchors=anchors, offset=offset,
                          jepa_ratio=a.jepa_ratio, log_path=os.path.join(run_dir, "log.jsonl"))
    t0 = time.time()
    result = trainer.train()
    secs = time.time() - t0
    summary = {"train_seconds": round(secs), "steps": trainer.state.global_step,
               "sec_per_step": round(secs / max(trainer.state.global_step, 1), 2),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), **result.metrics}
    print(json.dumps(summary, indent=2))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if a.smoke:
        print(f"Smoke done. Full run estimate: ~{summary['sec_per_step'] * steps_per_epoch * a.epochs / 60:.0f} min")
        return
    trainer.model.save_pretrained(os.path.join(run_dir, "adapter"))
    tok.save_pretrained(os.path.join(run_dir, "adapter"))
    if a.merge_to:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(a.merge_to, safe_serialization=True)
        tok.save_pretrained(a.merge_to)
        print("merged model ->", a.merge_to)
    print("adapter ->", os.path.join(run_dir, "adapter"))


if __name__ == "__main__":
    main()