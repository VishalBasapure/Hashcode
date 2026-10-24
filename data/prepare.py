"""BhashaJEPA data prep: Bactrian-X Hindi joined to English twins by id.

Outputs (in --out):
  train_hi.jsonl    3000 rows  {id, anchor_idx, hi_prompt, hi_response, en_prompt}
  val_hi.jsonl       300 rows  {id, hi_prompt, hi_response, en_prompt}
  replay_en.jsonl    300 rows  {id, en_prompt, en_response}   English twins of train ids (B2)
  probe_tune.jsonl   100 rows  {id, en_prompt}                 for choosing lambda only
  probe_test.jsonl   200 rows  {id, en_prompt}                 for the reported drift plot
  stats.json, DATA_HASH

Usage:
  python prepare.py --tokenizer meta-llama/Llama-3.2-1B-Instruct --out data
"""
import argparse, hashlib, json, os, random

from datasets import load_dataset
from transformers import AutoTokenizer

DATE = "26 Jul 2024"  # fixed so the Llama chat template never changes tokens mid-hackathon


def load_lang(lang):
    try:
        return load_dataset("MBZUAI/Bactrian-X", lang, split="train")
    except Exception as e:  # loading-script problems on new `datasets` versions
        print(f"[{lang}] default loader failed ({e}); trying parquet branch")
        return load_dataset("parquet",
                            data_files=f"hf://datasets/MBZUAI/Bactrian-X@~parquet/{lang}/train/*.parquet",
                            split="train")


def prompt_of(row):
    inp = (row.get("input") or "").strip()
    return row["instruction"].strip() + (("\n\n" + inp) if inp else "")


def deva_ratio(text):
    deva = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
    other = sum(1 for c in text if c.isalpha() and not (0x0900 <= ord(c) <= 0x097F))
    return deva / max(deva + other, 1)


def latin_ratio(text):
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if c.isascii()) / max(len(letters), 1)


def n_tokens(tok, user, assistant=None):
    msgs = [{"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=assistant is None,
                                       date_string=DATE))


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--n_replay", type=int, default=300)
    ap.add_argument("--n_probe_tune", type=int, default=100)
    ap.add_argument("--n_probe_test", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--min_deva", type=float, default=0.8)
    ap.add_argument("--min_latin", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    hi = {r["id"]: r for r in load_lang("hi")}
    en = {r["id"]: r for r in load_lang("en")}
    stats = {"hi_rows": len(hi), "en_rows": len(en)}

    # cheap filters first: both sides present, non-empty, right script
    ids = sorted(set(hi) & set(en))
    stats["joined"] = len(ids)
    ok_hi = [i for i in ids if hi[i]["output"].strip() and deva_ratio(hi[i]["output"]) >= a.min_deva]
    stats["hi_script_ok"] = len(ok_hi)
    random.Random(a.seed).shuffle(ok_hi)

    # length filter (drop, never truncate) until train + val are filled
    need = a.n_train + a.n_val
    picked, too_long = [], 0
    for i in ok_hi:
        hp, hr, ep = prompt_of(hi[i]), hi[i]["output"].strip(), prompt_of(en[i])
        if n_tokens(tok, hp, hr) > a.max_len or n_tokens(tok, ep) > a.max_len:
            too_long += 1
            continue
        picked.append({"id": i, "hi_prompt": hp, "hi_response": hr, "en_prompt": ep})
        if len(picked) == need:
            break
    stats["too_long_skipped"] = too_long
    if len(picked) < need:
        raise SystemExit(f"Only {len(picked)} usable rows; lower --min_deva or raise --max_len and record it")

    train, val = picked[:a.n_train], picked[a.n_train:]
    for k, r in enumerate(train):
        r["anchor_idx"] = k
    used = {r["id"] for r in picked}

    # B2 replay: English twins (with English responses) of train ids
    replay = []
    for r in train:
        er = en[r["id"]]["output"].strip()
        if er and latin_ratio(er) >= a.min_latin and n_tokens(tok, r["en_prompt"], er) <= a.max_len:
            replay.append({"id": r["id"], "en_prompt": r["en_prompt"], "en_response": er})
        if len(replay) == a.n_replay:
            break

    # drift probes: English prompts from ids never used in training or validation
    rest = [i for i in ids if i not in used]
    random.Random(a.seed + 1).shuffle(rest)
    probes = []
    for i in rest:
        ep = prompt_of(en[i])
        if latin_ratio(ep) >= a.min_latin and n_tokens(tok, ep) <= a.max_len:
            probes.append({"id": i, "en_prompt": ep})
        if len(probes) == a.n_probe_tune + a.n_probe_test:
            break
    probe_tune, probe_test = probes[:a.n_probe_tune], probes[a.n_probe_tune:]

    files = {"train_hi.jsonl": train, "val_hi.jsonl": val, "replay_en.jsonl": replay,
             "probe_tune.jsonl": probe_tune, "probe_test.jsonl": probe_test}
    h = hashlib.sha256()
    for name, rows in files.items():
        write_jsonl(os.path.join(a.out, name), rows)
        stats[name] = len(rows)
        with open(os.path.join(a.out, name), "rb") as f:
            h.update(f.read())
    stats["settings"] = vars(a) | {"date_string": DATE}
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(a.out, "DATA_HASH"), "w") as f:
        f.write(h.hexdigest() + "\n")

    # leakage guard
    assert not ({r["id"] for r in probes} & used), "probe ids overlap train/val"

    print(json.dumps(stats, indent=2, default=str))
    print("DATA_HASH", h.hexdigest())
    for r in train[:3]:
        print("\n--- id", r["id"], "\nHI:", r["hi_prompt"][:200], "\nEN:", r["en_prompt"][:200])


if __name__ == "__main__":
    main()