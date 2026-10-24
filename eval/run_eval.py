"""Evaluate one model with the same settings every time, then write metrics.json.

English (lm-eval, in its own venv): IFEval (chat template), MMLU (--limit 18 per subject),
HellaSwag (--limit 1000). Hindi/Kannada: Belebele hin + kan. Hindi metrics: eval/hindi_eval.py.

  python eval/run_eval.py --model /tmp/llama --name P1
  python eval/run_eval.py --model /tmp/merged/B1_s0 --name B1_s0
"""
import argparse, glob, json, os, subprocess, sys, time

TASKS = [  # (tasks, extra flags, limit)
    ("ifeval", ["--apply_chat_template", "--gen_kwargs", "max_gen_toks=512"], None),
    ("belebele_hin_Deva,belebele_kan_Knda", [], None),
    ("mmlu", [], 18),
    ("hellaswag", [], 1000),
]
KEEP = {"ifeval": "prompt_level_strict_acc,none", "mmlu": "acc,none", "hellaswag": "acc_norm,none",
        "belebele_hin_Deva": "acc,none", "belebele_kan_Knda": "acc,none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="/kaggle/working/evals")
    ap.add_argument("--lm_eval", default="/tmp/ev/bin/lm_eval")
    ap.add_argument("--only", default="", help="comma list: ifeval,belebele,mmlu,hellaswag,hindi")
    a = ap.parse_args()
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    only = set(filter(None, a.only.split(",")))
    times = {}

    for tasks, extra, limit in TASKS:
        key = tasks.split("_")[0].split(",")[0]
        if only and key not in only:
            continue
        cmd = [a.lm_eval, "--model", "hf", "--model_args", f"pretrained={a.model},dtype=float16",
               "--tasks", tasks, "--num_fewshot", "0", "--seed", "0",
               "--batch_size", "16" if tasks == "ifeval" else "auto",
               "--log_samples", "--output_path", os.path.join(out, "lm_eval")] + extra
        if limit:
            cmd += ["--limit", str(limit)]
        t0 = time.time()
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        times[tasks] = round(time.time() - t0)

    if not only or "hindi" in only:
        t0 = time.time()
        subprocess.run([sys.executable, "eval/hindi_eval.py", "--model", a.model, "--out", out], check=True)
        times["hindi"] = round(time.time() - t0)

    metrics = {"name": a.name, "model": a.model}
    for f in glob.glob(os.path.join(out, "lm_eval", "**", "results_*.json"), recursive=True):
        res = json.load(open(f))["results"]
        for task, metric in KEEP.items():
            if task in res and metric in res[task]:
                metrics[task] = res[task][metric]
    hpath = os.path.join(out, "hindi.json")
    if os.path.exists(hpath):
        h = json.load(open(hpath, encoding="utf-8"))
        metrics["hindi_val_nll"], metrics["hindi_script_rate"] = h["val_nll"], h["script_rate"]
    eng = [metrics[k] for k in ("ifeval", "mmlu", "hellaswag") if k in metrics]
    if len(eng) == 3:
        metrics["EA"] = sum(eng) / 3
    metrics["eval_seconds"] = times
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()