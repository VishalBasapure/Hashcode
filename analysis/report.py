"""Make every figure and the results table from whatever runs/evals exist so far.

  python analysis/report.py            # on Kaggle; re-run any time, it only plots what exists

Reads   /kaggle/working/runs/*/log.jsonl            (training curves)
        /kaggle/working/evals/*/metrics.json         (scores)
        /kaggle/working/evals/*/lm_eval/**/samples_ifeval*.jsonl  (per-prompt IFEval, for CIs)
Writes  /kaggle/working/figures/*.png, results.csv, results.md
"""
import argparse, glob, json, os, random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"P1": "#888780", "B1": "#D85A30", "B2": "#BA7517", "B4": "#378ADD", "J1": "#1D9E75", "J2": "#7F77DD"}
LABELS = {"P1": "P1 original", "B1": "B1 naive CFT", "B2": "B2 English replay", "B4": "B4 plain alignment",
          "J1": "J1 BhashaJEPA", "J2": "J2 shared target"}


def method_of(name):
    return name.split("_")[0][:2]


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def bootstrap_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def ifeval_items(eval_dir):
    files = glob.glob(os.path.join(eval_dir, "lm_eval", "**", "samples_ifeval*.jsonl"), recursive=True)
    if not files:
        return None
    vals = []
    for line in open(sorted(files)[-1]):
        d = json.loads(line)
        v = d.get("prompt_level_strict_acc")
        if v is not None:
            vals.append(float(v))
    return vals or None


def training_curves(runs_dir, fig_dir):
    runs = sorted(d for d in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(d) and "smoke" not in d)
    curves = {}
    for d in runs:
        p = os.path.join(d, "log.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        rows = [r for r in rows if "lm_loss" in r]
        if rows:
            curves[os.path.basename(d)] = rows
    if not curves:
        print("no training logs yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, rows in curves.items():
        m = method_of(name)
        lab = f"{LABELS.get(m, m)} ({name.split('_')[1]})"
        axes[0].plot([r["step"] for r in rows], [r["lm_loss"] for r in rows], color=COLORS.get(m), label=lab, lw=1.8)
        jr = [r for r in rows if "jepa_loss" in r]
        if jr:
            axes[1].plot([r["step"] for r in jr], [1 - r["jepa_loss"] for r in jr], color=COLORS.get(m), label=lab, lw=1.8)
    style(axes[0], "Learning Hindi: language-modelling loss (lower = better)", "training step", "LM loss")
    style(axes[1], "Hindi meaning vs frozen English meaning (cosine, higher = closer)", "training step", "cosine similarity")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "training_curves.png"), dpi=160)
    plt.close(fig)
    print("wrote training_curves.png")


def results(evals_dir, fig_dir, out_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(evals_dir, "*", "metrics.json"))):
        m = json.load(open(p))
        m["method"] = method_of(m["name"])
        items = ifeval_items(os.path.dirname(p))
        if items:
            m["ifeval_ci"] = bootstrap_ci(items)
        rows.append(m)
    if not rows:
        print("no metrics.json yet")
        return
    order = list(COLORS)
    rows.sort(key=lambda r: (order.index(r["method"]) if r["method"] in order else 99, r["name"]))

    # retention ratio vs P1 and mean B1 (only when both exist)
    def mean_of(meth, key):
        v = [r[key] for r in rows if r["method"] == meth and key in r]
        return sum(v) / len(v) if v else None
    for key, tag in (("EA", "R_EA"), ("ifeval", "R_ifeval")):
        p1, b1 = mean_of("P1", key), mean_of("B1", key)
        if p1 is not None and b1 is not None and abs(p1 - b1) > 1e-9:
            for r in rows:
                if key in r:
                    r[tag] = (r[key] - b1) / (p1 - b1)

    cols = ["name", "ifeval", "mmlu", "hellaswag", "EA", "belebele_hin_Deva", "belebele_kan_Knda",
            "hindi_val_nll", "hindi_script_rate", "R_ifeval", "R_EA"]
    with open(os.path.join(out_dir, "results.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]))
                             for c in cols) + "\n")
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join("" if r.get(c) is None else (f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]))
                                      for c in cols) + " |\n")
    print(open(os.path.join(out_dir, "results.md")).read())

    # 1) English instruction following, with 95% CI
    have = [r for r in rows if "ifeval" in r]
    if have:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, r in enumerate(have):
            y = r["ifeval"] * 100
            ax.bar(i, y, color=COLORS.get(r["method"]), width=0.6)
            if "ifeval_ci" in r:
                lo, hi = (v * 100 for v in r["ifeval_ci"])
                ax.errorbar(i, y, yerr=[[y - lo], [hi - y]], color="#2C2C2A", capsize=4, lw=1)
            ax.text(i, y / 2, f"{y:.1f}", ha="center", va="center", fontsize=10, color="white")
        p1 = [r for r in have if r["method"] == "P1"]
        if p1:
            ax.axhline(p1[0]["ifeval"] * 100, color=COLORS["P1"], ls="--", lw=1, label="original model (P1)")
            ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.set_xticks(range(len(have)), [r["name"] for r in have])
        style(ax, "English instruction following (IFEval, 95% CI)", None, "prompt-level strict accuracy (%)")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "english_ifeval.png"), dpi=160)
        plt.close(fig)
        print("wrote english_ifeval.png")

    # 2) all benchmarks, grouped bars
    bench = [("ifeval", "IFEval (EN)"), ("mmlu", "MMLU (EN)"), ("hellaswag", "HellaSwag (EN)"),
             ("belebele_hin_Deva", "Belebele Hindi"), ("belebele_kan_Knda", "Belebele Kannada")]
    bench = [b for b in bench if any(b[0] in r for r in rows)]
    if bench:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        w = 0.8 / len(rows)
        for j, r in enumerate(rows):
            xs = [i + j * w for i in range(len(bench))]
            ax.bar(xs, [r.get(b, 0) * 100 for b, _ in bench], width=w, color=COLORS.get(r["method"]), label=r["name"])
        ax.set_xticks([i + 0.4 - w / 2 for i in range(len(bench))], [lab for _, lab in bench])
        style(ax, "Every benchmark, every model", None, "accuracy (%)")
        ax.legend(frameon=False, fontsize=9, ncol=len(rows))
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "all_benchmarks.png"), dpi=160)
        plt.close(fig)
        print("wrote all_benchmarks.png")

    # 3) Pareto: English ability vs Hindi ability
    eng_key = "EA" if any("EA" in r for r in rows) else "ifeval"
    pts = [r for r in rows if eng_key in r and "hindi_val_nll" in r]
    if pts:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for r in pts:
            ax.scatter(r["hindi_val_nll"], r[eng_key] * 100, s=90, color=COLORS.get(r["method"]), zorder=3)
            ax.annotate(r["name"], (r["hindi_val_nll"], r[eng_key] * 100), textcoords="offset points", xytext=(8, 4), fontsize=9)
        ax.invert_xaxis()
        style(ax, "Trade-off: better Hindi (right) vs kept English (up)",
              "Hindi loss on held-out answers (lower = better, axis reversed)",
              "English ability (%)" if eng_key == "EA" else "IFEval (%)")
        ax.text(0.98, 0.97, "best corner", transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#0F6E56")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "pareto.png"), dpi=160)
        plt.close(fig)
        print("wrote pareto.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working")
    a = ap.parse_args()
    fig_dir = os.path.join(a.work, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    training_curves(os.path.join(a.work, "runs"), fig_dir)
    results(os.path.join(a.work, "evals"), fig_dir, a.work)


if __name__ == "__main__":
    main()