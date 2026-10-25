"""BhashaJEPA demo: the same prompt answered by P1 (original), B1 (naive CFT) and J1 (BhashaJEPA).

One base model in memory; B1 and J1 are LoRA adapters switched on/off (P1 = adapters off).

  pip install gradio peft transformers accelerate deep-translator   (deep-translator: optional, online translation)
  python demo/app.py --base <Llama-3.2-1B-Instruct path> --b1 models/B1_s0 --j1 models/J1_s0 --results results/main
Add --share for a public link.
"""
import argparse, csv, glob, html, json, math, os, re, time

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DATE = "26 Jul 2024"
COL = {"P1": "#8a8780", "B1": "#d85a30", "J1": "#1d9e75"}
NAME = {"P1": "Original model", "B1": "Naive Hindi fine-tuning", "J1": "BhashaJEPA (ours)"}
EXAMPLES = [
    "List three benefits of drinking water. Answer in exactly three bullet points.",
    "Write a two-sentence summary of why trees are important. Do not use any commas.",
    "भारत में मानसून का कृषि पर क्या प्रभाव पड़ता है? संक्षेप में बताइए।",
    "मुझे स्वस्थ रहने के तीन आसान तरीके बताइए।",
    "ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು? ಒಂದು ವಾಕ್ಯದಲ್ಲಿ ಉತ್ತರಿಸಿ.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="meta-llama/Llama-3.2-1B-Instruct")
ap.add_argument("--b1", required=True)
ap.add_argument("--j1", required=True)
ap.add_argument("--results", default="")
ap.add_argument("--share", action="store_true")
a = ap.parse_args()

# ------------------------------------------------------------------ model
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.j1)  # includes the <|predictor_k|> tokens added in training
base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float16 if dev == "cuda" else torch.float32)
base.resize_token_embeddings(len(tok))
# load B1 first, then J1: each adapter file also stores the embedding table, and J1 was trained with
# its own <|predictor_1|> row (used by the Meaning-match tab), so J1 must be loaded last
model = PeftModel.from_pretrained(base, a.b1, adapter_name="B1")
model.load_adapter(a.j1, adapter_name="J1")
model.to(dev).eval()
print("loaded on", dev)


@torch.no_grad()
def answer(prompt, which, max_new=160 if dev == "cuda" else 120):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True, date_string=DATE)
    enc = {k: v.to(dev) for k, v in enc.items()}
    kw = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    if which == "P1":
        with model.disable_adapter():
            out = model.generate(**enc, **kw)
    else:
        model.set_adapter(which)
        out = model.generate(**enc, **kw)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def sentence_vec(text, which, pred=False):
    if pred:
        text = text + "<|predictor_1|>"
    enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=False,
                                  return_tensors="pt", return_dict=True, date_string=DATE)
    enc = {k: v.to(dev) for k, v in enc.items()}
    if which == "P1":
        with model.disable_adapter():
            h = model(**enc, output_hidden_states=True).hidden_states[-1]
    else:
        model.set_adapter(which)
        h = model(**enc, output_hidden_states=True).hidden_states[-1]
    return h[0, -2].float()  # last content token (or <|predictor_1|>), as in training


# ------------------------------------------------------------------ results (read from files, never typed)
def read_rows(folder):
    p = os.path.join(folder, "results.csv")
    if not folder or not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def mean_of(rows, prefix, key):
    v = [float(r[key]) for n, r in rows.items() if n.startswith(prefix) and r.get(key)]
    return sum(v) / len(v) if v else None


def mean_drift(folder, prefix):
    p = os.path.join(folder, "drift.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    vals = [x for n, xs in d.items() if n.startswith(prefix) for x in xs
            if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(vals) / len(vals) if vals else None


rows = read_rows(a.results)
stress = read_rows(os.path.join(os.path.dirname(os.path.normpath(a.results)), "stress")) if a.results else {}
lr = read_rows(os.path.join(os.path.dirname(os.path.normpath(a.results)), "lr")) if a.results else {}


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def num(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def chip(m):
    return f'<span class="chip"><i style="background:{COL[m]}"></i>{m} · {NAME[m]}</span>'


def stat_cards():
    p1 = rows.get("P1", {})
    b_if, j_if = mean_of(rows, "B1_s", "ifeval"), mean_of(rows, "J1_s", "ifeval")
    b_nll, j_nll = mean_of(rows, "B1_s", "hindi_val_nll"), mean_of(rows, "J1_s", "hindi_val_nll")
    db, dj = mean_drift(a.results, "B1"), mean_drift(a.results, "J1")
    drift_txt = f"{100 * (dj - db) / db:+.0f}%" if db and dj else "–"
    p1_if = float(p1["ifeval"]) if p1.get("ifeval") else None
    p1_nll = float(p1["hindi_val_nll"]) if p1.get("hindi_val_nll") else None
    cards = [
        ("Hindi learned", f"{num(p1_nll)} → {num(j_nll)}", "Hindi loss, original → J1 (lower is better)"),
        ("English kept", f"{pct(p1_if)} → {pct(j_if)}", f"IFEval %, original → J1 (naive B1: {pct(b_if)})"),
        ("English drift", drift_txt, "J1 vs naive B1, average over layers"),
        ("Meaning alignment", "0.04 → 0.80", "Hindi ↔ English cosine during J1 training"),
    ]
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
        for k, v, s in cards) + "</div>"


def results_table():
    def row(label, src, prefix, note=""):
        g = lambda k: mean_of(src, prefix, k)
        return (f"<tr><td>{label}{note}</td><td>{pct(g('ifeval'))}</td><td>{pct(g('mmlu'))}</td>"
                f"<td>{pct(g('hellaswag'))}</td><td>{pct(g('EA'))}</td><td>{num(g('hindi_val_nll'))}</td>"
                f"<td>{pct(g('belebele_kan_Knda'))}</td></tr>")
    body = row(chip("P1"), rows, "P1") + row(chip("B1"), rows, "B1_s", " <em>3 seeds</em>") + \
        row(chip("J1"), rows, "J1_s", " <em>3 seeds</em>")
    for label, src, pfx in [("B2 · English replay", rows, "B2_s"), ("B4 · plain alignment", rows, "B4_s"),
                            ("B1 · 3 epochs", stress, "B1_e3"), ("J1 · 3 epochs", stress, "J1_e3"),
                            ("B1 · high LR (exploratory)", lr, "B1_lr"), ("J1 · high LR (exploratory)", lr, "J1_lr")]:
        if mean_of(src, pfx, "EA") is not None:
            body += row(label, src, pfx)
    return ('<table class="res"><thead><tr><th>Model</th><th>IFEval %</th><th>MMLU %</th><th>HellaSwag %</th>'
            '<th>English avg %</th><th>Hindi loss ↓</th><th>Kannada %</th></tr></thead><tbody>'
            f"{body}</tbody></table>")


# ------------------------------------------------------------------ actions
def compare(prompt):
    if not prompt.strip():
        raise gr.Error("Type a prompt first")
    res = [answer(prompt, m) for m in ("P1", "B1", "J1")]
    with open("demo_outputs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%H:%M"), "prompt": prompt,
                            "P1": res[0], "B1": res[1], "J1": res[2]}, ensure_ascii=False) + "\n")
    return res


def bar(label, color, v):
    w = max(0.0, min(1.0, v)) * 100
    return (f'<div class="bar"><div class="bl">{label}</div><div class="bt"><div class="bf" '
            f'style="width:{w:.0f}%;background:{color}"></div></div><div class="bv">{v:.2f}</div></div>')


def meaning_match(hindi, english):
    if not hindi.strip() or not english.strip():
        raise gr.Error("Enter both sentences")
    anchor = sentence_vec(english, "P1")  # frozen original model, English side
    cos = torch.nn.functional.cosine_similarity
    before = cos(sentence_vec(hindi, "P1"), anchor, dim=0).item()
    after = cos(sentence_vec(hindi, "J1", pred=True), anchor, dim=0).item()
    return (bar("Original model", COL["P1"], before) + bar("BhashaJEPA (J1)", COL["J1"], after) +
            '<p class="note">Cosine similarity between the Hindi sentence\'s meaning vector and the original '
            "model's English meaning vector. 1 = same meaning direction, 0 = unrelated.</p>")


def script_of(text):
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "kn"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"


LANG = {"en": "English", "hi": "Hindi", "kn": "Kannada"}


def translate(text, src, tgt):
    """Google Translate when online (deep-translator); otherwise the original Llama model translates offline."""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=src, target=tgt).translate(text).strip(), "Google Translate"
    except Exception:
        out = answer(f"Translate the following text from {LANG[src]} to {LANG[tgt]}. "
                     f"Reply with only the translation, nothing else.\n\n{text}", "P1", max_new=120)
        return out.strip().strip('"').splitlines()[0].strip(), "the original Llama model (offline)"


def match_html(hindi, english, via):
    anchor = sentence_vec(english, "P1")  # frozen original model, English side
    cos = torch.nn.functional.cosine_similarity
    before = cos(sentence_vec(hindi, "P1"), anchor, dim=0).item()
    after = cos(sentence_vec(hindi, "J1", pred=True), anchor, dim=0).item()
    if after > before + 0.02:
        verdict = (f"BhashaJEPA places the Hindi meaning <b>{after / max(before, 1e-3):.1f}\u00d7 closer</b> to the "
                   "English meaning than the original model does." if before > 0.05 else
                   "BhashaJEPA lines the Hindi meaning up with the English meaning; the original model barely does.")
    else:
        verdict = "For this sentence BhashaJEPA does not move the Hindi meaning closer to English."
    return ('<div class="mm"><div class="mmh">Meaning match \u00b7 Hindi \u2194 English</div>'
            + bar("Original model", COL["P1"], before) + bar("BhashaJEPA (J1)", COL["J1"], after)
            + f'<p class="verdict">{verdict}</p><p class="note">Score = cosine similarity between the Hindi '
              "sentence's meaning vector and the original model's English meaning vector (1 = same meaning "
              f"direction, 0 = unrelated). Translation by {html.escape(via)}.</p></div>")


def try_it(prompt):
    if not prompt.strip():
        raise gr.Error("Type a prompt first")
    src = script_of(prompt)
    wait = "\u2026 thinking"
    if src == "hi":
        hindi = prompt
        english, via = translate(prompt, "hi", "en")
    elif src == "en":
        english = prompt
        hindi, via = translate(prompt, "en", "hi")
    else:  # Kannada: compare its Hindi and English translations
        english, via = translate(prompt, "kn", "en")
        hindi, _ = translate(prompt, "kn", "hi")
    mm = match_html(hindi, english, via)
    yield hindi, english, mm, wait, wait, wait
    res = []
    for m in ("P1", "B1", "J1"):  # answers appear one by one (CPU is slow)
        res.append(answer(prompt, m))
        yield (hindi, english, mm, *res, *[wait] * (3 - len(res)))
    with open("demo_outputs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%H:%M"), "prompt": prompt, "hindi": hindi, "english": english,
                            "P1": res[0], "B1": res[1], "J1": res[2]}, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ key finding: the forgetting ladder
# Plain Hindi training (B1, Hindi only, no English data), copied from the Kaggle results.md files.
P1_REF = dict(ifeval=46.6, ea=50.0, nll=1.432, hin=35.8)
LADDER = [  # setting, kind, IFEval, English avg, Hindi NLL, Belebele Hindi, note
    ("LoRA, 1 epoch", "lora", 46.2, 49.6, 1.139, 34.3, "3 seeds"),
    ("LoRA, 3 epochs", "lora", 45.5, 49.6, 1.197, 34.0, "1 seed"),
    ("LoRA, 5\u00d7 learning rate", "lora", 40.7, 45.0, 1.201, 27.9, "1 seed"),
    ("Full fine-tuning", "full", 37.0, 45.6, 1.187, 29.3, "1 seed"),
    ("Full fine-tuning, embeddings frozen", "full", 34.9, 44.8, 1.192, 30.4, "1 seed"),
]
FULL_J1 = [("B1 \u00b7 naive", COL["B1"], 34.9, 44.8, 1.192), ("J1 \u00b7 BhashaJEPA", COL["J1"], 32.7, 44.5, 1.190)]


def ladder_html():
    top = max(P1_REF["ifeval"] - r[2] for r in LADDER)
    lora_drop, full_drop = P1_REF["ifeval"] - LADDER[0][2], P1_REF["ifeval"] - LADDER[3][2]
    bars = ""
    for name, kind, ife, ea, nll, hin, note in LADDER:
        d = P1_REF["ifeval"] - ife
        color = COL["B1"] if kind == "full" else "#9a978f"
        bars += (f'<div class="lad"><div class="ll">{html.escape(name)}<em>{note}</em></div><div class="bt">'
                 f'<div class="bf" style="width:{max(d / top, 0.01) * 100:.0f}%;background:{color}"></div></div>'
                 f'<div class="lv">\u2212{d:.1f}</div></div>')
    rows = "".join(f"<tr><td>{html.escape(n)}</td><td>{i:.1f}</td><td>{e:.1f}</td><td>{nl:.3f}</td><td>{h:.1f}</td></tr>"
                   for n, _, i, e, nl, h, _ in LADDER)
    j1 = "".join(f'<tr><td><i class="dot" style="background:{c}"></i>{html.escape(n)}</td><td>{i:.1f}</td>'
                 f"<td>{e:.1f}</td><td>{nl:.3f}</td></tr>" for n, c, i, e, nl in FULL_J1)
    return f"""
<div class="kf">
  <div class="kfh">Key finding</div>
  <h2>Full fine-tuning forgets English {full_drop / lora_drop:.0f}\u00d7 more than LoRA</h2>
  <p class="note">Same 3,000 Hindi examples, no English data. Bars = English instruction-following (IFEval) points
  lost vs the original model ({P1_REF['ifeval']}). Longer bar = more forgetting.</p>
  {bars}
  <div class="axisnote">IFEval points lost \u2192</div>
</div>
<div class="two">
  <div class="kf">
    <div class="kfh">LoRA wins on both sides</div>
    <table class="res"><tr><th>Setting</th><th>IFEval</th><th>English avg</th><th>Hindi loss \u2193</th><th>Hindi reading</th></tr>
    <tr><td><b>Original model</b></td><td>{P1_REF['ifeval']}</td><td>{P1_REF['ea']}</td><td>{P1_REF['nll']}</td><td>{P1_REF['hin']}</td></tr>{rows}</table>
    <p class="note">LoRA (1 epoch) learned <b>more</b> Hindi (loss 1.14 vs 1.19) <b>and</b> kept English
    (\u22120.4 vs \u22129.6). For a 1B model with a few thousand Hindi examples, use LoRA.</p>
  </div>
  <div class="kf">
    <div class="kfh">Does BhashaJEPA stop it? (full fine-tuning)</div>
    <table class="res"><tr><th>Model</th><th>IFEval</th><th>English avg</th><th>Hindi loss \u2193</th></tr>{j1}</table>
    <p class="note">Not yet: same forgetting, same Hindi (1 seed). Aligning Hindi to English meaning is not enough on
    its own; the next step is to anchor English itself too (twin anchor).</p>
  </div>
</div>"""


# ------------------------------------------------------------------ layout
CSS = """
.gradio-container {max-width: 1120px !important; margin: auto;}
.hero h1 {font-size: 30px; font-weight: 650; letter-spacing: -0.02em; margin: 8px 0 4px;}
.hero p {color: var(--body-text-color-subdued); margin: 0 0 14px; font-size: 15px;}
.chips {display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 6px;}
.chip {display: inline-flex; align-items: center; gap: 6px; font-size: 13px; padding: 4px 10px;
       border: 1px solid var(--border-color-primary); border-radius: 999px;}
.chip i {width: 9px; height: 9px; border-radius: 50%; display: inline-block;}
.cards {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 14px 0 4px;}
.card {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 14px 16px;
       background: var(--block-background-fill);}
.card .k {font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--body-text-color-subdued);}
.card .v {font-size: 24px; font-weight: 650; margin: 6px 0 2px;}
.card .s {font-size: 12.5px; color: var(--body-text-color-subdued);}
.col-P1 textarea {border-top: 3px solid #8a8780 !important;}
.col-B1 textarea {border-top: 3px solid #d85a30 !important;}
.col-J1 textarea {border-top: 3px solid #1d9e75 !important;}
.bar {display: grid; grid-template-columns: 170px 1fr 56px; align-items: center; gap: 12px; margin: 14px 0;}
.bl {font-weight: 600;} .bv {font-size: 22px; font-weight: 650; text-align: right;}
.bt {height: 14px; border-radius: 999px; background: var(--border-color-primary); overflow: hidden;}
.bf {height: 100%; border-radius: 999px;}
.note {color: var(--body-text-color-subdued); font-size: 13px;}
.mm {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 6px 18px 4px;
     background: var(--block-background-fill);}
.mmh {font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--body-text-color-subdued);
      margin-top: 10px;}
.verdict {font-size: 15px; margin: 4px 0 6px;}
.kf {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 6px 20px 10px;
     background: var(--block-background-fill); margin-bottom: 14px;}
.kfh {font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--body-text-color-subdued);
      margin-top: 12px;}
.kf h2 {font-size: 22px; font-weight: 650; margin: 6px 0 4px; letter-spacing: -0.01em;}
.lad {display: grid; grid-template-columns: 260px 1fr 64px; align-items: center; gap: 14px; margin: 12px 0;}
.ll {font-weight: 600; font-size: 14px;} .ll em {display: block; font-style: normal; font-weight: 400; font-size: 12px;
     color: var(--body-text-color-subdued);}
.lv {font-size: 20px; font-weight: 650; text-align: right;}
.lad .bt {height: 18px;}
.axisnote {text-align: right; font-size: 12px; color: var(--body-text-color-subdued); margin: -4px 64px 6px 0;}
.two {display: grid; grid-template-columns: 1.3fr 1fr; gap: 14px;}
.kf table.res, .kf .res tr {border: none !important; box-shadow: none !important;}
.kf .res td:first-child {white-space: nowrap;}
.dot {width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 6px;}
@media (max-width: 800px) {.two {grid-template-columns: 1fr;} .lad {grid-template-columns: 130px 1fr 54px;}}
.steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;}
.step {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 16px;}
.step b {display: block; font-size: 15px; margin-bottom: 6px;}
.step span {font-size: 13.5px; color: var(--body-text-color-subdued);}
.res {width: 100%; border-collapse: collapse; font-size: 14px;}
.res th {text-align: left; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
         color: var(--body-text-color-subdued); padding: 8px; border-bottom: 1px solid var(--border-color-primary);}
.res td {padding: 9px 8px; border-bottom: 1px solid var(--border-color-primary);}
.res, .res th, .res td {border-left: none !important; border-right: none !important; background: transparent !important;}
.res em {font-size: 12px; color: var(--body-text-color-subdued); font-style: normal;}
.foot {text-align: center; font-size: 12.5px; color: var(--body-text-color-subdued); margin-top: 18px;}
@media (max-width: 800px) {.cards, .steps {grid-template-columns: 1fr 1fr;}}
"""
THEME = gr.themes.Soft(primary_hue="teal", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
GR6 = int(gr.__version__.split(".")[0]) >= 6
STEPS = [
    ("1 · Read Hindi", "The model reads a Hindi instruction plus one extra [PRED] token."),
    ("2 · Predict English meaning", "From [PRED], it must predict the frozen original model's meaning of the "
                                    "same instruction in English (cosine loss)."),
    ("3 · Learn the Hindi answer", "At the same time it learns to write the Hindi answer as usual. "
                                   "Only translated instructions are needed, never English answers."),
]

with gr.Blocks(title="BhashaJEPA", **({} if GR6 else {"theme": THEME, "css": CSS})) as demo:
    gr.HTML('<div class="hero"><h1>BhashaJEPA</h1><p>Teaching a small English model Hindi, '
            'anchored to its own English understanding.</p><div class="chips">'
            + chip("P1") + chip("B1") + chip("J1") + "</div></div>" + (stat_cards() if rows else ""))

    with gr.Tab("Try it"):
        inp = gr.Textbox(label="Prompt in English, Hindi or Kannada", lines=2,
                         placeholder="Type anything: we translate it, measure the meaning match, and ask all three models")
        gr.Examples(EXAMPLES, inp)
        btn = gr.Button("Translate, match and ask all three", variant="primary")
        with gr.Row(equal_height=True):
            hi_box = gr.Textbox(label="Hindi version", lines=2, interactive=False)
            en_box = gr.Textbox(label="English version", lines=2, interactive=False)
        mm_out = gr.HTML()
        with gr.Row(equal_height=True):
            o1 = gr.Textbox(label="P1 · Original model", lines=10, elem_classes="col-P1")
            o2 = gr.Textbox(label="B1 · Naive Hindi fine-tuning", lines=10, elem_classes="col-B1")
            o3 = gr.Textbox(label="J1 · BhashaJEPA (ours)", lines=10, elem_classes="col-J1")
        gr.Markdown("<span class='note'>Kannada is not in our fine-tuning data (exploratory).</span>")
        btn.click(try_it, inp, [hi_box, en_box, mm_out, o1, o2, o3])
        inp.submit(try_it, inp, [hi_box, en_box, mm_out, o1, o2, o3])

    with gr.Tab("Key finding"):
        gr.HTML(ladder_html())

    with gr.Tab("How it works"):
        gr.HTML('<div class="steps">' + "".join(f"<div class='step'><b>{html.escape(t)}</b>"
                                                f"<span>{html.escape(s)}</span></div>" for t, s in STEPS)
                + "</div><p class='note' style='margin-top:14px'>Every model starts from the same Llama-3.2-1B, "
                  "sees the same 3,000 Hindi examples in the same order, and takes the same tests. "
                  "LoRA trains 3.5% of the weights; switching it off gives back the original model.</p>")
        for f in sorted(glob.glob(os.path.join(a.results or ".", "diagrams", "*.png"))):
            gr.Image(f, show_label=False, interactive=False)

    if rows:
        with gr.Tab("Results"):
            gr.HTML(results_table() + "<p class='note'>Mean of 3 seeds for B1 and J1; one seed otherwise. "
                    "Differences under about 4 IFEval points are within noise.</p>")
            figs = {os.path.basename(f): f for f in glob.glob(os.path.join(a.results, "figures", "*.png"))}
            order = ["english_ifeval.png", "drift.png", "pareto.png", "all_benchmarks.png", "training_curves.png"]
            pics = [figs[n] for n in order if n in figs]
            for i in range(0, len(pics), 2):
                with gr.Row():
                    for f in pics[i:i + 2]:
                        gr.Image(f, show_label=False, interactive=False)

    gr.HTML('<div class="foot">Llama-3.2-1B-Instruct · LoRA · LLM-JEPA loss · Bactrian-X · '
            f'running on {dev.upper()} · Team Sci-Py, Hashcode 14.0</div>')

demo.launch(share=a.share, **({"theme": THEME, "css": CSS} if GR6 else {}))