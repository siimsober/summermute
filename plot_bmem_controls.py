# plot_bmem_controls.py
# Builds plots/bmem_controls.html: Wikipedia validation loss of the B-mem experiment and its controls
# (est, 2 layers x 2 heads, vocab 512), from the CSV logs in models/exp-aabits/.
import csv
import json
import os

EXP = "models/exp-aabits"
OUT = "plots/bmem_controls.html"

# name: (label, short label at line ends, [(folder, step the next phase started from)], description)
RUNS = {
    "A2": ("A2: no warm-up", "A2", [("A2", None)], "10k Wikipedia steps"),
    "B": ("B: aabits 280 steps", "B", [("B1", 280), ("B2", None)], "aabits until held-out stories were best, then 10k Wikipedia"),
    "Bmem": ("B-mem: aabits memorized", "B-mem", [("Bmem1", 5000), ("Bmem2", None)], "5k aabits steps (memorized), then 10k Wikipedia"),
    "Wikimem": ("Ctrl: Wikipedia slice memorized", "Wiki slice", [("CtrlWikimem1", 5000), ("CtrlWikimem2", None)],
                "5k steps on a Wikipedia slice the size of aabits, then 10k Wikipedia"),
    "Ctrl15k": ("Ctrl: 15k Wikipedia", "15k Wiki", [("Ctrl15k", None)], "15k Wikipedia steps, no warm-up"),
}


def read_log(folder):
    with open(f"{EXP}/{folder}/sumerute-est-log.csv") as f:
        return [r for r in csv.DictReader(f) if r["step"] != "0"]


series = {}
for key, (label, short, phases, desc) in RUNS.items():
    offset = 0
    for folder, next_from in phases:
        rows = read_log(folder)
        if next_from is not None:        # warm-up phase: only its length counts
            offset += next_from
            continue
        series[key] = {
            "label": label, "short": short, "desc": desc, "offset": offset,
            "points": [[int(r["step"]), float(r["val_loss"])] for r in rows],
        }

# B-mem with seed 2 for the note (same recipe)
seed2 = read_log("Bmem2s2")[-1]

html = open("plots/template_bmem.html", encoding="utf-8").read()
html = html.replace("/*DATA*/null", json.dumps(series)).replace("/*SEED2*/null", seed2["val_loss"])
os.makedirs("plots", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print(f"✅ Wrote {OUT}")
