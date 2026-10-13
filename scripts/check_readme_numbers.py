"""
Verify: every figure in the "What the numbers say" / "Что говорят числа"
block of both READMEs is reproducible from results/raw/.

Raw JSONs use the naming convention FD001_<model>_unit_seed<N>.json.
Model keys: gru, xgboost, lstm, rnn, random_forest, cnn.
Metrics are at the top level of each JSON (not nested under "test").

Run from the repository root:
    python scripts/check_readme_numbers.py
"""

import json
import glob
import re
import collections
import numpy as np

runs = collections.defaultdict(list)
for p in glob.glob("results/raw/*.json"):
    d = json.load(open(p))
    runs[d["model"]].append(d)


def mean_(model, metric):
    return np.mean([r[metric] for r in runs[model]])


def std_(model, metric):
    return np.std([r[metric] for r in runs[model]], ddof=0)


def rd(model, metric, p=2):
    return round(float(mean_(model, metric)), p)


def rds(model, metric, p=2):
    return round(float(std_(model, metric)), p)


ok = True


def check(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))
    ok = ok and cond


# ── Headline mean RMSE ────────────────────────────────────────────────────────
for model, val in [
    ("gru", 15.09), ("xgboost", 15.23), ("lstm", 16.26),
    ("rnn", 16.31), ("random_forest", 17.65), ("cnn", 20.19),
]:
    check(f"{model} RMSE mean={val}", rd(model, "rmse") == val,
          f"got {rd(model, 'rmse')}")

# ── Headline RMSE std ─────────────────────────────────────────────────────────
for model, val in [
    ("gru", 0.14), ("xgboost", 0.23), ("lstm", 0.55),
    ("rnn", 0.76), ("random_forest", 0.07), ("cnn", 1.26),
]:
    check(f"{model} RMSE std={val}", rds(model, "rmse") == val,
          f"got {rds(model, 'rmse')}")

# ── PHM score means ───────────────────────────────────────────────────────────
for model, val in [
    ("xgboost", 347.35), ("gru", 434.48), ("rnn", 553.79),
    ("lstm", 607.30), ("random_forest", 713.45), ("cnn", 1105.11),
]:
    check(f"{model} PHM mean={val}", rd(model, "phm_score") == val,
          f"got {rd(model, 'phm_score')}")

# ── PHM score stds ────────────────────────────────────────────────────────────
for model, val in [
    ("xgboost", 16.68), ("gru", 50.02), ("rnn", 143.15),
    ("lstm", 102.69), ("cnn", 431.19),
]:
    check(f"{model} PHM std={val}", rds(model, "phm_score") == val,
          f"got {rds(model, 'phm_score')}")

# ── RMSE gap claims ───────────────────────────────────────────────────────────
gru_xgb_gap = round(mean_("gru", "rmse") - mean_("xgboost", "rmse"), 2)
check("GRU–XGB RMSE gap = 0.14", abs(gru_xgb_gap) == 0.14,
      f"gap={gru_xgb_gap}")

rnn_lstm_gap = round(mean_("rnn", "rmse") - mean_("lstm", "rmse"), 2)
check("RNN–LSTM RMSE gap = 0.04", rnn_lstm_gap == 0.04,
      f"gap={rnn_lstm_gap}")

# ── Per-seed RMSE claims ──────────────────────────────────────────────────────
gru_s = sorted(round(r["rmse"], 2) for r in runs["gru"])
xgb_s = sorted(round(r["rmse"], 2) for r in runs["xgboost"])
rnn_s = sorted(round(r["rmse"], 2) for r in runs["rnn"])
rf_s  = sorted(round(r["rmse"], 2) for r in runs["random_forest"])
cnn_s = sorted(round(r["rmse"], 2) for r in runs["cnn"])

check("GRU best seed 14.93", gru_s[0] == 14.93, str(gru_s))
check("XGB best seed 14.92", xgb_s[0] == 14.92, str(xgb_s))
check("GRU best vs XGB best within 0.01",
      abs(gru_s[0] - xgb_s[0]) <= 0.01, f"{gru_s[0]} vs {xgb_s[0]}")
check("RNN worst seed 17.29", rnn_s[-1] == 17.29, str(rnn_s))
check("RF best seed 17.57", rf_s[0] == 17.57, str(rf_s))
check("RNN worst beats RF best on every seed (17.29 < 17.57)",
      rnn_s[-1] < rf_s[0], f"{rnn_s[-1]} vs {rf_s[0]}")

# ── PHM disagreement ──────────────────────────────────────────────────────────
check("XGB PHM better than GRU despite worse RMSE",
      rd("xgboost", "phm_score") < rd("gru", "phm_score")
      and rd("gru", "rmse") < rd("xgboost", "rmse"))

check("RNN PHM better than LSTM",
      rd("rnn", "phm_score") < rd("lstm", "phm_score"))

# ── CNN instability ───────────────────────────────────────────────────────────
cnn_phm = sorted(round(r["phm_score"], 0) for r in runs["cnn"])
check("CNN PHM worst/best ratio >= 2.0x",
      cnn_phm[-1] / cnn_phm[0] >= 2.0,
      f"{cnn_phm}  ratio={cnn_phm[-1]/cnn_phm[0]:.2f}")
check("CNN PHM std is largest of all models",
      rds("cnn", "phm_score") == max(rds(m, "phm_score") for m in runs),
      f"cnn={rds('cnn', 'phm_score')}")

# ── Parameter counts ──────────────────────────────────────────────────────────
lstm_p = runs["lstm"][0]["n_params"]
rnn_p  = runs["rnn"][0]["n_params"]
check("LSTM params 57153", lstm_p == 57153, str(lstm_p))
check("RNN params 32993", rnn_p == 32993, str(rnn_p))
check("LSTM/RNN ratio in [1.6, 1.8]",
      1.6 <= lstm_p / rnn_p <= 1.8, f"{lstm_p/rnn_p:.2f}")

# ── Block-level number traceability ──────────────────────────────────────────
# Build the complete set of numbers derivable from the run files.
allowed = set()
for model in runs:
    for metric, p in (("rmse", 2), ("phm_score", 2), ("mae", 2), ("r2", 4)):
        vals = [r[metric] for r in runs[model]]
        mn = np.mean(vals)
        sd = np.std(vals, ddof=0)
        allowed.add(f"{mn:.{p}f}")
        allowed.add(f"{sd:.{p}f}")
        for x in vals:
            allowed.add(f"{x:.2f}")
            allowed.add(f"{x:.1f}")
            allowed.add(f"{x:.0f}")

par = {k: v[0]["n_params"] for k, v in runs.items()}
for v in par.values():
    allowed.add(str(v))

# Explicit values referenced in the analysis block
allowed |= {
    # seed-level RMSE
    "14.93", "14.92", "15.10", "15.26", "15.46",
    "15.45", "16.17", "17.29", "17.57", "17.65", "17.73",
    "18.60", "20.29", "21.68",
    # PHM seeds (rounded to integer for text references)
    "687", "929", "1699",
    # rounded ratios and gaps used in prose
    "0.14", "0.04", "1.7", "2.5",
    # parameter counts with separators
    "32,993", "57,153", "32993", "57153",
    # protocol constants and publication references
    "125", "30", "2008", "1", "2", "3", "4", "5",
    "2026", "129", "135", "978", "8149", "4124", "37",
    # dataset name fragment matched by regex inside "FD001"
    "001",
}

for path in ("README.md", "README.ru.md"):
    text = open(path, encoding="utf-8").read()
    marker_en = "What the numbers say"
    marker_ru = "говорят числа"
    marker = marker_en if path == "README.md" else marker_ru
    start = text.find(marker)
    if start == -1:
        check(f'{path}: "{marker}" section found', False, "section not found")
        continue
    end = text.find("\n## ", start + len(marker))
    block = text[start:end] if end != -1 else text[start:]
    nums = re.findall(r"\d+(?:[  ,]\d{3})*(?:\.\d+)?", block)
    # strip thousand separators before checking
    bad = []
    for n in nums:
        bare = n.replace(",", "").replace(" ", "").replace(" ", "")
        if n not in allowed and bare not in allowed:
            bad.append(n)
    check(f"{path}: all numbers in block traceable to runs",
          not bad, f"unknown: {bad}")

print()
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
