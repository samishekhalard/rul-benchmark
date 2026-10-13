# Remaining Useful Life prediction: an architecture comparison

*English · [Русский](README.ru.md)*

A reproducible comparison of neural and classical models for predicting the
Remaining Useful Life (RUL) of industrial equipment on the NASA C-MAPSS
dataset.

This repository accompanies two published papers and reproduces the whole
comparison, from raw data to the metrics table, with a single command.

---

## Why this exists

Papers and blog posts on RUL prediction usually report that some architecture
reached an RMSE of some value. Reproducing that number is typically
impossible: the window size is not given, the RUL labelling is not described,
it is not stated whether normalization statistics were computed on the training
split or on the whole dataset, and no seed-to-seed spread is reported.

Here every one of those decisions is fixed in code, identically for all
models, and written down in this README. The goal is not a record number but
a measurement of the difference between approaches under comparable conditions.

---

## Protocol

The decisions that influence the result most are made the same way for every
model:

**RUL labelling (training).** RUL is the difference between a unit's last
cycle and the current one, capped at 125 cycles. Before degradation begins,
the remaining life cannot be recovered from sensor readings, and a model
trained without the cap spends capacity predicting an uninformative segment.
The value 125 is taken from the literature and fixed.

**Test ground truth: no cap applied.** The RUL labels from `RUL_FDxxx.txt`
are used as-is for evaluation. Some implementations clip these at 125 to
match the training target; this repository does not. The convention is stated
here so results are comparable with code that makes the opposite choice.

**Train/validation split by unit, not by window.** Neighbouring windows of
the same engine are nearly identical. Splitting by window puts part of the
validation set into training and inflates the metrics. This is the most
common leak in C-MAPSS work. See the [Leakage experiment](#leakage-experiment)
section for a controlled measurement of the effect.

**Normalization fitted on the training part only.** Mean and variance are
computed on train, then applied to validation and test. Test statistics never
enter training.

**Feature selection.** Seven sensors on FD001 have zero variance and are
excluded: they carry no information but distort normalization.

**Test.** One last window per unit — that is the task as posed: predict the
remaining life at the moment the recording stops.

**Spread.** Each model is trained on three seeds (42, 1337, 2024); the table
reports mean and standard deviation. A difference of a point and a half in
RMSE between architectures can be explained entirely by initialization, and
without the spread that is invisible.

---

## Metrics

| Metric | Meaning |
|--------|---------|
| RMSE | The headline comparison metric in the C-MAPSS literature |
| MAE | More robust to outliers, shows the typical error |
| R² | Share of variance explained |
| PHM score | The asymmetric metric of the PHM 2008 challenge |

PHM score sits in the table separately and deserves attention: it penalizes a
late prediction harder than an early one. In operation, a missed failure costs
more than unnecessary maintenance, and RMSE cannot see that difference. A
model can win on RMSE and lose on PHM score — which means it more often
promises more remaining life than there is.

```
d = y_pred − y_true

if d < 0  (early / optimistic):  exp(−d / 13) − 1
if d ≥ 0  (late  / pessimistic): exp( d / 10) − 1
```

At |d| = 10 cycles: early penalty ≈ 1.16, late penalty ≈ 1.72 — roughly
50% heavier.

---

## Leakage experiment

Many C-MAPSS studies split train/validation by shuffling sliding windows.
Adjacent windows of the same engine are nearly identical, so the model
validates on samples it has effectively already seen. This repository
measures the effect rather than asserting it.

LSTM, FD001, 3 seeds, identical hyperparameters, only the split differs:

| Split     | Val RMSE     | Test RMSE    | Val–Test gap |
|-----------|--------------|--------------|--------------|
| by unit   | 13.53 ± 0.22 | 16.49 ± 0.41 | +2.97        |
| by window | 3.25 ± 0.19  | 19.01 ± 0.67 | +15.77       |

Reproduce with `bash scripts/leakage_experiment.sh`.

**A validation RMSE of 3.25 would look like a record.** Published results on
FD001 cluster around 11–13. A paper reporting 3.25 is not reporting a better
model — it is reporting its own split.

**The leak does not only flatter the report, it damages the model.** Test RMSE
is honest under both splits, and it gets *worse* with leakage: 16.49 → 19.01,
a 15% degradation. The mechanism is early stopping. It watches validation
loss, and under leakage validation keeps improving as the model memorises
near-duplicate windows, so training never stops when it should. Wall-clock
time confirms it: 24–37 s per run with the honest split, 105 s with the leaky
one — the same patience setting, three times the epochs.

> **Note:** in window mode, normalisation statistics are also computed over all
> training rows, including those backing validation windows. Both effects are
> part of the methodology being reproduced; the measured gap reflects their
> combination.

---

## Results

The table is assembled automatically from the run files:

```bash
python src/report.py --results-dir results/raw --output results/RESULTS.md
```

Current results: [results/RESULTS.md](results/RESULTS.md)

---

## What the numbers say

Below is the FD001 run, three seeds per model, mean ± standard deviation. The
full table is in [results/RESULTS.md](results/RESULTS.md).

**GRU leads, XGBoost is right behind — but the gap is noise.** GRU comes
first at 15.09 ± 0.14 RMSE; XGBoost second at 15.23 ± 0.23. The margin is
0.14 — exactly equal to GRU's own seed spread. Both models reached a
best-seed RMSE within 0.01 of each other (14.93 vs 14.92). Declaring a winner
here from three seeds would mean reporting initialization as a result.

**XGBoost wins on PHM despite losing on RMSE.** RMSE puts GRU ahead; PHM
score reverses it: 347.35 (XGBoost) against 434.48 (GRU). XGBoost's errors
lean early; GRU's lean late, where PHM punishes harder. A table with only
RMSE would hide this.

**A plain RNN matches LSTM and costs 1.7× less.** The RMSE gap is 0.04
(16.31 ± 0.76 vs 16.26 ± 0.55) — invisible inside the spreads. RNN beats
LSTM on PHM score (553.79 vs 607.30) and achieves this with 32,993 parameters
against LSTM's 57,153. Recurrence also beats the forest on every seed: RNN's
worst run (17.29) beats Random Forest's best (17.57).

**CNN is last and the least stable model by a wide margin.** 20.19 ± 1.26 —
nearly five RMSE points behind XGBoost. The PHM instability is starker:
±431.19 against ±143.15 for the next-most-variable model. CNN's PHM score
ran from 687 to 1699 across three seeds — a 2.5× range on the metric closest
to the operational cost of an error. Hyperparameters were not tuned
specifically for the convolutional architecture; read this as a statement
about this configuration, not about the approach in general.

**Short version:** on FD001 the choice between GRU, XGBoost, LSTM, and RNN
falls within the noise of initialization. The only differences this data can
resolve are the CNN deficit and the scale of seed-to-seed variation itself.

---

## Running it

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data

Download `CMAPSSData.zip` from the NASA Prognostics Data Repository and
extract the `.txt` files into `data/`. See [`data/README.md`](data/README.md)
for details.

### 3. Run the full benchmark (all models, 3 seeds)

```bash
bash scripts/run_all.sh data/ 100
```

### 4. Train a single model

```bash
# Neural model
python src/train.py --model lstm --subset FD001 --data-dir data/ --epochs 100

# Baseline
python src/train.py --model xgboost --subset FD001 --data-dir data/
```

### 5. Generate the results table

```bash
python src/report.py --results-dir results/raw --output results/RESULTS.md
```

### 6. Run tests

```bash
pytest tests/ -v
```

### 7. Lint

```bash
ruff check src/ tests/
```

---

## What's inside

```
rul-benchmark/
├── README.md
├── README.ru.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── default.yaml           Hyperparameters and protocol settings
├── data/
│   └── README.md              Download instructions (raw files not committed)
├── src/
│   ├── data.py                Loading, RUL labelling, windowing, normalization
│   ├── models.py              LSTM, GRU, 1D-CNN, RNN + registry
│   ├── baselines.py           XGBoost and Random Forest on flattened windows
│   ├── metrics.py             RMSE, MAE, R², PHM score
│   ├── train.py               Training loop, CLI, early stopping
│   └── report.py              Aggregates raw JSONs → results/RESULTS.md
├── tests/
│   ├── conftest.py            Synthetic C-MAPSS fixture (no download needed)
│   ├── test_data.py           Protocol correctness (17 tests)
│   ├── test_models.py         Forward pass and gradient tests
│   ├── test_metrics.py        Metric correctness and PHM asymmetry
│   └── test_report.py         Report aggregation tests
├── scripts/
│   ├── run_all.sh             Full benchmark runner
│   └── leakage_experiment.sh  Trains LSTM × 2 splits × 3 seeds; prints comparison table
└── results/
    ├── README.md
    ├── RESULTS.md             Generated by report.py (not committed)
    ├── raw/                   Per-run JSON files (not committed)
    └── leakage_experiment/    Leakage experiment run files
```

Network sizes are chosen so that the number of trainable parameters is
comparable across architectures. The baselines receive the same windows
flattened into vectors; the data and protocol are otherwise identical.

---

## Limitations

- **FD001 is the easiest subset:** one operating condition, one failure mode.
  The results do not transfer directly to FD002 and FD004, where there are
  six conditions. The code supports all four subsets, but the conclusions in
  this README apply to what was actually run.
- **Three seeds is few.** Enough to see the order of magnitude of the spread,
  not enough for a statistically significant claim that one architecture
  beats another.
- **Hyperparameters were not tuned per model.** Reasonable defaults were used.
  Careful tuning could change the ordering in the table.
- **C-MAPSS is a simulation, not production data.** The noise, gaps, and
  sensor faults one deals with in operation are absent here.

---

## Publications

The comparison methodology implemented in this repository was used in the
following papers:

1. Шейх Ал Ард М.С., Потехин В.В., Жуненков А.С. «Сравнительный анализ
   моделей LSTM и Random Forest для прогнозирования остаточного ресурса
   промышленного оборудования по данным многомерных временных рядов».
   Сборник «ИТ. Наука. Креатив», Т. 1, 2026, с. 129–135.
   ISBN 978-5-8149-4124-4, РИНЦ.

2. Shekh Al Ard M.S., Kovalchuk E.A., Potekhin V.V.
   «Proactive Monitoring of Equipment Condition in Multi-Agent Systems Using
   LSTM Neural Networks». 37th DAAAM International Symposium on Intelligent
   Manufacturing and Automation, Vienna, 2026.

> **Note:** these are the official citation strings. The English title in
> item 1 is a working translation; confirm the official English title before
> citing.

---

## Data

Turbofan Engine Degradation Simulation Data Set, NASA Prognostics Center of
Excellence. Distributed separately, not included in this repository.
See [`data/README.md`](data/README.md) for download instructions.

---

## License

MIT — see [LICENSE](LICENSE).
