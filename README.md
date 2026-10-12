# Remaining Useful Life prediction: an architecture comparison

*English · [Русский](README.ru.md)*

A reproducible comparison of neural and classical models for predicting the
Remaining Useful Life (RUL) of industrial equipment on the NASA C-MAPSS
dataset.

This repository accompanies two published papers and reproduces the whole
comparison, from raw data to the metrics table, with a single command.

## Why this exists

Papers and blog posts on RUL prediction usually report that some
architecture reached an RMSE of some value. Reproducing that number is
typically impossible: the window size is not given, the RUL labelling is not
described, it is not stated whether normalization statistics were computed on
the training split or on the whole dataset, and no seed-to-seed spread is
reported.

Here every one of those decisions is fixed in code, identically for all
models, and written down in this README. The goal is not a record number but
a measurement of the difference between approaches under comparable
conditions.

## Protocol

The decisions that influence the result most are made the same way for every
model:

**RUL labelling.** For the training split, RUL is the difference between a
unit's last cycle and the current one, capped at 125 cycles. Before
degradation begins, the remaining life cannot be recovered from sensor
readings, and a model trained without the cap spends capacity predicting an
uninformative segment. The value 125 is taken from the literature and fixed.

**Train/validation split by unit, not by window.** Neighbouring windows of
the same engine are nearly identical. Splitting by window puts part of the
validation set into training and inflates the metrics. This is the most
common leak in C-MAPSS work.

**Normalization fitted on the training part only.** Mean and variance are
computed on train, then applied to validation and test. Test statistics never
enter training.

**Feature selection.** Seven sensors on FD001 have zero variance and are
excluded: they carry no information but distort normalization.

**Test.** One last window per unit — that is the task as posed: predict the
remaining life at the moment the recording stops. Ground truth comes from
`RUL_FDxxx.txt`.

**Spread.** Each model is trained on three seeds; the table reports mean and
standard deviation. A difference of a point and a half in RMSE between
architectures can be explained entirely by initialization, and without the
spread that is invisible.

## Metrics

| Metric | Meaning |
|---|---|
| RMSE | The headline comparison metric in the C-MAPSS literature |
| MAE | More robust to outliers, shows the typical error |
| R² | Share of variance explained |
| PHM score | The asymmetric metric of the PHM 2008 challenge |

PHM score sits in the table separately and deserves attention: it penalizes a
late prediction harder than an early one. In operation, a missed failure
costs more than unnecessary maintenance, and RMSE cannot see that difference.
A model can win on RMSE and lose on PHM score — which means it more often
promises more remaining life than there is.

## Results

The table is assembled automatically from the run files:

```bash
python -m src.report --results-dir results/raw --out results/RESULTS.md
```

Current results: [results/RESULTS.md](results/RESULTS.md)

## What the numbers say

Below is the FD001 run, three seeds per model, mean ± standard deviation. The
full table is in [results/RESULTS.md](results/RESULTS.md).

**Gradient boosting is not a straw man.** XGBoost on flattened windows comes
first at 13.79 ± 0.16 RMSE, ahead of GRU (13.93 ± 0.48) and LSTM
(14.36 ± 0.43). Its worst seed (13.90) is still better than the best LSTM
seed (14.01). Whatever the recurrent models extract from the temporal order
inside a 30-cycle window, on FD001 it is not worth much.

**The gap between XGBoost and GRU is 0.14 RMSE — smaller than GRU's own
spread across seeds (± 0.48).** GRU's best seed (13.26) beats all three
XGBoost seeds. Declaring a winner here from a single run would be reporting
noise as a result.

**Recurrence beats the forest, but modestly.** RNN reaches 15.39 ± 0.11
against 16.27 ± 0.22 for Random Forest, and wins on every seed. A plain RNN
also has four times fewer parameters than LSTM (13 697 against 56 641) for a
1.03 RMSE deficit.

**1D-CNN trails clearly:** 18.82 ± 0.65, five points behind XGBoost.
Hyperparameters were left at their defaults and not tuned for the
convolutional model specifically — read this as a statement about that
configuration, not about the architecture as such.

**RMSE and PHM score disagree.** Random Forest has a clearly better RMSE than
1D-CNN (16.27 against 18.82) but a worse PHM score (659.07 against 648.50):
its errors land more often on the late side, where the metric punishes
harder. XGBoost is the best on both at once (312.53).

**Seed spread is the main caveat to the table.** GRU's PHM score ranges from
264.1 to 463.8 across three seeds: almost a twofold difference on the metric
closest to the operational cost of an error.

Short version: on FD001 the choice between LSTM, GRU and gradient boosting
falls within the noise. The only differences this data can resolve are the
1D-CNN deficit and the spread from initialization itself.

## Running it

```bash
git clone https://github.com/<username>/rul-benchmark.git
cd rul-benchmark
pip install -r requirements.txt
```

Download the C-MAPSS dataset and unpack it into `data/CMAPSS/` — details in
[data/README.md](data/README.md).

A single model:

```bash
python -m src.train --model lstm --subset FD001
python -m src.baselines --model xgboost --subset FD001
```

The full benchmark (4 neural architectures, 2 baselines, 3 seeds):

```bash
bash scripts/run_all.sh FD001
```

On a GPU the full run takes about half an hour; on CPU, a few hours.

## What's inside

```
src/
├── data.py        data preparation: RUL labelling, windowing, normalization
├── models.py      LSTM, GRU, 1D-CNN, RNN
├── baselines.py   XGBoost and Random Forest on flattened windows
├── metrics.py     RMSE, MAE, R², PHM score
├── train.py       training with early stopping and result saving
└── report.py      assembling the table from individual runs
```

Network sizes are chosen so that the number of trainable parameters is
comparable. Otherwise a comparison of architectures turns into a comparison
of capacity.

The baselines cannot consume sequences directly, so the window is flattened
into a vector. The data and the protocol stay the same — only the
representation differs.

## Limitations

Stated plainly, because they affect how the table should be read:

- **FD001 is the easiest subset:** one operating condition, one failure mode.
  The results do not transfer directly to FD002 and FD004, where there are
  six conditions. The code supports all four subsets, but the conclusions in
  this README apply to what was actually run.
- **Three seeds is few.** Enough to see the order of magnitude of the spread,
  not enough for a statistically significant claim that one architecture
  beats another.
- **Hyperparameters were not tuned per model.** Reasonable defaults were
  used. Careful tuning could change the ordering in the table.
- **C-MAPSS is a simulation, not production data.** The noise, gaps and
  sensor faults one deals with in operation are absent here.

## Publications

The comparison methodology implemented in this repository was used in the
following papers:

Shekh Al Ard M.S., Potekhin V.V., Zhunenkov A.S. "Comparative analysis of
LSTM and Random Forest models for predicting the remaining useful life of
industrial equipment from multivariate time series data". In: *IT. Science.
Creative*, Vol. 1, 2026, pp. 129–135. ISBN 978-5-8149-4124-4, RSCI.

Shekh Al Ard M.S., Kovalchuk E.A., Potekhin V.V. "Proactive Monitoring of
Equipment Condition in Multi-Agent Systems Using LSTM Neural Networks". 37th
DAAAM International Symposium on Intelligent Manufacturing and Automation,
Vienna, 2026.

## Data

Turbofan Engine Degradation Simulation Data Set, NASA Prognostics Center of
Excellence. Distributed separately, not included in this repository.

## License

MIT — see [LICENSE](LICENSE).
