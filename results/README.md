# Results

`RESULTS.md` is generated automatically by `src/report.py` after training runs
have completed. It aggregates all JSON files in `results/raw/` across seeds and
produces a ranked Markdown table.

To regenerate:
```bash
python src/report.py --results-dir results/raw --output results/RESULTS.md
```

Raw per-run JSON files (`results/raw/{subset}_{model}_seed{seed}.json`) and
prediction arrays (`*.npy`) are **not committed** to this repository.
