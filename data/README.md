# Data

## Download

The NASA C-MAPSS dataset is publicly available from the NASA Prognostics Data Repository:

**URL:** https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

Direct download (CMAPSSData.zip, ~7 MB):
```
https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip
```

After downloading, extract the `.txt` files into this directory so the layout is:

```
data/
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
├── train_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
├── RUL_FD001.txt
├── RUL_FD002.txt
├── RUL_FD003.txt
└── RUL_FD004.txt
```

The `.txt` files are **not committed** to this repository (see `.gitignore`).

## Format

Each `train_FDxxx.txt` / `test_FDxxx.txt` file has 26 space-separated columns,
no header:

| Columns | Meaning |
|---------|---------|
| 1 | unit id |
| 2 | cycle |
| 3–5 | op_setting_1, op_setting_2, op_setting_3 |
| 6–26 | sensor_1 … sensor_21 |

`RUL_FDxxx.txt` contains one integer per line: the true remaining useful life
at the last observed cycle of each test unit, in unit order.

## Subsets

| Subset | Train units | Test units | Op. conditions | Fault modes |
|--------|-------------|------------|----------------|-------------|
| FD001  | 100         | 100        | 1              | 1           |
| FD002  | 260         | 259        | 6              | 1           |
| FD003  | 100         | 100        | 1              | 2           |
| FD004  | 249         | 248        | 6              | 2           |

FD001 is the simplest subset and a common starting point for benchmarks.
