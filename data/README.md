# Данные

Датасет NASA C-MAPSS не включён в репозиторий (условия распространения
и размер). Скачайте его и распакуйте сюда.

Источник: NASA Prognostics Center of Excellence Data Set Repository,
раздел «Turbofan Engine Degradation Simulation Data Set».

Ожидаемая структура:

```
data/CMAPSS/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt
└── ...
```

Формат файлов train/test: 26 колонок без заголовка, разделитель — пробел:
`unit`, `cycle`, `op_setting_1..3`, `sensor_1..21`.

Файл `RUL_FDxxx.txt` содержит истинный остаточный ресурс для последнего
цикла каждого юнита из тестовой выборки — по одному числу на строку.
