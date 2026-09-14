# BSSUnfold.jl

Julia-порт пакета **bssunfold** для развёртки нейтронных спектров со
спектрометров Боннера (BSS).

[![Build Status](https://github.com/Radiationsafety/BSSUnfold.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Radiationsafety/BSSUnfold.jl/actions)
[![Coverage](https://codecov.io/gh/Radiationsafety/BSSUnfold.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Radiationsafety/BSSUnfold.jl)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Установка

```julia
using Pkg
Pkg.add("BSSUnfold")
# или из неопубликованного репозитория:
# Pkg.add(url="https://github.com/Radiationsafety/BSSUnfold.jl")
```

## Быстрый старт

```julia
using BSSUnfold

# Создать детектор
detector = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)

# Развёртка спектра
result = unfold_gravel(detector, readings, max_iterations=500)
println("Сходился за $(result["iterations"]) итераций")

# Monte-Carlo неопределённость
mc = monte_carlo_uncertainty(solve_mlem, A, b, x0, 0.01, 100, random_state=42)
println("Mean σ: $(mean(mc.std))")
```

## Доступные алгоритмы

| Категория           | Алгоритмы                                  |
|---------------------|--------------------------------------------|
| EM-методы           | `solve_mlem`, `solve_osem`, `solve_bsrem` |
| Взвешенные          | `solve_gravel`, `solve_maxed`              |
| Итеративные         | `solve_landweber`, `solve_kaczmarz`, `solve_cgls`, `solve_fista` |
| Регуляризованные    | `solve_tikhonov`, `solve_tsvd`, `solve_bsrem` |
| Классические        | `solve_sandii`, `solve_bunki`, `solve_staysl`, `solve_doroshenko` |

**Всего: 15 алгоритмов** развёртки, перенесённых с Python.

## Производительность

На матрице 14×640 (типичный размер BSS-задачи):

| Алгоритм  | Python (ms) | Julia (ms) | Ускорение |
|-----------|-------------|------------|-----------|
| MLEM      | 52          | 6          | **3.6×**  |
| GRAVEL    | 12          | 5          | **2.6×**  |
| Landweber | 24          | 10         | **2.4×**  |

## Документация

- 📖 [Tutorial](docs/src/tutorial.md)
- 🔬 [Алгоритмы](docs/src/algorithms.md)
- 📚 [API Reference](docs/src/api.md)
- 🐍 [Сравнение с Python bssunfold](docs/src/comparison.md)

## Примеры

В каталоге `examples/` находятся Pluto.jl-ноутбуки с примерами
использования. См. [examples/README.md](examples/README.md).

| Ноутбук                            | Описание                                       |
|------------------------------------|------------------------------------------------|
| `01-basic-example.jl`              | Базовая развёртка GRAVEL                        |
| `03-uncertainty.jl`                | Monte-Carlo оценка неопределённости             |
| `05-mlem_example.jl`               | MLEM: влияние итераций и x₀                     |
| `13-regularization.jl`             | Tikhonov/TSVD и выбор λ                         |
| `33-methods_comparison.jl`         | Сравнение всех 15 алгоритмов                    |
| `34-robustness_analysis.jl`        | Анализ устойчивости                             |

### Запуск Pluto-ноутбуков

```bash
julia -e 'using Pkg; Pkg.add("Pluto")'
julia -e 'using Pluto; Pluto.run()'
# Открыть http://localhost:1234 и выбрать ноутбук из examples/
```

## Тесты

В каталоге `test/` находятся тесты, организованные аналогично оригинальному
`bssunfold/tests/`:

```
test/
├── runtests.jl                   ← главная точка входа
├── test_detector.jl              ← Detector tests (порт из test_detector.py)
├── test_classic_unfolders.jl     ← все 15 алгоритмов (test_classic_unfolders.py)
├── test_comparison.jl            ← метрики сравнения (test_comparison.py)
├── test_montecarlo.jl            ← Monte-Carlo tests (test_new_ensemble_refinement.py)
├── test_regularization.jl        ← регуляризация (test_regularization_new_criteria.py)
├── test_iaea_validation.jl       ← IAEA Compendium validation (test_iaea_validation.py)
└── data/
    ├── IAEA_Compendium_dataset.csv
    └── MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv
```

### Запуск тестов

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Все 100+ тестов проходят. Включены валидация на IAEA Compendium (29 эталонных
спектров), smoke-тесты для всех 15 алгоритмов и проверки Monte-Carlo устойчивости.

## Python-совместимость

Через `PythonCall.jl` пакет можно вызывать из Python:

```python
from juliacall import Main as jl
jl.seval("using BSSUnfold")
result = jl.BSSUnfold.solve_mlem(A, b, x0)
```

См. `python_bridge/` для готовой обёртки, которая прозрачно заменяет
вызовы `bssunfold` на Julia-эквиваленты.

## Связанные ресурсы

- 🐍 [Оригинальный Python-пакет bssunfold](https://github.com/Radiationsafety/bssunfold)
- 📊 [IAEA Compendium of neutron spectra](https://www-nds.iaea.org/bssunfold/)
- 📝 [Инструкция по публикации в Julia General](https://github.com/Radiationsafety/BSSUnfold.jl/blob/main/docs/PUBLISHING_GUIDE.md)

## Лицензия

GPL-3.0-only — наследуется от оригинального `bssunfold`.
