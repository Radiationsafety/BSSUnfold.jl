# BSSUnfold.jl

Julia-порт пакета **bssunfold** для развёртки нейтронных спектров со спектрометров Боннера (BSS).

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

# Ответная матрица A (m × n): m сфер, n энергетических бинов
A = rand(14, 640)
# Истинный спектр
x_true = exp.(-collect(range(0, 10, length=640)))
b = A * x_true + 0.01 .* randn(14)

# MLEM
result = solve_mlem(A, b, ones(640) * 0.5, max_iterations=1000)
println("Сходился за $(result.iterations) итераций, ||res|| = $(result.residual_norm)")

# GRAVEL
result = solve_gravel(A, b, ones(640) * 0.5, max_iterations=500)

# Monte-Carlo неопределённость
mc = monte_carlo_uncertainty(solve_mlem, A, b, ones(640) * 0.5,
                             0.01, 100, random_state=42)
println("Std spectrum: ", mc.std)
```

## Доступные алгоритмы

| Алгоритм     | Функция                 | Тип                            |
|--------------|-------------------------|--------------------------------|
| MLEM         | `solve_mlem`            | Итеративный EM                 |
| GRAVEL       | `solve_gravel`          | Взвешенный log-likelihood      |
| Landweber    | `solve_landweber`       | Метод простой итерации         |
| MAXED        | `solve_maxed`           | Maximum entropy                |
| Sandii       | `solve_sandii`          | Итеративный                    |
| Bunki        | `solve_bunki`           | Вариант MLEM                   |
| Kaczmarz     | `solve_kaczmarz`        | Row-action метод               |
| CGLS         | `solve_cgls`            | Conjugate gradient LS          |
| FISTA        | `solve_fista`           | Proximal gradient              |
| BSREM        | `solve_bsrem`           | Block-sequential regularized  |
| OSEM         | `solve_osem`            | Ordered subset EM             |
| Tikhonov     | `solve_tikhonov`        | Регуляризация Тихонова         |
| TSVD         | `solve_tsvd`            | Truncated SVD                  |
| Staysl       | `solve_staysl`          | Байесовский с априорным спектром |
| Doroshenko   | `solve_doroshenko`      | Итеративный                    |

## Производительность

На матрице 14×640 (типичный размер BSS-задачи):
- MLEM: **3–25× быстрее** Python/NumPy
- GRAVEL: **2–12× быстрее**
- Landweber: **2–11× быстрее**

Подробности — в `/benchmark/` каталоге.

## Python-совместимость

Через `PythonCall.jl` пакет можно вызывать из Python:

```python
# В Python
from juliacall import Main as jl
jl.seval("using BSSUnfold")
result = jl.BSSUnfold.solve_mlem(A, b, x0)
```

См. `python_bridge/` для готовой обёртки, которая прозрачно заменяет
вызовы `bssunfold` на Julia-эквиваленты.

## Лицензия

GPL-3.0-only — наследуется от оригинального `bssunfold`.
