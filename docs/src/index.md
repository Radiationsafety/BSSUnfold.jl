# BSSUnfold.jl

**Julia-порт пакета `bssunfold` для развёртки нейтронных спектров
со спектрометров Боннера (BSS).**

[![CI](https://github.com/Radiationsafety/BSSUnfold.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Radiationsafety/BSSUnfold.jl/actions)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Установка

```julia
using Pkg
Pkg.add("BSSUnfold")
# или из git:
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

Полный список — в [Algorithms](@ref).

## Производительность

На матрице 14×640 (типичный размер BSS-задачи):

| Алгоритм  | Python (ms) | Julia (ms) | Ускорение |
|-----------|-------------|------------|-----------|
| MLEM      | 52          | 6          | **3.6×**  |
| GRAVEL    | 12          | 5          | **2.6×**  |
| Landweber | 24          | 10         | **2.4×**  |

## Связанные ресурсы

- [Оригинальный Python-пакет bssunfold](https://github.com/Radiationsafety/bssunfold)
- [IAEA Compendium of neutron spectra](https://www-nds.iaea.org/bssunfold/)
- [Pluto-ноутбуки с примерами](https://github.com/Radiationsafety/BSSUnfold.jl/tree/main/examples)
- [Инструкция по публикации](@ref)
