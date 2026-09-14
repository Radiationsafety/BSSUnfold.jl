# Сравнение с bssunfold (Python)

BSSUnfold.jl — это Julia-порт Python-пакета
[bssunfold](https://github.com/Radiationsafety/bssunfold). Этот документ
описывает различия и соответствие API.

## Перенесённые алгоритмы

| Python-имя                  | Julia-имя               | Статус          |
|-----------------------------|-------------------------|-----------------|
| `solve_mlem`                | `solve_mlem`            | ✅ Полный порт |
| `solve_gravel`              | `solve_gravel`          | ✅ Полный порт |
| `solve_landweber`           | `solve_landweber`       | ✅ Полный порт |
| `solve_maxed`               | `solve_maxed`           | ✅ Полный порт |
| `solve_tikhonov`            | `solve_tikhonov`        | ✅ Полный порт |
| `solve_tsvd`                | `solve_tsvd`            | ✅ Полный порт |
| `solve_sandii`              | `solve_sandii`          | ✅ Полный порт |
| `solve_bunki`               | `solve_bunki`           | ✅ Полный порт |
| `solve_kaczmarz`            | `solve_kaczmarz`        | ✅ Полный порт |
| `solve_cgls`                | `solve_cgls`            | ✅ Полный порт |
| `solve_fista`               | `solve_fista`           | ✅ Полный порт |
| `solve_bsrem`               | `solve_bsrem`           | ✅ Полный порт |
| `solve_osem`                | `solve_osem`            | ✅ Полный порт |
| `solve_staysl`              | `solve_staysl`          | ✅ Полный порт |
| `solve_doroshenko`          | `solve_doroshenko`      | ✅ Полный порт |

## Алгоритмы, не перенесённые на Julia

Эти алгоритмы требуют внешних Python-зависимостей и доступны через
`PythonCall.jl` bridge:

| Python-имя                  | Зависимость       | Причина отсутствия в Julia  |
|-----------------------------|-------------------|-----------------------------|
| `solve_mcmc`                | PyMC              | Turing.jl нужен            |
| `solve_genetic`             | mealpy            | Metaheuristics.jl нужен    |
| `solve_qubo`                | pyqubo, dwave-neal| Annealers.jl               |
| `solve_zfit`                | zfit, tensorflow  | Flux.jl + Distributions.jl |
| `solve_smt`                 | z3-solver         | Z3.jl                       |
| `solve_mystic`              | mystic            | Optim.jl                   |
| `solve_maeo`                | pymoo             | MOA.jl                      |
| `solve_interpret`           | pyoptexplain      | Custom                     |
| `solve_docplex`             | cplex, docplex    | CPLEX.jl                   |
| `solve_scip`                | pyscipopt         | SCIP.jl                    |
| `solve_nspline`             | custom             | В планах                   |
| `solve_nnksvd`              | custom             | В планах                   |
| `solve_parametric`          | custom             | В планах                   |
| `solve_bayesian_parametric` | custom             | В планах                   |

## Сравнение API

### Python

```python
from bssunfold import Detector
det = Detector.from_response_functions(df)
result = det.unfold_gravel(readings, max_iterations=500, calculate_errors=True)
spectrum = result["spectrum"]
```

### Julia

```julia
using BSSUnfold
det = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)
result = unfold_gravel(det, readings, max_iterations=500, calculate_errors=true)
spectrum = result["spectrum"]
```

### Ключевые отличия

1. **Типы**: Julia использует `Vector{Float64}` вместо `np.ndarray`,
   `Dict{String,Float64}` вместо Python dict.
2. **Именованные аргументы**: `max_iterations` вместо `max_iterations=` (но синтаксис похож).
3. **Возвращаемое значение**: `UnfoldResult` struct (immutable), а не dict;
   но `unfold_*` возвращает `Dict{String,Any}` для совместимости.
4. **Индексация**: 1-based в Julia vs 0-based в Python.

## Использование Python и Julia вместе

Через `PythonCall.jl` можно использовать оба API одновременно:

```julia
using PythonCall
bssunfold_py = pyimport("bssunfold")
result_py = bssunfold_py.Detector.from_response_functions(df).unfold_gravel(readings)

using BSSUnfold
result_jl = unfold_gravel(detector, readings)
```

См. `python_bridge/` для готовой обёртки.
