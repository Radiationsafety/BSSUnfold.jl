# Примеры BSSUnfold.jl

Этот каталог содержит Pluto.jl-ноутбуки с примерами использования BSSUnfold.jl.
Ноутбуки организованы по тому же принципу, что и в оригинальном Python-пакете
`bssunfold` (https://github.com/Radiationsafety/bssunfold/tree/main/examples),
но используют идиоматическую Julia.

## Список ноутбуков

| Ноутбук                          | Описание                                              |
|----------------------------------|-------------------------------------------------------|
| `01-basic-example.jl`            | Базовая развёртка: создание задачи, GRAVEL, графики   |
| `03-uncertainty.jl`              | Monte-Carlo оценка неопределённости                    |
| `05-mlem_example.jl`             | MLEM: влияние итераций и начального спектра           |
| `13-regularization.jl`            | Tikhonov/TSVD и выбор параметра регуляризации         |
| `33-methods_comparison.jl`        | Сравнение всех 15 алгоритмов на одной задаче          |
| `34-robustness_analysis.jl`        | Анализ устойчивости к шуму, x₀, случайному seed       |

## Запуск

### Вариант A: Pluto.jl (нативный, рекомендуется)

```bash
# Установить Pluto (один раз)
julia -e 'using Pkg; Pkg.add("Pluto")'

# Запустить сервер Pluto
julia -e 'using Pluto; Pluto.run()'

# Открыть ноутбук в браузере по адресу http://localhost:1234
```

### Вариант B: Выполнение как скрипта

```bash
# Без открытия браузера
julia -e 'using Pluto; Pluto.Configuration.notebook_path = "examples/01-basic-example.jl"; include("examples/01-basic-example.jl")'

# Или конвертировать в HTML
julia -e 'using PlutoStaticHTML; html_notebook("examples/01-basic-example.jl")'
```

### Вариант C: IJulia / Jupyter

Ноутбуки Pluto можно конвертировать в Jupyter:

```bash
julia -e 'using Pluto, PlutoNotebookHelpers;
          Pluto.save_notebook("examples/01-basic-example.jl", "01-basic-example.ipynb")'
```

## Структура ноутбука

Каждый ноутбук следует стандартной структуре:

1. **Markdown-ячейка**: заголовок и описание
2. **Код-ячейка**: импорт пакетов и подготовка данных
3. **Markdown**: описание алгоритма с формулой
4. **Код**: вызов `solve_*` или `unfold_*`
5. **Код**: визуализация результата
6. **Markdown**: интерпретация и резюме

## Используемые пакеты

Ноутбуки зависят от:

- `BSSUnfold` — основной пакет (этот репозиторий)
- `Plots.jl` — визуализация
- `LinearAlgebra`, `Statistics`, `Random` — стандартная библиотека

Установка зависимостей:

```bash
julia --project=. -e 'using Pkg; Pkg.add(["Plots", "Pluto"])'
```

## Соответствие оригинальному bssunfold

| Python (bssunfold)                | Julia (BSSUnfold.jl)                |
|-----------------------------------|-------------------------------------|
| `01-basic-example.ipynb`          | `01-basic-example.jl`                |
| `03-uncertainty.ipynb`             | `03-uncertainty.jl`                  |
| `05-mlem_example.ipynb`           | `05-mlem_example.jl`                 |
| `13-Bayes_statreg.ipynb`          | `13-regularization.jl` (Tikhonov)    |
| `14-Maxed.ipynb`                  | часть `13-regularization.jl`        |
| `33-methods_comparison.ipynb`     | `33-methods_comparison.jl`           |
| `34-robustness_analysis.ipynb`    | `34-robustness_analysis.jl`          |

## Зависимости от Python-примеров

Следующие ноутбуки оригинала требуют Python-зависимостей, которых ещё нет в
BSSUnfold.jl. Они будут перенесены в будущем:

- `06-features.ipynb` — Detector features (нужна инфраструктура данных)
- `07-QP_solvers.ipynb` — Convex.jl порт
- `08-combined_algorithm.ipynb`
- `10-lmfit.ipynb` — LMFIT.jl
- `16-Parametric.ipynb` — параметрическая развёртка
- `22-Genetic_mealpy.ipynb` — Metaheuristics.jl
- `24-interpret.ipynb`
- `29-MCMC_example.ipynb` — Turing.jl
- `30-smt.ipynb` — Z3.jl
- `41-nspline.ipynb` — N-spline метод
- `43-all_methods_example.ipynb` — все методы

См. `python_bridge/` для использования этих алгоритмов из Julia через PyCall.
