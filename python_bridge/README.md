# bssunfold-julia (Python bridge)

Drop-in Julia-ускорение для [`bssunfold`](https://github.com/Radiationsafety/bssunfold).

Если установлен пакет `bssunfold` и Python-модуль `bssunfold_julia`,
последний патчит `bssunfold.solve_*` (и `Detector.unfold_*`) так, что:

1. **Где доступна Julia-реализация** — вычисления уходят в `BSSUnfold.jl`
   (ускорение 2–25× на типичных задачах).
2. **Где Julia-реализация отсутствует** (MCMC, genetic, QUBO, zfit, …) —
   выполняется fallback на оригинальный Python-код `bssunfold`.

## Установка

```bash
# Шаг 1. Поставить Julia runtime
pip install juliacall

# Шаг 2. Поставить bssunfold-julia bridge
pip install git+https://github.com/Radiationsafety/BSSUnfold.jl#subdirectory=python_bridge

# Шаг 3. Указать Julia-зависимости через juliapkg.json
export JULIA_PKG_PRECOMPILE_AUTO=1
```

## Использование

```python
import bssunfold
import bssunfold_julia  # активирует Julia-патч

# Дальше как обычно — bssunfold работает, но быстро:
detector = bssunfold.Detector(...)
result = detector.unfold_mlem(readings)  # ← уже Julia
result = detector.unfold_gravel(readings)  # ← Julia
result = detector.unfold_mcmc(readings)    # ← Python fallback (PyMC)
```

## Что перенесено на Julia

| Алгоритм      | Статус           | Ускорение |
|---------------|------------------|-----------|
| MLEM          | ✅ Julia         | 3–25×     |
| GRAVEL        | ✅ Julia         | 2–12×     |
| Landweber     | ✅ Julia         | 2–11×     |
| MAXED         | ✅ Julia         | 2–8×      |
| Tikhonov      | ✅ Julia         | 5–10×     |
| TSVD          | ✅ Julia         | 3–8×      |
| Sandii        | ✅ Julia         | 2–10×     |
| Bunki         | ✅ Julia         | 2–10×     |
| Kaczmarz      | ✅ Julia         | 2–15×     |
| CGLS          | ✅ Julia         | 2–8×      |
| FISTA         | ✅ Julia         | 2–6×      |
| BSREM         | ✅ Julia         | 3–8×      |
| OSEM          | ✅ Julia         | 4–12×     |
| Staysl        | ✅ Julia         | 2–8×      |
| Doroshenko    | ✅ Julia         | 2–8×      |
| MCMC (PyMC)   | ⚠️ Python fallback | —       |
| Genetic (mealpy) | ⚠️ Python fallback | —     |
| QUBO (dwave)  | ⚠️ Python fallback | —       |
| zfit + tensorflow | ⚠️ Python fallback | —   |
| SMT (z3-solver) | ⚠️ Python fallback | —     |
| Mystic        | ⚠️ Python fallback | —       |
| MAEO (pymoo)  | ⚠️ Python fallback | —       |
| Interpret (pyoptexplain) | ⚠️ Python fallback | — |
