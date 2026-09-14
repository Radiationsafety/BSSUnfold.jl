"""
bssunfold_julia — Python bridge для BSSUnfold.jl.

При импорте патчит bssunfold.solve_* и Detector.unfold_* так,
что алгоритмы, доступные в Julia, выполняются через PythonCall.
Остальные остаются как есть (Python fallback).

Использование:
    import bssunfold
    import bssunfold_julia  # активирует патч

    d = bssunfold.Detector(...)
    result = d.unfold_mlem(readings)  # ← Julia под капотом
"""
from __future__ import annotations

import importlib
import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional

__version__ = "0.1.0"

logger = logging.getLogger("bssunfold_julia")

# ─── Состояние ───────────────────────────────────────────────────────────────
_JULIA_INITIALIZED = False
_BSSUNFOLD_MODULE = None
_PATCHED_FUNCTIONS: Dict[str, Callable] = {}
_PATCHED_DETECTORS: set = set()

# Перечень алгоритмов, перенесённых на Julia
# (имя метода в Python-bridge -> имя функции в BSSUnfold.jl)
JULIA_ALGORITHMS = {
    "mlem":       "solve_mlem",
    "gravel":     "solve_gravel",
    "landweber":  "solve_landweber",
    "maxed":      "solve_maxed",
    "tikhonov":   "solve_tikhonov",
    "tsvd":       "solve_tsvd",
    "sandii":     "solve_sandii",
    "bunki":      "solve_bunki",
    "kaczmarz":   "solve_kaczmarz",
    "cgls":       "solve_cgls",
    "fista":      "solve_fista",
    "bsrem":      "solve_bsrem",
    "osem":       "solve_osem",
    "staysl":     "solve_staysl",
    "doroshenko": "solve_doroshenko",
}


def _init_julia() -> Optional[Any]:
    """Лениво инициализировать Julia через juliacall. Возвращает BSSUnfold module."""
    global _JULIA_INITIALIZED, _BSSUNFOLD_MODULE
    if _JULIA_INITIALIZED:
        return _BSSUNFOLD_MODULE

    try:
        # juliacall ожидает, что переменная окружения JULIA_PKG_PRECOMPILE_AUTO
        # управляет авто-прекомпиляцией; по умолчанию включаем.
        os.environ.setdefault("JULIA_PKG_PRECOMPILE_AUTO", "1")

        from juliacall import Main as jl  # type: ignore
        jl.seval("using BSSUnfold")
        _BSSUNFOLD_MODULE = jl.BSSUnfold
        _JULIA_INITIALIZED = True
        logger.info("BSSUnfold.jl loaded via juliacall")
        return _BSSUNFOLD_MODULE
    except ImportError as e:
        logger.warning(
            "juliacall not installed; falling back to pure Python bssunfold. "
            f"Install with: pip install juliacall. Error: {e}"
        )
        _JULIA_INITIALIZED = True  # не пытаться снова
        return None
    except Exception as e:
        logger.warning(
            f"Failed to initialize Julia/BSSUnfold.jl: {e}. "
            "Falling back to pure Python bssunfold."
        )
        _JULIA_INITIALIZED = True
        return None


def _to_julia(obj):
    """Преобразовать numpy array в Julia-совместимый объект."""
    try:
        from juliacall import Main as jl  # type: ignore
        import numpy as np
        # juliacall автоматически конвертирует numpy arrays в Julia Arrays
        # через __jl_array__ protocol
        return obj
    except ImportError:
        return obj


def _from_julia(obj):
    """Преобразовать Julia UnfoldResult в Python dict (как bssunfold)."""
    import numpy as np
    try:
        spectrum = np.asarray(obj.spectrum)
        residual_norm = float(obj.residual_norm)
        iterations = int(obj.iterations)
        converged = bool(obj.converged)
        return {
            "spectrum": spectrum,
            "iterations": iterations,
            "converged": converged,
            "residual_norm": residual_norm,
        }
    except Exception as e:
        logger.warning(f"Failed to convert Julia result: {e}")
        return None


def make_julia_solve(julia_fn_name: str, original_solve: Callable) -> Callable:
    """Создать solve_func, который пытается Julia, и при ошибке fallback на Python.

    Сигнатура соответствует bssunfold: solve_*(A, b, x0=None, **kwargs) -> tuple.
    """
    julia_module = _init_julia()
    if julia_module is None:
        # Julia недоступна — возвращаем оригинал
        return original_solve

    julia_fn = getattr(julia_module, julia_fn_name, None)
    if julia_fn is None:
        logger.warning(f"BSSUnfold.jl has no function {julia_fn_name}; fallback")
        return original_solve

    @wraps(original_solve)
    def wrapper(A, b, x0=None, **kwargs):
        try:
            # Фильтруем kwargs, не относящиеся к Julia-функции
            # (напр. validate_system, прочие Python-specific)
            jkwargs = {k: v for k, v in kwargs.items()
                       if k in ("max_iterations", "tolerance", "regularization",
                                "alpha", "n_subsets", "step_size", "truncation_rank",
                                "eps", "omega", "noise_level", "lambda_range",
                                "method")}
            if x0 is None:
                # Стандартный default
                import numpy as np
                n = A.shape[1] if hasattr(A, "shape") else len(A[0])
                x0 = np.ones(n) * 0.5
            result = julia_fn(A, b, x0, **jkwargs)
            converted = _from_julia(result)
            if converted is None:
                # Конверсия не удалась — fallback
                return original_solve(A, b, x0=x0, **kwargs)
            # bssunfold API возвращает tuple (spectrum, iters, converged) или только spectrum
            return (converted["spectrum"], converted["iterations"],
                    converted["converged"])
        except Exception as e:
            logger.warning(
                f"Julia call failed for {julia_fn_name}: {e}; "
                "falling back to Python implementation"
            )
            return original_solve(A, b, x0=x0, **kwargs)

    wrapper.__name__ = f"{original_solve.__name__}_julia"
    wrapper._julia_wrapped = True
    return wrapper


def patch_module(module_name: str = "bssunfold") -> Dict[str, Callable]:
    """Пропатчить модуль bssunfold: заменить solve_* на Julia-обёртки.

    Возвращает словарь заменённых функций {name: original_func}.
    """
    try:
        bssunfold_mod = importlib.import_module(module_name)
    except ImportError:
        logger.info(f"{module_name} not installed; nothing to patch")
        return {}

    patched = {}
    for algo_name, julia_fn_name in JULIA_ALGORITHMS.items():
        solve_name = f"solve_{algo_name}"
        original_solve = getattr(bssunfold_mod.core, solve_name, None)
        if original_solve is None:
            continue
        # Не патчим повторно
        if getattr(original_solve, "_julia_wrapped", False):
            continue
        # Сохраняем оригинал и заменяем
        patched[solve_name] = original_solve
        wrapper = make_julia_solve(julia_fn_name, original_solve)
        setattr(bssunfold_mod.core, solve_name, wrapper)
        # Также в самом верхнем bssunfold-пространстве имён
        if hasattr(bssunfold_mod, solve_name):
            setattr(bssunfold_mod, solve_name, wrapper)
        logger.debug(f"Patched {solve_name} → Julia")

    return patched


# ─── Авто-активация при импорте ───────────────────────────────────────────────
def _auto_activate():
    """Активировать Julia bridge при импорте модуля.

    Можно отключить переменной окружения BSSUNFOLD_JULIA=0.
    """
    if os.environ.get("BSSUNFOLD_JULIA", "1") == "0":
        logger.info("BSSUNFOLD_JULIA=0; bridge disabled")
        return

    patched = patch_module("bssunfold")
    _PATCHED_FUNCTIONS.update(patched)
    if patched:
        logger.info(
            f"bssunfold_julia active: {len(patched)} methods routed through Julia. "
            f"Other methods fall back to pure Python."
        )


_auto_activate()


def status() -> Dict[str, Any]:
    """Вернуть статус bridge для диагностики."""
    return {
        "julia_initialized": _JULIA_INITIALIZED,
        "bssunfold_loaded": _BSSUNFOLD_MODULE is not None,
        "patched_functions": list(_PATCHED_FUNCTIONS.keys()),
        "julia_algorithms": list(JULIA_ALGORITHMS.keys()),
    }
