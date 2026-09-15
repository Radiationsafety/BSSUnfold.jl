"""
bssunfold_julia — Python bridge for BSSUnfold.jl.

On import it patches bssunfold.solve_* and Detector.unfold_* so that
algorithms available in Julia are executed via PythonCall.
The rest are kept as-is (Python fallback).

Usage:
    import bssunfold
    import bssunfold_julia  # activates the patch

    d = bssunfold.Detector(...)
    result = d.unfold_mlem(readings)  # ← Julia under the hood
"""
from __future__ import annotations

import importlib
import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional

__version__ = "0.1.0"

logger = logging.getLogger("bssunfold_julia")

# ─── State ───────────────────────────────────────────────────────────────────
_JULIA_INITIALIZED = False
_BSSUNFOLD_MODULE = None
_PATCHED_FUNCTIONS: Dict[str, Callable] = {}
_PATCHED_DETECTORS: set = set()

# List of algorithms ported to Julia
# (Python-bridge method name -> BSSUnfold.jl function name)
JULIA_ALGORITHMS = {
    # basic (v0.1)
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
    # v0.2 extensions
    "lanczos":                "solve_lanczos",
    "iterative_refinement":   "solve_iterative_refinement",
    "randomized_kaczmarz":    "solve_randomized_kaczmarz",
    "cvxpy":                  "solve_cvxpy",
    "qpsolvers":              "solve_qpsolvers",
    # bssunfold port (v0.3) — all pure algorithms
    "amaxed":                 "solve_amaxed",
    "amaxed_regularization":  "solve_amaxed_regularization",
    "imaxed":                 "solve_imaxed",
    "sart":                   "solve_sart",
    "mapem":                  "solve_mapem",
    "mlem_stop":              "solve_mlem_stop",
    "bunkiut":                "solve_bunkiut",
    "rebunki":                "solve_rebunki",
    "directed_divergence":    "solve_directed_divergence",
    "ferdor":                 "solve_ferdor",
    "scipy_direct":           "solve_scipy_direct",
    "tikhonov_tv":            "solve_tikhonov_tv",
    "tikhonov_legendre":      "solve_tikhonov_legendre",
    "statreg":                "solve_statreg",
    "reconst":                "solve_reconst",
    "bayes":                  "solve_bayes",
    "bayes_spline":           "solve_bayes_spline",
    "eki":                    "solve_eki",
    "express":                "solve_express",
    "crystal_ball":           "solve_crystal_ball",
    "ensemble":               "solve_ensemble",
    "cs":                     "solve_cs",
    "binned":                 "solve_binned",
    "gks":                    "solve_gks",
    "maeo":                   "solve_maeo",
    "nsduaz":                 "solve_nsduaz",
    "nnksvd":                 "solve_nnksvd",
    "nspline":                "solve_nspline",
    "hybrid_gmres":           "solve_hybrid_gmres",
    "hybrid_parametric":      "solve_hybrid_parametric",
    "parametric":             "solve_parametric",
    "parametric2":            "solve_parametric2",
    # bssunfold port v0.3 (merged with remote main)
    "mcmc":                   "solve_mcmc",
    "genetic":                "solve_genetic",
    "qubo":                   "solve_qubo",
}

# Highly-port dependencies (PyMC, mealpy, dwave, zfit/tensorflow, z3-solver,
# docplex, CPLEX-SCIP, pyoptexplain, ODL) with complex backends execute via
# Python fallback — see README.
# Methods with heavy Python-only dependencies (docplex, CPLEX-SCIP,
# z3-solver, pyoptexplain, ODL, tensorflow/zfit, mealpy) run via the
# Python fallback — see README. MCMC in the Julia implementation uses
# Turing.jl lazily (graceful degradation without it).
PYTHON_FALLBACK = [
    "scip", "docplex", "lmfit", "zfit", "smt", "interpret",
    "odl_advanced", "mlem_odl", "fruit_like",
]


def _init_julia() -> Optional[Any]:
    """Lazily initialize Julia via juliacall. Returns the BSSUnfold module."""
    global _JULIA_INITIALIZED, _BSSUNFOLD_MODULE
    if _JULIA_INITIALIZED:
        return _BSSUNFOLD_MODULE

    try:
        # juliacall expects the JULIA_PKG_PRECOMPILE_AUTO environment variable
        # to control auto-precompilation; enable it by default.
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
        _JULIA_INITIALIZED = True  # do not retry
        return None
    except Exception as e:
        logger.warning(
            f"Failed to initialize Julia/BSSUnfold.jl: {e}. "
            "Falling back to pure Python bssunfold."
        )
        _JULIA_INITIALIZED = True
        return None


def _to_julia(obj):
    """Convert a numpy array to a Julia-compatible object."""
    try:
        from juliacall import Main as jl  # type: ignore
        import numpy as np
        # juliacall converts numpy arrays to Julia Arrays automatically
        # via the __jl_array__ protocol
        return obj
    except ImportError:
        return obj


def _from_julia(obj):
    """Convert a Julia UnfoldResult into a Python dict (like bssunfold)."""
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
    """Create a solve_func that tries Julia first and falls back to Python on error.

    Signature matches bssunfold: solve_*(A, b, x0=None, **kwargs) -> tuple.
    """
    julia_module = _init_julia()
    if julia_module is None:
        # Julia unavailable — return the original function
        return original_solve

    julia_fn = getattr(julia_module, julia_fn_name, None)
    if julia_fn is None:
        logger.warning(f"BSSUnfold.jl has no function {julia_fn_name}; fallback")
        return original_solve

    @wraps(original_solve)
    def wrapper(A, b, x0=None, **kwargs):
        try:
            # Filter out kwargs that do not belong to the Julia function
            # (e.g. validate_system and other Python-specific options)
            jkwargs = {k: v for k, v in kwargs.items()
                       if k in ("max_iterations", "tolerance", "regularization",
                                "alpha", "n_subsets", "step_size", "truncation_rank",
                                "eps", "omega", "noise_level", "lambda_range",
                                "method")}
            if x0 is None:
                # Standard default
                import numpy as np
                n = A.shape[1] if hasattr(A, "shape") else len(A[0])
                x0 = np.ones(n) * 0.5
            result = julia_fn(A, b, x0, **jkwargs)
            converted = _from_julia(result)
            if converted is None:
                # Conversion failed — fallback
                return original_solve(A, b, x0=x0, **kwargs)
            # bssunfold API returns a tuple (spectrum, iters, converged) or just the spectrum
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
    """Patch the bssunfold module: replace solve_* with Julia wrappers.

    Returns a dict of replaced functions {name: original_func}.
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
        # Do not patch twice
        if getattr(original_solve, "_julia_wrapped", False):
            continue
        # Keep the original and replace it
        patched[solve_name] = original_solve
        wrapper = make_julia_solve(julia_fn_name, original_solve)
        setattr(bssunfold_mod.core, solve_name, wrapper)
        # Also in the top-level bssunfold namespace
        if hasattr(bssunfold_mod, solve_name):
            setattr(bssunfold_mod, solve_name, wrapper)
        logger.debug(f"Patched {solve_name} → Julia")

    return patched


# ─── Auto-activation on import ───────────────────────────────────────────────
def _auto_activate():
    """Activate the Julia bridge on module import.

    Can be disabled with the environment variable BSSUNFOLD_JULIA=0.
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
    """Return the bridge status for diagnostics."""
    return {
        "julia_initialized": _JULIA_INITIALIZED,
        "bssunfold_loaded": _BSSUNFOLD_MODULE is not None,
        "patched_functions": list(_PATCHED_FUNCTIONS.keys()),
        "julia_algorithms": list(JULIA_ALGORITHMS.keys()),
    }
