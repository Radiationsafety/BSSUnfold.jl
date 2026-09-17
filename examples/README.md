# BSSUnfold.jl examples

This directory contains Pluto.jl notebooks demonstrating how to use
BSSUnfold.jl. The notebooks follow the same organization as the original
Python package `bssunfold`
(https://github.com/Radiationsafety/bssunfold/tree/main/examples),
but use idiomatic Julia.

## Notebook list

| Notebook                          | Description                                            |
|-----------------------------------|---------------------------------------------------------|
| `01-basic-example.jl`             | Basic unfolding: problem setup, GRAVEL, plots           |
| `03-uncertainty.jl`               | Monte-Carlo uncertainty estimation                       |
| `05-mlem_example.jl`              | MLEM: effect of iterations and initial spectrum         |
| `13-regularization.jl`            | Tikhonov/TSVD and regularization parameter selection     |
| `33-methods_comparison.jl`        | Comparison of all solvers on one problem                |
| `34-robustness_analysis.jl`       | Robustness to noise, x₀, random seed                     |
| `40-real-spectra.jl`              | Real IAEA spectra: RF, reference spectra and dose rates |
| `45-seapearl.jl`                  | SeaPearl CP unfolding of an IAEA spectrum: feasible-set intervals, dose check, optional CP+RL learned heuristic |

## SeaPearl CP+RL training pipeline (`seapearl_training/`)

`examples/45-seapearl.jl` ships with a self-contained training pipeline for
the RL value-selection heuristic (methodology:
[learning-generic-csp](https://github.com/corail-research/learning-generic-csp)):

| File | Purpose |
|------|---------|
| `seapearl_training/bss_generator.jl` | `SeaPearl.AbstractModelGenerator`s: randomized BSS instances + the CP encoding mirrored from `solve_seapearl` |
| `seapearl_training/agent_builder.jl` | DQN + CPNN agent construction, parameter loading, inference-mode helper |
| `seapearl_training/train_seapearl_bss.jl` | Training script (Julia 1.9 side-environment with SeaPearl 0.4.5) |
| `seapearl_training/eval_seapearl_bss.jl` | Learned-vs-BasicHeuristic benchmark on held-out instances + the IAEA instance |
| `seapearl_training/materialize_iaea_instance.jl` | Exports the IAEA CP problem as JSON (run on Julia 1.10, bridges the two environments) |
| `data/seapearl_bss_agent.ser` | Pretrained network parameters (62 KiB) |
| `data/seapearl_bss_training_metrics.json` | Training/evaluation metrics of the shipped agent |
| `data/seapearl_iaea_instance.json` | Materialized IAEA reference instance (10 spheres × 15 bins) |

```bash
# Training (Julia 1.9 side-environment; SeaPearl 0.4.x needs ≤ 1.9)
julia-1.9 --project=<env-with-SeaPearl> examples/seapearl_training/train_seapearl_bss.jl \
    --episodes 100 --timeout 2400
julia-1.9 --project=<env-with-SeaPearl> examples/seapearl_training/eval_seapearl_bss.jl

# Regenerate the IAEA instance JSON (Julia ≥ 1.10, repository environment)
julia --project=. examples/seapearl_training/materialize_iaea_instance.jl
```

## Running

### Option A: Pluto.jl (native, recommended)

```bash
# Install Pluto (once)
julia -e 'using Pkg; Pkg.add("Pluto")'

# Start the Pluto server
julia -e 'using Pluto; Pluto.run()'

# Open a notebook in the browser at http://localhost:1234
```

### Option B: Run as a script

```bash
# Without opening a browser
julia -e 'using Pluto; Pluto.Configuration.notebook_path = "examples/01-basic-example.jl"; include("examples/01-basic-example.jl")'

# Or convert to HTML
julia -e 'using PlutoStaticHTML; html_notebook("examples/01-basic-example.jl")'
```

### Option C: IJulia / Jupyter

Pluto notebooks can be converted to Jupyter:

```bash
julia -e 'using Pluto, PlutoNotebookHelpers;
          Pluto.save_notebook("examples/01-basic-example.jl", "01-basic-example.ipynb")'
```

## Notebook structure

Each notebook follows a standard structure:

1. **Markdown cell**: title and description
2. **Code cell**: package imports and data preparation
3. **Markdown**: algorithm description with a formula
4. **Code**: call `solve_*` or `unfold_*`
5. **Code**: visualization of the result
6. **Markdown**: interpretation and summary

## Packages used

The notebooks depend on:

- `BSSUnfold` — the main package (this repository)
- `Plots.jl` — visualization
- `LinearAlgebra`, `Statistics`, `Random` — standard library

Installing dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.add(["Plots", "Pluto"])'
```

## Correspondence to the original bssunfold

| Python (bssunfold)                | Julia (BSSUnfold.jl)            |
|-----------------------------------|---------------------------------|
| `01-basic-example.ipynb`          | `01-basic-example.jl`            |
| `03-uncertainty.ipynb`            | `03-uncertainty.jl`              |
| `05-mlem_example.ipynb`           | `05-mlem_example.jl`             |
| `13-Bayes_statreg.ipynb`          | `13-regularization.jl` (Tikhonov)|
| `14-Maxed.ipynb`                  | part of `13-regularization.jl`   |
| `33-methods_comparison.ipynb`     | `33-methods_comparison.jl`       |
| `34-robustness_analysis.ipynb`    | `34-robustness_analysis.jl`      |

## Python-example dependencies

The following original notebooks require Python dependencies that were ported
using pure-Julia implementations:

- `07-QP_solvers.ipynb` — Convex.jl + SCS / OSQP.jl (`solve_cvxpy`, `solve_qpsolvers`)
- `16-Parametric.ipynb` — `solve_parametric` / `solve_parametric2`
- `22-Genetic_mealpy.ipynb` — native PSO/GA/DE/GWO/NSGA-II (`solve_genetic`)
- `29-MCMC_example.ipynb` — Turing.jl (`solve_mcmc`, lazy load)
- `41-nspline.ipynb` — `solve_nspline` / `solve_nspline_full`

Methods requiring ecosystem-specific Python stacks (CPLEX, SCIP, z3-solver,
zfit + TensorFlow, ODL, lmfit-based variants) can still be used through the
Python fallback — see `python_bridge/`.
