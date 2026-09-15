# Algorithms

BSSUnfold.jl implements **55+ neutron spectrum unfolding solvers** (64
including auxiliary and helper variants such as `solve_direct` and the
`*_combined`/`*_full`/`*_dictionary` flavours). All solvers follow a single
interface: `solve_<algorithm>(A, b, x0; kwargs...) -> UnfoldResult`.

Convex/SCS/OSQP/JSON are hard dependencies of the package (v0.4.0).
`solve_mcmc` uses Turing.jl lazily and degrades gracefully (zero spectrum with
a warning) if Turing is not installed.

## Iterative EM methods

### MLEM — Maximum Likelihood Expectation Maximization

```math
x_{k+1} = x_k \odot (A^T (b / (A x_k)))
```

- **Preserves** non-negativity
- **Monotonically** increases the likelihood
- Converges slowly; typically 500–5000 iterations

```julia
result = solve_mlem(A, b, x0, max_iterations=2000, tolerance=1e-8)
```

Related solvers: `solve_mlem_stop` (with a stopping criterion), `solve_mapem`
(maximum a posteriori EM).

### OSEM — Ordered Subset Expectation Maximization

Groups measurements into subsets and updates $x$ per subset. Speeds up
convergence by a factor of `n_subsets`.

```julia
result = solve_osem(A, b, x0, max_iterations=50, n_subsets=4)
```

### BSREM — Block-Sequential Regularized EM

OSEM with built-in regularization (L₂ by default):

```julia
result = solve_bsrem(A, b, x0, max_iterations=50, n_subsets=4, regularization=1e-3)
```

### SART

Simultaneous Algebraic Reconstruction Technique — a row-action histogram
method with backward correction averaging:

```julia
result = solve_sart(A, b, x0, max_iterations=200)
```

## Weighted (BSS classics)

### GRAVEL

Weighted log-likelihood method. The most popular BSS unfolding algorithm.

```math
x_{k+1}[j] = x_k[j] \exp\left(\frac{\sum_i W_{ij} \ln(b_i/(Ax_k)_i)}{\sum_i W_{ij}}\right)
```

```julia
result = solve_gravel(A, b, x0, max_iterations=500, tolerance=1e-8)
```

### MAXED family — Maximum Entropy Deconvolution

Maximizes Shannon entropy subject to $Ax = b$. Variants: `solve_maxed`
(classic), `solve_amaxed` (adaptive), `solve_amaxed_regularization` (with
regularization term), `solve_imaxed` (incremental).

```julia
result = solve_maxed(A, b, x0, max_iterations=500)
```

## Direct and regularized methods

### Tikhonov regularization

```math
\min_x \|Ax - b\|^2 + \lambda \|Lx\|^2
```

Solved via the normal equations `(A^T A + λI) x = A^T b`. Variants:
`solve_tikhonov` (base), `solve_tikhonov_nnls` (non-negative least squares
constraint via Lawson–Hanson), `solve_tikhonov_tv` (total-variation penalty),
`solve_tikhonov_legendre` (legende penality on differences).

```julia
result = solve_tikhonov(A, b, x0, regularization=1e-3)
```

### TSVD — Truncated Singular Value Decomposition

Discards singular values below a threshold.

```julia
result = solve_tsvd(A, b, x0, truncation_rank=8)
```

### Direct / scipy-style

`solve_direct` solves the pseudoinverse directly; `solve_scipy_direct` ports
the scipy-based direct approach; `solve_statreg` uses statistical
regularization; `solve_reconst` reconstructs the spectrum iteratively from the
counts.

## Iterative least-squares methods

### Landweber

```math
x_{k+1} = x_k + \omega A^T (b - A x_k)
```

```julia
result = solve_landweber(A, b, x0, max_iterations=500, omega=0.0)
```

### Kaczmarz

Row-action method: updates $x$ one row of $A$ at a time. Variants:
`solve_kaczmarz` (randomized order), `solve_randomized_kaczmarz`
(probability-weighted row sampling).

```julia
result = solve_kaczmarz(A, b, x0, max_iterations=100)
```

### CGLS — Conjugate Gradient Least Squares

Applies CG to the normal equations without forming them explicitly.

```julia
result = solve_cgls(A, b, x0, max_iterations=200)
```

### FISTA — Fast Iterative Shrinkage-Thresholding

Proximal gradient method with $O(1/k^2)$ acceleration and L1 shrinkage.

```julia
result = solve_fista(A, b, x0, max_iterations=200, regularization=1e-4)
```

### Matrix Krylov / refinement

`solve_lanczos` (Krylov projection), `solve_iterative_refinement`
(floating-point accurate refinement), `solve_hybrid_gmres` (GMRES with
restarts).

```julia
result = solve_lanczos(A, b, x0, max_iterations=100)
result = solve_iterative_refinement(A, b, x0, max_iterations=20)
result = solve_hybrid_gmres(A, b, x0, max_iterations=200)
```

## Classical BSS methods

### Sandii (1970)

Iterative EM-like algorithm that preserves the spectrum integral.

```julia
result = solve_sandii(A, b, x0, max_iterations=500)
```

### Bunki

Modified MLEM with relaxation factor $\alpha$:

```math
x_{k+1} = x_k \cdot (1 + \alpha (A^T (b/Ax) - 1))
```

Variants: `solve_bunki`, `solve_bunkiut` (uncertainty-transformed weights),
`solve_rebunki` (with reconstruction monitoring).

```julia
result = solve_bunki(A, b, x0, max_iterations=500, alpha=0.8)
```

### Staysl (1982)

Bayesian method with a prior spectrum:

```julia
result = solve_staysl(A, b, x0, max_iterations=500)
```

### Doroshenko (1986)

Iterative method that preserves the integral.

```julia
result = solve_doroshenko(A, b, x0, max_iterations=500)
```

### Ferdor

Forward–reverse dual correction method:

```julia
result = solve_ferdor(A, b, x0, max_iterations=500)
```

### Directed divergence

Divergence-minimizing update (`solve_directed_divergence`).

## Parametric and hybrid methods

```julia
result = solve_parametric(A, b, x0; params)
result = solve_parametric2(A, b, x0; params)
result = solve_hybrid_parametric(A, b, x0; params)
```

`solve_parametric_cvxpy` and `solve_parametric_qpsolvers` combine parametric
capturing with the convex stacks (Convex.jl/SCS and OSQP.jl respectively);
`solve_parametric_combined` merges several parametric evaluations.

## N-splines (Islamgulov & Lartsev, 2008)

Spline-based unfolding with automatic knot selection:

```julia
result = solve_nspline(A, b, x0, bn_knots=6)
result = solve_nspline_full(A, b, x0; auto_knots=...)
```

Helper exports: `auto_knots`, `build_continuity_matrix`, `fit_nspline`,
`nspline_eval`, `NSPLINE_KNOT_PRESETS`.

## Catalogue + SPUNIT (NSDUAZ)

`solve_nsduaz` automatically selects the initial spectrum from a built-in
catalogue of NPP/referenced shapes and then refines it:

```julia
result = solve_nsduaz(A, b, x0)
initial = select_catalogue_initial(A, b)  # standalone catalogue selection
```

Helpers: `builtin_catalogue`, `nsduaz_builtin_catalogue`,
`nsduaz_reference_index`, `nsduaz_select_catalogue_initial`.

## Sparse / dictionary methods

`solve_omp` (Orthogonal Matching Pursuit), `solve_ksvd` (K-SVD dictionary
learning; also `solve_ksvd`/`solve_ksvd_dictionary` variants), `solve_nn_omp`
(non-negative OMP), `solve_nnksvd` (non-negative K-SVD;
`*_dictionary` helper builds the dictionary), `solve_nnls_topk` (top-k
activation on NNLS), `solve_sl0` (smoothed L0), `solve_cs` (compressed
sensing).

## Global optimization and ensembles

- `solve_genetic` — native PSO/GA/DE/GWO/NSGA-II metaheuristics (no mealpy
  dependency).
- `solve_qubo` — binary spectrum encoding + simulated annealing.
- `solve_eki` — Ensemble Kalman Inversion.
- `solve_ensemble` — ensemble of restarts with averaging.
- `solve_maeo` / `solve_maeo_ensemble` — MAEO assimilation.

## Convex optimization

Hard dependency (Convex.jl + SCS + OSQP.jl are in `Project.toml`):

```julia
result = solve_cvxpy(A, b, x0)        # Convex.jl + SCS
result = solve_qpsolvers(A, b, x0)    # OSQP.jl
```

## Bayesian

```julia
result = solve_bayes(A, b, x0, prior=:piecewise)
result = solve_bayes_spline(A, b, x0, n_knots=6)
result = solve_mcmc(A, b, x0, n_samples=1000)   # NUTS; requires Turing.jl (lazy)
```

## Summary table

| Algorithm    | Type                    | Regularization | Speed       | Accuracy |
|--------------|-------------------------|----------------|-------------|----------|
| MLEM         | EM iterative            | None           | Slow        | High   |
| OSEM         | EM subset               | None           | Fast        | Medium |
| BSREM        | EM subset + reg         | Yes            | Fast        | High   |
| SART         | Row-action              | None           | Fast        | Medium |
| GRAVEL       | Weighted                | Weak           | Medium      | High   |
| MAXED/AMAXED | Maximum entropy         | Built-in       | Medium      | Medium |
| Tikhonov     | Direct                  | Strong         | Very fast   | Low(negatives possible) |
| tsvd         | Direct                  | Strong         | Very fast   | Low    |
| Landweber    | Iterative               | None           | Medium      | Medium |
| Kaczmarz     | Row-action              | None           | Fast        | Medium |
| CGLS         | CG                      | Weak           | Fast        | High   |
| FISTA        | Proximal gradient       | L1             | Fast        | Medium |
| Lanczos      | Krylov                  | Implicit       | Fast        | High   |
| Sandii       | EM variant              | None           | Medium      | Medium |
| Bunki        | Relaxed EM              | None           | Medium      | Medium |
| Staysl       | Bayesian                | Prior          | Medium      | High   |
| Doroshenko   | Iterative               | None           | Medium      | Medium |
| parametric   | Parametric              | Parametric form| Fast        | High   |
| nspline      | Spline                  | Spline smoothness | Fast     | High   |
| nsduaz       | Catalogue + iterative   | Selection      | Fast        | High   |
| genetic      | Metaheuristic           | None           | Slow        | Medium-High |
| qubo         | Binary + annealed       | Binary structure | Slow     | Medium |
| mcmc         | Bayesian NUTS           | Prior          | Slow        | High   |
| cvxpy/qpsolvers | Convex/QP           | Trust region   | Fast        | Medium |
| omp/nnksvd/NN-omp | Sparse dictionary  | Sparsity       | Fast        | Medium |
| eki/ensemble | Monte-Carlo ensemble    | Prior          | Slow        | Medium |

The exact set of keyword arguments and published tunings can be verified by
running `examples/33-methods_comparison.jl` with `benchmark_unfold_methods`,
which ranks all methods on IAEA reference spectra by 52 metrics.
