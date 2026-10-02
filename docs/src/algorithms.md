# Algorithms

BSSUnfold.jl implements **64+ neutron spectrum unfolding solvers** (73
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

## First-order optimization family

Solvers sharing the `solve_pgd` proximal/gradient skeleton; all accept
`max_iterations`, `tolerance` and the penalty weights of their Python
counterparts.

```julia
result = solve_pgd(A, b, x0, regularization=1e-3)                    # projected gradient
result = solve_pgd(A, b, x0, constraint="simplex", total_fluence=F)  # fluence-constrained
result = solve_coordinate_descent(A, b, x0, l1_penalty=1e-3, selection="cyclic")
result = solve_subgradient(A, b, x0, step_policy="diminishing", tv_penalty=1e-2)
result = solve_extragradient(A, b, x0, noise_level=0.02)             # Korpelevich 1976
result = solve_frank_wolfe(A, b, x0, total_fluence=F, away_steps=true)
result = solve_admm(A, b, x0, l1_penalty=1e-3, tv_penalty=1e-3)
result = solve_lbfgsb(A, b, x0, smoothness=1e-3, x_max=0.5)
```

- `solve_pgd` projects onto `nonnegative`, `box` (needs `x_max`) or `simplex`
  (needs `total_fluence`); `backtracking=true` replaces the fixed `1/L` step.
- `solve_admm` returns `primal_residual`, `dual_residual` and `rho` in
  `result.extra`; with both penalties at zero it reduces to a single NNLS solve,
  exactly as in Python. Its x-update solves the augmented NNLS on the Gram
  system `G = A'A + rho I + rho D'D` with an incremental Cholesky factor of the
  passive block, so the augmented matrix is never built and a pivot costs
  `O(k^2)` instead of a fresh least-squares solve — the reason the solver stays
  usable on realistic grids (n ≈ 640).
- `solve_lbfgsb` is a **native** L-BFGS-B implementation (two-loop recursion +
  Cauchy-point projection and Armijo backtracking) — this method does not use
  the optional JuMP/Optim backend. Bounds are `[x_min, x_max]`.
- `solve_subgradient` is genuinely slow: it does not reach a small chi-square by
  default, matching Python iteration for iteration.

## Additional classical BSS codes

```julia
result = solve_rfsp(A, b, x0, weights=nothing)   # Fischer RFSP, damped normal equations
result = solve_louhi(A, b, x0, smoothness=1.0, smooth_order=1)
result = solve_louhi(A, b, x0, smooth_order=2, auto_smooth=true)  # selects lambda
```

`x0` is the a-priori spectrum. LOUHI (Routti & Sandberg 1980, CPC
doi:10.1016/0010-4655(80)90021-4) solves the Hildreth coordinate QP for the
generalized-smoothing weighted least-squares functional, not a maximum-entropy
problem despite the classical pedigree.

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
| pgd/coordinate_descent | Proximal gradient | L1/L2, set constraint | Medium | High |
| admm/frank_wolfe  | Convex, constrained | L1 + TV / simplex | Medium    | High   |
| lbfgsb            | Quasi-Newton, bounded | Smoothness     | Fast        | High   |
| subgradient/extragradient | First-order | L1/TV, discrepancy | Slow     | Medium |
| rfsp/louhi        | Classical weighted LS | Smoothing / none | Medium   | Medium |
| eki/ensemble | Monte-Carlo ensemble    | Prior          | Slow        | Medium |

The exact set of keyword arguments and published tunings can be verified by
running `examples/33-methods_comparison.jl` with `benchmark_unfold_methods`,
which ranks all methods on IAEA reference spectra by 52 metrics.

## Optional JuMP backend (bucket-C unlock)

Six methods reach LP/QP/MIP engines that are not built into Julia Base —
`docplex` (CPLEX), `scip` (pyscipopt), `commercial` (Gurobi/MOSEK/CPLEX/COPT/
Xpress), `interval` (Shary LP), `nnqp` and `qpmad`. In the base install they
degrade gracefully (warn + zeros). Install JuMP + HiGHS in a separate
environment to activate them:

```julia
# from the repository root
using Pkg
Pkg.activate("env/jump")
Pkg.instantiate()
Pkg.develop(path=".")
```

Then, from the same environment, `BSSUnfold.has_jump()` returns `true` and
`solve_docplex`/`solve_scip`/`solve_commercial`/`solve_interval`/`solve_nnqp`/
`solve_qpmad` build the canonical QP `min ½‖Ax−b‖² + α‖Lx‖² (or α‖x‖² or αΣx)
s.t. 0 ≤ x ≤ ub` on JuMP and dispatch to HiGHS (or Clarabel, or the installed
commercial engine). Set `BSSUNFOLD_JL_BACKEND=0` to opt out and force the
graceful-degradation path (useful when juliacall must not pay precompile
cost). Parity gates by engine:

| engine         | method                 | typical pointwise relL2 | gate                          |
|----------------|------------------------|--------------------------|-------------------------------|
| `:highs`       | active-set QP          | 1e-9 … 1e-11             | cos>0.99995, obj rel<1e-8     |
| `:clarabel`    | homogeneous IPM        | 1e-7 … 1e-9              | obj rel<1e-6                  |
| `:cosmo`       | ADMM + acceleration    | 1e-4 … 1e-6              | obj rel<1e-6                  |
| `:osqp`        | ADMM (ECOS)            | 1e-4 … 1e-6              | obj rel<1e-6                  |
| `:scs`         | IPM                    | ~1e-3                    | obj rel<1e-4                  |
| LP (`interval`)| HiGHS simplex          | 1e-8 … 1e-10             | pointwise rel<1e-8            |

Native active-set ports (`solve_nnls`, `solve_admm` x-update, etc.) keep the
tighter 1e-13…1e-15 parity; the bucket-C methods are additive and never
replace them.

## Native-portable additions (bucket-D)

Five algorithms from the Python package have no external dependency — pure
linear algebra, first-order iteration or a self-contained sampler — so they
ship native and run in the base environment without JuMP.

```julia
result = solve_lavrentiev(A, b, x0; alpha=1e-3, form="gram")        # (AAᵀ+αI)y=b, z=Aᵀy
result = solve_lavrentiev(A, b, x0; alpha=1e-3, form="iterated", n_iterations=10, q=0.5)
result = solve_mirror_descent(A, b, x0; mirror_map="entropy", total_fluence=F, max_iterations=500)
result = solve_mirror_descent(A, b, x0; mirror_map="l2", step_size=1e-2, line_search=false)
result = solve_osem_anlm(A, b, x0; n_subsets=4, n_iterations=30, anlm_mode="each", alpha=1.0)
result = solve_tikhonov_sobolev_dp(A, b; noise_level=0.02, penalty=:sobolev, method=:brent)
result = solve_bayesian_parametric(E_MeV, readings; n_samples=4000, burn_in=1000, random_state=0)
```

- `solve_lavrentiev` exposes four forms (`"gram"`, `"padded"`, `"iterated"`,
  `"direct"`). `"direct"` rejects rectangular `A`; `"iterated"` is
  Bakushinskiy's scheme `y_{k+1} = y_k + (B + α·qᵏ I)⁻¹(b − B y_k)` with
  `q ∈ (0, 1]`.
- `solve_mirror_descent` uses Bregman proximal steps with `entropy`,
  `log`, `l2` or `pnorm` (p>1) mirrors. `entropy`/`log` require `total_fluence`
  `F > 0` and return a simplex point (`Σx = F`); per-iteration step length
  uses golden-section line search unless `step_size` is fixed.
- `solve_osem_anlm` combines OSEM subset splitting (`np.array_split` semantics:
  first `rem` subsets get one extra index) with an Adaptive Non-Local Means
  filter (`anlm_filter_1d`) and Lavrentiev noise estimate (`estimate_noise_1d`,
  Immerkaer MAD on second differences). `anlm_mode` selects `"each"`,
  `"post"` or `"none"`. Two-stage filter: `h1 = σ/2`, `h2 = σ·‖w₁(i,·)‖₂`.
- `solve_tikhonov_sobolev_dp` is the generalized discrepancy principle:
  `α*` is the root of `ρ(α) = ‖A z(α) − b‖² − δ²` bracketed on
  `[alpha_range]` and refined by Brent (bisection on log10 α) or
  Newton–Kantorovich using `dρ/dt = dρ/dα · ln(10) · α`. Penalty is
  `:sobolev` (order-1), `:curvature` (order-2) or `:identity`. Status codes
  0/1/2 mirror Python (`1` = least-regularised overfits, `2` = maximal
  regularisation underfits). The raw `z` is returned without clipping.
- `solve_bayesian_parametric` runs a single-pass Metropolis–Hastings sampler
  over the five-parameter Maxwellian + 1/E + evaporation form
  (`parametric_model_fp`) with uniform priors. `random_state` seeds a
  `MersenneTwister`, so Julia-vs-Julia reproduction holds; the Python
  reference uses `default_rng` (PCG64) and is not bit-exact. `result.extra`
  carries `mean_params`, `sigma`, `n_samples`, `burn_in`, `proposal_scale`,
  `accepted`.

Helper exports: `anlm_filter_1d`, `estimate_noise_1d`, `parametric_model_fp`.
