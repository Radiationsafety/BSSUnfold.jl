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

### BSREM — Block-Sequential Regularized Expectation Maximization

Relaxed additive update over ordered subsets of the detector readings
(port of the Python `solve_bsrem`):

```math
x \leftarrow x + x \cdot \frac{\alpha}{\omega \sum_i A_{ij}}
        \left( A_S^T \frac{b_S}{A_S x} - \sum_i A_{S,ij} - \omega \nabla V(x) \right)
```

with a floor clamp (`addition_after_iteration`, default 1e-4),
relaxation `α(n)` (constant 1 by default), `ω = |S|/m` and optional
nearest-neighbour priors `V` (`quadratic`, `logcosh`,
`relative_difference`; `prior="none"` by default).

```julia
result = solve_bsrem(A, b, x0, max_iterations=50, n_subsets=1, prior="none")
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

Applies CG to the normal equations without forming them explicitly;
the solution is regularized by early stopping (normal-equation residual
`‖Aᵀ(b − Ax)‖ ≤ tolerance·‖Aᵀb‖`, or the discrepancy principle when
`noise_level` is given).  Nonnegativity is enforced by a single clamping
pass after the iteration.

```julia
result = solve_cgls(A, b, x0, max_iterations=100, tolerance=1e-12)
```

### FISTA — Fast Iterative Shrinkage-Thresholding

Accelerated proximal gradient (IRtools IRfista-style) for
`0.5‖Ax−b‖² + 0.5·regularization‖x‖² + l1‖x‖₁ + tv‖Dx‖₁` with
nonnegativity projection and optional box constraints; step `1/L` with
`L = ‖A‖₂²`:

```julia
result = solve_fista(A, b, x0, max_iterations=500, tolerance=1e-8,
                     regularization=0.0, l1_penalty=0.0, tv_penalty=0.0)
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

Logarithmic SAND-II update with per-detector weights
(`x_new_j = x_j · exp(Σᵢ Wᵢⱼ log Rᵢ / Σᵢ Wᵢⱼ)`); stops when the χ² of
the fit is not greater than the number of detectors
(`chi_fac=1`, `relative_uncertainty=0.1`).

```julia
result = solve_sandii(A, b, x0, max_iterations=50)
```

### Bunki

SPUNIT iteration (the BUNKI core): the working spectrum `spl` starts at
ones over the response matrix scaled by the initial guess
(`aleth = A · x0`), is multiplicatively corrected with the `ss`
normalization and 3-point smoothed, and the final spectrum is
`x = spl · x0`:

```math
spll_j = spl_j \cdot \frac{\sum_i aleth_{ij} / bcc_i}{\sum_i aleth_{ij} / b_i},
\qquad bcc = aleth \cdot spl
```

Variants: `solve_bunki` (SPUNIT, port of the Python `solve_bunki`),
`solve_bunkiut` (uncertainty-transformed weights), `solve_rebunki`
(with reconstruction monitoring).

```julia
result = solve_bunki(A, b, x0, max_iterations=1000, smoothing=0.1)
```

### Staysl (1982)

Single-step Bayesian linear update of a prior spectrum with full
measurement/prior covariance information (`Cb = diag((rel·b)²)`,
`Cx = diag((prior·x0)²)`, `gain = Cx Aᵀ (Cb + A Cx Aᵀ)⁻¹`):

```julia
result = solve_staysl(A, b, x0, relative_uncertainty=0.1, prior_uncertainty=1.0)
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

## Constraint programming (SeaPearl.jl)

`solve_seapearl` — CP formulation of the unfolding problem solved with
[SeaPearl.jl](https://github.com/corail-research/SeaPearl.jl) (optional,
lazy-loaded like Turing.jl for `solve_mcmc`):

- every bin flux is quantized on `n_levels` levels;
- every sphere reading yields two `kσ`-compatibility constraints
  `b_i - k·σ_i ≤ Σ_j A_ij φ_j ≤ b_i + k·σ_i` (integer linear sums);
- an adjacent-bin smoothness constraint bounds quantized flux jumps;
- the solver **enumerates the whole feasible set** (up to `max_solutions`)
  instead of returning a single estimate.

The result carries what no other method provides — per-bin **interval
estimates** (`"spectrum_lower"` / `"spectrum_upper"`: min/max over the
feasible set) that directly characterize the null-space uncertainty, plus the
minimum-χ² member of the set. The default `BasicHeuristic` (lowest level
first) works without any trained model; a pre-trained SeaPearl
`LearnedHeuristic` can be injected via `learned_heuristic` (e.g. trained with
[learning-generic-csp](https://github.com/corail-research/learning-generic-csp))
to obtain a full CP + RL pipeline. If no feasible spectrum exists, the noise
level (`relax_step`) and then the flux ceiling (`expand_factor`) are relaxed
automatically before falling back to a flux-matched spectrum.

```julia
res = solve_seapearl(A, b, x0; n_levels=16, k_sigma=2.0, max_solutions=256)
res.extra["spectrum_lower"]   # per-bin lower bound of the feasible set
res.extra["spectrum_upper"]   # per-bin upper bound
res.extra["chi2_best"]        # χ² of the returned spectrum
```

SeaPearl 0.4.x supports Julia 1.8–1.9: install it on Julia 1.9 with
`Pkg.add(name="SeaPearl", version="0.4.5")` (or use a compat-patched fork on
newer Julia). Without SeaPearl the solver degrades gracefully (zero spectrum
+ warning), and `seapearl_available()` reports availability.

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

## Dev-branch methods (v0.5.0)

Six additional solvers live on the `develop` branch first and are integrated
into the main API from v0.5.0:

### RFSP-JUL (`solve_rfsp_jul`)

Independent reimplementation of the RFSP-JUL algorithm from its published
description (Fischer; the 1981 unfolding-codes review): iterative,
Marquardt-style damped least squares with relative-change damping
(`phi - phi_prev` penalty), solved in closed form from symmetric
positive-definite normal equations at each iteration.

```julia
result = solve_rfsp_jul(A, b, x0; max_iterations=200, tolerance=1e-4)
```

### Preconditioned Krylov (`solve_amg`)

Analogue of `Rlinsolve`/pyamg-style preconditioning: the (optionally
damped) normal equations are solved with PCG, preconditioned BiCGSTAB or
GMRES and Jacobi / Gauss-Seidel / SOR / SSOR preconditioners; non-negativity
is enforced by projected outer restarts.

```julia
result = solve_amg(A, b, nothing; method="cg", preconditioner="jacobi",
                   tolerance=1e-10, nonnegativity=true)
```

### Uno NLP presets (`solve_uno`, `solve_uno_full`)

Analogue of the R package `Uno` 2.x: Lagrange-Newton solution of the
non-negativity-constrained unfolding NLP with two presets — `"filter_sqp"`
(exact Hessian SQP with the Fletcher-Leyffer filter globalisation) and
`"ipopt_like"` (primal-dual interior point with a geometric barrier
schedule and optional BFGS Hessian).

```julia
diag = solve_uno_full(A, b, x0; preset="ipopt_like", hessian="bfgs")
result = solve_uno(A, b, x0)          # UnfoldResult wrapper
```

### SSR sign-parsimony (`solve_ssr`, `solve_ssr_full`)

Port of the R package `sisireg` 1.2.1 (Metzner): the spectrum is the most
parsimonious regression function whose folded residual signs pass the
partial-sum and maximum-run adequacy criteria; one MLEM update alternates
with a quantised Gauss-Seidel (QSOR) sweep, and the sign threshold follows
Metzner's minimum statistic (`fn="auto"` ladder).

```julia
diag = solve_ssr_full(A, b; x0=x0, E_MeV=E, fn="auto")
result = solve_ssr(A, b, x0)
```

Helpers: `ssr`, `ssr_ne`, `ssr_min_statistic`, `ssr_min_statistic_ne`,
`ssr_predict`, `max_run_quantile`, `partial_sum_quantile`,
`number_of_extrema`, `partial_sum_valid`, `run_valid`.

### MLEM-BS (`solve_mlem_bs`, `solve_mlem_bs_full`)

B-spline MLEM with sieve regularisation (Mazankova et al., CNDGS'2026): the
spectrum is a non-negative B-spline combination (`RB = A * B`), the
second-derivative penalty enters the MLEM denominator and `N_s`, `beta` and
the iteration count can be selected automatically by minimising the
goodness-of-fit statistic K_S (paper Eq. 6).

```julia
result = solve_mlem_bs(A, b, x0; n_basis=10, beta_relative=1e-3)
diag   = solve_mlem_bs_full(A, b, x0; auto_params=true)
```

Helpers: `build_bspline_basis`, `second_difference_matrix`, `ks_statistic`.

### P-spline REML (`solve_pspline_reml`, `solve_pspline_reml_full`)

Analogue of the R package `LMMsolver` (Boer 2023): P-spline representation
with the smoothness selected by restricted maximum likelihood in the
mixed-model reparameterisation (Wand & Ormerod 2008); the profile is
optimised by golden-section search on a scale-free relative-lambda axis.

```julia
result = solve_pspline_reml(A, b; x0=x0, n_basis=10, spline_order=3)
diag   = solve_pspline_reml_full(A, b; x0=x0)   # lam, ed, reml_loglik, ...
```

## Summary table

| Algorithm    | Type                    | Regularization | Speed       | Accuracy |
|--------------|-------------------------|----------------|-------------|----------|
| MLEM         | EM iterative            | None           | Slow        | High   |
| OSEM         | EM subset               | None           | Fast        | Medium |
| BSREM        | Relaxed subset update   | Optional prior | Fast        | Medium |
| SART         | Row-action              | None           | Fast        | Medium |
| GRAVEL       | Weighted                | Weak           | Medium      | High   |
| MAXED/AMAXED | Maximum entropy         | Built-in       | Medium      | Medium |
| Tikhonov     | Direct                  | Strong         | Very fast   | Low(negatives possible) |
| tsvd         | Direct                  | Strong         | Very fast   | Low    |
| Landweber    | Iterative               | None           | Medium      | Medium |
| Kaczmarz     | Row-action              | None           | Fast        | Medium |
| CGLS         | CG (normal eqs)         | Early stopping | Fast        | High   |
| FISTA        | Accelerated proximal    | L1/L2/TV       | Fast        | Medium |
| Lanczos      | Krylov                  | Implicit       | Fast        | High   |
| Sandii       | Logarithmic SAND-II     | None           | Medium      | Medium |
| Bunki        | SPUNIT (BUNKI core)     | Smoothing      | Medium      | High   |
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
| rfsp_jul     | Damped least squares    | Marquardt damp | Very fast   | Medium |
| amg          | Preconditioned Krylov   | Tikhonov damp  | Fast        | Medium |
| uno          | SQP / interior point    | Roughness ridge| Fast        | High   |
| ssr          | Sign-parsimony (QSOR)   | Sign criteria  | Medium      | Medium |
| mlem_bs      | Spline MLEM             | Sieve + D2     | Fast        | High   |
| pspline_reml | Mixed-model P-spline    | REML-selected  | Fast        | High   |

The exact set of keyword arguments and published tunings can be verified by
running `examples/33-methods_comparison.jl` with `benchmark_unfold_methods`,
which ranks all methods on IAEA reference spectra by 52 metrics.
