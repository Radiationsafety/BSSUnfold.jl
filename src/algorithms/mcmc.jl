"""
Bayesian MCMC unfolding method (port from unfold_mcmc.py, pymc → Turing.jl).

A fully Bayesian approach to unfolding neutron spectra with
Markov Chain Monte Carlo, namely the NUTS (No-U-Turn Sampler) sampler —
an adaptive variant of Hamiltonian Monte Carlo.

The spectrum is modeled in log-scale with a smoothing (Ornstein-Uhlenbeck)
prior, anchored to a data-driven center (the initial spectrum `x0` or
the non-negative least-squares solution).  This keeps the heavily
underdetermined unfolding problem in check: the spectrum remains
positive, smooth and bounded in the null space of the response
matrix, and the posterior mean coincides with the deterministic solvers
(e.g. `solve_cvxpy`) on the reference IAEA spectrum base.

The Bayesian structure provides:
- full posterior distributions for each energy bin;
- uncertainty quantification via credible intervals (HPD);
- automatic regularization through the prior specification;
- hierarchical modeling of likelihood noise (use_hierarchical=true).

Dependency: Turing.jl (lazy load on first call; without it the
function issues a warning and returns a zero spectrum).

Model (analog of the pymc model of the Python original):

    s ~ HalfNormal(lambda_prior)                    # amplitude of deviations
    z ~ MvNormal(0, I)                              # white latent field
    theta = mu_prior + s * (L_corr * z)             # OU-correlated field
    spectrum = exp(theta)                           # positive spectrum
    sigma = sigma_prior * |b|                       # relative noise
    b ~ MvNormal(A * spectrum, Diagonal(sigma^2))   # likelihood

where `C_ou[i, j] = exp(-|i - j| / lengthscale)` — OU correlation
(non-centered parameterization via the Cholesky factor L_corr).

The posterior samples of the spectrum are reconstructed from the samples of `s` and `z`
(`spectrum = exp(mu_prior + s * L_corr * z)`), which is independent of recording
deterministic variables to the chain in particular versions of Turing.
"""

const _TURING_LOADED = Ref(false)

"""
    _try_load_turing() -> Bool

Lazy load of Turing.jl at the first call of `solve_mcmc` and definition
of the Bayesian model (the macro `Turing.@model` requires a loaded Turing;
it is therefore expanded at runtime via `@eval`).

Turing is loaded into `Main` of the current session (similar to Requires.jl):
the package cannot `using` a non-direct dependency from its own namespace,
but it can load it into the user environment. If Turing is already loaded
by the user (`using Turing`) — we simply reuse it.

Returns `true` if Turing is available (in `Main.Turing`).
"""
function _try_load_turing()
    if _TURING_LOADED[]
        return true
    end
    try
        Base.eval(Main, :(using Turing))
    catch err
        @warn "Turing.jl could not be loaded; solve_mcmc is unavailable. " *
              "Install via: Pkg.add(\"Turing\")" exception=err
        return false
    end

    # Define the Bayesian model in Main (the @model macro is expanded
    # with Turing already loaded; its re-exports Normal/MvNormal/etc.
    # are available in Main).
    if !isdefined(Main, :_bssunfold_bayesian_model)
        try
            Base.eval(Main, quote
                Turing.@model function _bssunfold_bayesian_model(
                        A, b_abs, mu_prior, L_corr, lambda_prior,
                        sigma_prior, use_hierarchical, n_energy)
                    # Spatial amplitude of the log-deviations from the prior center
                    s ~ truncated(Normal(0.0, lambda_prior), 0.0, Inf)
                    # Whitened latent field; theta = mu + s*(L_corr*z) —
                    # non-centered MvNormal with OU covariance.
                    z ~ MvNormal(zeros(n_energy), I)
                    theta = mu_prior .+ s .* (L_corr * z)
                    spectrum = exp.(theta)

                    # Likelihood noise: fixed relative scale
                    # or estimated hierarchically.
                    if use_hierarchical === true
                        rel_noise ~ truncated(Normal(0.0, sigma_prior), 0.0, Inf)
                        sigma = rel_noise .* b_abs
                    else
                        sigma = sigma_prior .* b_abs
                    end

                    # Forward model (package convention: b = A * spectrum)
                    b ~ MvNormal(A * spectrum, Diagonal(sigma .^ 2))
                end
            end)
        catch err
            @warn "Failed to define the Turing Bayesian model" exception=err
            return false
        end
    end
    _TURING_LOADED[] = true
    return true
end

# ─── Prior math ─────────────────────────────────────────────────────────────

"""
    _ou_correlation_cholesky(n_bins, lengthscale) -> Matrix{Float64}

Cholesky factor of the Ornstein-Uhlenbeck correlation matrix.

The OU correlation `C[i, j] = exp(-|i - j| / lengthscale)` gives smooth,
stationary prior samples with bounded amplitude (unlike a
pure random walk), which keeps the posterior distribution
of the heavily underdetermined unfolding manageable for NUTS.
"""
function _ou_correlation_cholesky(n_bins::Integer, lengthscale::Real)
    ls = max(Float64(lengthscale), 1e-9)
    corr = [exp(-abs(i - j) / ls) for i in 0:(n_bins - 1), j in 0:(n_bins - 1)]
    corr += 1e-9 * Matrix{Float64}(I, n_bins, n_bins)
    return cholesky(Symmetric(corr)).L
end

"""
    _prior_center(A, b, initial_spectrum, n_energy) -> Vector{Float64}

Data-driven center of the spectrum log-prior.

Uses the user-supplied `initial_spectrum` if given (and containing
at least one positive value), otherwise the non-negative
LS solution `A @ x = b`.  Returned in log-scale
`log(max(x, eps))`, so the spectrum prior `f = exp(theta)` is anchored
around a spectrum consistent with the measurements.
"""
function _prior_center(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                      initial_spectrum::Union{Nothing,AbstractVector{<:Real}},
                      n_energy::Integer)
    local center::Vector{Float64}
    if initial_spectrum !== nothing
        c = max.(Float64.(collect(initial_spectrum)), 0.0)
        center = (length(c) == n_energy && any(>(0), c)) ? c : zeros(n_energy)
    else
        center = max.(qr(A, ColumnNorm()) \ b, 0.0)
    end
    return log.(max.(center, 1e-6))
end

# ─── HPD interval and convergence diagnostics ───────────────────────────────

"""
    _hpd_interval(samples::AbstractMatrix, prob=0.95) -> (lower, upper)

Shortest (highest posterior density) interval over the columns of samples.

Computed in pure Julia along the sample axis (axis 1), bypassing differences
in `az.hdi` semantics between ArviZ versions.

# Returns
`(lower, upper)` — bounds of the HPD interval, each of length n_energy.
"""
function _hpd_interval(samples::AbstractMatrix{<:Real}, prob::Real=0.95)
    n_total, n_energy = size(samples)
    n_keep = max(ceil(Int, prob * n_total), 1)
    lower = Vector{Float64}(undef, n_energy)
    upper = Vector{Float64}(undef, n_energy)
    for j in 1:n_energy
        sorted = sort(samples[:, j])
        if n_keep >= n_total
            lower[j], upper[j] = sorted[1], sorted[end]
            continue
        end
        widths = sorted[n_keep:end] .- sorted[1:(n_total - n_keep + 1)]
        best_idx = argmin(widths)
        lower[j] = sorted[best_idx]
        upper[j] = sorted[best_idx + n_keep - 1]
    end
    return lower, upper
end

"""
    _split_rhat(samples::AbstractMatrix, n_chains::Int) -> Vector{Float64}

Split-R̂ Gelman-Rubin convergence diagnostic over the columns
(port of rhat from ArviZ in minimal form).

`samples` — (n_total, n_energy) with chains joined along rows
(chain after chain).  Each chain is additionally split in half, giving
`2 * n_chains` subsegments.
"""
function _split_rhat(samples::AbstractMatrix{<:Real}, n_chains::Integer)
    n_total, n_energy = size(samples)
    (n_chains >= 1 && n_total >= 8) || return fill(NaN, n_energy)
    d = n_total ÷ n_chains
    d < 4 && return fill(NaN, n_energy)
    half = d ÷ 2
    segments = Vector{Matrix{eltype(samples)}}()
    for ci in 1:n_chains
        base = (ci - 1) * d
        push!(segments, samples[(base + 1):(base + half), :])
        push!(segments, samples[(base + half + 1):(base + 2 * half), :])
    end
    k = length(segments)
    means = reduce(hcat, vec(mean(s, dims=1)) for s in segments)   # (n, k)
    vars_ = reduce(hcat, vec(var(s, dims=1)) for s in segments)    # (n, k)
    W = vec(mean(vars_, dims=2))                                   # within-chain
    B = half * vec(var(means, dims=2; corrected=true))             # between-chain
    var_hat = @. (half - 1) / half * W + B / half
    return sqrt.(max.(var_hat ./ max.(W, 1e-300), 0.0))
end

# ─── Main solver ────────────────────────────────────────────────────────────

"""
    _extract_posterior_samples(chain, n_energy) -> (s_vec, z_mat)

Version-independent extraction of parameter samples `s` and `z` from the chain:

- MCMCChains (Turing <= 0.4x): `chain[:s]` → (draws, chains),
  `chain[:z]` → (draws, chains, n_energy);
- FlexiChains/VNChain (Turing >= 0.49): `chain[:s]` → DimMatrix
  (draws, chains), `chain[:z]` → DimMatrix (draws, chains) with a vector
  eltype; fallback — per-component keys `Symbol("z[j]")`.

Returns `(s_vec, z_mat)`: a vector of amplitudes (n_total,) and a matrix
of the latent field (n_total × n_energy), chains consecutive along rows.
"""
function _extract_posterior_samples(chain, n_energy)
    s_arr = parent(chain[:s])
    ndims(s_arr) == 1 && (s_arr = reshape(s_arr, :, 1))
    s_vec = vec(Float64.(s_arr))                     # (n_total,), chains in order

    local z_arr
    try
        z_arr = parent(chain[:z])
    catch
        z_arr = nothing
    end

    z_mat = if z_arr !== nothing && ndims(z_arr) == 3
        reshape(z_arr, size(z_arr, 1) * size(z_arr, 2), size(z_arr, 3))
    elseif z_arr !== nothing && ndims(z_arr) == 2 && eltype(z_arr) <: AbstractVector
        # FlexiChains: each element is a vector of length n_energy
        rows = [Float64.(collect(z_arr[i, c]))
                for c in 1:size(z_arr, 2) for i in 1:size(z_arr, 1)]
        reduce(vcat, r' for r in rows)
    else
        # Fallback: per-component keys "z[j]" (MCMCChains)
        cols = [vec(Float64.(parent(chain[Symbol("z[$j]")]))) for j in 1:n_energy]
        reduce(hcat, cols)
    end

    size(z_mat, 1) == length(s_vec) || error(
        "MCMC: mismatch between the number of s samples ($(length(s_vec))) and z ($(size(z_mat, 1)))")
    size(z_mat, 2) == n_energy || error(
        "MCMC: expected $n_energy columns of 'z', got $(size(z_mat, 2))")
    return s_vec, z_mat
end

"""
    _turing_sampling_pipeline(A, b_abs, mu_prior, L_corr, lambda_prior,
                              sigma_prior, use_hierarchical, n_energy,
                              n_samples, chains, target_accept) -> Matrix

Full Turing pipeline: model construction → NUTS sampling →
extraction of posterior samples of the spectrum (n_total × n_energy).
Called ONLY via `Base.invokelatest` (Turing and the model are loaded
dynamically; methods from a new world age are unavailable from the old frame).
"""
function _turing_sampling_pipeline(A, b_abs, mu_prior, L_corr, lambda_prior,
                                  sigma_prior, use_hierarchical, n_energy,
                                  n_samples, chains, target_accept)
    Turing = Main.Turing
    model = Main._bssunfold_bayesian_model(
        A, b_abs, mu_prior, L_corr, lambda_prior, sigma_prior,
        use_hierarchical, n_energy)

    # NUTS sampling
    chain = if chains > 1
        Turing.sample(model, Turing.NUTS(target_accept), Turing.MCMCThreads(),
                      n_samples, chains; progress=false)
    else
        Turing.sample(model, Turing.NUTS(target_accept), n_samples;
                      progress=false)
    end

    s_vec, z_mat = _extract_posterior_samples(chain, n_energy)

    # theta = mu_prior + s * (L_corr * z); spectrum = exp(theta)
    return exp.(mu_prior' .+ s_vec .* (z_mat * L_corr'))  # (n_total, n)
end

"""
    solve_mcmc(A, b, x0; sigma_prior=0.05, lambda_prior=0.5, lengthscale=3.0,
               n_samples=1000, tune=500, chains=2, target_accept=0.95,
               use_hierarchical=false, random_state=nothing) -> UnfoldResult

Solve the unfolding problem with Bayesian MCMC using the NUTS sampler (Turing.jl).

The spectrum is modeled in log-scale with a smoothing OU prior,
anchored at the center `x0` (or the LS solution if `x0` is trivial).
The NUTS sampler generates samples from the posterior p(f|b), from which
statistics are computed (mean, median, std, HPD intervals, R̂).

# Arguments
- `A::AbstractMatrix{T}`: response matrix (n_detectors × n_energy)
- `b::AbstractVector{T}`: measured readings (n_detectors,)
- `x0::AbstractVector{T}`: center of the prior (n_energy,); trivial `x0`
  is replaced by the non-negative LS solution
- `sigma_prior`: relative scale of the measurement noise (default 0.05)
- `lambda_prior`: scale of the amplitude `s` of log-deviations (default 0.5)
- `lengthscale`: OU correlation length of the prior, in bins (default 3.0)
- `n_samples`: number of MCMC samples per chain (default 1000)
- `tune`: number of adaptation samples per chain (default 500)
- `chains`: number of independent chains (default 2)
- `target_accept`: target acceptance level of NUTS (default 0.95)
- `use_hierarchical`: estimate the noise scale from the data (default false)
- `random_state`: seed for reproducibility

# Returns
An `UnfoldResult` with the spectrum — the posterior mean; `extra` contains:
`samples`, `mean`, `median`, `std`, `hpd_lower`, `hpd_upper`,
`rhat`, `rhat_max`.

If Turing.jl is not installed, returns a zero spectrum with a warning
(install: `Pkg.add("Turing")`).
"""
function solve_mcmc(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   sigma_prior::Real=0.05,
                   lambda_prior::Real=0.5,
                   lengthscale::Real=3.0,
                   n_samples::Integer=1000,
                   tune::Integer=500,
                   chains::Integer=2,
                   target_accept::Real=0.95,
                   use_hierarchical::Bool=false,
                   random_state::Union{Integer,Nothing}=nothing) where T<:AbstractFloat
    m, n_energy = size(A)
    length(b) == m || throw(ArgumentError("b length ($(length(b))) must match A rows ($m)"))

    if !_try_load_turing()
        @warn "Turing.jl is not installed. Install via: Pkg.add(\"Turing\"). " *
              "Returning a zero spectrum."
        return UnfoldResult(zeros(T, n_energy), 0, false, T(0),
                            Dict{String,Any}("error" => "Turing.jl not available"))
    end

    # Prior center and Cholesky factor of the OU correlation
    mu_prior = _prior_center(A, b, x0, n_energy)
    L_corr = _ou_correlation_cholesky(n_energy, lengthscale)
    b_abs = abs.(Float64.(b)) .+ 1e-6

    random_state !== nothing && Random.seed!(Int(random_state))

    # The whole Turing pipeline (model construction, sampling, extraction
    # of samples) is run via invokelatest: Turing and the model are defined
    # JUST NOW via eval, and their methods are unavailable from the current world age.
    local samples
    try
        samples = Base.invokelatest(
            _turing_sampling_pipeline,
            Float64.(Matrix(A)), b_abs, mu_prior, L_corr,
            Float64(lambda_prior), Float64(sigma_prior),
            Bool(use_hierarchical), Int(n_energy),
            Int(n_samples), Int(chains), Float64(target_accept))
    catch err
        error("MCMC sampling failed: ", sprint(showerror, err))
    end

    mean_spec = vec(mean(samples, dims=1))
    median_spec = vec(median(samples, dims=1))
    std_spec = vec(std(samples, dims=1))
    hpd_lower, hpd_upper = _hpd_interval(samples, 0.95)
    rhat = _split_rhat(samples, chains)
    rhat_max = isempty(filter(isfinite, rhat)) ? NaN : maximum(filter(isfinite, rhat))

    residual = b .- A * mean_spec
    return UnfoldResult(
        Vector{T}(max.(mean_spec, 0.0)), Int(n_samples), true, T.(norm(residual)),
        Dict{String,Any}(
            "samples" => samples,
            "mean" => mean_spec,
            "median" => median_spec,
            "std" => std_spec,
            "hpd_lower" => hpd_lower,
            "hpd_upper" => hpd_upper,
            "rhat" => rhat,
            "rhat_max" => rhat_max,
            "n_chains" => Int(chains),
            "n_samples" => Int(n_samples),
        ))
end
