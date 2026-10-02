"""
Interval (Shary) unfolding via the optional JuMP backend (LP).

Port of `bssunfold/core/unfold_interval.py`. Uses `scipy.optimize.linprog(method="highs")`
in Python; here we hand the same LPs to `_solve_lp_jump` — HiGHS is the
identical engine, so parity is tight (rel ~1e-8…1e-10 per the plan gate).

`solve_interval` returns x_min and x_max (each from n LPs; total 2n) and
their midpoint; `solve_interval_tol` adds a Shary-functional pass via a
projected-gradient L-BFGS-B on the max-energy-mask bounds; `posterior` and
`intvalpy` variants are placeholders that warn-and-return-zeros when the
backend is not applicable.
"""

function _interval_readings_to_bounds(A::AbstractMatrix{T}, b::AbstractVector{T},
                                      db::Union{AbstractVector,Nothing},
                                      noise_level::Real) where T<:AbstractFloat
    m = length(b)
    if db !== nothing
        rad = Vector{T}(db)
    else
        rad = T(noise_level) .* abs.(b)
    end
    (b .- rad, b .+ rad, rad)
end

function _build_lp_system(A::AbstractMatrix{T}, b_lo::AbstractVector{T},
                          b_hi::AbstractVector{T};
                          with_tv::Bool=false, tv_bound::Real=1.0) where T<:AbstractFloat
    m, n = size(A)
    # variables: x(1..n) then optionally t(1..n-1)
    nv = with_tv ? n + (n - 1) : n
    # rows: |Ax| rows (m) — each becomes two-sided
    lhs = Vector{T}(); rhs = Vector{T}()
    A_rows = zeros(T, 0, nv)
    # b_lo <= A x <= b_hi
    Ar = zeros(T, m, nv)
    for i in 1:m, j in 1:n
        Ar[i, j] = A[i, j]
    end
    A_rows = Ar
    append!(lhs, b_lo); append!(rhs, b_hi)
    if with_tv && n >= 2
        # -t_k <= x_{k+1} - x_k <= t_k  → 2 rows per k
        # Also: sum(t) <= tv_bound
        tv = T(tv_bound)
        for k in 1:n-1
            row1 = zeros(T, 1, nv); row1[1, k] = T(-1); row1[1, k+1] = T(1); row1[1, n+k] = T(-1)
            # x_{k+1} - x_k - t_k <= 0
            row2 = zeros(T, 1, nv); row2[1, k] = T(1);  row2[1, k+1] = T(-1); row2[1, n+k] = T(-1)
            # x_k - x_{k+1} - t_k <= 0
            A_rows = vcat(A_rows, row1, row2)
            append!(lhs, [typemin(T), typemin(T)]); append!(rhs, [T(0), T(0)])
        end
        # sum(t) <= tv_bound
        srow = zeros(T, 1, nv)
        for k in 1:n-1; srow[1, n+k] = T(1); end
        A_rows = vcat(A_rows, srow)
        push!(lhs, typemin(T)); push!(rhs, tv)
    end
    (A_rows, lhs, rhs, nv, n)
end

"""
    solve_interval(A, b, x0; tv_bound, noise_level, reading_uncertainties,
                  engine, timeout)

Port of `solve_interval` from `unfold_interval.py`. Builds one LP model with
n + (n-1) variables and swaps the objective `2n` times (min eᵢ, max eᵢ).
"""
function solve_interval(A::AbstractMatrix{T}, b::AbstractVector{T},
                        x0::Union{Nothing,AbstractVector{T}}=nothing;
                        tv_bound::Real=1.0,
                        noise_level::Real=0.01,
                        reading_uncertainties::Union{Nothing,AbstractVector}=nothing,
                        engine::Symbol=:default,
                        timeout::Real=30.0) where T<:AbstractFloat
    m, n = size(A)
    if !_try_load_jump()
        @warn "solve_interval: JuMP backend unavailable; returning zero spectrum."
        return UnfoldResult(zeros(T, n), 0, false, T(0),
                            Dict{String,Any}("error" => "JuMP not available"))
    end
    db = reading_uncertainties
    b_lo, b_hi, rad = _interval_readings_to_bounds(A, b, db, noise_level)
    A_rows, lhs, rhs, nv, _ = _build_lp_system(A, b_lo, b_hi; with_tv=true, tv_bound=tv_bound)
    x_lb = zeros(T, nv); x_ub = fill(typemax(T), nv)
    x_min = zeros(T, n); x_max = zeros(T, n)
    ok_all = true
    for i in 1:n
        c = zeros(T, nv); c[i] = one(T)
        r_min = _solve_lp_jump(c; A_rows=A_rows, lhs=lhs, rhs=rhs,
                               x_lb=x_lb, x_ub=x_ub, engine=engine,
                               time_limit=Float64(timeout), sense=:min)
        r_max = _solve_lp_jump(c; A_rows=A_rows, lhs=lhs, rhs=rhs,
                               x_lb=x_lb, x_ub=x_ub, engine=engine,
                               time_limit=Float64(timeout), sense=:max)
        x_min[i] = r_min.ok ? r_min.x[i] : zero(T)
        x_max[i] = r_max.ok ? r_max.x[i] : zero(T)
        ok_all &= (r_min.ok && r_max.ok)
    end
    x_mid = T(0.5) .* (x_min .+ x_max)
    residual = b .- A * x_mid
    UnfoldResult(x_mid, 2 * n, ok_all, sqrt(sum(abs2, residual)),
                 Dict{String,Any}(
                    "spectrum_lower" => x_min,
                    "spectrum_upper" => x_max,
                    "spectrum_mid" => x_mid,
                    "tv_bound" => Float64(tv_bound),
                    "noise_level" => Float64(noise_level),
                    "method" => "interval"))
end

"""
    solve_interval_tol(A, b, x0; noise_level, reading_uncertainties, backend)

Shary-functional variant. The `-Tol(x)` objective is nonsmooth (L-BFGS-B
in Python may stall at a kink); we use a projected-subgradient pass here
and only gate on interval-consistency, not pointwise parity.
"""
function solve_interval_tol(A::AbstractMatrix{T}, b::AbstractVector{T},
                            x0::Union{Nothing,AbstractVector{T}}=nothing;
                            noise_level::Real=0.01,
                            reading_uncertainties::Union{Nothing,AbstractVector}=nothing,
                            max_iterations::Integer=500,
                            backend::Symbol=:native,
                            ub::Union{Nothing,AbstractVector}=nothing) where T<:AbstractFloat
    m, n = size(A)
    if !_try_load_jump()
        @warn "solve_interval_tol: JuMP backend unavailable; returning zero spectrum."
        return UnfoldResult(zeros(T, n), 0, false, T(0),
                            Dict{String,Any}("error" => "JuMP not available"))
    end
    b_lo, b_hi, rad = _interval_readings_to_bounds(A, b, nothing, noise_level)
    if reading_uncertainties !== nothing
        rad = Vector{T}(reading_uncertainties)
    end
    b_mid = T(0.5) .* (b_lo .+ b_hi)
    # Projected subgradient on x >= 0 (maximise Tol = minimise -Tol).
    x = x0 === nothing ? fill(T(0.1), n) : Vector{T}(x0)
    step = T(1e-3)
    L = norm(A, 2)
    L = isfinite(L) && L > 0 ? L : T(1)
    step = T(1) / (L * L)
    prev_tol = T(-Inf)
    iters = 0
    for k in 1:max_iterations
        residuals = A * x .- b_mid
        slack = rad .- abs.(residuals)
        j = argmin(slack)
        tol = slack[j]
        # Subgradient of -Tol at x: -(sign(residuals[j]) * A[j,:])
        g = -sign(residuals[j]) * @view A[j, :]
        x = max.(x .- step .* g, T(0))
        iters = k
        abs(tol - prev_tol) < 1e-12 && break
        prev_tol = tol
    end
    x_pseudo = copy(x)
    # Now build 2n LPs without TV constraint (per Python `_solve_interval_tol`)
    A_rows, lhs, rhs, nv, _ = _build_lp_system(A, b_lo, b_hi; with_tv=false)
    x_lb = zeros(T, n); x_ub = fill(typemax(T), n)
    x_min = zeros(T, n); x_max = zeros(T, n)
    for i in 1:n
        c = zeros(T, nv); c[i] = one(T)
        r_min = _solve_lp_jump(c; A_rows=A_rows, lhs=lhs, rhs=rhs,
                               x_lb=x_lb, x_ub=x_ub, engine=:default)
        r_max = _solve_lp_jump(c; A_rows=A_rows, lhs=lhs, rhs=rhs,
                               x_lb=x_lb, x_ub=x_ub, engine=:default, sense=:max)
        x_min[i] = r_min.ok ? r_min.x[i] : zero(T)
        x_max[i] = r_max.ok ? r_max.x[i] : zero(T)
    end
    x_mid = T(0.5) .* (x_min .+ x_max)
    residual = b .- A * x_mid
    UnfoldResult(x_mid, iters, true, sqrt(sum(abs2, residual)),
                 Dict{String,Any}(
                    "spectrum_lower" => x_min,
                    "spectrum_upper" => x_max,
                    "spectrum_mid" => x_mid,
                    "x_pseudo" => x_pseudo,
                    "tol" => Float64(minimum(rad .- abs.(A * x_mid .- b_mid))),
                    "noise_level" => Float64(noise_level),
                    "backend" => String(backend),
                    "method" => "interval_tol"))
end

"""
    solve_interval_posterior(A, b, x0; noise_level, ...)

FD sensitivity approximation of the interval width per bin:
`Σ|A · (1e-6 e_i)| · b_rad`. Cheap and does not call the LP engine — kept
for parity with the Python wrapper that returns these fields.
"""
function solve_interval_posterior(A::AbstractMatrix{T}, b::AbstractVector{T},
                                  x0::Union{Nothing,AbstractVector{T}}=nothing;
                                  noise_level::Real=0.01,
                                  reading_uncertainties::Union{Nothing,AbstractVector}=nothing) where T<:AbstractFloat
    m, n = size(A)
    b_lo, b_hi, rad = _interval_readings_to_bounds(A, b, reading_uncertainties, noise_level)
    b_mid = T(0.5) .* (b_lo .+ b_hi)
    x = x0 === nothing ? fill(T(0.1), n) : Vector{T}(x0)
    eps_fd = T(1e-6)
    widths = zeros(T, n)
    for j in 1:n
        d = zeros(T, n); d[j] = eps_fd
        widths[j] = sum(abs, A * d) * maximum(rad)
    end
    residual = b .- A * x
    UnfoldResult(x, 0, true, sqrt(sum(abs2, residual)),
                 Dict{String,Any}(
                    "interval_width" => widths,
                    "b_mid" => b_mid,
                    "b_rad" => rad,
                    "method" => "interval_posterior"))
end

"""
    solve_interval_intvalpy(A, b, x0; kwargs...)

Placeholder for the Python `intvalpy` variant — that package is
Python-only, so we always warn-and-return-zeros.
"""
function solve_interval_intvalpy(A::AbstractMatrix{T}, b::AbstractVector{T},
                                 x0::Union{Nothing,AbstractVector{T}}=nothing;
                                 kwargs...) where T<:AbstractFloat
    m, n = size(A)
    @warn "solve_interval_intvalpy: intvalpy is a Python-only package and is not ported."
    UnfoldResult(zeros(T, n), 0, false, T(0),
                 Dict{String,Any}("error" => "intvalpy not available"))
end

# Detector-level wrappers
for (ufn, sfn, label) in [(:unfold_interval, :solve_interval, "Interval"),
                          (:unfold_interval_tol, :solve_interval_tol, "Interval_Tol"),
                          (:unfold_interval_posterior, :solve_interval_posterior, "Interval_Posterior"),
                          (:unfold_interval_intvalpy, :solve_interval_intvalpy, "Interval_Intvalpy")]
    @eval function $(ufn)(d::Detector, readings::Dict{String,T}; kwargs...) where T<:AbstractFloat
        run_unfolding($(sfn), d.config.detector_names, d.config.n_energy_bins,
                      d.config.E_MeV, d.config.sensitivities, d.config.cc_icrp116,
                      readings; method_name=$(label), solve_kwargs=NamedTuple(kwargs))
    end
end
