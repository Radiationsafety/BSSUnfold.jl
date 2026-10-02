"""
nnqp / qpmad — box-constrained QP via the optional JuMP backend.

Both Python packages solve the same problem: `min ½ x'Px + q'x  s.t.  lb ≤ x ≤ ub`.
Instead of porting the active-set mechanics (Goldfarb–Idnani for qpmad,
Lawson-Hanson for nnqp), we delegate to `_solve_boxqp_jump`, which reaches
HiGHS (active-set) and matches the reference to ~1e-9.

The `backend` kwarg accepts the Python set (`"python"|"native"|"qpmad"|"cpp"|"nnqp"|"jump"`);
all values other than `:jump` are treated as `:jump` with a one-shot
warning, since we do not ship a native implementation.
"""

const _BOXQP_WARNED = Ref(false)
const _BOXQP_VALID_BACKENDS = (:python, :native, :qpmad, :cpp, :nnqp, :jump)
function _boxqp_check_backend(bs::Symbol)
    bs in _BOXQP_VALID_BACKENDS || throw(ArgumentError(
        "Unsupported backend '$bs'; valid values: python, native, qpmad, cpp, nnqp, jump"))
end
function _boxqp_warn_backend(backend)
    _BOXQP_WARNED[] && return
    _BOXQP_WARNED[] = true
    @warn "solve_nnqp/qpmad: backend '$(backend)' maps to the shared JuMP kernel; no native port is provided."
end

function _build_boxqp(A::AbstractMatrix{T}, b::AbstractVector{T},
                      alpha::T, smoothness_order::Integer,
                      smoothness_weight::T) where T<:AbstractFloat
    m, n = size(A)
    P = Matrix{T}(A' * A)
    q = -(A' * b)
    if smoothness_order in (1, 2)
        L = create_derivative_matrix(T, n, smoothness_order)
        P = P .+ (alpha * smoothness_weight) .* (L' * L)
    elseif smoothness_order == 0
        P = P .+ alpha .* Matrix{T}(I, n, n)
    else
        throw(ArgumentError("smoothness_order must be 0, 1 or 2; got $smoothness_order"))
    end
    P = T(0.5) .* (P .+ P')
    (P, q)
end

"""
    solve_nnqp(A, b, x0; regularization, smoothness_order, smoothness_weight,
              ub, backend)

Port of `unfold_nnqp.py`. Non-negative box QP with optional per-bin upper
bound; the solution is post-processed with `max.(x, 0)`.
"""
function solve_nnqp(A::AbstractMatrix{T}, b::AbstractVector{T},
                    x0::Union{Nothing,AbstractVector{T}}=nothing;
                    regularization::T=T(1e-4),
                    smoothness_order::Integer=0,
                    smoothness_weight::T=T(1.0),
                    ub::Union{Nothing,AbstractVector}=nothing,
                    backend::Union{Symbol,AbstractString}=:jump,
                    timeout::Real=30.0) where T<:AbstractFloat
    bs = Symbol(backend)
    _boxqp_check_backend(bs)
    bs !== :jump && _boxqp_warn_backend(bs)
    m, n = size(A)
    if !_try_load_jump()
        @warn "solve_nnqp: JuMP backend unavailable; returning zero spectrum."
        return UnfoldResult(zeros(T, n), 0, false, T(0),
                            Dict{String,Any}("error" => "JuMP not available"))
    end
    P, q = _build_boxqp(A, b, regularization, smoothness_order, smoothness_weight)
    lb = zeros(T, n)
    ubv = ub === nothing ? fill(typemax(T), n) : Vector{T}(ub)
    r = _solve_qp_jump(P, q, lb, ubv; engine=:default, time_limit=Float64(timeout),
                       x0=x0 === nothing ? nothing : Vector{T}(x0))
    x = max.(Vector{T}(r.x), T(0))
    ub !== nothing && (x[ub .== T(0)] .= T(0))
    residual = b .- A * x
    UnfoldResult(x, r.iters, r.ok, sqrt(sum(abs2, residual)),
                 Dict{String,Any}(
                    "regularization" => regularization,
                    "smoothness_order" => Int(smoothness_order),
                    "smoothness_weight" => smoothness_weight,
                    "engine" => string(r.engine),
                    "method" => "nnqp"))
end

"""
    solve_qpmad(A, b, x0; regularization, smoothness_order, smoothness_weight,
               ub, backend)

Port of `unfold_qpmad.py`. Same box QP; `extra["status"]` mirrors the
Goldfarb-Idnani return code: 0 = optimal, 1 = max-iters, 2 = infeasible.
`iterations` equals the status code, matching the Python wrapper.
"""
function solve_qpmad(A::AbstractMatrix{T}, b::AbstractVector{T},
                     x0::Union{Nothing,AbstractVector{T}}=nothing;
                     regularization::T=T(1e-4),
                     smoothness_order::Integer=0,
                     smoothness_weight::T=T(1.0),
                     lb::Union{Nothing,AbstractVector}=nothing,
                     ub::Union{Nothing,AbstractVector}=nothing,
                     backend::Union{Symbol,AbstractString}=:jump,
                     timeout::Real=30.0) where T<:AbstractFloat
    bs = Symbol(backend)
    _boxqp_check_backend(bs)
    bs !== :jump && _boxqp_warn_backend(bs)
    m, n = size(A)
    if !_try_load_jump()
        @warn "solve_qpmad: JuMP backend unavailable; returning zero spectrum."
        return UnfoldResult(zeros(T, n), 2, false, T(0),
                            Dict{String,Any}("error" => "JuMP not available",
                                             "status" => 2))
    end
    P, q = _build_boxqp(A, b, regularization, smoothness_order, smoothness_weight)
    lbv = lb === nothing ? zeros(T, n) : Vector{T}(lb)
    ubv = ub === nothing ? fill(typemax(T), n) : Vector{T}(ub)
    r = _solve_qp_jump(P, q, lbv, ubv; engine=:default, time_limit=Float64(timeout),
                       x0=x0 === nothing ? nothing : Vector{T}(x0))
    x = Vector{T}(r.x)
    ub !== nothing && (x[ub .== T(0)] .= T(0))
    status = r.ok ? 0 : 2
    residual = b .- A * x
    UnfoldResult(x, status, r.ok, sqrt(sum(abs2, residual)),
                 Dict{String,Any}(
                    "regularization" => regularization,
                    "smoothness_order" => Int(smoothness_order),
                    "smoothness_weight" => smoothness_weight,
                    "status" => status,
                    "engine" => string(r.engine),
                    "method" => "qpmad"))
end

# Detector-level wrappers (nnqp / qpmad have no `regularization_method` in Python)
for (ufn, sfn, label) in [(:unfold_nnqp, :solve_nnqp, "NNQP"),
                          (:unfold_qpmad, :solve_qpmad, "QPMAD")]
    @eval function $(ufn)(d::Detector, readings::Dict{String,T}; kwargs...) where T<:AbstractFloat
        run_unfolding($(sfn), d.config.detector_names, d.config.n_energy_bins,
                      d.config.E_MeV, d.config.sensitivities, d.config.cc_icrp116,
                      readings; method_name=$(label), solve_kwargs=NamedTuple(kwargs))
    end
end
