"""
CPLEX/docplex-style unfolding via the optional JuMP backend.

Port of `bssunfold/core/unfold_docplex.py`. The CPLEX-specific model layer
is replaced by the shared `_bucket_c_qp` kernel — a faithful match, because
docplex here never uses integer/SOS/callback features; the QP is continuous.
"""

"""
    solve_docplex(A, b, x0; regularization, norm, timeout, smoothness_order,
                 smoothness_weight, nonneg, random_state, ub, solver)

Core QP solver (see `_bucket_c_qp`). `solver` selects the JuMP engine
(:default → HiGHS; see JUMP_ENGINES).
"""
function solve_docplex(A::AbstractMatrix{T}, b::AbstractVector{T},
                       x0::Union{Nothing,AbstractVector{T}}=nothing;
                       regularization::T=T(1e-4),
                       norm::Integer=2,
                       timeout::Real=10.0,
                       smoothness_order::Integer=0,
                       smoothness_weight::T=T(1.0),
                       nonneg::Bool=true,
                       random_state::Union{Integer,Nothing}=nothing,
                       ub::Union{Nothing,AbstractVector}=nothing,
                       solver::Symbol=:default) where T<:AbstractFloat
    r = _bucket_c_qp(A, b, x0;
                     regularization=regularization, norm=norm, timeout=timeout,
                     smoothness_order=smoothness_order,
                     smoothness_weight=smoothness_weight, nonneg=nonneg,
                     ub=ub, engine=solver)
    random_state !== nothing && (r.extra["random_state"] = Int(random_state))
    r.extra["method"] = "docplex"
    return r
end

"""
    unfold_docplex(d::Detector, readings; kwargs...)

Detector-level wrapper (mirrors `unfold_docplex` in Python).
"""
function unfold_docplex(d::Detector, readings::Dict{String,T}; kwargs...) where T<:AbstractFloat
    _bucket_c_unfold(:solve_docplex, "docplex", d, readings; kwargs...)
end
