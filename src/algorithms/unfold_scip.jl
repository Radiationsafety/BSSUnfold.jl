"""
SCIP-based unfolding via the optional JuMP backend.

Port of `bssunfold/core/unfold_scip.py`. The Python path uses pyscipopt as
a general-purpose MIP/NLP engine, but the model it builds is a plain
continuous QP — the same one docplex and cvxpy/commercial describe. Here
the model is handed to `_bucket_c_qp` and `x0` is forwarded as a warm
start when the underlying engine supports it (HiGHS ignores it; Mosek
honours it).
"""

function solve_scip(A::AbstractMatrix{T}, b::AbstractVector{T},
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
    r.extra["method"] = "scip"
    r.extra["note"] = "SCIP-as-MIP not exercised: QP is continuous"
    return r
end

function unfold_scip(d::Detector, readings::Dict{String,T}; kwargs...) where T<:AbstractFloat
    _bucket_c_unfold(:solve_scip, "SCIP", d, readings; kwargs...)
end
