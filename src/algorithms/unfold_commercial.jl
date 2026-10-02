"""
Commercial-license QP solvers (Gurobi / MOSEK / CPLEX / COPT / XPRESS)
via the optional JuMP backend.

Port of `bssunfold/core/unfold_commercial.py` + `_commercial_qp.py`. Python
drives each engine through cvxpy's interface; here the same canonical QP
goes to `_bucket_c_qp` and reaches whichever engine the user has installed
via JuMP extensions (e.g. `Gurobi.jl`, `MosekTools.jl`, `Cplex.jl`).
Aliases: gurobi → :gurobi, mosek → :mosek, cplex → :cplex, copt → :copt,
xpress → :xpress. If the requested engine is not installed, warn-and-return
zeros (matching Python's `commercial_solver_info(available=false)` path).
"""

const COMMERCIAL_SOLVER_ALIASES = Dict{Symbol,Symbol}(
    :gurobi => :gurobi,
    :mosek  => :mosek,
    :cplex  => :cplex,
    :copt   => :copt,
    :xpress => :xpress,
)

const COMMERCIAL_SOLVER_JL_PACKAGES = Dict{Symbol,Symbol}(
    :gurobi => :Gurobi,
    :mosek  => :MosekTools,
    :cplex  => :Cplex,
    :copt   => :Copt,
    :xpress => :Xpress,
)

function solve_commercial(A::AbstractMatrix{T}, b::AbstractVector{T},
                          x0::Union{Nothing,AbstractVector{T}}=nothing;
                          regularization::T=T(1e-4),
                          norm::Integer=2,
                          solver::Symbol=:gurobi,
                          timeout::Real=10.0,
                          smoothness_order::Integer=0,
                          smoothness_weight::T=T(1.0),
                          nonneg::Bool=true,
                          random_state::Union{Integer,Nothing}=nothing,
                          ub::Union{Nothing,AbstractVector}=nothing) where T<:AbstractFloat
    haskey(COMMERCIAL_SOLVER_ALIASES, solver) || throw(ArgumentError(
        "Unknown commercial solver '$(solver)'. Supported: gurobi, mosek, cplex, copt, xpress"))
    r = _bucket_c_qp(A, b, x0;
                     regularization=regularization, norm=norm, timeout=timeout,
                     smoothness_order=smoothness_order,
                     smoothness_weight=smoothness_weight, nonneg=nonneg,
                     ub=ub, engine=solver)
    r.extra["solver"] = String(solver)
    r.extra["license_required"] = true
    random_state !== nothing && (r.extra["random_state"] = Int(random_state))
    return r
end

# Per-engine aliases (Python exposes solve_gurobi / solve_mosek / ... as
# partials of solve_commercial).
for (fn, eng) in [(:solve_gurobi, :gurobi), (:solve_mosek, :mosek),
                  (:solve_cplex, :cplex), (:solve_copt, :copt),
                  (:solve_xpress, :xpress)]
    @eval function $(fn)(A::AbstractMatrix{T}, b::AbstractVector{T},
                         x0::Union{Nothing,AbstractVector{T}}=nothing;
                         kwargs...) where T<:AbstractFloat
        solve_commercial(A, b, x0; solver=$(QuoteNode(eng)), kwargs...)
    end
end

function unfold_commercial(d::Detector, readings::Dict{String,T};
                           solver::Symbol=:gurobi, kwargs...) where T<:AbstractFloat
    _bucket_c_unfold(:solve_commercial, String(solver), d, readings;
                     solver=solver, kwargs...)
end

for (ufn, eng) in [(:unfold_gurobi, :gurobi), (:unfold_mosek, :mosek),
                   (:unfold_cplex, :cplex), (:unfold_copt, :copt),
                   (:unfold_xpress, :xpress)]
    @eval function $(ufn)(d::Detector, readings::Dict{String,T};
                          kwargs...) where T<:AbstractFloat
        _bucket_c_unfold(:solve_$(QuoteNode(eng)), String($(QuoteNode(eng))),
                         d, readings; kwargs...)
    end
end
