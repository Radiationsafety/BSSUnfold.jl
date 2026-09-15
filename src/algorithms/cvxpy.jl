"""
Convex-optimization-based unfolding (a full port of unfold_cvxpy.py).

Uses `Convex.jl` to solve the problem:

    min  ||A*x - b||₂ + α * ||x||_p    subject to  x ≥ 0, x ≤ ub

where `p` is the norm order (1 for L1, 2 for L2).

The Python original used `cvxpy` with the ECOS/SCS/CLARABEL solvers.
The Julia port uses `Convex.jl` plus one of the available solvers:
SCS.jl, ECOS.jl, Clarabel.jl, COSMO.jl.
"""

# Convex and SCS are declared package dependencies (see Project.toml),
# so we use a static import: this is more reliable than runtime loading
# (@eval / Base.require), which causes world-age problems on Julia 1.12+.
import Convex
import SCS

"""
    solve_cvxpy(A, b, x0; regularization, norm, solver, ub)

Solve the unfolding problem via convex optimization:

    min  ||A*x - b||₂ + α * ||x||_p    subject to  x ≥ 0, x ≤ ub

# Arguments
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: measurements (m,)
- `x0::AbstractVector{T}`: unused (API)
- `regularization::T`: parameter α (default 1e-4)
- `norm::Integer`: 1 for L1, 2 for L2 (default 2)
- `solver::Symbol`: `:SCS`, `:ECOS`, `:Clarabel`, `:COSMO`, `:default` (default `:default`)
- `ub::Union{Nothing,Vector{T}}`: upper bounds (default `nothing`)

# Returns
- `UnfoldResult{T}` with the spectrum
"""
function solve_cvxpy(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   regularization::T=T(1e-4),
                   norm::Integer=2,
                   solver::Symbol=:default,
                   ub::Union{Nothing,AbstractVector{T}}=nothing) where T<:AbstractFloat
    # Validate norm early
    if norm != 1 && norm != 2
        throw(ArgumentError("Unsupported norm: $norm. Use 1 or 2."))
    end

    n = size(A, 2)

    # Decision variable: x >= 0
    x_var = Convex.Variable(n, Convex.Positive())

    # Objective: minimize ||Ax - b||₂ + α * ||x||_p
    residual = A * x_var - b
    if norm == 1
        objective = Convex.norm2(residual) + regularization * Convex.norm_1(x_var)
    else  # norm == 2 (validated early)
        objective = Convex.norm2(residual) + regularization * Convex.norm2(x_var)
    end

    # Constraints: x <= ub (if provided)
    constraints = Convex.Constraint[]
    if ub !== nothing
        finite_mask = isfinite.(ub)
        if any(finite_mask)
            push!(constraints, x_var[finite_mask] <= ub[finite_mask])
        end
    end

    problem = Convex.minimize(objective, constraints)

    # Solver selection: SCS is a declared dependency; the others are optional.
    chosen_solver = if solver == :default
        :SCS
    else
        solver
    end

    solver_mod = if chosen_solver == :SCS
        SCS
    else
        try
            Base.require(@__MODULE__, chosen_solver)
        catch
            @warn "Solver $chosen_solver unavailable, using SCS"
            chosen_solver = :SCS
            SCS
        end
    end

    # Solve via Convex.solve!
    try
        Convex.solve!(problem, solver_mod.Optimizer)
    catch err
        @warn "Solver $chosen_solver failed: $err"
        return UnfoldResult(zeros(T, n), 0, false, T(0),
                           Dict{String,Any}("error" => string(err)))
    end

    if problem.status != Convex.MOI.OPTIMAL && problem.status != Convex.MOI.ALMOST_OPTIMAL
        @warn "CVXPY problem status: $(problem.status). Returning zero spectrum."
        return UnfoldResult(zeros(T, n), 0, false, T(0),
                           Dict{String,Any}("status" => string(problem.status)))
    end

    x = Vector{T}(Convex.evaluate(x_var))
    if ub !== nothing
        x[ub .== T(0)] .= T(0)
    end

    residual_vec = b .- A * x
    return UnfoldResult(
        max.(x, T(0)), 1, true, sqrt(sum(abs2, residual_vec)),
        Dict{String,Any}(
            "norm" => norm,
            "solver" => String(chosen_solver),
            "regularization" => regularization,
        )
    )
end
