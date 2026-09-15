"""
Convex-optimization-based unfolding (полный порт unfold_cvxpy.py).

Использует `Convex.jl` для решения задачи:

    min  ||A*x - b||₂ + α * ||x||_p    subject to  x ≥ 0, x ≤ ub

где `p` — порядок нормы (1 для L1, 2 для L2).

В Python-оригинале использовался `cvxpy` с солверами ECOS/SCS/CLARABEL.
В Julia-порте используется `Convex.jl` + один из доступных солверов:
SCS.jl, ECOS.jl, Clarabel.jl, COSMO.jl.
"""

# Convex и SCS — заявленные зависимости пакета (см. Project.toml),
# поэтому используем статический импорт: это надёжнее рантайм-загрузки
# (@eval / Base.require), которая порождает проблемы world age за Julia 1.12+.
import Convex
import SCS

"""
    solve_cvxpy(A, b, x0; regularization, norm, solver, ub)

Решить задачу развёртки через выпуклую оптимизацию:

    min  ||A*x - b||₂ + α * ||x||_p    subject to  x ≥ 0, x ≤ ub

# Аргументы
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: измерения (m,)
- `x0::AbstractVector{T}`: не используется (API)
- `regularization::T`: параметр α (default 1e-4)
- `norm::Integer`: 1 для L1, 2 для L2 (default 2)
- `solver::Symbol`: `:SCS`, `:ECOS`, `:Clarabel`, `:COSMO`, `:default` (default `:default`)
- `ub::Union{Nothing,Vector{T}}`: верхние границы (default `nothing`)

# Возвращает
- `UnfoldResult{T}` со спектром
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

    # Solver selection: SCS — заявленная зависимость; остальные — опциональны.
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
            @warn "Солвер $chosen_solver недоступен, использую SCS"
            chosen_solver = :SCS
            SCS
        end
    end

    # Решить через Convex.solve!
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
