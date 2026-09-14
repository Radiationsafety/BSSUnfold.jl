"""
Convex-optimization-based unfolding (полный порт unfold_cvxpy.py).

Использует `Convex.jl` для решения задачи:

    min  ||A*x - b||₂ + α * ||x||_p    subject to  x ≥ 0, x ≤ ub

где `p` — порядок нормы (1 для L1, 2 для L2).

В Python-оригинале использовался `cvxpy` с солверами ECOS/SCS/CLARABEL.
В Julia-порте используется `Convex.jl` + один из доступных солверов:
SCS.jl, ECOS.jl, Clarabel.jl, COSMO.jl.

Это расширение (extension) основного пакета: активируется при наличии
`Convex` и хотя бы одного конического солвера. Если их нет —
функция выдаёт предупреждение и возвращает нулевой спектр.
"""

# Опциональные зависимости — загружаем через Requires-style механизм
# во время первого вызова. Это позволяет BSSUnfold.jl работать без Convex.jl.
const _CONVEX_LOADED = Ref(false)
const _CONVEX_SOLVERS = Ref{Vector{Symbol}}(Symbol[])

function _try_load_convex()
    if _CONVEX_LOADED[]
        return _CONVEX_SOLVERS[]
    end

    loaded = Symbol[]
    # SCS
    try
        @eval using SCS
        push!(loaded, :SCS)
    catch
    end
    # ECOS
    try
        @eval using ECOS
        push!(loaded, :ECOS)
    catch
    end
    # Clarabel
    try
        @eval using Clarabel
        push!(loaded, :Clarabel)
    catch
    end
    # COSMO
    try
        @eval using COSMO
        push!(loaded, :COSMO)
    catch
    end

    if !isempty(loaded)
        try
            @eval using Convex
            pushfirst!(loaded, :Convex)
        catch err
            @warn "Convex.jl не удалось загрузить; solve_cvxpy недоступен" exception=err
            empty!(loaded)
        end
    end

    _CONVEX_SOLVERS[] = loaded
    _CONVEX_LOADED[] = true
    return loaded
end


"""
    solve_cvxpy(A, b, x0; regularization, norm, solver, ub)

Решить задачу развёртки через выпуклую оптимизацию.

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
    # Validate norm early (before loading Convex.jl)
    if norm != 1 && norm != 2
        throw(ArgumentError("Unsupported norm: $norm. Use 1 or 2."))
    end

    available = _try_load_convex()
    if isempty(available)
        @warn "Convex.jl + хотя бы один солвер (SCS, ECOS, Clarabel, COSMO) не установлены. " *
              "Установите через: Pkg.add([\"Convex\", \"SCS\"]). Возвращаю нулевой спектр."
        return UnfoldResult(zeros(T, size(A, 2)), 0, false, T(0),
                           Dict{String,Any}("error" => "Convex.jl not available"))
    end

    m, n = size(A)
    Convex = Base.get(Main, :Convex, nothing)
    Convex === nothing && error("Convex module not loaded")

    # Decision variable: x >= 0
    x_var = Convex.Variable(n, Positive())

    # Objective: minimize ||Ax - b||₂ + α * ||x||_p
    residual = A * x_var - b
    if norm == 1
        objective = Convex.norm2(residual) + regularization * Convex.norm1(x_var)
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

    # Solver selection
    chosen_solver = if solver == :default
        # Попробовать в порядке приоритета
        for cand in (:SCS, :ECOS, :Clarabel, :COSMO)
            if cand in available
                chosen = cand
                break
            end
        end
        chosen
    else
        solver
    end

    if chosen_solver ∉ available
        @warn "Запрошенный солвер $chosen_solver недоступен. Доступные: $(filter(!=(:Convex), available))"
        # Бертём первый доступный
        chosen_solver = first(filter(!=(:Convex), available))
    end

    # Решить через Convex.solve! или solve!
    solver_mod = getproperty(Main, chosen_solver)
    try
        Convex.solve!(problem, solver_mod.Optimizer; verbose=false)
    catch err
        @warn "Solver $chosen_solver failed: $err"
        return UnfoldResult(zeros(T, n), 0, false, T(0),
                           Dict{String,Any}("error" => string(err)))
    end

    if problem.status !== Convex.OPTIMAL && problem.status !== Convex.OPTIMAL_INACCURATE
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
        max.(x, T(0)), 1, true, norm(residual_vec),
        Dict{String,Any}(
            "norm" => norm,
            "solver" => String(chosen_solver),
            "regularization" => regularization,
        )
    )
end
