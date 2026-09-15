"""
Quadratic programming unfolding (полный порт unfold_qpsolvers.py).

Решает задачу:

    min  (1/2) * xᵀ P x + qᵀ x     subject to  x ≥ 0, x ≤ ub

где:
- `P = AᵀA + α * reg_matrix` (reg_matrix = identity для L2 / smoothness penalty)
- `q = -Aᵀb` (для L2) или `q = -Aᵀb + α * ones(n)` (для L1 при x ≥ 0)

В Python-оригинале использовался `qpsolvers` (OSQP, ECOS, ProxQP, Clarabel).
В Julia-порте используется `OSQP.jl` (или `Clarabel.jl` как fallback).

Это extension-зависимый алгоритм. Без установленного `OSQP.jl` / `Clarabel.jl`
функция выдаёт предупреждение и возвращает нулевой спектр.
"""

const _QP_SOLVERS_LOADED = Ref(false)
const _QP_SOLVERS_AVAILABLE = Ref{Vector{Symbol}}(Symbol[])

function _try_load_qp_solvers()
    if _QP_SOLVERS_LOADED[]
        return _QP_SOLVERS_AVAILABLE[]
    end

    loaded = Symbol[]
    # Опциональные пакеты загружаются в Main текущей сессии (аналогично
    # Requires.jl) — см. комментарий в cvxpy.jl.
    try
        Base.eval(Main, :(using OSQP))
        push!(loaded, :OSQP)
    catch
    end
    try
        Base.eval(Main, :(using Clarabel))
        push!(loaded, :Clarabel)
    catch
    end
    try
        Base.eval(Main, :(using COSMO))
        push!(loaded, :COSMO)
    catch
    end
    try
        Base.eval(Main, :(using ProxSDP))
        push!(loaded, :ProxSDP)
    catch
    end

    _QP_SOLVERS_AVAILABLE[] = loaded
    _QP_SOLVERS_LOADED[] = true
    return loaded
end


# Построить smoothness penalty matrix для производных 1-го и 2-го порядка.
function _smoothness_penalty(n::Integer, alpha::T, order::Integer, weight::T) where T
    if order == 0
        return alpha * Matrix{T}(I, n, n)
    elseif order == 1
        # First-order difference L1 (n-1 × n)
        L1 = zeros(T, n-1, n)
        for i in 1:n-1
            L1[i, i] = 1
            L1[i, i+1] = -1
        end
        return alpha * weight * (L1' * L1)
    elseif order == 2
        # Second-order difference L2 (n-2 × n)
        L2 = zeros(T, n-2, n)
        for i in 1:n-2
            L2[i, i] = 1
            L2[i, i+1] = -2
            L2[i, i+2] = 1
        end
        return alpha * weight * (L2' * L2)
    else
        throw(ArgumentError("smoothness_order must be 0, 1, or 2; got $order"))
    end
end


"""
    solve_qpsolvers(A, b, x0; regularization, norm, solver, smoothness_order, smoothness_weight, ub)

Решить задачу развёртки через quadratic programming.

# Аргументы
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: измерения (m,)
- `x0::AbstractVector{T}`: не используется (API)
- `regularization::T`: параметр α (default 1e-4)
- `norm::Integer`: 1 для L1, 2 для L2 (default 2)
- `solver::Symbol`: `:OSQP`, `:Clarabel`, `:default` (default `:default`)
- `smoothness_order`: 0, 1, или 2 (default 0)
- `smoothness_weight::T`: вес smoothness (default 1.0)
- `ub::Union{Nothing,Vector{T}}`: верхние границы (default `nothing`)

# Возвращает
- `UnfoldResult{T}` со спектром
"""
function solve_qpsolvers(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                       regularization::T=T(1e-4),
                       norm::Integer=2,
                       solver::Symbol=:default,
                       smoothness_order::Integer=0,
                       smoothness_weight::T=T(1.0),
                       ub::Union{Nothing,AbstractVector{T}}=nothing) where T<:AbstractFloat
    # Validate parameters early
    if norm != 1 && norm != 2
        throw(ArgumentError("Unsupported norm: $norm. Use 1 or 2."))
    end
    if smoothness_order < 0 || smoothness_order > 2
        throw(ArgumentError("smoothness_order must be 0, 1, or 2; got $smoothness_order"))
    end

    available = _try_load_qp_solvers()
    if isempty(available)
        @warn "Один из QP-солверов (OSQP, Clarabel, COSMO, ProxSDP) не установлен. " *
              "Установите через: Pkg.add([\"OSQP\"]) или Pkg.add([\"Clarabel\"]). Возвращаю нулевой спектр."
        return UnfoldResult(zeros(T, size(A, 2)), 0, false, T(0),
                           Dict{String,Any}("error" => "QP solver not available"))
    end

    m, n = size(A)

    # Build P (positive semidefinite) and q
    P = A' * A
    q_base = -(A' * b)

    # Smoothness penalty
    pen = _smoothness_penalty(n, regularization, smoothness_order, smoothness_weight)

    if norm == 2
        P = P + pen
        q = q_base
    else  # norm == 1 (validated early)
        # L1 with x >= 0: alpha * sum(x) is linear, shifts q
        P = P + pen - regularization * Matrix{T}(I, n, n)  # remove the alpha*I from smoothness_penalty
        q = q_base .+ regularization
        # Re-add smoothness part without the alpha*I
        if smoothness_order > 0
            P += _smoothness_penalty(n, regularization, smoothness_order, smoothness_weight)
        end
    end

    # Symmetrize P for solver
    P = (P + P') ./ T(2)

    # Bounds
    lb = zeros(T, n)
    ub_vec = ub === nothing ? T(Inf) .* ones(T, n) : Vector{T}(ub)

    chosen_solver = if solver == :default
        :OSQP in available ? :OSSP : (:Clarabel in available ? :Clarabel : first(available))
    else
        solver
    end

    if chosen_solver ∉ available
        @warn "Запрошенный солвер $chosen_solver недоступен. Доступные: $available"
        chosen_solver = first(available)
    end

    # Solve via chosen QP solver
    x = if chosen_solver == :OSQP
        _solve_osqp(P, q, lb, ub_vec)
    elseif chosen_solver == :Clarabel
        _solve_clarabel(P, q, lb, ub_vec)
    elseif chosen_solver == :COSMO
        _solve_cosmo(P, q, lb, ub_vec)
    elseif chosen_solver == :ProxSDP
        _solve_proxsdp(P, q, lb, ub_vec)
    else
        @warn "Unknown QP solver: $chosen_solver"
        zeros(T, n)
    end

    if all(x .== 0)
        return UnfoldResult(zeros(T, n), 0, false, T(0),
                           Dict{String,Any}("error" => "QP solver failed"))
    end

    if ub !== nothing
        x[ub .== T(0)] .= T(0)
    end
    x = max.(x, T(0))

    residual = b .- A * x
    return UnfoldResult(
        x, 1, true, norm(residual),
        Dict{String,Any}(
            "norm" => norm,
            "solver" => String(chosen_solver),
            "regularization" => regularization,
            "smoothness_order" => smoothness_order,
            "smoothness_weight" => smoothness_weight,
        )
    )
end


# ═══ QP Solver wrappers ══════════════════════════════════════════════════════

function _solve_osqp(P::Matrix{T}, q::Vector{T}, lb::Vector{T}, ub::Vector{T}) where T
    # OSQP требует sparse P
    P_sparse = SparseArrays.sparse(P)
    try
        OSQP = Main.OSQP
        model = OSQP.Model()
        OSQP.setup!(model; P=P_sparse, q=q, lb=lb, ub=ub, verbose=false)
        OSQP.warm_start!(model, lb .* 0)
        results = OSQP.solve!(model)
        return Vector{T}(results.x)
    catch err
        @warn "OSQP failed: $err"
        return zeros(T, length(q))
    end
end


function _solve_clarabel(P::Matrix{T}, q::Vector{T}, lb::Vector{T}, ub::Vector{T}) where T
    try
        Clarabel = Main.Clarabel
        # Clarabel: min (1/2) x' P x + q' x, subject to lb <= x <= ub
        # Convert to cone format: -x ≤ -lb  AND  x ≤ ub
        n = length(q)
        P_sparse = SparseArrays.sparse(P)
        # Stacked: x <= ub, -x <= -lb  =>  A_cone x + s = b_cone, s in cone
        A_cone = SparseArrays.sparse([Matrix{T}(I, n, n); -Matrix{T}(I, n, n)])
        b_cone = [ub; -lb]
        cones = [Clarabel.NonnegativeConeT(n), Clarabel.NonnegativeConeT(n)]
        settings = Clarabel.Settings(verbose=false)
        solver = Clarabol.Solver(P_sparse, q, A_cone, b_cone, cones, settings)
        result = Clarabel.solve!(solver)
        return Vector{T}(result.x)
    catch err
        @warn "Clarabel failed: $err"
        return zeros(T, length(q))
    end
end


function _solve_cosmo(P::Matrix{T}, q::Vector{T}, lb::Vector{T}, ub::Vector{T}) where T
    try
        COSMO = Main.COSMO
        n = length(q)
        P_sparse = SparseArrays.sparse(P)
        A_cone = SparseArrays.sparse([Matrix{T}(I, n, n); -Matrix{T}(I, n, n)])
        b_cone = [ub; -lb]
        constraints = [COSMO.Nonnegatives(n), COSMO.Nonnegatives(n)]
        model = COSMO.Model()
        COSMO.assemble!(model, P_sparse, q, A_cone, b_cone, constraints; settings=COSMO.Settings(verbose=false))
        result = COSMO.optimize!(model)
        return Vector{T}(result.x)
    catch err
        @warn "COSMO failed: $err"
        return zeros(T, length(q))
    end
end


function _solve_proxsdp(P::Matrix{T}, q::Vector{T}, lb::Vector{T}, ub::Vector{T}) where T
    try
        ProxSDP = Main.ProxSDP
        n = length(q)
        # ProxSDP solves SDPs; for QP we use it as fallback
        # min 0.5 x'Px + q'x  s.t. lb <= x <= ub
        # Equivalent: A = [I; -I], b = [ub; -lb], cones = NonNegative
        # Use Convex-like formulation via ProxSDP result
        model = ProxSDP.Model()
        ProxSDP.set_constraint_matrix!(model, SparseArrays.sparse([Matrix{T}(I, n, n); -Matrix{T}(I, n, n)]))
        ProxSDP.set_objective!(model, P, q)
        ProxSDP.set_rhs!(model, [ub; -lb])
        ProxSDP.optimize!(model; verbose=false)
        return Vector{T}(model.x)
    catch err
        @warn "ProxSDP failed: $err"
        return zeros(T, length(q))
    end
end
