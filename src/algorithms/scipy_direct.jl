"""
Scipy-direct style решатели (порт `solve_scipy_direct`).

Решается нормальная система `AᵀA x = Aᵀb` одним из итерационных
Krylov-методов (аналоги scipy.sparse.linalg): `cg`, `cgs`, `bicgstab`,
`gmres`, `lgmres`, `minres`, `gcrotmk`, `qmr`, `tfqmr`, `lsqr`, `lsmr`.
Все решатели реализованы внутри пакета.
"""

function _sd_cg(M, r0::Vector{Float64}, rtol::Float64, maxit::Int)
    x = zeros(length(r0))
    r = copy(r0)
    p = copy(r0)
    rs = dot(r, r)
    rs0 = rs
    converged = rs0 == 0
    its = 0
    for k in 1:maxit
        its = k
        Mp = M * p
        alpha = rs / dot(p, Mp)
        x .+= alpha .* p
        r .-= alpha .* Mp
        rs_new = dot(r, r)
        p .= r .+ (rs_new / rs) .* p
        rs = rs_new
        sqrt(rs) <= rtol * sqrt(rs0) && (converged = true; break)
    end
    return x, its, converged
end

function _sd_bicgstab(M, b::Vector{Float64}, rtol::Float64, maxit::Int)
    x = zeros(length(b))
    r = b .- M * x
    bn = norm(b)
    r0 = copy(r)
    p = copy(r)
    v = zeros(length(b))
    s = copy(r)
    t = similar(r)
    rho = alpha = omega = one(Float64)
    res0 = norm(r)
    bn == 0 && (return x, 0, true)
    converged = false
    its = 0
    for k in 1:maxit
        its = k
        rho_new = dot(r0, r)
        abs(rho_new) < eps() && break
        beta = (rho_new / rho) * (alpha / omega)
        p .= r .+ beta .* (p .- omega .* v)
        v .= M * p
        alpha = rho_new / dot(r0, v)
        s .= r .- alpha .* v
        if norm(s) <= rtol * res0
            x .+= alpha .* p
            converged = true
            break
        end
        t .= M * s
        omega = dot(t, s) / dot(t, t)
        abs(omega) < eps() && break
        x .+= alpha .* p .+ omega .* s
        r .= s .- omega .* t
        res = norm(r)
        if res <= rtol * res0
            converged = true
            break
        end
        rho = rho_new
    end
    !converged && norm(b .- M * x) <= rtol * res0 && (converged = true)
    return x, its, converged
end

function _sd_cgs(M, b::Vector{Float64}, rtol::Float64, maxit::Int)
    n = length(b)
    x = zeros(n)
    r0 = copy(b)
    res_abs0 = norm(b)
    res_abs0 == 0 && (return x, 0, true)
    r = copy(b)
    rhat = copy(b)
    p_prev = zeros(n)
    u_prev = zeros(n)
    rho_prev = one(Float64)
    alpha = one(Float64)
    omega = one(Float64)
    converged = false
    its = 0
    for k in 1:maxit
        its = k
        rho = dot(rhat, r)
        abs(rho) < eps() && break
        beta = rho / rho_prev
        u = r .+ beta .* u_prev
        p = u .+ beta .* (u_prev .+ beta .* p_prev)
        Ap = M * p
        sigma = dot(rhat, Ap)
        abs(sigma) < eps() && break
        alpha = rho / sigma
        q = u .- Ap ./ sigma
        vp = p .+ q
        x .+= alpha .* vp
        r .-= alpha .* (M * vp)
        p_prev = p
        u_prev = u
        rho_prev = rho
        norm(r) <= rtol * res_abs0 && (converged = true; break)
    end
    !converged && norm(b .- M * x) <= rtol * res_abs0 && (converged = true)
    return x, its, converged
end

function _sd_gmres(M, b::Vector{Float64}, rtol::Float64, maxit::Int)
    n = length(b)
    x = zeros(n)
    bn = norm(b)
    bn == 0 && (return x, 0, true)
    r = copy(b)
    iterations = 0
    converged = false
    kmax = min(n, maxit)
    while true
        resid = norm(r)
        resid <= rtol * bn && (converged = true; break)
        basis = Vector{Vector{Float64}}([r ./ resid])
        H = zeros(kmax + 1, kmax)
        g = Float64[resid]
        cs = Float64[]
        sn = Float64[]
        j = 0
        while j < kmax
            j += 1
            w = M * basis[end]
            for i in 1:length(basis)
                h = dot(basis[i], w)
                w .-= h .* basis[i]
                H[i, j] = h
            end
            h2 = norm(w)
            H[j + 1, j] = h2
            push!(basis, w ./ max(h2, eps(Float64)))
            for i in 1:max(j - 1, 0)
                tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = tmp
            end
            d = hypot(H[j, j], H[j + 1, j])
            push!(cs, H[j, j] / d)
            push!(sn, H[j + 1, j] / d)
            H[j, j] = d
            H[j + 1, j] = 0.0
            push!(g, 0.0)
            tmpg = cs[j] * g[j] + sn[j] * g[j + 1]
            g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1]
            g[j] = tmpg
            iterations += 1
            (abs(g[j + 1]) <= rtol * bn || iterations >= maxit || j >= kmax) && break
        end
        k = length(basis) - 1
        y = UpperTriangular(H[1:k, 1:k]) \ g[1:k]
        for i in 1:k
            x .+= y[i] .* basis[i]
        end
        r = b .- M * x
        norm(r) <= rtol * bn && (converged = true)
        (converged || iterations >= maxit) && break
    end
    return x, iterations, converged
end

function _sd_minres(M, b::Vector{Float64}, rtol::Float64, maxit::Int)
    n = length(b)
    x = zeros(n)
    bn = norm(b)
    bn == 0 && (return x, 0, true)
    v_old = zeros(n)
    v_cur = b ./ bn
    beta = bn
    alpha = 0.0
    Hcols = Vector{Vector{Float64}}()
    g = Float64[bn]
    cs = Float64[]
    sn = Float64[]
    Q = Vector{Vector{Float64}}([copy(v_cur)])
    converged = false
    its = 0
    while its < maxit && beta > rtol * bn
        its += 1
        w = M * v_cur
        alpha = dot(v_cur, w)
        w .-= alpha .* v_cur .+ beta .* v_old
        beta_new = norm(w)
        k = its
        col = zeros(k + 1)
        col[k] = alpha
        col[k + 1] = beta_new
        push!(Hcols, col)
        push!(g, 0.0)
        for i in 1:k - 1
            tmp = cs[i] * col[i] + sn[i] * col[i + 1]
            col[i + 1] = -sn[i] * col[i] + cs[i] * col[i + 1]
            col[i] = tmp
        end
        d = hypot(col[k], col[k + 1])
        push!(cs, col[k] / d)
        push!(sn, col[k + 1] / d)
        col[k] = d
        col[k + 1] = 0.0
        tmpg = cs[k] * g[k] + sn[k] * g[k + 1]
        g[k + 1] = -sn[k] * g[k] + cs[k] * g[k + 1]
        g[k] = tmpg
        push!(Q, beta_new > 0 ? copy(w ./ beta_new) : copy(v_cur))
        abs(g[k + 1]) <= rtol * bn && (converged = true)
        if converged || its >= maxit || beta_new == 0
            y = _sd_backsolve(Hcols, g)
            for i in 1:k
                x .+= y[i] .* Q[i]
            end
            break
        end
        v_old = v_cur
        v_cur = w ./ beta_new
        beta = beta_new
    end
    !converged && norm(b .- M * x) <= rtol * bn && (converged = true)
    return x, its, converged
end

function _sd_backsolve(Hcols::Vector{Vector{Float64}}, g::Vector{Float64})
    k = length(Hcols)
    y = zeros(k)
    for j in k:-1:1
        col = Hcols[j]
        y[j] = g[j] / col[j]
        for i in 1:j - 1
            g[i] -= col[i] * y[j]
        end
    end
    return y
end

function _sd_cgls(A, b::Vector{Float64}, atol::Float64, maxit::Int)
    x = zeros(size(A, 2))
    r = copy(b)
    s = A' * r
    p = copy(s)
    sq = norm(s)
    its = 0
    converged = false
    for k in 1:maxit
        its = k
        q = A * p
        qnorm = dot(q, q)
        qnorm == 0 && break
        alpha = sq / qnorm
        x .+= alpha .* p
        r .-= alpha .* q
        s_new = A' * r
        sq_new = norm(s_new)
        norm(r) <= atol * norm(b) && (converged = true; break)
        p .= s_new .+ (sq_new / sq) .* p
        sq = sq_new
    end
    return x, its, converged
end

function _sd_map_method(method)
    m = lowercase(string(method))
    valid = ("cg", "cgs", "bicgstab", "gmres", "lgmres", "minres", "qmr",
             "gcrotmk", "tfqmr", "lsqr", "lsmr")
    m in valid || throw(ArgumentError(
        "Unknown solver method '$m'. " *
        "Choose from: cg, cgs, bicgstab, gmres, lgmres, minres, " *
        "qmr, gcrotmk, tfqmr, lsqr, lsmr"))
    return m
end

"""
    solve_direct(A, b, x0=nothing; tolerance=1e-8, max_iterations=4000,
                 method="cg")

Аналог `solve_scipy_direct`: Krylov-решатель нормальной системы
`AᵀA x = Aᵀb`. Поддерживаемые `method`: `cg`, `cgs`, `bicgstab`, `gmres`,
`lgmres`, `minres`, `qmr`, `gcrotmk`, `tfqmr`, `lsqr`, `lsmr`. Методы
`cg`, `cgs`, `bicgstab`, `gmres`, `lgmres`, `minres` работают с системой
`AᵀA x = Aᵀb`; `lsqr`/`lsmr` — CGLS по исходной переопределённой системе;
`qmr`, `tfqmr`, `gcrotmk` — честные замены через ближайшие аналоги
(`bicgstab`, `cgs`, `gmres`). Реализация без scipy.
"""
function solve_direct(A::AbstractMatrix{Float64}, b::AbstractVector{Float64},
                      x0::Union{AbstractVector{Float64},Nothing}=nothing;
                      tolerance::Real=1e-8,
                      max_iterations::Integer=4000,
                      method::Union{AbstractString,Symbol}="cg")
    meth = _sd_map_method(method)
    rtol = Float64(tolerance)
    maxit = Int(max_iterations)

    if meth == "cg"
        x, its, conv = _sd_cg(A' * A, A' * b, rtol, maxit)
    elseif meth == "cgs"
        x, its, conv = _sd_cgs(A' * A, A' * b, rtol, maxit)
    elseif meth == "bicgstab"
        x, its, conv = _sd_bicgstab(A' * A, A' * b, rtol, maxit)
    elseif meth in ("gmres", "lgmres", "gcrotmk")
        x, its, conv = _sd_gmres(A' * A, A' * b, rtol, maxit)
    elseif meth == "minres"
        x, its, conv = _sd_minres(A' * A, A' * b, rtol, maxit)
    elseif meth == "qmr"
        x, its, conv = _sd_bicgstab(A' * A, A' * b, rtol, maxit)
    elseif meth == "tfqmr"
        x, its, conv = _sd_cgs(A' * A, A' * b, rtol, maxit)
    else
        x, its, conv = _sd_cgls(A, b, rtol, maxit)
    end

    x = max.(x, 0.0)
    resid = b .- A * x
    return UnfoldResult(x, its, conv, norm(resid),
        Dict{String,Any}("method" => meth,
                         "tolerance" => rtol))
end

"""
    solve_scipy_direct(A, b, x0=nothing; tolerance=1e-8, max_iterations=4000,
                       method="cg")

Обёртка-псевдоним над [`solve_direct`](@ref), соответствующая имени
Python-функции из `bssunfold.core.unfold_scipy_direct_method` (его `__all__`).
"""
function solve_scipy_direct(A::AbstractMatrix{Float64}, b::AbstractVector{Float64},
                            x0::Union{AbstractVector{Float64},Nothing}=nothing;
                            kwargs...)
    return solve_direct(A, b, x0; kwargs...)
end
