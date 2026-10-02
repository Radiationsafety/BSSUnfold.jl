"""
ADMM — Alternating Direction Method of Multipliers unfolding for neutron spectra.

Solves: min_x 1/2||Ax - b||² + l1_penalty*||x||_1 + tv_penalty*||Dx||_1  s.t.  x >= 0

by the consensus splitting min_{x,z1,z2} with x = z1, D x = z2, x >= 0, whose
iterations are fully decoupled: the x-update is an exact NNLS on the augmented
design [A; sqrt(rho) I; sqrt(rho) D] (so non-negativity holds at every step),
z1/z2 are element-wise soft-thresholdings (exact prox of the L1 / TV terms),
and u1/u2 are scaled dual ascent. rho is optionally retuned every 10
iterations from the primal/dual residual ratio (Boyd et al., sec. 3.4.1).

The augmented design is never materialised: its NNLS is stated on the Gram
system (`_admm_nnls_gram`), where the passive block keeps an incremental
Cholesky factor and full-space products use the factored operator
`v -> A'(A v) + rho v + rho D'(D v)`. Handing the augmented matrix to
`lawson_hanson` instead refactorises it from scratch at every pivot, which costs
O(n^4) in the support size and stalls on realistic grids (n > 200).

References: Gabay & Mercier (1976); Boyd, Parikh, Chu, Peleato & Eckstein,
"Distributed Optimization and Statistical Learning via ADMM", 2011 (MIPT
optimization course, lecture 11 / homework 18).
Port of bssunfold core/unfold_admm.py.
"""

"""
    _admm_soft_threshold(v, t)

Element-wise soft-thresholding (prox of the L1 norm): `sign(v) * max(|v| - t, 0)`.
Returns a copy when `t <= 0`. Port of `soft_threshold` in unfold_admm.py.
"""
function _admm_soft_threshold(v::AbstractVector{T}, t::T) where T<:AbstractFloat
    t <= 0 && return copy(v)
    return sign.(v) .* max.(abs.(v) .- t, T(0))
end


"""
    _admm_difference_matrix(n, T)

First-order difference operator D with `(D x)_i = x_{i+1} - x_i`, i.e. row i
has -1 in column i and +1 in column i+1. Port of `_difference_matrix`.
"""
function _admm_difference_matrix(n::Int, ::Type{T}) where T<:AbstractFloat
    D = zeros(T, max(n - 1, 0), n)
    for i in 1:max(n - 1, 0)
        D[i, i] = T(-1)
        D[i, i + 1] = T(1)
    end
    return D
end


"""
    _AdmmGramOperator

Gram-vector product `v -> G v` of the augmented design `M = [A; sqrt(rho) I;
sqrt(rho) D]`, applied through its factors as `A'(A v) + rho v + rho D'(D v)`:
`O(m n)` per call instead of the `O(n^2)` of the dense `G * v`.
"""
struct _AdmmGramOperator{T}
    A::Matrix{T}
    At::Adjoint{T, Matrix{T}}
    D::Matrix{T}
    Dt::Adjoint{T, Matrix{T}}
    rho::T
    use_tv::Bool
end

function (g::_AdmmGramOperator{T})(v::AbstractVector{T}) where T<:AbstractFloat
    out = g.At * (g.A * v)
    rho = g.rho
    @inbounds @simd for i in eachindex(out)
        out[i] += rho * v[i]
    end
    if g.use_tv
        band = g.Dt * (g.D * v)
        @inbounds for i in eachindex(out)
            out[i] += rho * band[i]
        end
    end
    return out
end


"""
    _admm_gram(AtA, DtD, rho, use_tv)

Dense Gram `G = A'A + rho I + rho D'D` of the augmented design, needed only for
the passive-set blocks; rebuilt solely when rho is retuned.
"""
function _admm_gram(AtA::Matrix{T}, DtD::Matrix{T}, rho::T,
                    use_tv::Bool) where T<:AbstractFloat
    G = copy(AtA)
    @inbounds for j in 1:size(G, 1)
        G[j, j] += rho
    end
    if use_tv
        @inbounds for idx in eachindex(G, DtD)
            G[idx] += rho * DtD[idx]
        end
    end
    return G
end


"""
    _admm_chol_column(L, g)

Forward substitution `L y = g` for the `k x k` lower Cholesky factor `L`.
"""
function _admm_chol_column(L::Matrix{T}, g::AbstractVector{T}) where T<:AbstractFloat
    k = size(L, 1)
    y = zeros(T, k)
    for i in 1:k
        s = g[i]
        for j in 1:i - 1
            s -= L[i, j] * y[j]
        end
        y[i] = s / L[i, i]
    end
    return y
end


"""
    _admm_chol_solve(L, b)

Solve `(L L') z = b` through the lower Cholesky factor `L` by forward and back
substitution.
"""
function _admm_chol_solve(L::Matrix{T}, b::AbstractVector{T}) where T<:AbstractFloat
    k = size(L, 1)
    y = copy(b)
    for i in 1:k
        s = y[i]
        for j in 1:i - 1
            s -= L[i, j] * y[j]
        end
        y[i] = s / L[i, i]
    end
    for i in k:-1:1
        s = y[i]
        for j in i + 1:k
            s -= L[j, i] * y[j]
        end
        y[i] = s / L[i, i]
    end
    return y
end


"""
    _admm_chol_factor(Gsub)

Lower Cholesky factor of a dense symmetric block, or `nothing` when the block is
numerically not positive definite (the caller then uses a general solve).
"""
function _admm_chol_factor(Gsub::Matrix{T}) where T<:AbstractFloat
    try
        return Matrix(cholesky(Symmetric(Gsub, :L)).L)
    catch
        return nothing
    end
end


"""
    _admm_nnls_gram(gv, G, c, max_iterations)

Lawson-Hanson for `min_x ||M x - r||^2` over `x >= 0`, stated on the Gram system
`G = M'M`, `c = M'r` (objective `x'Gx - 2c'x + constant`), which is the shape of
the ADMM x-update. The active set, the pivot test (`sqrt(eps)` on the gradient)
and the alpha-step are those of `lawson_hanson`, so both paths return the same
solution; only the linear algebra differs:

* full-space products go through `gv`, the factored operator of `G`;
* accepting a pivot is a rank-1 Cholesky update of the passive block, one
  forward substitution of cost `O(k^2)`, instead of a fresh least-squares solve
  of `M[:, passive]` costing `O(m k^2)`;
* dropping indices after an alpha-step invalidates the factor, which is then
  rebuilt from `G`.

# Returns
`x`: the non-negative least-squares solution.
"""
function _admm_nnls_gram(gv::_AdmmGramOperator{T}, G::Matrix{T},
                         c::AbstractVector{T}, max_iterations::Int) where T<:AbstractFloat
    n = length(c)
    tol = sqrt(eps(T))

    x = zeros(T, n)
    Prl = Int[]                      # passive set, in the order of L
    inpass = falses(n)
    L = zeros(T, 0, 0)

    iter = 0
    while iter < max_iterations
        iter += 1
        w = c .- gv(x)
        best = 0
        best_w = tol
        for j in 1:n
            if !inpass[j] && w[j] > best_w
                best = j
                best_w = w[j]
            end
        end
        best == 0 && break

        # ---- rank-1 update of the factor with the new pivot ----------------
        k = length(Prl)
        col = _admm_chol_column(L, view(G, Prl, best))
        d2 = G[best, best] - dot(col, col)
        push!(Prl, best)
        inpass[best] = true
        if d2 > 0
            Lnew = zeros(T, k + 1, k + 1)
            Lnew[1:k, 1:k] = L
            Lnew[k + 1, 1:k] = col
            Lnew[k + 1, k + 1] = sqrt(d2)
            L = Lnew
        else
            L = _admm_chol_factor(G[Prl, Prl])
        end

        # ---- inner loop: correction of negative entries ----------------------
        while true
            isempty(Prl) && break
            cP = Vector{T}(view(c, Prl))
            if size(L, 1) == length(Prl)
                zP = _admm_chol_solve(L, cP)
            else
                L = _admm_chol_factor(G[Prl, Prl])
                zP = L === nothing ? G[Prl, Prl] \ cP : _admm_chol_solve(L, cP)
            end
            z = zeros(T, n)
            z[Prl] = zP
            if all(j -> z[j] > zero(T), Prl)
                x = z
                break
            end
            α = T(Inf)
            for j in Prl
                if z[j] <= zero(T)
                    d = x[j] - z[j]
                    d > 0 && (α = min(α, x[j] / d))
                end
            end
            isfinite(α) || break
            for j in 1:n
                x[j] += α * (z[j] - x[j])
                if x[j] <= 10 * eps(T)
                    x[j] = zero(T)
                end
            end
            keep = Int[j for j in Prl if x[j] > zero(T)]
            if length(keep) != length(Prl)
                Prl = keep
                fill!(inpass, false)
                for j in Prl
                    inpass[j] = true
                end
                L = zeros(T, 0, 0)    # the factor no longer matches Prl
            end
        end
    end

    return x
end


function solve_admm(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                    max_iterations::Int=500,
                    tolerance::T=T(1e-6),
                    l1_penalty::T=T(0.0),
                    tv_penalty::T=T(0.0),
                    rho::Union{Nothing,T}=nothing,
                    adaptive_rho::Bool=true,
                    abstol::T=T(1e-10),
                    reltol::T=T(1e-6)) where T<:AbstractFloat
    validate_system(A, b; x0=x0, max_iterations=max_iterations, tolerance=tolerance)

    m, n = size(A)
    l1 = max(T(l1_penalty), T(0))
    tv = max(T(tv_penalty), T(0))

    x = max.(x0, T(0))

    if l1 == 0 && tv == 0
        # Degenerate ADMM: consensus with no nonsmooth terms reduces to a
        # single (augmented) NNLS solve; perform it directly.
        x_opt, _ = lawson_hanson(A, b; max_iterations=10 * n)
        return UnfoldResult(x_opt, 1, true, norm(b .- A * x_opt))
    end

    D = _admm_difference_matrix(n, T)
    use_tv = tv > 0

    if rho === nothing
        fro2 = sum(abs2, A)                                  # ||A||_F^2, np.linalg.norm(A)
        scale = max(norm(b) / max(sqrt(fro2), T(1e-30)), T(1e-12))
        rho_cur = T(scale * max(fro2 / T(length(A)), T(1e-12)))
    else
        rho_cur = T(rho)
    end
    rho_cur = max(rho_cur, T(1e-12))

    rows_M = m + n + (use_tv ? max(n - 1, 0) : 0)   # sqrt(rows_M) * abstol in eps_pri

    # Gram data of the x-update; the rho-independent parts are hoisted out.
    AD = Matrix{T}(A)
    At = AD'
    Dt = D'
    Atb = At * b
    AtA = At * AD
    AtA = (AtA + AtA') / T(2)                  # blocks of G are read as symmetric
    DtD = zeros(T, n, n)
    if use_tv
        Q = Dt * D
        DtD = (Q + Q') / T(2)
    end
    gv = _AdmmGramOperator(AD, At, D, Dt, rho_cur, use_tv)
    G = _admm_gram(AtA, DtD, rho_cur, use_tv)

    z1 = copy(x)
    z2 = D * x
    u1 = zeros(T, n)
    u2 = zeros(T, size(D, 1))

    converged = false
    iterations = 0
    primal_residual = T(NaN)
    dual_residual = T(NaN)

    for k in 1:max_iterations
        z1_prev = copy(z1)
        z2_prev = copy(z2)

        # ---- x-update: exact NNLS on the augmented system ------------------
        c = Atb .+ rho_cur .* (z1 .- u1)
        if use_tv
            c .+= rho_cur .* (Dt * (z2 .- u2))
        end
        x = _admm_nnls_gram(gv, G, c, 10 * n)

        Dx = D * x

        # ---- z-updates: exact proximal operators ---------------------------
        z1 = _admm_soft_threshold(x .+ u1, l1 / rho_cur)
        if use_tv
            z2 = _admm_soft_threshold(Dx .+ u2, tv / rho_cur)
        end

        # ---- dual updates --------------------------------------------------
        u1 = u1 .+ x .- z1
        if use_tv
            u2 = u2 .+ Dx .- z2
        end

        iterations = k

        # ---- Boyd primal/dual residual stopping rule -----------------------
        pri = norm(x .- z1)
        if use_tv
            pri = hypot(pri, norm(Dx .- z2))
        end
        dz1 = z1 .- z1_prev
        dual = norm(dz1)
        if use_tv
            dual = hypot(dual, norm(Dt * (z2 .- z2_prev)))
        end
        dual *= rho_cur
        primal_residual = pri
        dual_residual = dual

        n_pri = max(norm(x), norm(z1), use_tv ? norm(Dx) : T(0))
        n_dual = norm(u1) + (use_tv ? norm(Dt * u2) : T(0))
        eps_pri = sqrt(T(rows_M)) * abstol + reltol * max(n_pri, T(1e-30))
        eps_dual = T(n) * abstol + reltol * max(n_dual, T(1e-30)) * rho_cur

        if pri <= eps_pri && dual <= eps_dual
            converged = true
            break
        end

        # ---- adaptive rho (Boyd et al., sec. 3.4.1) ------------------------
        if adaptive_rho && k % 10 == 0
            rho_new = pri > 10 * dual ? 2 * rho_cur : (dual > 10 * pri ? rho_cur / 2 : rho_cur)
            if rho_new != rho_cur
                # Rescale the duals so the state stays consistent with the new
                # penalty (equivalent to u <- (rho/rho_new) u).
                factor = rho_cur / rho_new
                u1 .*= factor
                u2 .*= factor
                rho_cur = rho_new
                gv = _AdmmGramOperator(AD, At, D, Dt, rho_cur, use_tv)
                G = _admm_gram(AtA, DtD, rho_cur, use_tv)
            end
        end
    end

    extra = Dict{String,Any}(
        "primal_residual" => Float64(primal_residual),
        "dual_residual"   => Float64(dual_residual),
        "rho"             => Float64(rho_cur),
        "l1_penalty"      => Float64(l1),
        "tv_penalty"      => Float64(tv),
        "adaptive_rho"    => adaptive_rho,
    )
    return UnfoldResult(x, iterations, converged, norm(b .- A * x), extra)
end
