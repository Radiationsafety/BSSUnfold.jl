"""
GKS (Generalized Krylov Subspace) unfolding.

Golub-Kahan bidiagonalization of `A` builds a Krylov subspace; both `A`
and the regularization operator `L` (identity or derivative matrix) are
projected onto it.  At each iteration the projected Tikhonov problem
`min ||RA*y - bhat||^2 + lam*||RL*y||^2` is solved, with `lam` selected
automatically on the projected problem by GCV, the Discrepancy Principle
(DP) or the L-curve.  Port of the TRIPs-Py / IRtools GKS implementation.

`regularization_method`: `"gcv"`, `"dp"`, `"lcurve"` or `"manual"`.
"""
function _gks_reg_operator(n::Int, smoothness_order::Int)
    smoothness_order == 0 && return Matrix{Float64}(I, n, n)
    0 < smoothness_order < 3 || throw(ArgumentError("Unsupported smoothness_order: $smoothness_order. Use 0, 1 or 2."))
    if smoothness_order == 1
        L = zeros(n - 1, n)
        for i in 1:(n-1)
            L[i, i] = -1.0
            L[i, i+1] = 1.0
        end
        return L
    end
    L = zeros(n - 2, n)
    for i in 1:(n-2)
        L[i, i] = 1.0
        L[i, i+1] = -2.0
        L[i, i+2] = 1.0
    end
    return L
end

function _gks_projected_gcv(RA::Matrix{Float64}, bhat::Vector{Float64};
                            n_lambdas::Int=200, lambda_range::Tuple{Float64,Float64}=(1e-12, 1e2))
    U, s, _ = svd(RA)
    c = U' * bhat
    s2 = s .^ 2
    m_proj = size(RA, 1)
    lambdas = 10.0 .^ range(log10(lambda_range[1]), log10(lambda_range[2]), length=n_lambdas)
    gcv_values = similar(lambdas)
    for i in eachindex(lambdas)
        lam = lambdas[i]
        filt = s2 ./ (s2 .+ lam)
        residual_coeff = lam ./ (s2 .+ lam)
        residual_sq = sum((residual_coeff .* c) .^ 2)
        trace_term = sum(filt)
        gcv_values[i] = residual_sq / (m_proj - trace_term)^2
    end
    _, idx = findmin(gcv_values)
    return lambdas[idx]
end

function _gks_projected_dp(RA::Matrix{Float64}, bhat::Vector{Float64}, noise_level::Float64;
                           n_lambdas::Int=200, lambda_range::Tuple{Float64,Float64}=(1e-12, 1e2))
    U, _, Vt = svd(RA)
    c = U' * bhat
    s = svdvals(RA)
    s2 = s .^ 2
    m_proj = size(RA, 1)
    target = noise_level * sqrt(m_proj)
    lambdas = 10.0 .^ range(log10(lambda_range[1]), log10(lambda_range[2]), length=n_lambdas)
    residuals = similar(lambdas)
    for i in eachindex(lambdas)
        lam = lambdas[i]
        filt = s ./ (s2 .+ lam)
        x = Vt' * (filt .* c)
        residuals[i] = norm(RA * x - bhat)
    end
    _, idx = findmin(abs.(residuals .- target))
    return lambdas[idx]
end

function _gks_projected_lcurve(RA::Matrix{Float64}, bhat::Vector{Float64};
                               n_lambdas::Int=200, lambda_range::Tuple{Float64,Float64}=(1e-12, 1e2))
    U, _, Vt = svd(RA)
    c = U' * bhat
    s = svdvals(RA)
    s2 = s .^ 2
    lambdas = 10.0 .^ range(log10(lambda_range[1]), log10(lambda_range[2]), length=n_lambdas)
    n_lam = length(lambdas)
    residuals = similar(lambdas)
    norms = similar(lambdas)
    for i in 1:n_lam
        lam = lambdas[i]
        filt = s ./ (s2 .+ lam)
        x = Vt' * (filt .* c)
        residuals[i] = norm(RA * x - bhat)
        norms[i] = norm(x)
    end
    if n_lam < 3
        return lambdas[n_lam ÷ 2]
    end
    log_res = log.(max.(residuals, 1e-300))
    log_norm = log.(max.(norms, 1e-300))
    p1 = [log_res[1], log_norm[1]]
    p2 = [log_res[end], log_norm[end]]
    d21 = p2 .- p1
    denom = max(norm(d21), 1e-300)
    distances = abs.(d21[1] .* (p1[2] .- log_norm) .- d21[2] .* (p1[1] .- log_res)) ./ denom
    _, idx = findmax(distances)
    return lambdas[idx]
end

"""
    solve_gks(A, b, x0=nothing; smoothness_order=0, regularization_method="gcv",
              max_iterations=nothing, regularization=1e-8, noise_level=nothing)
      -> UnfoldResult

Golub-Kahan GKS развёртка: спроектированная задача Тихонова с автоматическим
выбором `lam` (GCV / DP / L-curve / manual).  `x0` принимается для
совместимости API (не используется).  `converged` означает, что
Крыловское пространство было полностью построено либо достигнута
неподвижная точка проекции.
"""
function solve_gks(A::AbstractMatrix, b::AbstractVector, x0::Union{Nothing,AbstractVector}=nothing;
                   smoothness_order::Integer=0,
                   regularization_method::String="gcv",
                   max_iterations::Union{Nothing,Integer}=nothing,
                   regularization::Real=1e-8,
                   noise_level::Union{Nothing,Real}=nothing)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    m, n = size(AF)
    max_k = max_iterations === nothing ? min(m, n) : max(1, Int(max_iterations))
    L = _gks_reg_operator(n, Int(smoothness_order))

    beta = norm(bf)
    if beta == 0.0
        res = UnfoldResult(zeros(n), 0, true, 0.0)
        return res
    end

    reg = Float64(regularization)
    method = lowercase(regularization_method)

    Ucols = Vector{Vector{Float64}}([bf ./ beta])
    V = Matrix{Float64}(undef, n, 0)
    alphas = Float64[]
    betas = Float64[]

    best_x = zeros(n)
    iterations = 0
    converged = false

    for k in 1:max_k
        u = Ucols[end]
        if k == 1
            v = AF' * u
        else
            v = AF' * u .- betas[end] .* V[:, end]
        end
        alpha = norm(v)
        if alpha <= 1e-14
            converged = true
            break
        end
        v = v ./ alpha
        V = hcat(V, v)

        u2 = AF * v .- alpha .* u
        new_beta = norm(u2)
        if new_beta <= 1e-14
            converged = true
        else
            push!(Ucols, u2 ./ new_beta)
        end

        append!(alphas, [alpha])
        append!(betas, [new_beta])

        p = length(Ucols)

        B = zeros(p, k)
        for i in 1:k
            B[i, i] = alphas[i]
        end
        if k > 1
            sub = min(k - 1, p - 1)
            for i in 1:sub
                B[i+1, i] = betas[i]
            end
        end

        bhat = zeros(p)
        bhat[1] = beta

        U_full = hcat(Ucols...)
        RA = U_full' * (AF * V)
        RL = L * V

        if method == "gcv"
            lam = _gks_projected_gcv(RA, bhat)
        elseif method == "dp"
            nl = noise_level === nothing ? 0.01 : Float64(noise_level)
            lam = _gks_projected_dp(RA, bhat, nl)
        elseif method == "lcurve"
            lam = _gks_projected_lcurve(RA, bhat)
        elseif method == "manual"
            lam = reg
        else
            throw(ArgumentError("Unsupported regularization method: $regularization_method. Choose from 'gcv', 'dp', 'lcurve', 'manual'."))
        end

        if !isfinite(lam) || lam <= 0
            lam = reg
        end

        lhs = vcat(RA, sqrt(lam) .* RL)
        rhs = vcat(bhat, zeros(size(RL, 1)))
        y = lhs \ rhs
        x = V * y
        best_x = x

        iterations = k

        if converged
            break
        end
    end

    best_x = max.(best_x, 0.0)
    residual = bf .- AF * best_x
    return UnfoldResult(best_x, iterations, converged, norm(residual))
end
