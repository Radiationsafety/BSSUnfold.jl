"""
ReBUNKI (SPUNIT) — современная открытая реимплементация кода BUNKI
(Lacerda et al., 2018; оригинал BUNKI — Naval Research Laboratory, 1984,
SPUNIT-алгоритм, RSICC PSR-266). SPUNIT работает на летаргия-взвешенной
матрице отклика, масштабированной начальным спектром (`aleth = A * x0`),
со стартовым спектром `spl = 1`:

    ss_j    = Σ_i aleth_ij / b_i
    bcc_i   = Σ_j aleth_ij * spl_j
    spll_j  = spl_j * (Σ_i aleth_ij / bcc_i) / ss_j
    spl     ← 3-точечное сглаживание spll (по умолчанию на бины 1,-1 также)

Итоговый спектр — обратный пересчёт `x = spl * x0`. Допуск сходимости по
умолчанию ≈1% (рекомендация документации ReBUNKI).
"""
function solve_rebunki(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                       smoothing::Real=T(0.1),
                       max_iterations::Integer=1000,
                       tolerance::Real=T(0.01),
                       lethargy_weights::Union{Nothing,AbstractVector{T}}=nothing) where T<:AbstractFloat
    m, n = size(A)

    if lethargy_weights !== nothing
        length(lethargy_weights) == n ||
            throw(ArgumentError("lethargy_weights must have length ($n,)"))
        A = A .* reshape(lethargy_weights, 1, n)
    end

    any(b .< 0) && throw(ArgumentError("BUNKI requires strictly positive measurements"))

    keep = b .> 0
    if any(keep .== 0)
        A = A[keep, :]
        b = b[keep]
        isempty(b) && throw(ArgumentError("BUNKI requires strictly positive measurements"))
    end
    m = size(A, 1)

    x0_safe = max.(x0, T(0))
    aleth = A .* reshape(x0_safe, 1, n)

    spl = ones(T, n)
    bcc = aleth * spl

    inv_b = ifelse.(b .> 0, 1 ./ max.(b, eps(T)), T(0))
    ss = vec(aleth' * inv_b)
    inv_ss = ifelse.(ss .> 0, 1 ./ max.(ss, T(1e-37)), T(0))
    denom_s = 1.0 + 2.0 * smoothing

    converged = false
    iters = 0

    @inbounds for k in 1:max_iterations
        iters = k

        inv_bcc = ifelse.(bcc .> 0, 1 ./ max.(bcc, T(1e-37)), T(0))
        spll = spl .* (aleth' * inv_bcc) .* inv_ss
        @. spll = ifelse(spll < T(1e-37), T(0), spll)
        @. spll = ifelse(spl <= T(0), T(0), spll)

        new_spl = copy(spll)
        if n > 2
            new_spl[2:n-1] .= (T(smoothing) .* spll[1:n-2] .+ spll[2:n-1] .+
                               T(smoothing) .* spll[3:n]) ./ T(denom_s)
        end
        bcc = aleth * new_spl

        rel = norm(new_spl .- spl) / (norm(spl) + 1e-12)
        spl = new_spl
        if rel < T(tolerance)
            converged = true
            break
        end
    end

    spectrum = spl .* x0_safe
    residual = b .- A * spectrum
    return UnfoldResult(spectrum, iters, converged, norm(residual))
end
