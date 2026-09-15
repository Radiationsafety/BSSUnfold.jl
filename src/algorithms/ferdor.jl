"""
FERDOR (Ferret) unfolding — ORNL Burrus (ORNL-4154, 1965).

Взвешенная МНК с шумовым ограничением и регуляризацией вторыми разностями:

    phi_hat = argmin { 1/2 ||Sigma^{-1/2}(A phi - b)||^2 + alpha/2 ||D2 phi||^2 },
    phi >= 0.

Вес сглаживания `alpha` подбирается бисекцией так, чтобы приведённый
xи-квадрат равнялся `chi_squared_target` (диспропорция-принцип FERDOR).
Прямое ограниченное МНК-решение, поэтому результат не зависит от `x0`
(принимается для совместимости и используется как запасной вариант).
"""

function _fdm_d2(n::Integer)
    if n > 2
        L = zeros(Float64, n - 2, n)
        for j in 1:n - 2
            L[j, j] = 1.0
            L[j, j + 1] = -2.0
            L[j, j + 2] = 1.0
        end
        return L, L' * L
    end
    return zeros(0, n), zeros(n, n)
end

function _ferdor_wls(ATA::Matrix{Float64}, ATb::Vector{Float64}, LTL::Matrix{Float64},
                    alpha::Float64, Aw, bw)
    if alpha < 1e-20
        if Aw !== nothing && bw !== nothing
            try
                return solve_nnls(Aw, bw)
            catch
            end
        end
        try
            return max.(ATA \ ATb, 0.0)
        catch
            return nothing
        end
    end

    P = ATA .+ alpha .* LTL

    x = try
        max.(P \ ATb, 0.0)
    catch
        nothing
    end
    x !== nothing && return x

    x = try
        max.(pinv(P) * ATb, 0.0)
    catch
        nothing
    end
    x !== nothing && return x

    if Aw !== nothing && bw !== nothing
        if alpha > 0 && any(!iszero, LTL)
            La = try
                cholesky(Symmetric((LTL + LTL') ./ 2)).L .* sqrt(alpha)
            catch
                nothing
            end
            if La !== nothing
                Aw_aug = [Aw; La]
                bw_aug = [bw; zeros(size(La, 1))]
                try
                    return solve_nnls(Aw_aug, bw_aug)
                catch
                end
            end
        end
        try
            return solve_nnls(Aw, bw)
        catch
        end
    end
    return nothing
end

"""
    solve_ferdor(A, b, x0; max_iterations=100, tolerance=1e-3,
                 smoothing=1e-3, chi_squared_target=1.0,
                 relative_uncertainty=0.1, sigma=nothing,
                 min_alpha=1e-12, max_alpha=1e12)

Развёртка методом FERDOR. `alpha` подбирается бисекцией так, чтобы
приведённый xи-квадрат `chi2/dof` приближался к `chi_squared_target`.

# Аргументы
- `A` — матрица откликов `m×n`
- `b` — вектор измерений `m`
- `x0` — начальное приближение (используется только при отказе прямого решения)
- `max_iterations` — максимум итераций подбора веса
- `tolerance` — относительный допуск на приведённый xи-квадрат
- `smoothing` — начальный вес сглаживания `alpha`
- `chi_squared_target` — целевой xи-квадрат на степень свободы
- `relative_uncertainty` — относительная погрешность измерений (если нет `sigma`)
- `sigma` — явные погрешности `m` (перекрывают `relative_uncertainty`)
- `min_alpha`, `max_alpha` — границы бракетинга веса
"""
function solve_ferdor(A::AbstractMatrix{Float64}, b::AbstractVector{Float64},
                      x0::AbstractVector{Float64};
                      max_iterations::Integer=100,
                      tolerance::Real=1e-3,
                      smoothing::Real=1e-3,
                      chi_squared_target::Real=1.0,
                      relative_uncertainty::Real=0.1,
                      sigma::Union{AbstractVector{Float64},Nothing}=nothing,
                      min_alpha::Real=1e-12,
                      max_alpha::Real=1e12)
    m, n = size(A)
    m == 0 && throw(ArgumentError("Measurement vector b is empty"))
    all(b .> 0) || throw(ArgumentError(
        "FERDOR requires at least one strictly positive measurement"))

    sigma_v = sigma !== nothing ? max.(collect(float.(sigma)), 1e-12) :
              relative_uncertainty .* max.(abs.(b), 1e-12)
    length(sigma_v) == m || throw(ArgumentError("sigma must have shape ($m,)"))

    Wsqrt = 1.0 ./ sigma_v
    Aw = A .* Wsqrt
    bw = b .* Wsqrt

    ATA = Aw' * Aw
    ATb = Aw' * bw

    Ld, LTL = _fdm_d2(n)

    dof = max(m - 1, 1)

    lo = Float64(min_alpha)
    hi = Float64(max_alpha)
    alpha_raw = clamp(Float64(smoothing), lo, hi)

    best = max.(x0, 0.0)
    converged = false
    iterations = 0
    chi_ratio = NaN

    x_init = _ferdor_wls(ATA, ATb, LTL, 0.0, Aw, bw)
    if x_init !== nothing
        r_init = (A * x_init) .- b
        chi2_init = dot(r_init ./ sigma_v, r_init ./ sigma_v)
        if chi2_init / dof <= chi_squared_target
            resid = b .- A * x_init
            return UnfoldResult(x_init, 1, true, norm(resid),
                Dict{String,Any}("chi_squared" => chi2_init / dof,
                                 "alpha" => 0.0))
        end
    end

    x_final = best
    for it in 1:Int(max_iterations)
        iterations = it
        x = _ferdor_wls(ATA, ATb, LTL, alpha_raw, Aw, bw)
        x === nothing && break
        x_final = x

        residual = A * x .- b
        chi2 = dot(residual ./ sigma_v, residual ./ sigma_v)
        chi_ratio = chi2 / dof

        if abs(chi_ratio - chi_squared_target) <= tolerance * max(abs(chi_squared_target), 1.0)
            converged = true
            break
        end

        if chi_ratio > chi_squared_target
            hi = alpha_raw
        else
            lo = alpha_raw
        end

        if hi <= lo
            converged = true
            break
        end

        new_alpha = sqrt(lo * hi)
        if abs(new_alpha - alpha_raw) <= tolerance * max(abs(alpha_raw), 1e-30)
            converged = true
            break
        end
        alpha_raw = new_alpha
    end

    resid = b .- A * x_final
    return UnfoldResult(x_final, iterations, converged, norm(resid),
        Dict{String,Any}("chi_squared" => chi_ratio,
                         "chi_squared_target" => Float64(chi_squared_target),
                         "relative_uncertainty" => Float64(relative_uncertainty),
                         "smoothing" => Float64(smoothing),
                         "alpha" => alpha_raw))
end
