"""
Noise-constrained Tikhonov–TV unfolding (порт `automatic_Tikhonov_TV`
Gazzola & Gholami, адаптированный к 1D).

    min f(m)  subject to  ||A m - b||^2 = epsilon

где `f` одно из: `TT` — ||D1 m||₁ + β/2·||D1̄ D1 m||², `TV` — чистая TV,
`T` — чистый Тихонов. Решается ADMM-схемой; допустимость шума
поддерживается мультипликативным множителем `gamma` (e-подзадача),
вычисляемым аналитически через старший вещественный корень кубического
уравнения gamma³ + p·gamma + q = 0 (формула Кардано — вместо `np.roots`).
"""

function _tvtv_d1(n::Integer)
    D1 = zeros(Float64, n - 1, n)
    for i in axes(D1, 1)
        D1[i, i] = -1.0
        D1[i, i + 1] = 1.0
    end
    return D1
end

function _tvtv_d2(n::Integer)
    n <= 2 && return zeros(0, n)
    L = zeros(Float64, n - 2, n)
    for j in 1:n - 2
        L[j, j] = 1.0
        L[j, j + 1] = -2.0
        L[j, j + 2] = 1.0
    end
    return L
end

function _tvtv_zscore_max(p::AbstractVector{Float64}, a::Float64)
    psort = sort(abs.(p))
    isempty(psort) && return 0.0
    upper = psort[div(length(psort), 2)+1:end]
    mu = sum(upper) / length(upper)
    mad = 1.4826 * (sum(abs.(upper .- mu)) / length(upper) + eps(Float64))
    med = length(upper) % 2 == 1 ? upper[(length(upper)+1)÷2] :
          (upper[length(upper)÷2] + upper[length(upper)÷2 + 1]) / 2
    z = (upper .- med) ./ mad
    absz = abs.(z)
    idx = findall(<(a), absz)
    isempty(idx) && return 0.0
    return maximum(upper[idx])
end

function _tvtv_gamma_from_cubic(pp::Float64, qq::Float64)
    if abs(pp) < 1e-300
        return qq <= 0 ? cbrt(-qq) : -cbrt(qq)
    end
    disc = qq^2 / 4 + pp^3 / 27
    if disc > 0
        c1 = cbrt(-qq / 2 + sqrt(disc))
        c2 = cbrt(-qq / 2 - sqrt(disc))
        return c1 + c2
    else
        r = sqrt(-pp / 3)
        arg = clamp(3 * qq / (2 * pp) * r, -1.0, 1.0)
        φ = acos(arg)
        roots = [2r * cos((φ - 2π * k) / 3) for k in 0:2]
        return maximum(roots)
    end
end

"""
    solve_tikhonov_tv(A, b, x0=nothing; epsilon=nothing, mu=(1.0,1.0,1.0),
                      max_iterations=100, type_="TT", beta=1.0, zthr=2.5,
                      tolerance=1e-4)

Развёртка noise-constrained Tikhonov–TV (Gazzola & Gholami, ADMM).

- `epsilon` — оценка квадрата нормы шума; по умолчанию вычисляется
  метод наименьших квадратов без регуляризации.
- `mu` — штрафные параметры (mu1, mu2, mu3).
- `type_` — задача: "TT" (TV+Tikhonov), "TV", "T".
- `beta` — балансирующий параметр; строка/символ `"adapt"` — адаптивная
  оценка (только для type_ = "TT").
- `zthr` — порог адаптивной оценки beta.
- `tolerance` — критерий остановки по относительному изменению решения.
"""
function solve_tikhonov_tv(A::AbstractMatrix{Float64}, b::AbstractVector{Float64},
                           x0::Union{AbstractVector{Float64},Nothing}=nothing;
                           epsilon::Union{Real,Nothing}=nothing,
                           mu::Tuple{Real,Real,Real}=(1.0, 1.0, 1.0),
                           max_iterations::Integer=100,
                           type_::Union{AbstractString,Symbol}="TT",
                           beta::Union{Real,Symbol,AbstractString}=1.0,
                           zthr::Real=2.5,
                           tolerance::Real=1e-4)
    type_s = uppercase(string(type_))
    type_s in ("TT", "TV", "T") || throw(ArgumentError(
        "Unsupported type_: $type_. Choose from 'TT', 'TV', 'T'."))

    m, n = size(A)
    mu1, mu2, mu3 = Float64.(mu)

    D1 = _tvtv_d1(n)
    D1_bar = _tvtv_d2(max(n - 1, 3))

    eps_val = if epsilon === nothing
        x_ls = (A' * A) \ (A' * b)
        norm(b .- A * x_ls)^2
    else
        Float64(epsilon)
    end
    eps_val <= 0 && (eps_val = 1e-12)

    B = mu1 .* (D1' * D1) .+ mu2 .* (A' * A)
    B_inv = try
        inv(B)
    catch
        pinv(B)
    end

    m_vec = zeros(n)
    g1 = zeros(n - 1)
    g2 = zeros(n - 1)
    e = zeros(m)
    lambda_1 = zeros(n - 1)
    lambda_2 = zeros(m)
    lambda_3 = 0.0

    adapt_beta = beta isa Symbol ?
        (beta == :adapt) :
        (beta isa AbstractString && String(beta) == "adapt")
    beta_k = adapt_beta ? Float64(1.0) : Float64(beta)
    type_s == "TV" && (beta_k = 0.0)

    converged = false
    stopit = Int(max_iterations)
    m_prev = nothing
    eye = Matrix{Float64}(I, n - 1, n - 1)

    for k in 1:Int(max_iterations)
        rhs = mu1 .* (D1' * (g1 .+ g2 .+ lambda_1)) .+
              mu2 .* (A' * (b .+ e .+ lambda_2))
        m_vec = B_inv * rhs

        if k == 1
            m_prev = copy(m_vec)
        else
            norm_prev = norm(m_prev)
            diffm = norm_prev > 0 ? norm(m_vec .- m_prev) / norm_prev : norm(m_vec)
            m_prev = copy(m_vec)
            if diffm < tolerance && !converged
                stopit = k
                converged = true
                break
            end
        end

        if type_s in ("TT", "TV")
            y1 = (D1 * m_vec) .- g2 .- lambda_1
            g1 .= sign.(y1) .* max.(abs.(y1) .- 1 / mu1, 0.0)
        end

        if type_s in ("TT", "T")
            y2 = (D1 * m_vec) .- g1 .- lambda_1
            if beta_k > 0 && mu1 > 0
                lhs = eye .+ (beta_k / mu1) .* (D1_bar' * D1_bar)
                g2 .= lhs \ y2
            else
                g2 .= y2
            end
        end

        y = A * m_vec .- b .- lambda_2
        E = dot(y, y)
        if E > 0
            pp = (mu2 - 2 * mu3 * (eps_val + lambda_3)) / (2 * mu3 * E)
            qq = -mu2 / (2 * mu3 * E)
            gamma = _tvtv_gamma_from_cubic(pp, qq)
        else
            gamma = 0.0
        end

        e .= gamma .* y

        lambda_1 .+= g1 .+ g2 .- D1 * m_vec
        lambda_2 .+= b .+ e .- A * m_vec
        lambda_3 += eps_val - dot(e, e)

        if adapt_beta && type_s == "TT"
            grad = D1 * m_vec
            target = _tvtv_zscore_max(grad, Float64(zthr))
            value = isempty(g2) ? 0.0 : maximum(abs.(g2))
            denom = value + target
            if denom > 0
                beta_k = 2.0 * value / denom * beta_k
            end
        elseif type_s == "T" && !adapt_beta
            beta_k = Float64(beta)
        end
    end

    spectrum = max.(m_vec, 0.0)
    resid = b .- A * spectrum
    return UnfoldResult(spectrum, stopit, converged, norm(resid),
        Dict{String,Any}("type_" => type_s,
                         "epsilon" => eps_val,
                         "beta" => adapt_beta ? "adapt" : beta_k))
end
