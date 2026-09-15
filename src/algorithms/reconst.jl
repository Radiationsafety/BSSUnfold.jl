"""
Statistical regularization — RECONST.FOR / STREG1 (Turchin 1967), порт
`solve_reconst`.

Решается система `(B·β + Ω·α)·f = A_vec·β` с автоматическим подбором
`α`/`β`: если параметр отрицателен/нулев (`alpha < 0`, `beta ≤ 0`)
используется поиск нуля функционалов `omega(α)` и `delta(β)`
(сканирование + бисекция, как в оригинальном коде STREG1).
"""

const _RECONST_AINF = [1.01, 1.01, 0.01, 0.01, 0.0]

function _reg_o(n::Int, pp::Float64)
    XX = collect(Float64(1.0):Float64(n + 2.0))

    AA = zeros(n + 2)
    BB = zeros(n + 3)
    CC = zeros(n + 3)

    for ii in 2:n - 1
        AA[ii + 1] = 1.0 / (XX[ii + 1] - XX[ii])
        CC[ii + 1] = 1.0 / (XX[ii] - XX[ii - 1])
        BB[ii + 1] = -(AA[ii + 1] + CC[ii + 1])
    end

    OMO = zeros(5, n)
    for ii in 0:n - 1
        j = ii + 1
        OMO[1, j] = AA[ii + 1] * CC[ii + 1]
        OMO[2, j] = AA[ii + 1] * BB[ii + 1] + BB[ii + 2] * CC[ii + 2]
        OMO[3, j] = AA[ii + 1]^2 + BB[ii + 2]^2 + CC[ii + 3]^2 +
                    pp * (XX[ii + 2] - XX[ii + 1])
    end
    return OMO
end

function _reg_o_full(OMO::Matrix{Float64}, n::Int)
    Omega = zeros(n, n)
    for i in 1:n
        Omega[i, i] = OMO[3, i]
        i > 1 && (Omega[i, i - 1] = OMO[2, i])
        i > 2 && (Omega[i, i - 2] = OMO[1, i])
        i < n && (Omega[i, i + 1] = OMO[2, i + 1])
        i < n - 1 && (Omega[i, i + 2] = OMO[1, i + 2])
    end
    return Omega
end

function _reg_inv(D::Matrix{Float64})
    n = size(D, 1)
    for reg in (0.0, 1e-6, 1e-4, 1e-2)
        Dr = reg == 0.0 ? D : D + Matrix{Float64}(I, n, n) .* reg
        invD = try
            inv(Dr)
        catch
            nothing
        end
        if invD !== nothing && all(isfinite, vec(invD))
            return invD
        end
    end
    return pinv(D)
end

function _reg_reg1(B::Matrix{Float64}, OMO::Matrix{Float64},
                   A_vec::Vector{Float64}, n::Int, alpha::Float64,
                   beta::Float64, ich::Int)
    Omega = _reg_o_full(OMO, n)
    D = beta .* B .+ alpha .* Omega
    if ich > 0
        D_inv = _reg_inv(D)
        FI = D_inv * (A_vec .* beta)
        SIGMA = sqrt.(abs.(diag(D_inv)))
        return D_inv, FI, SIGMA
    end
    return D, nothing, nothing
end

function _reg_omega(OMO::Matrix{Float64}, D_inv::Matrix{Float64},
                    FI::Vector{Float64}, n::Int, alpha::Float64)
    Omega = _reg_o_full(OMO, n)
    code_trace = 0.0
    for i in 1:n
        j_start = i >= 4 ? 1 : (i == 3 ? 2 : i)
        j_end = i <= n - 2 ? i + 2 : (i == n - 1 ? i + 1 : i)
        for j in j_start:j_end
            abs(i - j) <= 2 && (code_trace += Omega[i, j] * D_inv[j, i])
        end
    end
    fof = dot(FI, Omega * FI)
    return Float64(n) / alpha - (code_trace + fof)
end

function _reg_delta(B::Matrix{Float64}, D_inv::Matrix{Float64},
                    FI::Vector{Float64}, A_vec::Vector{Float64},
                    F::Vector{Float64}, S::Vector{Float64},
                    n::Int, m::Int, beta::Float64)
    d1 = tr(B * D_inv)
    d2 = dot(FI, B * FI)
    d3 = dot(A_vec, FI)
    d4 = sum((F ./ S) .^ 2)
    delta = d1 + d2 - 2.0 * d3 + d4
    return Float64(m) / beta - delta
end

function _reg_def_alpha(B::Matrix{Float64}, OMO::Matrix{Float64},
                        A_vec::Vector{Float64}, n::Int, m::Int,
                        alpha::Float64, beta::Float64,
                        omega_init::Float64, ainf::Vector{Float64})
    alm = 4.0^(omega_init >= 0 ? 1.0 : -1.0)
    als = omega_init

    for _ in 1:50
        alpha *= alm
        D_inv, FI, _ = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
        omega = _reg_omega(OMO, D_inv, FI, n, alpha)
        omega * als <= 0 && break
    end

    aln = (alpha + alpha / alm) / 5.0
    alk = 4.0 * aln

    for _ in 1:100
        alpha = (aln + alk) / 2.0
        D_inv, FI, _ = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
        omega = _reg_omega(OMO, D_inv, FI, n, alpha)
        if omega < 0
            alk = alpha
        else
            aln = alpha
        end
        alk <= aln * ainf[1] && break
    end
    return alpha
end

function _reg_def_beta(B::Matrix{Float64}, OMO::Matrix{Float64},
                       A_vec::Vector{Float64}, F::Vector{Float64},
                       S::Vector{Float64}, n::Int, m::Int,
                       alpha::Float64, beta::Float64,
                       delta_init::Float64, ainf::Vector{Float64})
    betm = 4.0^(delta_init >= 0 ? 1.0 : -1.0)
    bets = delta_init

    for _ in 1:50
        beta *= betm
        D_inv, FI, _ = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
        delta = _reg_delta(B, D_inv, FI, A_vec, F, S, n, m, beta)
        delta * bets <= 0 && break
    end

    betn = (beta + beta / betm) / 5.0
    betk = 4.0 * betn

    for _ in 1:100
        beta = (betn + betk) / 2.0
        D_inv, FI, _ = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
        delta = _reg_delta(B, D_inv, FI, A_vec, F, S, n, m, beta)
        if delta < 0
            betk = beta
        else
            betn = beta
        end
        betk <= betn * ainf[2] && break
    end
    return beta
end

function _reconst_streg1(AK::Matrix{Float64}, F::Vector{Float64},
                         S::Vector{Float64}, n::Int, m::Int,
                         alpha::Float64, beta::Float64,
                         pp::Float64, ainf::Vector{Float64})
    sa = exp(sum(log.(max.(S, 1e-300))) / length(S))
    S_norm = S ./ sa

    W = AK ./ S_norm
    B = W' * W

    A_vec = AK' * (F ./ S_norm .^ 2)

    OMO = _reg_o(n, pp)

    if beta > 0.0
        beta = beta / sa^2
        if alpha >= 0.0
            D_inv, FI, SIGMA = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
        else
            alpha = -alpha
            D_inv, FI, _ = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
            omega_init = _reg_omega(OMO, D_inv, FI, n, alpha)
            alpha = _reg_def_alpha(B, OMO, A_vec, n, m, alpha, beta, omega_init, ainf)
            D_inv, FI, SIGMA = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
        end
    else
        beta = 1.0 / sa^2
        if alpha >= 0.0
            D_inv, FI, _ = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
            delta_init = _reg_delta(B, D_inv, FI, A_vec, F, S_norm, n, m, beta)
            beta = _reg_def_beta(B, OMO, A_vec, F, S_norm, n, m, alpha, beta, delta_init, ainf)
            D_inv, FI, SIGMA = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
        else
            alpha = -alpha
            bet_saved = beta
            D_inv, FI, _ = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
            for _ in 1:30
                omega_init = _reg_omega(OMO, D_inv, FI, n, alpha)
                alpha = _reg_def_alpha(B, OMO, A_vec, n, m, alpha, beta, omega_init, ainf)
                D_inv, FI, _ = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
                delta_init = _reg_delta(B, D_inv, FI, A_vec, F, S_norm, n, m, beta)
                beta = _reg_def_beta(B, OMO, A_vec, F, S_norm, n, m, alpha, beta, delta_init, ainf)
                abs(bet_saved - beta) <= beta * ainf[3] && break
                _, FI, _ = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
                bet_saved = beta
            end

            cors = 1.0 / (sqrt(beta) * sa)
            sa *= cors
            S_norm .*= cors

            _, FI, SIGMA = _reg_reg1(B, OMO, A_vec, n, alpha, beta, 2)
        end
    end

    FI = max.(FI, 0.0)
    return FI, SIGMA, (alpha = alpha, beta = beta)
end

"""
    solve_reconst(A, b, x0=nothing; E_MeV=nothing, pp=1e-3, alpha=-1.0,
                  beta=0.0, sigma_b=nothing)

Порт RECONST.FOR (STREG1): статистическая регуляризация Тургина
(Turchin). Решается `(B·β + Ω·α)·f = A_vec·β`.

- `pp` — параметр PP (вес строки Ω).
- `alpha` — регуляризация: >0 фиксирована, <0 авто (по умолчанию −1).
- `beta` — фиделити: >0 фиксировано, ≤0 авто (по умолчанию 0).
- `sigma_b` — погрешности измерений `m`; по умолчанию `sqrt(max(b,1e-10))`.
- `E_MeV`, `x0` — игнорируются (совместимость API).
"""
function solve_reconst(A::AbstractMatrix{Float64}, b::AbstractVector{Float64},
                       x0::Union{AbstractVector{Float64},Nothing}=nothing;
                       E_MeV::Union{AbstractVector{Float64},Nothing}=nothing,
                       pp::Real=1e-3,
                       alpha::Real=-1.0,
                       beta::Real=0.0,
                       sigma_b::Union{AbstractVector{Float64},Nothing}=nothing)
    M, N = size(A)
    m = M
    n = N
    F = copy(collect(float.(b)))

    S = sigma_b !== nothing ? max.(collect(float.(sigma_b)), 1e-300) :
        sqrt.(max.(F, 1e-10))

    ainf = copy(_RECONST_AINF)

    FI, SIGMA, ab = _reconst_streg1(float.(A), F, S, n, m, Float64(alpha),
                                    Float64(beta), Float64(pp), ainf)

    spectrum = max.(FI, 0.0)
    resid = b .- A * spectrum
    return UnfoldResult(spectrum, 0, true, norm(resid),
        Dict{String,Any}("pp" => Float64(pp),
                         "alpha" => ab.alpha,
                         "beta" => ab.beta,
                         "sigma" => SIGMA))
end
