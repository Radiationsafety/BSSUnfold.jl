"""
Hybrid GMRES unfolding (по IRtools `IRhybrid_gmres`, Gazzola et al.).

Бидиагонализация Гольба-Кахана от остатка `r0 = b - A x0` строит
Krylov-базисы U (пространство спектра) и V (пространство данных); на
каждой глубине k рассматривается проекционная задача

    min || B_k y - beta0*e1 ||^2 + lambda * ||y||^2,

которая решается как Tikhonov-задача (B_k — bidiagonal, p x k), причём
`lambda` подбирается автоматически по GCV (`"gcv"`/`"modgcv"`), по
принципу невязки (`"discrep"`, threshold = `eta * noise_level * ||b||`,
бинарный	SCOPEuскальный подбор делением/умножением вдвое) или
фиксируется (`"manual"`).  Активна полная реортогонализация
(`reorthogonalization::Bool`); останов — рост текущего минимума GCV
более чем на 1% (GCV stabilization).  Лучший раствор — с минимальным
GCV; неотрицательность спектра обеспечена `max.(x, 0)`.
"""

function _hybgmres_gcv(lambda_val::Float64, B_k::Matrix{Float64}, beta::Vector{Float64})
    k = size(B_k, 2)
    if lambda_val < 1e-14
        x_lambda = pinv(B_k) * beta[1:k]
        residual = beta[1:k] .- B_k * x_lambda
    else
        B_reg = vcat(B_k, lambda_val .* Matrix{Float64}(I, k, k))
        rhs = vcat(beta, zeros(k))
        x_lambda = pinv(B_reg) * rhs
        residual = beta .- B_k * x_lambda
    end
    numerator = sum(abs2, residual)
    if k < 50
        denominator = (length(beta) - tr(B_k * pinv(B_k)))^2
    else
        denominator = max(length(beta) * 0.1, 1.0)
    end
    denominator < 1e-10 && return 1e10
    return numerator / denominator
end

function _hybgmres_regsolve(B_k::Matrix{Float64}, rhs_proj::Vector{Float64}, lam::Float64)
    L_reg = vcat(B_k, lam .* Matrix{Float64}(I, size(B_k, 2), size(B_k, 2)))
    rhs_reg = vcat(rhs_proj, zeros(size(B_k, 2)))
    return pinv(L_reg) * rhs_reg
end

"""
    solve_hybrid_gmres(A, b, x0=nothing; max_iterations=90,
                       regularization_method="gcv", regularization=0.0,
                       noise_level=nothing, eta=1.01, reorthogonalization=true)
      -> UnfoldResult

Гибридный GMRES: на каждой глубине Krylov-пространства решается
регуляризованная проекция, параметр регуляризации подбирается по GCV /
принципу невязки / вручную; лучший раствор выбирается по минимальному
GCV.  `x0 = nothing` означает нулевой старт; если остаток `b - A x0`
пренебрежимо мал, возвращается `x0` (обрезанный сверху нулём) с нулевой
невязкой.  Параметр подпишет history GCV / lambda в `extra`.
"""
function solve_hybrid_gmres(A::AbstractMatrix, b::AbstractVector, x0::Union{Nothing,AbstractVector}=nothing;
                            max_iterations::Integer=90,
                            regularization_method::String="gcv",
                            regularization::Real=0.0,
                            noise_level::Union{Nothing,Real}=nothing,
                            eta::Real=1.01,
                            reorthogonalization::Bool=true)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    n_detectors, n_energy = size(AF)
    max_krylov = min(Int(max_iterations), n_detectors)
    x_init = x0 === nothing ? zeros(n_energy) : Vector{Float64}(x0)

    r0 = bf .- AF * x_init
    b0 = norm(r0)

    if b0 < 1e-14
        spectrum = max.(x_init, 0.0)
        residual = bf .- AF * spectrum
        return UnfoldResult(spectrum, 0, true, norm(residual))
    end

    U = zeros(n_energy, max_krylov + 1)
    V = zeros(n_detectors, max_krylov + 1)
    alpha = zeros(max_krylov + 1)
    betas = zeros(max_krylov + 1)

    betas[1] = b0
    V[:, 1] .= r0 ./ b0
    u1 = AF' * V[:, 1]
    alpha[1] = norm(u1)
    if alpha[1] < 1e-14
        spectrum = max.(x_init, 0.0)
        residual = bf .- AF * spectrum
        return UnfoldResult(spectrum, 0, true, norm(residual))
    end
    U[:, 1] .= u1 ./ alpha[1]

    gcv_values = Float64[]
    reg_params = Float64[]
    residual_norms = Float64[]
    solution_norms = Float64[]
    best_solution = nothing
    best_gcv = Inf
    best_gcv_val = Inf
    stop_iteration = max_krylov
    actual_k = max_krylov

    for kk in 0:(max_krylov-1)
        v_new = AF * U[:, kk+1] .- alpha[kk+1] .* V[:, kk+1]
        if reorthogonalization && kk > 0
            for j in 1:kk
                v_new .-= dot(V[:, j], v_new) .* V[:, j]
            end
        end
        betas[kk+2] = norm(v_new)

        if betas[kk+2] < 1e-14
            actual_k = kk + 1
            break
        end
        V[:, kk+2] .= v_new ./ betas[kk+2]

        u_new = AF' * V[:, kk+2] .- betas[kk+2] .* U[:, kk+1]
        if reorthogonalization && kk > 0
            for j in 1:kk
                u_new .-= dot(U[:, j], u_new) .* U[:, j]
            end
        end
        if kk < max_krylov - 1
            alpha[kk+2] = norm(u_new)
            if alpha[kk+2] < 1e-14
                actual_k = kk + 1
                break
            end
            U[:, kk+2] .= u_new ./ alpha[kk+2]
        end

        current_k = kk + 1

        B_k = zeros(current_k + 1, current_k)
        for i in 1:current_k
            B_k[i, i] = alpha[i]
            B_k[i+1, i] = betas[i+1]
        end
        rhs_proj = zeros(current_k + 1)
        rhs_proj[1] = betas[1]

        lam = Float64(regularization)
        method = lowercase(regularization_method)
        if method in ("gcv", "modgcv")
            lambda_candidates = 10.0 .^ range(-10, 2, length=50)
            local_best_lambda = Float64(regularization)
            local_best_gcv = Inf
            for lc in lambda_candidates
                gv = _hybgmres_gcv(lc, B_k, rhs_proj)
                if gv < local_best_gcv
                    local_best_gcv = gv
                    local_best_lambda = lc
                end
            end
            lam = local_best_lambda
            best_gcv_val = local_best_gcv
            push!(gcv_values, best_gcv_val)

            if length(gcv_values) >= 3
                recent_min = minimum(gcv_values[end-2:end])
                if best_gcv_val > recent_min * 1.01
                    stop_iteration = current_k
                end
            end
        elseif method == "discrep" && noise_level !== nothing
            threshold = Float64(eta) * Float64(noise_level) * norm(bf)
            lam = max(Float64(regularization), 1e-12)
            for _ in 1:20
                y_try = _hybgmres_regsolve(B_k, rhs_proj, lam)
                x_try = x_init .+ U[:, 1:current_k] * y_try
                rn = norm(AF * x_try .- bf)
                if rn > threshold
                    lam *= 2
                else
                    lam /= 2
                end
            end
            best_gcv_val = Inf
        else
            lam = Float64(regularization)
        end
        push!(reg_params, lam)

        y_lambda = _hybgmres_regsolve(B_k, rhs_proj, lam)
        x_lambda = x_init .+ U[:, 1:current_k] * y_lambda
        res_norm = norm(AF * x_lambda .- bf)
        push!(residual_norms, res_norm)
        push!(solution_norms, norm(x_lambda))

        if method in ("gcv", "modgcv")
            if best_gcv_val < best_gcv
                best_gcv = best_gcv_val
                best_solution = copy(x_lambda)
            end
        else
            best_solution = copy(x_lambda)
        end

        if stop_iteration <= current_k
            break
        end
    end

    spectrum = best_solution === nothing ? max.(x_init, 0.0) : max.(best_solution, 0.0)
    residual = bf .- AF * spectrum
    extra = Dict{String,Any}(
        "method" => "Hybrid_GMRES",
        "regularization_parameters" => reg_params,
        "gcv_values" => gcv_values,
        "solution_norms" => solution_norms,
        "residual_norms_history" => residual_norms,
    )
    return UnfoldResult(spectrum, stop_iteration, false, norm(residual), extra)
end
