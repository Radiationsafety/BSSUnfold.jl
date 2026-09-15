"""
Non-negative K-SVD unfolding (Xu et al., NIM A, 2026, BNCT).

Двухступенчатый пайплайн: (1) K-SVD с неотрицательным усечением атомов
словаря, (2) разреженное кодирование по выученному словарю.  Поддержаны
три стратегии разреженного кодирования: `nnls_topk` (глобальный NNLS
→ top-K скрининг → локальный NNLS), `omp` (классический OMP с
неограниченным LS, заменяет nnmp) и `nn_omp` (OMP с NNLS-шагом).
Регуляризованная NNLS (Eq. 2.5) решается через дополненную матрицу
(Eq. 2.6) вызовом `solve_nnls(A, b)` пакета BSSUnfold.
"""
function _nnksvd_normalize_columns(D::Matrix{Float64})
    norms = vec(sqrt.(sum(abs2, D, dims=1)))
    norms = replace(norms, 0.0 => 1.0)
    return D ./ reshape(norms, 1, :)
end

"""
    solve_tikhonov_nnls(M_norm, y; lambda_tik=0.01, prior_wt=0.0,
                        alpha_prior=nothing, max_iter=nothing) -> Vector{Float64}

Тихоновская NNLS (Eq. 2.5 article) via дополненной матрицы (Eq. 2.6):

    min || [y; 0] - [M_norm; sqrt(lambda_tik) I] alpha ||^2,  alpha >= 0

При `prior_wt > 0` добавляется ограничение на отклонение от
`alpha_prior` (training-sample-driven prior, section 2.2.2).
"""
function solve_tikhonov_nnls(M_norm::AbstractMatrix, y::AbstractVector;
                             lambda_tik::Real=0.01,
                             prior_wt::Real=0.0,
                             alpha_prior::Union{Nothing,AbstractVector}=nothing,
                             max_iter::Union{Nothing,Integer}=nothing)
    M = Matrix{Float64}(M_norm)
    yv = Vector{Float64}(y)
    p = size(M, 2)

    A_aug = vcat(M, sqrt(max(Float64(lambda_tik), 0.0)) .* Matrix{Float64}(I, p, p))
    b_aug = vcat(yv, zeros(p))

    if prior_wt > 0.0
        alpha_prior === nothing &&
            throw(ArgumentError("alpha_prior must be provided when prior_wt > 0"))
        ap = Vector{Float64}(alpha_prior)
        length(ap) == p || throw(ArgumentError("alpha_prior length ($(length(ap))) must match number of dictionary atoms ($p)"))
        A_aug = vcat(A_aug, sqrt(Float64(prior_wt)) .* Matrix{Float64}(I, p, p))
        b_aug = vcat(b_aug, sqrt(Float64(prior_wt)) .* ap)
    end

    return solve_nnls(A_aug, b_aug)
end

"""
    solve_nn_omp(D, y, sparsity; tolerance=1e-6) -> Vector{Float64}

Неотрицательный Orthogonal Matching Pursuit: на каждом шаге выбирается
атом с наибольшей *положительной* проекцией на остаток, затем NNLS на
выбранной опоре (вместо неограниченного LS).
"""
function solve_nn_omp(D::AbstractMatrix, y::AbstractVector, sparsity::Integer;
                      tolerance::Real=1e-6)
    Df = Matrix{Float64}(D)
    yv = Vector{Float64}(y)
    n, p = size(Df)
    alpha = zeros(p)
    residual = copy(yv)
    support = Int[]

    norms = vec(sqrt.(sum(abs2, Df, dims=1)))
    norms = replace(norms, 0.0 => 1.0)
    D_norm = Df ./ reshape(norms, 1, :)

    for _ in 1:min(Int(sparsity), p)
        correlations = D_norm' * residual
        for idx in support
            correlations[idx] = -Inf
        end
        mv, mk = findmax(correlations)
        if mv <= 0 || !isfinite(mv)
            break
        end
        push!(support, mk)

        D_s = Df[:, support]
        coefs = solve_nnls(D_s, yv)
        alpha[support] = coefs
        residual = yv .- D_s * coefs

        norm(residual) < tolerance && break
    end

    return alpha
end

function _nnksvd_omp(D::AbstractMatrix, y::AbstractVector, sparsity::Integer; tolerance::Real=1e-6)
    Df = Matrix{Float64}(D)
    yv = Vector{Float64}(y)
    n, p = size(Df)
    alpha = zeros(p)
    residual = copy(yv)
    support = Int[]

    for _ in 1:min(Int(sparsity), p)
        correlations = Df' * residual
        for idx in support
            correlations[idx] = -Inf
        end
        mv, mk = findmax(correlations)
        if mv <= 0 || !isfinite(mv)
            break
        end
        push!(support, mk)
        D_s = Df[:, support]
        coefs = D_s \ yv
        alpha[support] = coefs
        residual = yv .- D_s * coefs
        norm(residual) < tolerance && break
    end

    return alpha
end

"""
    solve_nnls_topk(M_norm, y, sparsity; lambda_tik=0.01, prior_wt=0.0,
                    alpha_prior=nothing, max_iter=nothing) -> Vector{Float64}

NNLS+TopK (proposed method статьи Xu et al. 2026): (1) глобальный NNLS
черновой раствор, (2) скрининг top-`K` атомов по величине коэффициентов,
(3) локальный NNLS на отобранной поддержки для уточнённого
K-разреженного неотрицательного решения.
"""
function solve_nnls_topk(M_norm::AbstractMatrix, y::AbstractVector, sparsity::Integer;
                         lambda_tik::Real=0.01,
                         prior_wt::Real=0.0,
                         alpha_prior::Union{Nothing,AbstractVector}=nothing,
                         max_iter::Union{Nothing,Integer}=nothing)
    M = Matrix{Float64}(M_norm)
    yv = Vector{Float64}(y)
    p = size(M, 2)
    K = max(0, min(Int(sparsity), p))
    K == 0 && return zeros(p)

    alpha_full = solve_tikhonov_nnls(M, yv; lambda_tik=lambda_tik,
                                     prior_wt=prior_wt, alpha_prior=alpha_prior,
                                     max_iter=max_iter)

    K >= p && return alpha_full
    topk_idx = sortperm(alpha_full, rev=true)[1:K]

    M_topk = M[:, topk_idx]
    alpha_prior_topk = nothing
    if prior_wt > 0.0 && alpha_prior !== nothing
        alpha_prior_topk = Vector{Float64}(alpha_prior)[topk_idx]
    end
    alpha_topk = solve_tikhonov_nnls(M_topk, yv; lambda_tik=lambda_tik,
                                     prior_wt=prior_wt, alpha_prior=alpha_prior_topk,
                                     max_iter=max_iter)

    alpha = zeros(p)
    alpha[topk_idx] = alpha_topk
    return alpha
end

"""
    solve_nnksvd_dictionary(signals, n_atoms; n_iterations=80, sparsity=2,
                            lambda_tik=0.01, prior_wt=0.5,
                            sparse_coder="nnls_topk", random_state=nothing,
                            tolerance=1e-6) -> (D, alpha_prior)

Неотрицательное K-SVD обучение словаря (аналог python `solve_nnksvd`):
разреженное кодирование выбранной стратегией + обновление словаря
через rank-1 SVD усечения ошибки с неотрицательным усечением атома и
коэффициентов.  Возвращает нормированный словарь `D` (n x p) и средний
разреженный код `alpha_prior` как training-sample-driven prior.
"""
function solve_nnksvd_dictionary(signals::AbstractMatrix, n_atoms::Integer;
                                 n_iterations::Integer=80,
                                 sparsity::Integer=2,
                                 lambda_tik::Real=0.01,
                                 prior_wt::Real=0.5,
                                 sparse_coder::String="nnls_topk",
                                 random_state::Union{Nothing,Integer}=nothing,
                                 tolerance::Real=1e-6)
    sparse_coder in ("nnls_topk", "omp", "nn_omp") ||
        throw(ArgumentError("Unknown sparse_coder '$sparse_coder'. Expected 'nnls_topk', 'omp' or 'nn_omp'."))

    S = Matrix{Float64}(signals)
    n, m = size(S)
    S = max.(S, 0.0)

    rng = MersenneTwister(random_state === nothing ? 0 : Int(random_state))

    p = max(1, min(Int(n_atoms), m))
    idx = randperm(rng, m)[1:p]
    D = S[:, idx]
    col_norms = vec(sqrt.(sum(abs2, D, dims=1)))
    for j in findall(==(0.0), col_norms)
        bump = max.(randn(rng, n), 0.0)
        norm(bump) == 0 && (bump = ones(n))
        D[:, j] = bump
    end
    D = _nnksvd_normalize_columns(D)

    coefficients = zeros(p, m)

    for _ in 1:Int(n_iterations)
        D_prev = copy(D)

        for j in 1:m
            y_j = S[:, j]
            if sparse_coder == "omp"
                coefficients[:, j] = _nnksvd_omp(D, y_j, Int(sparsity), tolerance=tolerance)
            elseif sparse_coder == "nn_omp"
                coefficients[:, j] = solve_nn_omp(D, y_j, Int(sparsity), tolerance=tolerance)
            else
                coefficients[:, j] = solve_nnls_topk(D, y_j, Int(sparsity), lambda_tik=Float64(lambda_tik), prior_wt=0.0)
            end
        end

        for atom in 1:p
            used = findall(!=(0.0), coefficients[atom, :])
            if isempty(used)
                j_new = rand(rng, 1:m)
                new_atom = max.(S[:, j_new], 0.0)
                nm = norm(new_atom)
                if nm == 0
                    new_atom = ones(n)
                    nm = sqrt(n)
                end
                D[:, atom] = new_atom ./ nm
                continue
            end

            D_restricted = copy(D)
            D_restricted[:, atom] .= 0.0
            E = S[:, used] .- D_restricted * coefficients[:, used]

            Uk, sk, Vtk = svd(E, full=false)
            new_atom = max.(Uk[:, 1], 0.0)
            new_coef = max.(sk[1] .* (Vtk[1, :]), 0.0)

            nm = norm(new_atom)
            if nm > 0
                scale = nm
                D[:, atom] = new_atom ./ nm
                for (k, jj) in enumerate(used)
                    coefficients[atom, jj] = new_coef[k] * scale
                end
            else
                j_new = rand(rng, 1:m)
                new_atom = max.(S[:, j_new], 0.0)
                nm = norm(new_atom)
                if nm == 0
                    new_atom = ones(n)
                    nm = sqrt(n)
                end
                D[:, atom] = new_atom ./ nm
                for jj in used
                    coef_u = solve_nnls(reshape(D[:, atom], :, 1), S[:, jj])
                    coefficients[atom, jj] = coef_u[1]
                end
            end
        end

        norm(D .- D_prev) < tolerance * max(1.0, norm(D_prev)) && break
    end

    D = _nnksvd_normalize_columns(max.(D, 0.0))
    alpha_prior = vec(mean_dim1(coefficients))
    return D, alpha_prior
end

function mean_dim1(X::Matrix{Float64})
    return vec(sum(X, dims=2)) ./ size(X, 2)
end

function _nnksvd_training_signals(n::Int, n_basis::Int)
    signals = zeros(n, n_basis + 1)
    t = collect(range(0.0, 1.0, length=n))
    nb = max(1, n_basis)
    for i in 1:nb
        center = (i) / (nb + 1)
        col = @. exp(-((t - center)^2) / (2 * (1.0 / nb)^2))
        nm = norm(col)
        nm > 0 && (col = col ./ nm)
        signals[:, i] = col
    end
    return signals
end

function _nnksvd_training_signals_log(n::Int, n_basis::Int, E_MeV::Vector{Float64})
    log_E = log10.(max.(E_MeV, 1e-15))
    width = max((log_E[end] - log_E[1]) / max(n_basis * 1.5, 1.0), 1e-6)
    signals = zeros(n, n_basis + 1)
    centers = range(log_E[1], log_E[end], length=n_basis)
    for (i, c) in enumerate(centers)
        col = @. exp(-((log_E - $c)^2) / (2 * $width^2))
        nm = norm(col)
        nm > 0 && (col = col ./ nm)
        signals[:, i] = col
    end
    return signals
end

function _nnksvd_equivalent_dictionary(R::Matrix{Float64}, D::Matrix{Float64})
    M = R * D
    return _nnksvd_normalize_columns(M)
end

"""
    solve_nnksvd(A, b, x0=nothing; n_atoms=15, sparsity=2, dictionary=nothing,
                 training_signals=nothing, n_dictionary_iterations=80,
                 lambda_tik=0.01, prior_wt=0.5, sparse_coder="nnls_topk",
                 random_state=nothing, tolerance=1e-6, E_MeV=nothing)
      -> UnfoldResult

Развёртка через пайплайн non-negative K-SVD: спектр представляется как
`phi = D @ alpha` по выученному неотрицательному словарю `D`
(online-обучение при отсутствии `dictionary`/`training_signals`;
training signals — лог-расставленные гауссовы бампы по энергосетке).
Разреженное кодирование выполняется на эквивалентном словаре детектора
`M_norm = normalize(A @ D)` выбранной стратегией.  Компенсация
нормировки словаря + выравнивание масштаба по показаниям.
"""
function solve_nnksvd(A::AbstractMatrix, b::AbstractVector, x0::Union{Nothing,AbstractVector}=nothing;
                      n_atoms::Integer=15,
                      sparsity::Integer=2,
                      dictionary::Union{Nothing,AbstractMatrix}=nothing,
                      training_signals::Union{Nothing,AbstractMatrix}=nothing,
                      n_dictionary_iterations::Integer=80,
                      lambda_tik::Real=0.01,
                      prior_wt::Real=0.5,
                      sparse_coder::String="nnls_topk",
                      random_state::Union{Nothing,Integer}=nothing,
                      tolerance::Real=1e-6,
                      E_MeV::Union{Nothing,AbstractVector}=nothing)
    random_state === nothing && (random_state = 42)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    m, n = size(AF)

    alpha_prior = nothing
    if dictionary !== nothing
        D = Matrix{Float64}(dictionary)
        size(D, 1) == n || throw(ArgumentError("Dictionary first dimension ($(size(D,1))) must match the number of energy bins ($n)."))
        D = _nnksvd_normalize_columns(max.(D, 0.0))
    else
        if training_signals !== nothing
            signals = max.(Matrix{Float64}(training_signals), 0.0)
            size(signals, 1) == n || throw(ArgumentError("Training signals first dimension ($(size(signals,1))) must match the number of energy bins ($n)."))
        else
            base = x0 === nothing ? ones(n) ./ sqrt(n) : max.(Vector{Float64}(x0), 0.0)
            nm0 = norm(base)
            base = nm0 > 0 ? base ./ nm0 : ones(n) ./ sqrt(n)

            nb = max(Int(n_atoms) * 2, 8)
            nb = min(nb, n)
            if E_MeV !== nothing && any(>(0), collect(E_MeV))
                signals = _nnksvd_training_signals_log(n, nb, collect(Float64.(E_MeV)))
            else
                signals = _nnksvd_training_signals(n, nb)
            end
            signals[:, end] = base
        end

        D, alpha_prior = solve_nnksvd_dictionary(signals, n_atoms;
                                                 n_iterations=n_dictionary_iterations,
                                                 sparsity=sparsity,
                                                 lambda_tik=lambda_tik,
                                                 prior_wt=prior_wt,
                                                 sparse_coder=sparse_coder,
                                                 random_state=random_state,
                                                 tolerance=tolerance)
    end

    M = _nnksvd_equivalent_dictionary(AF, D)
    effective_prior_wt = alpha_prior === nothing ? 0.0 : Float64(prior_wt)

    if sparse_coder == "nnls_topk"
        alpha = solve_nnls_topk(M, bf, Int(sparsity); lambda_tik=Float64(lambda_tik),
                                prior_wt=effective_prior_wt, alpha_prior=alpha_prior)
    elseif sparse_coder == "omp"
        alpha = _nnksvd_omp(M, bf, Int(sparsity), tolerance=tolerance)
    elseif sparse_coder == "nn_omp"
        alpha = solve_nn_omp(M, bf, Int(sparsity), tolerance=tolerance)
    else
        throw(ArgumentError("Unknown sparse_coder '$sparse_coder'. Expected 'nnls_topk', 'omp' or 'nn_omp'."))
    end

    unnorm = AF * D
    atom_norms = vec(sqrt.(sum(abs2, unnorm, dims=1)))
    atom_norms = replace(atom_norms, 0.0 => 1.0)
    phi = D * (alpha .* atom_norms)
    phi = max.(phi, 0.0)

    computed = AF * phi
    if norm(computed) > 0 && norm(bf) > 0
        scale = dot(bf, computed) / (dot(computed, computed) + 1e-12)
        scale > 0 && (phi = phi .* scale)
    end

    residual = norm(AF * phi .- bf)
    converged = residual < tolerance * max(1.0, norm(bf))
    extra = Dict{String,Any}("n_atoms" => n, "sparse_coder" => sparse_coder)
    return UnfoldResult(phi, n_dictionary_iterations, converged, residual, extra)
end
