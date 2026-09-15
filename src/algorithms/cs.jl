"""
    solve_omp(D, y, sparsity; tolerance=1e-6)

Orthogonal Matching Pursuit: поиск разреженного коэффициентного вектора
`alpha` (не более `sparsity` ненулевых компонент), аппроксимирующего
`y ≈ D * alpha`.

На каждом шаге выбирается атом словаря, наиболее коррелированный с
остатком, затем коэффициенты на текущей поддержке уточняются решением
МНК (`lstsq`).  Остановка — по исчерпанию sparsity или по остатку
меньше `tolerance`.

# Возвращает
Разреженный вектор коэффициентов длины `k = size(D, 2)`.
"""
function solve_omp(D::AbstractMatrix{T}, y::AbstractVector{T}, sparsity::Integer;
                   tolerance::T=T(1e-6)) where T<:AbstractFloat
    _, k = size(D)
    alpha = zeros(T, k)
    residual = copy(y)
    support = Int[]

    norms = vec(norm.(eachcol(D)))
    norms = [nz > 0 ? nz : one(T) for nz in norms]
    D_norm = D ./ reshape(norms, 1, k)

    for _ in 1:min(sparsity, k)
        correlations = abs.(D_norm' * residual)
        for idx in support
            correlations[idx] = T(-1)
        end
        idx = argmax(correlations)
        if correlations[idx] <= 0
            break
        end
        push!(support, idx)

        D_s = D[:, support]
        coefs = _lstsq(D_s, y)
        residual = y .- D_s * coefs

        if norm(residual) < tolerance
            break
        end
    end

    if !isempty(support)
        D_s = D[:, support]
        coefs = _lstsq(D_s, y)
        for (i, s) in enumerate(support)
            alpha[s] = coefs[i]
        end
    end

    return alpha
end

"""
    _lstsq(D, y)

Решение переопределённой/недоопределённой задачи МНК через SVD
(аналог `np.linalg.lstsq` с `rcond=None`).
"""
function _lstsq(D::AbstractMatrix{T}, y::AbstractVector{T}) where T<:AbstractFloat
    F = svd(Matrix(D))
    smax = F.S[1]
    rcond = eps(real(T)) * max(size(D)...) * (smax > 0 ? smax : one(T))
    sinv = [s > rcond ? inv(s) : zero(T) for s in F.S]
    return F.V * (sinv .* (F.U' * y))
end

"""
    solve_ksvd(signals, n_atoms; n_iterations=20, sparsity=5, random_state=nothing)

Обучение словаря алгоритмом K-SVD.

Сигналы подаются столбцами (`n × m`).  Словарь инициализируется
случайными обучающими сигналами с нормировкой столбцов; на каждой
итерации выполняется разреженное кодирование (OMP) и обновление атомов
последовательным SVD матриц ошибок (с сохранением разреженности
коэффициентов).  `random_state` задаёт воспроизводимость.

# Возвращает
Обученный словарь (`n × n_atoms`) с единичной нормой столбцов.
"""
function solve_ksvd(signals::AbstractMatrix{T}, n_atoms::Integer;
                    n_iterations::Integer=20,
                    sparsity::Integer=5,
                    random_state::Union{Nothing,Integer}=nothing) where T<:AbstractFloat
    rng = random_state === nothing ? Random.default_rng() : MersenneTwister(random_state)
    n, m = size(signals)

    n_atoms_eff = min(n_atoms, m)
    idx = sort(randperm(rng, m)[1:n_atoms_eff])
    D = Matrix{T}(signals[:, idx])

    norms = vec(norm.(eachcol(D)))
    norms = [nz > 0 ? nz : one(T) for nz in norms]
    D ./= reshape(norms, 1, n_atoms_eff)

    coefficients = zeros(T, n_atoms_eff, m)

    for _ in 1:n_iterations
        for j in 1:m
            coefficients[:, j] .= solve_omp(D, signals[:, j], sparsity)
        end

        for atom in 1:n_atoms_eff
            used = findall(!=(0), coefficients[atom, :])
            isempty(used) && continue

            D_restricted = copy(D)
            D_restricted[:, atom] .= T(0)
            E = signals[:, used] .- D_restricted * coefficients[:, used]

            F = svd(Matrix(E))
            new_atom = F.U[:, 1]
            new_coef = F.S[1] .* F.Vt[1, :]

            D[:, atom] .= new_atom
            coefficients[atom, used] .= new_coef
        end

        norms = vec(norm.(eachcol(D)))
        norms = [nz > 0 ? nz : one(T) for nz in norms]
        D ./= reshape(norms, 1, n_atoms_eff)
    end

    return D
end

"""
    solve_sl0(A, b; sigma_min=0.01, sigma_decrease_factor=0.5, mu_0=1.0,
              L=3, max_iterations=1000, tolerance=1e-6)

SL0 (Smoothed L0): восстановление разреженного решения
недоопределённой системы `b = A x`.

L0-норма аппроксимируется гауссовой суррогатной функцией
`Σ (1 - exp(-x²/(2σ²)))`; выполняется градиентный спуск с проекцией
на допустимое множество `{x : A x = b}` (через псевдообратную
матрицу), σ уменьшается геометрически от `2·max|x|` до `sigma_min`.

# Возвращает
Разреженный вектор `x` длины `n = size(A, 2)`.
"""
function solve_sl0(A::AbstractMatrix{T}, b::AbstractVector{T};
                   sigma_min::T=T(0.01),
                   sigma_decrease_factor::T=T(0.5),
                   mu_0::T=T(1.0),
                   L::Integer=3,
                   max_iterations::Integer=1000,
                   tolerance::T=T(1e-6)) where T<:AbstractFloat
    x = pinv(A) * b
    pinv_AT = A' * pinv(A * A')

    sigma = T(2.0) * maximum(abs.(x))
    sigma = sigma == 0 ? one(T) : sigma
    sigma = max(sigma, sigma_min)

    for _ in 1:max_iterations
        x_prev = copy(x)

        for _ in 1:L
            exp_term = exp.(-(x .^ 2) ./ (T(2.0) * sigma^2))
            @. x = x - mu_0 * x * exp_term
            x = x .- pinv_AT * (A * x .- b)
        end

        sigma *= sigma_decrease_factor
        sigma < sigma_min && break

        if norm(x .- x_prev) < tolerance * max(one(T), norm(x))
            break
        end
    end

    return x
end

"""
    solve_cs(A, b, x0=nothing; n_atoms=nothing, sparsity=nothing, dictionary=nothing,
             n_dictionary_iterations=20, sigma_min=0.01, sigma_decrease_factor=0.5,
             mu_0=1.0, L=3, max_iterations=1000, tolerance=1e-6, random_state=nothing)

Compressive Sensing (CS) развёртка нейтронного спектра.

Спектр `x` представляется разреженно в обученном словаре `D`:
`x = D * alpha`.  Уравнение измерений принимает вид
`b = (A * D) * alpha` и решается относительно разреженного `alpha`
алгоритмом SL0; спектр восстанавливается как `x = D * alpha` с
неотрицательной проекцией и нормировкой масштаба по данным.

Словарь обучается K-SVD на тренировочных сигналах (косинусный базис
плюс начальное приближение).  Можно передать готовый `dictionary`
(размер `n × n_atoms`) — тогда обучение пропускается.

# Возвращает
`UnfoldResult` с восстановленным спектром.
"""
function solve_cs(A::AbstractMatrix{T}, b::AbstractVector{T},
                  x0::Union{Nothing,AbstractVector{T}}=nothing;
                  n_atoms::Union{Nothing,Integer}=nothing,
                  sparsity::Union{Nothing,Integer}=nothing,
                  dictionary::Union{Nothing,AbstractMatrix{T}}=nothing,
                  n_dictionary_iterations::Integer=20,
                  sigma_min::T=T(0.01),
                  sigma_decrease_factor::T=T(0.5),
                  mu_0::T=T(1.0),
                  L::Integer=3,
                  max_iterations::Integer=1000,
                  tolerance::T=T(1e-6),
                  random_state::Union{Nothing,Integer}=nothing) where T<:AbstractFloat
    m, n = size(A)

    n_atoms_eff = n_atoms === nothing ? max(n, 2 * m) : n_atoms
    sparsity_eff = sparsity === nothing ? max(1, n ÷ 20) : sparsity

    base = if x0 !== nothing && any(x0 .!= 0)
        bx = max.(float.(x0), T(0))
        bx ./ (norm(bx) + T(1e-12))
    else
        fill(one(T) / sqrt(n), n)
    end

    t = collect(range(T(0), T(π), length=n))
    n_basis = min(n, max(2 * m, 8))
    signals = zeros(T, n, n_basis + 1)
    for i in 0:(n_basis - 1)
        col = cos.(i .* t)
        nrm = norm(col)
        nrm > 0 && (col ./= nrm)
        signals[:, i + 1] .= col
    end
    signals[:, end] .= base

    D = if dictionary !== nothing
        Dict_mat = Matrix{T}(dictionary)
        if size(Dict_mat, 1) != n
            throw(ArgumentError("Dictionary first dimension ($(size(Dict_mat, 1))) must match " *
                                "number of energy bins ($n)"))
        end
        Dict_mat
    else
        solve_ksvd(signals, n_atoms_eff;
                   n_iterations=n_dictionary_iterations,
                   sparsity=sparsity_eff,
                   random_state=random_state)
    end

    Phi = A * D

    alpha = solve_sl0(Phi, b;
                      sigma_min=sigma_min,
                      sigma_decrease_factor=sigma_decrease_factor,
                      mu_0=mu_0,
                      L=L,
                      max_iterations=max_iterations,
                      tolerance=tolerance)

    x = max.(D * alpha, T(0))

    computed = A * x
    if norm(computed) > 0 && norm(b) > 0
        scale = dot(b, computed) / (dot(computed, computed) + T(1e-12))
        x .*= scale
    end

    residual = norm(A * x .- b)
    converged = residual < tolerance * max(one(T), norm(b))
    return UnfoldResult(x, max_iterations, converged, residual,
                        Dict{String,Any}("n_atoms" => size(D, 2)))
end
