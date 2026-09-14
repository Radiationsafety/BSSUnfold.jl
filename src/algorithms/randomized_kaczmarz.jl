"""
Randomized Kaczmarz unfolding method.

Порт из `bssunfold/src/bssunfold/core/unfold_randomized_kaczmarz.py`.

Рандомизированный вариант алгоритма Качмажа: строки выбираются
вероятностно с вероятностью, пропорциональной квадрату нормы строки.
Это даёт более быструю сходимость на плохо обусловленных системах.

Ссылка: Strohmer & Vershynin (2009), "A Randomized Kaczmarz Algorithm
with Exponential Convergence".
"""

"""
    solve_randomized_kaczmarz(A, b, x0; max_iterations, omega, tolerance, random_state)

Рандомизированный алгоритм Качмажа.

# Аргументы
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: измерения (m,)
- `x0::AbstractVector{T}`: начальный спектр (n,)
- `max_iterations`: макс. число итераций (default 1000)
- `omega::T`: параметр релаксации, 0 < ω ≤ 2 (default 1.0)
- `tolerance::T`: критерий сходимости `||x_k - x_{k-1}||` после полного прохода
- `random_state`: seed для воспроизводимости (по умолч. `nothing` = случайный)

# Возвращает
- `UnfoldResult{T}` со спектром
"""
function solve_randomized_kaczmarz(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                                  max_iterations::Integer=1000,
                                  omega::T=T(1.0),
                                  tolerance::T=T(1e-6),
                                  random_state::Union{Integer,Nothing}=nothing,
                                  eps::T=T(1e-30)) where T<:AbstractFloat
    m, n = size(A)
    x = max.(copy(x0), T(0))

    rng = random_state === nothing ? MersenneTwister() : MersenneTwister(random_state)

    # Квадраты норм строк для вероятностного выбора
    row_norms_sq = vec(sum(A .^ 2, dims=2))
    total_norm_sq = sum(row_norms_sq)
    if total_norm_sq == 0
        return UnfoldResult(x, 0, true, T(0))
    end
    probabilities = row_norms_sq ./ total_norm_sq

    # Кумулятивные вероятности для выборки
    cum_probs = cumsum(probabilities)

    converged = false
    iterations = 0
    x_old = copy(x)

    @inbounds for k in 1:max_iterations
        # Сэмплировать индекс строки
        u = rand(rng)
        i = searchsortedfirst(cum_probs, u)
        i = clamp(i, 1, m)

        if row_norms_sq[i] > eps
            ai = view(A, i, :)
            update = (b[i] - dot(ai, x)) / row_norms_sq[i]
            x .+= omega .* update .* ai
            x .= max.(x, T(0))
        end

        # Проверка сходимости после каждого полного цикла (m итераций)
        if k % m == 0
            diff_norm = norm(x .- x_old)
            if diff_norm < tolerance
                converged = true
                iterations = k
                break
            end
            x_old .= x
        end
    end

    if !converged
        iterations = max_iterations
    end

    residual = b .- A * x
    return UnfoldResult(x, iterations, converged, norm(residual))
end
