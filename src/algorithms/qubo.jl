"""
QUBO-based neutron spectrum unfolding using quantum-inspired annealing
(порт из unfold_qubo.py, pyqubo + dwave-neal → нативный симулированный отжиг).

Модуль реализует QUBO (Quadratic Unconstrained Binary Optimization)
формулировку задачи развёртки нейтронных спектров, решаемую
квантово-вдохновлённым симулированным отжигом (в Python-оригинале —
D-Wave Neal) или другими QUBO-солверами.

Подход дискретизирует спектр в бинарные переменные и формулирует
развёртку как:

    min_x ||A x - b||^2 + λ * R(x)      subject to x >= 0,

где спектр представлен в бинарном кодировании для QUBO-совместимости.

Каждый энергетический бин кодируется `n_bits` бинарными переменными
(дробное бинарное разложение): `x[i] = max_value * Σ_j b_ij * 2^-(j+1)`.

Задача сводится к QUBO-гамильтониану

    E(q) = q' Q q + l' q,
    Q = (A T)' (A T) + regularization * I,   l = -2 (A T)' b,

где T — матрица перехода от битов к непрерывному спектру.

В Python-оригинале использовались `pyqubo` (компиляция гамильтониана) и
`dwave-neal` (симулированный отжиг).  В Julia-порте оба этапа реализованы
**нативно** (без внешних зависимостей): гамильтониан собирается матрично,
а сэмплер — классический симулированный отжиг с Metropolis-критерием и
геометрическим температурным расписанием, адаптированным к масштабу
энергий задачи (аналог SimulatedAnnealingSampler из dwave-neal).
"""

# ─── Бинарное кодирование (порт _spectrum_to_binary/_binary_to_spectrum) ────

"""
    spectrum_to_binary(spectrum; n_bits=8, max_value=nothing) -> Vector{Int}

Перевести непрерывный спектр в бинарное представление
(длина `n_bins * n_bits`).
"""
function spectrum_to_binary(spectrum::AbstractVector{<:Real};
                           n_bits::Integer=8,
                           max_value::Union{Nothing,Real}=nothing)
    mv = max_value === nothing ? maximum(spectrum) : Float64(max_value)
    mv <= 0 && (mv = 1.0)
    n_bins = length(spectrum)
    binary = zeros(Int, n_bins * n_bits)
    for i in 1:n_bins
        val = clamp(spectrum[i] / mv, 0.0, 1.0)
        for j in 1:n_bits
            # NB: как и в Python-оригинале, для val == 1.0 первая «цифра»
            # будет 2 (int(2.0)); decoder binary_to_spectrum обрабатывает
            # это корректно (2 * 2^-1 = 1.0).
            bit = floor(Int, val * 2)
            binary[(i - 1) * n_bits + j] = bit
            val = (val * 2) % 1
        end
    end
    return binary
end

"""
    binary_to_spectrum(binary, n_bins; n_bits=6, max_value=1.0) -> Vector{Float64}

Перевести бинарное представление обратно в непрерывный спектр.
"""
function binary_to_spectrum(binary::AbstractVector{<:Real}, n_bins::Integer;
                           n_bits::Integer=6, max_value::Real=1.0)
    spectrum = zeros(n_bins)
    for i in 1:n_bins
        val = 0.0
        for j in 1:n_bits
            val += binary[(i - 1) * n_bits + j] * 2.0^(-j)
        end
        spectrum[i] = val * max_value
    end
    return max.(spectrum, 0.0)
end

# ─── Симулированный отжиг ───────────────────────────────────────────────────

"""
    _simulated_annealing_qubo(Q, l; num_reads=10, num_sweeps=1000, rng)

Симулированный отжиг для QUBO `E(q) = q'Qq + l'q`, q ∈ {0,1}^N.

Metropolis-критерий с геометрическим температурным расписанием,
адаптированным к масштабу задачи: T0 оценивается по разбросу энергий
случайных состояний, T1 = 1e-4 * T0.  Возвращает (лучший q, лучшая энергия).
"""
function _simulated_annealing_qubo(Q::AbstractMatrix{Float64},
                                  l::Vector{Float64};
                                  num_reads::Integer=10,
                                  num_sweeps::Integer=1000,
                                  rng::AbstractRNG=MersenneTwister())
    N = length(l)

    # Начальная температура: разброс энергий случайных состояний
    sample_energies = Float64[]
    for _ in 1:min(50, max(10, num_reads))
        q = rand(rng, 0:1, N)
        E = dot(q, Q * q) + dot(l, q)
        push!(sample_energies, E)
    end
    E_std = isempty(sample_energies) ? 1.0 : std(sample_energies)
    T0 = max(E_std * 1.5, 1e-3)
    T1 = T0 * 1e-4
    rate = (T1 / T0)^(1 / max(num_sweeps - 1, 1))

    best_q = zeros(Int, N)
    best_E = Inf

    for _read in 1:num_reads
        q = rand(rng, 0:1, N)
        h = Q * q .+ l          # h_i = (Qq)_i + l_i
        E = dot(q, Q * q) + dot(l, q)
        T = T0
        for _sweep in 1:num_sweeps
            for i in 1:N
                # ΔE переворота бита i: (1-2q_i) * (2h_i - l_i + Q_ii)
                dE = (1 - 2 * q[i]) * (2 * h[i] - l[i] + Q[i, i])
                if dE <= 0 || rand(rng) < exp(-dE / T)
                    flip = 1 - 2 * q[i]
                    q[i] = 1 - q[i]
                    E += dE
                    # Инкрементальное обновление h: h += Q[:, i] * flip
                    axpy!(flip, view(Q, :, i), h)
                end
            end
            T *= rate
        end
        if E < best_E
            best_E = E
            best_q = copy(q)
        end
    end
    return best_q, best_E
end

# ─── Основной солвер ────────────────────────────────────────────────────────

"""
    solve_qubo(A, b, x0; n_bits=6, max_value=nothing, regularization=0.01,
               max_iterations=1000, annealing_time=1000, num_reads=10,
               random_state=nothing) -> UnfoldResult

Решить задачу развёртки через QUBO-формулировку с симулированным отжигом.

# Аргументы
- `A::AbstractMatrix{T}`: ответная матрица (m × n)
- `b::AbstractVector{T}`: измерения (m,)
- `x0::AbstractVector{T}`: начальная догадка для масштабирования
  (используется для оценки `max_value`, если он не задан)
- `n_bits`: бит на энергетический бин (default 6)
- `max_value`: максимальное значение спектра для масштабирования;
  `nothing` — оценка из данных (2 * max(x0) или псевдоинверсия)
- `regularization`: параметр регуляризации (default 0.01)
- `max_iterations`: максимальное число итераций (возвращается как
  `iterations`, default 1000)
- `annealing_time`: число проходов отжига (sweeps, default 1000)
- `num_reads`: число независимых считываний (default 10)
- `random_state`: seed для воспроизводимости

# Возвращает
`UnfoldResult` со спектром; в `extra` — `n_bits`, `energy`, `num_reads`,
`annealing_time`.
"""
function solve_qubo(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   n_bits::Integer=6,
                   max_value::Union{Nothing,Real}=nothing,
                   regularization::Real=0.01,
                   max_iterations::Integer=1000,
                   annealing_time::Integer=1000,
                   num_reads::Integer=10,
                   random_state::Union{Integer,Nothing}=nothing) where T<:AbstractFloat
    m, n_bins = size(A)
    length(b) == m || throw(ArgumentError("b length ($(length(b))) must match A rows ($m)"))
    n_bits >= 1 || throw(ArgumentError("n_bits must be >= 1, got $n_bits"))

    rng = random_state === nothing ? MersenneTwister() : MersenneTwister(Int(random_state))

    # Оценка max_value, если не задана
    mv = if max_value !== nothing
        Float64(max_value)
    elseif any(>(0), x0)
        2 * maximum(x0)
    else
        # Грубая оценка из псевдоинверсии
        try
            x_est = qr(A, ColumnNorm()) \ b
            2 * maximum(abs.(x_est))
        catch
            1.0
        end
    end
    mv <= 0 && (mv = 1.0)

    # Матрица перехода битов в непрерывный спектр:
    # x_cont[i] = Σ_j (binary[i*n_bits + j] * 2^-(j+1)) * max_value
    n_binary = n_bins * n_bits
    Tmat = zeros(n_bins, n_binary)
    for i in 1:n_bins, j in 1:n_bits
        Tmat[i, (i - 1) * n_bits + j] = 2.0^(-j) * mv
    end

    # Эффективная ответная матрица на вектор битов: A_scaled = A * Tmat
    A_scaled = A * Tmat

    # QUBO-гамильтониан: ||A_scaled q - b||^2 + reg * ||q||^2
    # = q'(A'A)q - 2 b'A q + b'b  +  reg * q'q
    Q = A_scaled' * A_scaled + Float64(regularization) * Matrix{Float64}(I, n_binary, n_binary)
    l = vec(-2 * (A_scaled' * b))

    # Симулированный отжиг
    best_q, best_E = _simulated_annealing_qubo(Q, l;
                                               num_reads=num_reads,
                                               num_sweeps=annealing_time,
                                               rng=rng)

    # Декодирование бинарного решения в непрерывный спектр
    spectrum = binary_to_spectrum(best_q, n_bins; n_bits=Int(n_bits), max_value=mv)
    spectrum = max.(spectrum, 0.0)

    residual = b .- A * spectrum
    return UnfoldResult(
        Vector{T}(spectrum), Int(max_iterations), isfinite(best_E), T(norm(residual)),
        Dict{String,Any}(
            "n_bits" => Int(n_bits),
            "energy" => best_E,
            "num_reads" => Int(num_reads),
            "annealing_time" => Int(annealing_time),
            "max_value" => mv,
        ))
end
