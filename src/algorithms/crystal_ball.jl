"""
    solve_crystal_ball(A, b, x0=nothing; regularization=0.0)

Алгоритм CRYSTAL BALL — одношаговая (безытерационная) развёртка.

Спектр представляется линейной комбинацией ответных функций детекторов
(строк матрицы A):  phi = Σ_i alpha_i * A_i.  Подстановка в уравнение
измерений b = A * phi даёт нормальные уравнения

    (A Aᵀ + λ I) alpha = b,

после чего спектр восстанавливается как `phi = Aᵀ alpha`.  Это
эквивалентно аппроксимации дельта-оператора линейной комбинацией
интегральных операторов отклика (Kam & Stallmann).

`x0` не используется (принимается для единообразия сигнатуры).
`regularization` — параметр Тихонова λ для стабилизации плохо
обусловленной граммовой матрицы.

# Возвращает
`UnfoldResult` (iterations = 1, converged = true — одношаговый метод).
"""
function solve_crystal_ball(A::AbstractMatrix{T}, b::AbstractVector{T},
                            x0::Union{Nothing,AbstractVector{T}}=nothing;
                            regularization::T=T(0.0)) where T<:AbstractFloat
    m, n = size(A)

    if isempty(A) || isempty(b)
        throw(ArgumentError("Response matrix and measurements must be non-empty"))
    end
    if all(b .<= 0)
        throw(ArgumentError("All measurements are zero or negative"))
    end

    G = A * A'
    if regularization > 0
        G .+= regularization .* Matrix{T}(I, m, m)
    end

    alpha = G \ b
    spectrum = A' * alpha

    x = max.(spectrum, T(0))
    residual = b .- A * x
    return UnfoldResult(x, 1, true, norm(residual),
                        Dict{String,Any}("regularization" => regularization))
end
