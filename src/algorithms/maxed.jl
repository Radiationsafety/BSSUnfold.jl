"""
MAXED — Maximum Entropy Deconvolution.

Итеративный алгоритм, максимизирующий энтропию Шеннона
H(x) = -Σ x_j ln(x_j / x0_j) при ограничениях A x = b.

Используется модифицированный множитель Лагранжа с
двойственной формулировкой (Reginski et al., 1981).
"""
function solve_maxed(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                    max_iterations::Integer=1000,
                    tolerance::T=T(1e-6),
                    eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)
    x = max.(copy(x0), eps)
    AT = Matrix(A')

    converged = false
    iters = 0

    @inbounds for k in 1:max_iterations
        iters = k
        Ax = A * x
        Ax = max.(Ax, eps)
        # Двойственные переменные λ: λ_i = ln(Ax_i / b_i)
        # Обновление: x_j = x0_j * exp(-Σ_i A_ij * λ_i)
        # Здесь упрощённая схема: использовать невязку как λ
        ratio = log.(Ax ./ max.(b, eps))
        # x_new = x0 .* exp.(-Aᵀ * ratio), но для стабильности используем partial update
        correction = AT * ratio
        x_new = x .* exp.(-correction .* T(0.5))
        x_new = max.(x_new, eps)
        diff = norm(x_new .- x) / (norm(x) + eps)
        x = x_new
        if diff < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
