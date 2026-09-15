"""
SART (simultaneous algebraic reconstruction technique) — развёртка BSS.

Релаксированный взвешенный МНК-алгебраический reconstruction:

    x^{n+1} = x^n + alpha(n)/(Aᵀ 1 + eps) * Aᵀ ( (b - A x^n) / (A 1 + eps) )

Порт SART из PyTomography (MIT), адаптированный к развёртке нейтронных спектров.
Бин с наименьшей энергией (нулевое отклик детектора) удерживается на значении
начального приближения.
"""
function solve_sart(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                    max_iterations::Integer=50,
                    tolerance::Real=1e-6,
                    relaxation::Union{Nothing,Real,Function}=nothing) where T<:AbstractFloat
    m, n = size(A)
    eps = T(1e-11)
    x = max.(copy(x0), T(0))
    x0_first = x[1]

    relax_at = if relaxation === nothing
        k -> T(0.8)
    elseif relaxation isa Function
        k -> T(relaxation(k))
    else
        r = T(relaxation)
        k -> r
    end

    norm_back = vec(sum(A, dims=1))
    norm_forward = vec(sum(A, dims=2))
    norm_back_safe = norm_back .+ eps
    norm_forward_safe = norm_forward .+ eps

    x_old = copy(x)
    update = similar(x)
    residual = similar(b)
    AT = Matrix(A')
    converged = false
    iters = 0

    @inbounds for k in 1:max_iterations
        iters = k
        x_old_norm = norm(x_old)
        α = relax_at(k)

        mul!(residual, A, x)
        @. residual = b - residual
        @. residual = residual / norm_forward_safe
        mul!(update, AT, residual)
        @. x += α * update / norm_back_safe
        @. x = max(x, T(0))
        x[1] = x0_first

        rel = norm(x .- x_old) / (x_old_norm + eps)
        if rel < tolerance
            converged = true
            break
        end
        copyto!(x_old, x)
    end

    residual_final = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual_final))
end
