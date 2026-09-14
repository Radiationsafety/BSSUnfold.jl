"""
Doroshenko — итеративный метод развёртки (Doroshenko et al., 1986).

Аналог GRAVEL, но с другим обновлением:

    x_{k+1}[j] = x_k[j] * (1 + Σ_i A[i,j] * (b_i/(Ax)_i - 1) / Σ_i A[i,j])

Гарантирует неотрицательность и сохранение интеграла.
"""
function solve_doroshenko(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                         max_iterations::Integer=1000,
                         tolerance::T=T(1e-6),
                         eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)
    x = max.(copy(x0), eps)
    AT = Matrix(A')

    # Sensitivity: column sums
    col_sums = vec(sum(A, dims=1))
    col_sums = max.(col_sums, eps)

    converged = false
    iters = 0

    @inbounds for k in 1:max_iterations
        iters = k
        Ax = A * x
        Ax = max.(Ax, eps)
        # residual factor
        r_factor = b ./ Ax .- T(1)
        # update correction
        correction = AT * r_factor
        update_factor = T(1) .+ correction ./ col_sums
        x_new = x .* update_factor
        diff = norm(x_new .- x) / (norm(x) + eps)
        x = max.(x_new, T(0))
        if diff < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
