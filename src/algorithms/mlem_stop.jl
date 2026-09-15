"""
    calculate_j_factor(measurements, estimate)

J-factor from Bouallegue et al.: `sum((measurements - estimate)^2) / sum(estimate)`.
If `sum(estimate) <= 0`, returns `Inf`.
"""
function calculate_j_factor(measurements::AbstractVector{T},
                            estimate::AbstractVector{T}) where T<:AbstractFloat
    denominator = sum(estimate)
    denominator <= 0 && return T(Inf)
    return sum(abs2, measurements .- estimate) / denominator
end

"""
    solve_mlem_stop(A, b, x0; max_iterations=15000, cps_crossover=30000.0, j_threshold=nothing)

MLEM-STOP — MLEM with an early stopping criterion based on the J-factor
(Bouallegue; Montgomery et al., NIM A 957 (2020) 163400).

J-factor: `sum((measurements - estimate)^2) / sum(estimate)`. Iterations
stop as soon as J drops below a threshold; by default the threshold
is computed as `mean(b)/cps_crossover`.
"""
function solve_mlem_stop(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                         max_iterations::Integer=15000,
                         cps_crossover::Real=30000.0,
                         j_threshold::Union{Nothing,Real}=nothing) where T<:AbstractFloat
    threshold = j_threshold === nothing ? sum(b) / length(b) / T(cps_crossover) : T(j_threshold)

    x = max.(copy(x0), T(1e-10))
    AT = Matrix(A')

    @inbounds for i in 1:max_iterations
        Ax = A * x
        j_factor = calculate_j_factor(b, Ax)
        if j_factor <= threshold
            residual = b .- A * x
            return UnfoldResult(x, i, true, norm(residual))
        end
        @. Ax = max(Ax, T(1e-10))
        ratio = b ./ Ax
        correction = AT * ratio
        x = max.(x .* correction, T(0))
    end

    j_final = calculate_j_factor(b, A * x)
    residual = b .- A * x
    return UnfoldResult(x, max_iterations, j_final <= threshold, norm(residual))
end
