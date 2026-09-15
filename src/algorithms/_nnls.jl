"""
NNLS (Lawson–Hanson, семантика `scipy.optimize.nnls`).

    min ||A x - b||₂   subject to   x ≥ 0

Активный набор решается обычным LS; нарушенная неотрицательность
исправляется шагом α по направлению `z - x` (алгоритм Lawson–Hanson).
"""

"""
    lawson_hanson(A, b; tol=sqrt(eps(T)), max_iterations=30n)

Решить NNLS-задачу.

# Возвращает
`(x, w)` где `x` — неотрицательное решение, `w` — вектор лагранжианов
(Aᵀ(b - A x)).
"""
function lawson_hanson(A::AbstractMatrix{T}, b::AbstractVector{T};
                       tol::Real=sqrt(eps(T)),
                       max_iterations::Integer=30 * max(size(A, 2), 1)) where
                      {T<:AbstractFloat}
    m, n = size(A)
    m == length(b) || throw(ArgumentError("b length must match rows of A"))
    AT = Matrix(A')
    ATb = AT * b
    x = zeros(T, n)
    z = zeros(T, n)
    Prl = Int[]                      # пассивный набор
    iter = 0
    lim = max(Int(max_iterations), 0)
    tol_T = T(tol)

    while iter < lim
        iter += 1
        w = ATb .- (AT * (A * x))
        isactive = falses(n)
        for j in Prl
            isactive[j] = true
        end
        best = 0
        best_w = tol_T
        for j in 1:n
            if !isactive[j] && w[j] > best_w
                best = j
                best_w = w[j]
            end
        end
        best == 0 && break
        push!(Prl, best)

        # внутренний цикл: коррекция отрицательных z
        while true
            AP = A[:, Prl]
            zP = AP \ b
            fill!(z, T(0))
            for (k, j) in enumerate(Prl)
                z[j] = zP[k]
            end
            if all(z[Prl] .> 0)
                copy!(x, z)
                break
            end
            α = Inf
            for j in Prl
                if z[j] <= 0
                    d = x[j] - z[j]
                    d > 0 && (α = min(α, x[j] / d))
                end
            end
            isfinite(α) || break
            for j in 1:n
                x[j] += T(α) * (z[j] - x[j])
                if x[j] <= 10 * eps(T)
                    x[j] = T(0)
                end
            end
            keep = Int[j for j in Prl if x[j] > 0]
            isempty(keep) && break
            Prl = keep
        end
    end

    return x, ATb .- (AT * (A * x))
end

"""
    solve_nnls(A, b)

NNLS через `lawson_hanson`; возвращает только `x`.
"""
function solve_nnls(A::AbstractMatrix{T}, b::AbstractVector{T}) where
                      {T<:AbstractFloat}
    x, _ = lawson_hanson(A, b)
    return x
end
