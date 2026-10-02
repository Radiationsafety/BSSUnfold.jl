"""
Lavrentiev regularization — shift-regularized solution of the ill-posed
operator equation `A z = b`.

Port of `bssunfold/core/unfold_lavrentiev.py`. Four forms:
* `:gram` (default) — Lavrentiev on the m×m Gram operator `B = A Aᵀ`:
  `(B + α I) y = b`, `z = Aᵀ y`. Algebraically identical to
  zeroth-order Tikhonov via push-through, but cheap when m ≪ n.
* `:direct` — classical `(A + α I) z = b`; requires m == n.
* `:padded` — zero-pad A to a square max(m,n)×max(m,n) operator and
  apply the direct scheme. Legitimate but forces `z_i = 0` for i > m
  when m < n.
* `:iterated` — Bakushinsky a-priori α_k = α·q^k defect correction on
  the Gram operator: `y_{k+1} = y_k + (B + α_k I)⁻¹ (b − B y_k)`.

Deterministic (no RNG); result clipped to nonnegative.
"""
function solve_lavrentiev(A::AbstractMatrix{T}, b::AbstractVector{T},
                          x0::Union{Nothing,AbstractVector{T}}=nothing;
                          alpha::Real=T(0.05),
                          form::Union{Symbol,String}=:gram,
                          q::Real=T(0.5),
                          n_iterations::Integer=5) where T<:AbstractFloat
    α = T(alpha)
    α >= 0 || throw(ArgumentError("alpha must be non-negative, got $alpha"))
    m, n = size(A)
    b = Vector{T}(b)
    fs = Symbol(form)
    eye_m = Matrix{T}(I, m, m)
    z = zeros(T, n)

    if fs === :direct
        m == n || throw(ArgumentError(
            "direct Lavrentiev requires a square response matrix, got A.size=$m×$n"))
        M = A .+ α .* Matrix{T}(I, n, n)
        z = _safe_solve(M, b)
    elseif fs === :gram
        B = A * A'
        M = B .+ α .* eye_m
        y = _safe_solve(M, b)
        z = A' * y
    elseif fs === :padded
        s = max(m, n)
        A_pad = zeros(T, s, s); A_pad[1:m, 1:n] .= A
        b_pad = zeros(T, s);    b_pad[1:m]     .= b
        M = A_pad .+ α .* Matrix{T}(I, s, s)
        z_full = _safe_solve(M, b_pad)
        z = z_full[1:n]
    elseif fs === :iterated
        qv = T(q)
        (0 < qv ≤ 1) || throw(ArgumentError("q must satisfy 0 < q <= 1, got $q"))
        n_iterations >= 1 || throw(ArgumentError("n_iterations must be >= 1, got $n_iterations"))
        B = A * A'
        y = zeros(T, m)
        for k in 0:(n_iterations-1)
            αk = α * (qv ^ k)
            r = b .- B * y
            y = y .+ _safe_solve(B .+ αk .* eye_m, r)
        end
        z = A' * y
    else
        throw(ArgumentError(
            "form must be :gram, :direct, :padded or :iterated, got $form"))
    end
    z = max.(z, T(0))
    UnfoldResult(z, Int(fs === :iterated ? n_iterations : 1), true, norm(b .- A * z))
end

function _safe_solve(M::AbstractMatrix{T}, rhs::AbstractVector{T}) where T<:AbstractFloat
    try
        return Vector{T}(M \ rhs)
    catch err
        if err isa LinearAlgebra.SingularException || err isa LinearAlgebra.NoPivotException
            F = qr(Matrix(M); pivot=:none, rtol=0.0)
            return Vector{T}(F \ rhs)
        end
        rethrow(err)
    end
end
