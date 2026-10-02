"""
Matrix helpers (port of `bssunfold/core/_matrix_utils.py`).

Finite-difference operators shared by the regularized solvers. Only the
grid-independent (`E_MeV = nothing`) variant of `create_derivative_matrix`
is ported: every solver that calls it passes `E_MeV = nothing`.
"""

"""
    create_derivative_matrix([T], n, order) -> Matrix{T}

`(n - order) × n` finite-difference operator with unit-spaced rows: `[-1, 1]`
for `order = 1` and `[1, -2, 1]` for `order = 2`. The penalty `‖L x‖²` is
invariant to the row sign convention, so this agrees with both the Python
original and the older in-package copies.

Port of `create_derivative_matrix(n, order, E_MeV=None)`.
"""
function create_derivative_matrix(::Type{T}, n::Integer, order::Integer) where T<:AbstractFloat
    order in (1, 2) || throw(ArgumentError("Unsupported derivative order: $order"))
    rows = n - order
    rows >= 1 || throw(ArgumentError("n must exceed order, got n=$n"))
    L = zeros(T, rows, n)
    if order == 1
        for i in 1:rows
            L[i, i] = T(-1)
            L[i, i + 1] = T(1)
        end
    else
        for i in 1:rows
            L[i, i] = T(1)
            L[i, i + 1] = T(-2)
            L[i, i + 2] = T(1)
        end
    end
    return L
end

create_derivative_matrix(n::Integer, order::Integer) = create_derivative_matrix(Float64, n, order)
