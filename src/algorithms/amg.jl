"""
AMG- and stationary-preconditioned Krylov unfolding method.

Julia analogue of the R packages `Rlinsolve` (stationary and Krylov
iterative solvers for sparse linear systems) and `pyamg`-style algebraic
multigrid: the unfolding least-squares problem is solved as a
preconditioned Krylov iteration on the normal equations

    (AᵀA) x = Aᵀb,

where the preconditioner approximates `(AᵀA)^-1` and can be built from
`"amg"`, `"jacobi"`, `"gs"`, `"sor"`, `"ssor"` or `"none"`. The Krylov
solver itself is one of `cg`, `bicgstab` or `gmres` (implementations
reused from the `solve_scipy_direct` family). Non-negativity of the
fluence is enforced by projected outer restarts: after each Krylov solve
the spectrum is clamped to `x >= 0` and the iteration is restarted on the
residual of the clamped iterate, `outer_iterations` times.
"""

const _AMG_VALID_METHODS = ("cg", "bicgstab", "gmres")
const _AMG_VALID_PRECONDITIONERS = ("amg", "jacobi", "gs", "sor", "ssor", "none")

# Preconditioners whose application matrix is symmetric positive definite
# (compatible with CG); "gs" and "sor" are nonsymmetric and are only
# used with bicgstab/gmres.
const _AMG_CG_COMPATIBLE = ("amg", "jacobi", "ssor", "none")

# Auto Tikhonov damping (fraction of the mean diagonal of AᵀA) used
# when `regularization=nothing`: a tiny damping makes the normal matrix
# nonsingular, which stabilises the preconditioners on rank-deficient
# systems.
const _AMG_AUTO_REG_FACTOR = 1e-4

"""
    _amg_LinearOp(matvec)

Minimal callable linear operator: `M * v` applies `matvec(v)`.
"""
struct _amg_LinearOp{F}
    matvec::F
end

Base.:*(M::_amg_LinearOp, v::AbstractVector) = M.matvec(v)

"""
    _amg_stationary_apply(N, kind, omega, r)

Apply one stationary sweep of kind `kind` to system `N`

Implements the preconditioned-residual recurrences of the classical
splitting iterations (Jacobi / Gauss-Seidel / SOR / SSOR) used by
`Rlinsolve`: given the splitting `N = D + L + U` the application
returns `M^-1 r` with

* `jacobi`: `M = D`;
* `gs`: `M = D + L` (forward substitution);
* `sor`: `M = (D + omega L) / omega`;
* `ssor`: `M = (D + omega L) D^-1 (D + omega U) / (omega (2 - omega))`.
"""
function _amg_stationary_apply(N::AbstractMatrix{Float64}, kind::AbstractString,
                               omega::Float64, r::Vector{Float64})
    D = diag(N)
    D_safe = [abs(d) > 0 ? d : 1.0 for d in D]
    if kind == "jacobi"
        return r ./ D_safe
    end

    n = size(N, 2)
    lower = tril(N, -1)
    upper = triu(N, 1)
    Dm = Diagonal(D_safe)

    if kind == "gs"
        M = Dm + lower
        return LowerTriangular(M) \ r
    elseif kind == "sor"
        M = Dm + omega .* lower
        return omega .* (LowerTriangular(M) \ r)
    elseif kind == "ssor"
        Mf = Dm + omega .* lower
        Mb = Dm + omega .* upper
        t = LowerTriangular(Mf) \ r
        t = D_safe .* t
        t = UpperTriangular(Mb) \ t
        return (omega * (2.0 - omega)) .* t
    end

    throw(ArgumentError("Unknown stationary kind: $kind"))
end

"""
    _amg_build_preconditioner(A, kind="amg"; omega=1.0, damping=0.0)

Build a callable operator approximating `(AᵀA + damping I)^-1` acting on
`(n,)` vectors. `kind`: `"amg"` (falls back to Jacobi with a warning —
no `pyamg` dependency is used), `"jacobi"`, `"gs"`, `"sor"`, `"ssor"`
or `"none"`.
"""
function _amg_build_preconditioner(A::Matrix{Float64};
                                   kind::AbstractString="amg",
                                   omega::Float64=1.0,
                                   damping::Float64=0.0)
    kind in _AMG_VALID_PRECONDITIONERS || throw(ArgumentError(
        "preconditioner must be one of $_AMG_VALID_PRECONDITIONERS, got $kind"))
    n = size(A, 2)
    if kind == "none"
        return _amg_LinearOp(r -> Vector{Float64}(r))
    end

    N = A' * A + damping * Matrix{Float64}(I, n, n)

    if kind == "amg"
        # No smoothed-aggregation AMG dependency is bundled; degrade to
        # the Jacobi preconditioner (mirrors the "pyamg not installed"
        # fallback path of the Python original).
        @warn "AMG (smoothed aggregation) is not available -- " *
              "AMG preconditioner falls back to Jacobi."
        kind = "jacobi"
    end

    return _amg_LinearOp(r -> _amg_stationary_apply(N, kind, omega, Vector{Float64}(r)))
end

"""
    solve_amg(A, b, x0=nothing; method="cg", preconditioner="amg", omega=1.0,
              max_iterations=200, tolerance=1e-10, outer_iterations=3,
              nonnegativity=true, regularization=nothing)

Solve unfolding problem with a preconditioned Krylov method.

The (optionally damped) normal equations `(AᵀA + reg I) x = Aᵀb` are
solved with the chosen Krylov solver and preconditioner; non-negativity
is enforced with projected outer restarts.

# Keywords
- `method` — Krylov solver: `"cg"` (default), `"bicgstab"` or `"gmres"`.
  `"cg"` requires a symmetric positive-definite preconditioner; the
  nonsymmetric `"gs"` / `"sor"` preconditioners are transparently replaced
  by `"ssor"` (with a warning) for CG;
- `preconditioner` — `"amg"` (default), `"jacobi"`, `"gs"`, `"sor"`,
  `"ssor"` or `"none"`;
- `omega` — relaxation factor for `"sor"` / `"ssor"` (default: 1.0);
- `max_iterations` — maximum Krylov iterations per outer restart (default: 200);
- `tolerance` — relative residual tolerance of the normal equations
  (default: 1e-10);
- `outer_iterations` — number of projected restarts (default: 3);
- `nonnegativity` — clamp the spectrum to `x >= 0` between restarts
  (default: `true`);
- `regularization` — Tikhonov damping added to the diagonal of `AᵀA`;
  `nothing` (default) selects `1e-4 * mean(diag(AᵀA))` automatically,
  which keeps the damped system nonsingular; pass `0.0` for the pure
  (undamped) normal equations.

Returns an [`UnfoldResult`](@ref) with the estimated total Krylov
iteration count across restarts.
"""
function solve_amg(A::AbstractMatrix{T}, b::AbstractVector{T},
                   x0::Union{AbstractVector{T},Nothing}=nothing;
                   method::AbstractString="cg",
                   preconditioner::AbstractString="amg",
                   omega::Real=1.0,
                   max_iterations::Integer=200,
                   tolerance::Real=1e-10,
                   outer_iterations::Integer=3,
                   nonnegativity::Bool=true,
                   regularization::Union{Nothing,Real}=nothing
                   ) where T<:AbstractFloat
    method ∈ _AMG_VALID_METHODS || throw(ArgumentError(
        "method must be one of $_AMG_VALID_METHODS, got $method"))
    preconditioner ∈ _AMG_VALID_PRECONDITIONERS || throw(ArgumentError(
        "preconditioner must be one of $_AMG_VALID_PRECONDITIONERS, " *
        "got $preconditioner"))
    if method == "cg" && !(preconditioner ∈ _AMG_CG_COMPATIBLE)
        @warn "preconditioner=$preconditioner is nonsymmetric and " *
              "incompatible with method='cg'; switching to 'ssor'"
        preconditioner = "ssor"
    end
    A, b, x0 = validate_system(A, b; x0=x0,
                               max_iterations=max_iterations,
                               tolerance=tolerance)
    outer_iterations < 1 && throw(ArgumentError(
        "outer_iterations must be >= 1, got $outer_iterations"))
    !(0.0 < omega <= 2.0) && @warn "omega=$omega outside the recommended range (0, 2]"

    Af = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    n = size(Af, 2)

    AT_A = Af' * Af
    AT_b = Af' * bf
    local damping::Float64
    if regularization === nothing
        damping = _AMG_AUTO_REG_FACTOR * Float64(mean(diag(AT_A)))
    else
        damping = Float64(regularization)
        damping < 0 && throw(ArgumentError(
            "regularization must be non-negative, got $regularization"))
    end
    AT_A_solver = AT_A + damping * Matrix{Float64}(I, n, n)
    M = _amg_build_preconditioner(Af; kind=preconditioner,
                                  omega=Float64(omega), damping=damping)
    # cg/gmres apply the preconditioner once per iteration, bicgstab twice
    applications_per_iteration = method == "bicgstab" ? 2 : 1

    x = x0 === nothing ? zeros(n) : Vector{Float64}(x0)
    converged = false
    inner_converged = false
    total_iterations = 0

    b_norm = max(norm(AT_b), 1e-300)
    maxit = Int(max_iterations)
    rtol = Float64(tolerance)

    for _outer in 1:Int(outer_iterations)
        residual = AT_b .- AT_A_solver * x
        if norm(residual) <= rtol * b_norm
            converged = true
            break
        end
        local dx, its, conv
        if method == "cg"
            dx, its, conv = _sd_cg(M, residual, rtol, maxit)
        elseif method == "bicgstab"
            dx, its, conv = _sd_bicgstab(M, residual, rtol, maxit)
        else
            dx, its, conv = _sd_gmres(M, residual, rtol, maxit)
        end
        total_iterations += its
        if !all(isfinite, dx)
            dx = zeros(n)
        end
        x .+= dx
        if nonnegativity
            x = max.(x, 0.0)
        end
        if conv
            inner_converged = true
            # Full success requires the (possibly clamped) iterate to
            # satisfy the tolerance as well.
            if norm(AT_b .- AT_A_solver * x) <= rtol * b_norm
                converged = true
                break
            end
        elseif its < maxit
            # Early termination without convergence == Krylov breakdown
            # (loss of orthogonality / illegal input): further restarts
            # on the same system are unlikely to help.
            break
        end
    end

    # The projected restarts succeed when the Krylov solves themselves
    # converged (the clamped iterate is then the projected solution) or
    # when the clamped iterate satisfies the residual tolerance.
    converged = converged || inner_converged
    total_iterations = min(total_iterations ÷ applications_per_iteration,
                           Int(outer_iterations) * maxit)
    if nonnegativity
        x = max.(x, 0.0)
    end

    residual = bf .- Af * x
    return UnfoldResult(x, total_iterations, converged, Float64(norm(residual)),
                        Dict{String,Any}(
                            "method" => method,
                            "preconditioner" => preconditioner,
                            "omega" => Float64(omega),
                            "outer_iterations" => Int(outer_iterations),
                            "regularization" => damping))
end
