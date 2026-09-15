"""
    solve_omp(D, y, sparsity; tolerance=1e-6)

Orthogonal Matching Pursuit: find a sparse coefficient vector
`alpha` (at most `sparsity` nonzero components) approximating
`y ≈ D * alpha`.

At each step, the dictionary atom most correlated with the residual is
selected, then the coefficients on the current support are refined by
solving the least-squares problem (`lstsq`).  Stopping occurs when
sparsity is exhausted or the residual falls below `tolerance`.

# Returns
Sparse coefficient vector of length `k = size(D, 2)`.
"""
function solve_omp(D::AbstractMatrix{T}, y::AbstractVector{T}, sparsity::Integer;
                   tolerance::T=T(1e-6)) where T<:AbstractFloat
    _, k = size(D)
    alpha = zeros(T, k)
    residual = copy(y)
    support = Int[]

    norms = vec(norm.(eachcol(D)))
    norms = [nz > 0 ? nz : one(T) for nz in norms]
    D_norm = D ./ reshape(norms, 1, k)

    for _ in 1:min(sparsity, k)
        correlations = abs.(D_norm' * residual)
        for idx in support
            correlations[idx] = T(-1)
        end
        idx = argmax(correlations)
        if correlations[idx] <= 0
            break
        end
        push!(support, idx)

        D_s = D[:, support]
        coefs = _lstsq(D_s, y)
        residual = y .- D_s * coefs

        if norm(residual) < tolerance
            break
        end
    end

    if !isempty(support)
        D_s = D[:, support]
        coefs = _lstsq(D_s, y)
        for (i, s) in enumerate(support)
            alpha[s] = coefs[i]
        end
    end

    return alpha
end

"""
    _lstsq(D, y)

Solve the over-/under-determined least-squares problem via SVD
(the analogue of `np.linalg.lstsq` with `rcond=None`).
"""
function _lstsq(D::AbstractMatrix{T}, y::AbstractVector{T}) where T<:AbstractFloat
    F = svd(Matrix(D))
    smax = F.S[1]
    rcond = eps(real(T)) * max(size(D)...) * (smax > 0 ? smax : one(T))
    sinv = [s > rcond ? inv(s) : zero(T) for s in F.S]
    return F.V * (sinv .* (F.U' * y))
end

"""
    solve_ksvd(signals, n_atoms; n_iterations=20, sparsity=5, random_state=nothing)

Dictionary training with the K-SVD algorithm.

Signals are supplied as columns (`n × m`).  The dictionary is initialized
with random training signals and normalized columns; at each
iteration, sparse coding (OMP) and atom updates are performed via
sequential SVD of the error matrices (preserving the sparsity
of the coefficients).  `random_state` controls reproducibility.

# Returns
The trained dictionary (`n × n_atoms`) with unit-norm columns.
"""
function solve_ksvd(signals::AbstractMatrix{T}, n_atoms::Integer;
                    n_iterations::Integer=20,
                    sparsity::Integer=5,
                    random_state::Union{Nothing,Integer}=nothing) where T<:AbstractFloat
    rng = random_state === nothing ? Random.default_rng() : MersenneTwister(random_state)
    n, m = size(signals)

    n_atoms_eff = min(n_atoms, m)
    idx = sort(randperm(rng, m)[1:n_atoms_eff])
    D = Matrix{T}(signals[:, idx])

    norms = vec(norm.(eachcol(D)))
    norms = [nz > 0 ? nz : one(T) for nz in norms]
    D ./= reshape(norms, 1, n_atoms_eff)

    coefficients = zeros(T, n_atoms_eff, m)

    for _ in 1:n_iterations
        for j in 1:m
            coefficients[:, j] .= solve_omp(D, signals[:, j], sparsity)
        end

        for atom in 1:n_atoms_eff
            used = findall(!=(0), coefficients[atom, :])
            isempty(used) && continue

            D_restricted = copy(D)
            D_restricted[:, atom] .= T(0)
            E = signals[:, used] .- D_restricted * coefficients[:, used]

            F = svd(Matrix(E))
            new_atom = F.U[:, 1]
            new_coef = F.S[1] .* F.Vt[1, :]

            D[:, atom] .= new_atom
            coefficients[atom, used] .= new_coef
        end

        norms = vec(norm.(eachcol(D)))
        norms = [nz > 0 ? nz : one(T) for nz in norms]
        D ./= reshape(norms, 1, n_atoms_eff)
    end

    return D
end

"""
    solve_sl0(A, b; sigma_min=0.01, sigma_decrease_factor=0.5, mu_0=1.0,
              L=3, max_iterations=1000, tolerance=1e-6)

SL0 (Smoothed L0): recovery of a sparse solution of the
underdetermined system `b = A x`.

The L0 norm is approximated by a Gaussian surrogate function
`Σ (1 - exp(-x²/(2σ²)))`; gradient descent with projection
onto the feasible set `{x : A x = b}` (via the pseudo-inverse
matrix) is performed, and σ decreases geometrically from `2·max|x|`
to `sigma_min`.

# Returns
Sparse vector `x` of length `n = size(A, 2)`.
"""
function solve_sl0(A::AbstractMatrix{T}, b::AbstractVector{T};
                   sigma_min::T=T(0.01),
                   sigma_decrease_factor::T=T(0.5),
                   mu_0::T=T(1.0),
                   L::Integer=3,
                   max_iterations::Integer=1000,
                   tolerance::T=T(1e-6)) where T<:AbstractFloat
    x = pinv(A) * b
    pinv_AT = A' * pinv(A * A')

    sigma = T(2.0) * maximum(abs.(x))
    sigma = sigma == 0 ? one(T) : sigma
    sigma = max(sigma, sigma_min)

    for _ in 1:max_iterations
        x_prev = copy(x)

        for _ in 1:L
            exp_term = exp.(-(x .^ 2) ./ (T(2.0) * sigma^2))
            @. x = x - mu_0 * x * exp_term
            x = x .- pinv_AT * (A * x .- b)
        end

        sigma *= sigma_decrease_factor
        sigma < sigma_min && break

        if norm(x .- x_prev) < tolerance * max(one(T), norm(x))
            break
        end
    end

    return x
end

"""
    solve_cs(A, b, x0=nothing; n_atoms=nothing, sparsity=nothing, dictionary=nothing,
             n_dictionary_iterations=20, sigma_min=0.01, sigma_decrease_factor=0.5,
             mu_0=1.0, L=3, max_iterations=1000, tolerance=1e-6, random_state=nothing)

Compressive Sensing (CS) unfolding of a neutron spectrum.

The spectrum `x` is represented sparsely in a trained dictionary `D`:
`x = D * alpha`.  The measurement equation becomes
`b = (A * D) * alpha` and is solved for the sparse `alpha`
by the SL0 algorithm; the spectrum is recovered as `x = D * alpha` with
nonnegative projection and scale normalization to the data.

The dictionary is trained with K-SVD on training signals (cosine basis
plus the initial approximation).  A ready-made `dictionary` can be
passed (of size `n × n_atoms`) — training is then skipped.

# Returns
`UnfoldResult` with the recovered spectrum.
"""
function solve_cs(A::AbstractMatrix{T}, b::AbstractVector{T},
                  x0::Union{Nothing,AbstractVector{T}}=nothing;
                  n_atoms::Union{Nothing,Integer}=nothing,
                  sparsity::Union{Nothing,Integer}=nothing,
                  dictionary::Union{Nothing,AbstractMatrix{T}}=nothing,
                  n_dictionary_iterations::Integer=20,
                  sigma_min::T=T(0.01),
                  sigma_decrease_factor::T=T(0.5),
                  mu_0::T=T(1.0),
                  L::Integer=3,
                  max_iterations::Integer=1000,
                  tolerance::T=T(1e-6),
                  random_state::Union{Nothing,Integer}=nothing) where T<:AbstractFloat
    m, n = size(A)

    n_atoms_eff = n_atoms === nothing ? max(n, 2 * m) : n_atoms
    sparsity_eff = sparsity === nothing ? max(1, n ÷ 20) : sparsity

    base = if x0 !== nothing && any(x0 .!= 0)
        bx = max.(float.(x0), T(0))
        bx ./ (norm(bx) + T(1e-12))
    else
        fill(one(T) / sqrt(n), n)
    end

    t = collect(range(T(0), T(π), length=n))
    n_basis = min(n, max(2 * m, 8))
    signals = zeros(T, n, n_basis + 1)
    for i in 0:(n_basis - 1)
        col = cos.(i .* t)
        nrm = norm(col)
        nrm > 0 && (col ./= nrm)
        signals[:, i + 1] .= col
    end
    signals[:, end] .= base

    D = if dictionary !== nothing
        Dict_mat = Matrix{T}(dictionary)
        if size(Dict_mat, 1) != n
            throw(ArgumentError("Dictionary first dimension ($(size(Dict_mat, 1))) must match " *
                                "number of energy bins ($n)"))
        end
        Dict_mat
    else
        solve_ksvd(signals, n_atoms_eff;
                   n_iterations=n_dictionary_iterations,
                   sparsity=sparsity_eff,
                   random_state=random_state)
    end

    Phi = A * D

    alpha = solve_sl0(Phi, b;
                      sigma_min=sigma_min,
                      sigma_decrease_factor=sigma_decrease_factor,
                      mu_0=mu_0,
                      L=L,
                      max_iterations=max_iterations,
                      tolerance=tolerance)

    x = max.(D * alpha, T(0))

    computed = A * x
    if norm(computed) > 0 && norm(b) > 0
        scale = dot(b, computed) / (dot(computed, computed) + T(1e-12))
        x .*= scale
    end

    residual = norm(A * x .- b)
    converged = residual < tolerance * max(one(T), norm(b))
    return UnfoldResult(x, max_iterations, converged, residual,
                        Dict{String,Any}("n_atoms" => size(D, 2)))
end
