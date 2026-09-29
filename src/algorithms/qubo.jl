"""
QUBO-based neutron spectrum unfolding using quantum-inspired annealing
(faithful port of unfold_qubo.py: pyqubo QUBO construction +
dwave.samplers.SimulatedAnnealingSampler).

The spectrum is discretized into binary variables (fractional binary
expansion): `x[i] = max_value * Σ_j b_ij * 2^-(j+1)`.

pyqubo stage (unfold_qubo.py builds the Hamiltonian term by term):

    E(q) = Σ_i (Q_ii + l_i) q_i + Σ_{i<j} Q_ij q_i q_j,
    Q = (A T)' (A T) + regularization * I,   l = -2 (A T)' b,

where T is the bit-to-continuous transition matrix.  Note that the Python
original adds each off-diagonal pair term Q_ij * x_i * x_j exactly once
(not 2 * Q_ij), and drops pairs with |Q_ij| <= 1e-10 before compiling;
this port reproduces both conventions.

dwave.samplers stage (v1.8, sequential Metropolis simulated annealing):
the QUBO is converted to Ising via q = (1 + s) / 2, a geometric beta
schedule spanning the default per-spin-bias-derived beta range is used
(num_betas = num_sweeps, one sweep per beta), each sweep updates all
spins in sequential order with the Metropolis criterion, proposals with
delta_energy >= 44.36142 / beta are skipped, and the lowest-energy
sample over all reads is returned.
"""

# ─── Binary encoding (port of _spectrum_to_binary/_binary_to_spectrum) ────

"""
    spectrum_to_binary(spectrum; n_bits=8, max_value=nothing) -> Vector{Int}

Convert a continuous spectrum into binary representation
(length `n_bins * n_bits`).
"""
function spectrum_to_binary(spectrum::AbstractVector{<:Real};
                           n_bits::Integer=8,
                           max_value::Union{Nothing,Real}=nothing)
    mv = max_value === nothing ? maximum(spectrum) : Float64(max_value)
    mv <= 0 && (mv = 1.0)
    n_bins = length(spectrum)
    binary = zeros(Int, n_bins * n_bits)
    for i in 1:n_bins
        val = clamp(spectrum[i] / mv, 0.0, 1.0)
        for j in 1:n_bits
            # NB: as in the Python original, for val == 1.0 the first "digit"
            # will be 2 (int(2.0)); the decoder binary_to_spectrum handles
            # this correctly (2 * 2^-1 = 1.0).
            bit = floor(Int, val * 2)
            binary[(i - 1) * n_bits + j] = bit
            val = (val * 2) % 1
        end
    end
    return binary
end

"""
    binary_to_spectrum(binary, n_bins; n_bits=6, max_value=1.0) -> Vector{Float64}

Convert the binary representation back into a continuous spectrum.
"""
function binary_to_spectrum(binary::AbstractVector{<:Real}, n_bins::Integer;
                           n_bits::Integer=6, max_value::Real=1.0)
    spectrum = zeros(n_bins)
    for i in 1:n_bins
        val = 0.0
        for j in 1:n_bits
            val += binary[(i - 1) * n_bits + j] * 2.0^(-j)
        end
        spectrum[i] = val * max_value
    end
    return max.(spectrum, 0.0)
end

# ─── dwave.samplers simulated annealing (port of cpu_sa.cpp + sampler.py) ──

"""
    _dwave_beta_range(h, Js) -> (hot, cold)

Port of `_default_ising_beta_range(h, J)` from dwave.samplers 1.8
(max_single_qubit_excitation_rate=0.01, scale_T_with_N=True).
`h` is the Ising field vector, `Js` the symmetric spin coupling matrix.
"""
function _dwave_beta_range(h::Vector{Float64}, Js::Matrix{Float64})
    N = length(h)
    sum_abs = abs.(h) .+ vec(sum(abs, Js; dims=2))
    max_eff = maximum(sum_abs; init=0.0)
    hot = max_eff == 0.0 ? 1.0 : log(2.0) / (2.0 * max_eff)

    # smallest non-zero |bias| touching each variable
    min_bias = fill(Inf, N)
    @inbounds for i in 1:N
        h[i] != 0.0 && (min_bias[i] = abs(h[i]))
    end
    @inbounds for j in 1:N, i in 1:j-1
        v = Js[i, j]
        if v != 0.0
            a = abs(v)
            a < min_bias[i] && (min_bias[i] = a)
            a < min_bias[j] && (min_bias[j] = a)
        end
    end
    finite = min_bias[isfinite.(min_bias)]
    if isempty(finite)
        # all biases zero: dwave.samplers falls back to [0.1, 1]
        return 0.1, hot
    end
    min_eff = minimum(finite)
    n_min_gaps = count(==(min_eff), finite)
    cold = log(n_min_gaps / 0.01) / (2.0 * min_eff)
    return hot, cold
end

"""
    _qubo_ising(L, Qpair, iidx, jidx) -> (h, Js)

Ising conversion of the QUBO `Σ L_i q_i + Σ_k Qpair_k q_{i_k} q_{j_k}`
via q = (1 + s) / 2: `h_i = L_i / 2 + Σ_j Js_ij`, `Js_ij = J_ij / 4`.
"""
function _qubo_ising(L::Vector{Float64}, Qpair::Vector{Float64},
                     iidx::Vector{Int}, jidx::Vector{Int})
    N = length(L)
    Js = zeros(N, N)
    @inbounds for k in eachindex(Qpair)
        i, j, v = iidx[k], jidx[k], Qpair[k] / 4.0
        Js[i, j] = v
        Js[j, i] = v
    end
    h = L / 2 .+ vec(sum(Js; dims=2))
    return h, Js
end

"""
    _dwave_sa(h, Js, L, Qpair, iidx, jidx, betas, rng) -> (q, energy)

Sequential-Metropolis simulated annealing on the QUBO
`E(q) = Σ L_i q_i + Σ_k Qpair_k q_{i_k} q_{j_k}`, q ∈ {0,1}^N,
port of dwave.samplers `general_simulated_annealing`
(randomize_order=False, proposal_acceptance_criteria="Metropolis",
num_sweeps_per_beta=1, one sweep per beta value in `betas`,
single read, uniform random initial state).  Returns final binary state
and its QUBO energy.
"""
function _dwave_sa(h::Vector{Float64}, Js::Matrix{Float64},
                   L::Vector{Float64}, Qpair::Vector{Float64},
                   iidx::Vector{Int}, jidx::Vector{Int},
                   betas::AbstractVector{Float64}, rng::AbstractRNG)
    N = length(L)

    s = [rand(rng) < 0.5 ? -1.0 : 1.0 for _ in 1:N]
    dE = -2.0 .* s .* (h .+ Js * s)

    @inbounds for beta in betas
        threshold = 44.36142 / beta
        for var in 1:N
            d = dE[var]
            d >= threshold && continue
            if d <= 0.0
                accept = true
            else
                accept = exp(-d * beta) > rand(rng)
            end
            if accept
                mult = 4.0 * s[var]
                for nbr in 1:N
                    nbr == var && continue
                    jv = Js[var, nbr]
                    jv == 0.0 && continue
                    dE[nbr] += mult * jv * s[nbr]
                end
                s[var] = -s[var]
                dE[var] = -d
            end
        end
    end

    q = [(1.0 + si) / 2 for si in s]
    E = dot(L, q)
    @inbounds for k in eachindex(Qpair)
        E += Qpair[k] * q[iidx[k]] * q[jidx[k]]
    end
    return q, E
end

# ─── Main solver ────────────────────────────────────────────────────────

"""
    solve_qubo(A, b, x0; n_bits=6, max_value=nothing, regularization=0.01,
               max_iterations=1000, annealing_time=1000, num_reads=10,
               random_state=nothing) -> UnfoldResult

Solve the unfolding problem via a QUBO formulation with simulated annealing
(port of `solve_qubo_unfold`; defaults match the Python signature).

# Arguments
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: measurements (m,)
- `x0::AbstractVector{T}`: initial guess used to estimate `max_value`
  (`2 * max(x0)`) when it is not given
- `n_bits`: bits per energy bin (default 6)
- `max_value`: maximum value of the spectrum for scaling;
  `nothing` — estimated from `x0`
- `regularization`: regularization parameter (default 0.01)
- `max_iterations`: iteration count reported in the result (default 1000)
- `annealing_time`: annealing sweeps = number of beta values (default 1000)
- `num_reads`: independent annealing reads; best-energy one is kept
  (default 10)
- `random_state`: seed for reproducibility (`random_state` in Python)

# Returns
`UnfoldResult`; `extra` contains `n_bits`, `energy`, `num_reads`,
`annealing_time`, `max_value`, `beta_range`.
"""
function solve_qubo(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   n_bits::Integer=6,
                   max_value::Union{Nothing,Real}=nothing,
                   regularization::Real=0.01,
                   max_iterations::Integer=1000,
                   annealing_time::Integer=1000,
                   num_reads::Integer=10,
                   random_state::Union{Integer,Nothing}=nothing) where T<:AbstractFloat
    m, n_bins = size(A)
    length(b) == m || throw(ArgumentError("b length ($(length(b))) must match A rows ($m)"))
    n_bits >= 1 || throw(ArgumentError("n_bits must be >= 1, got $n_bits"))

    rng = random_state === nothing ? MersenneTwister() : MersenneTwister(Int(random_state))

    # Estimate max_value if not given (Python: 2 * max(x0), fallback lstsq)
    mv = if max_value !== nothing
        Float64(max_value)
    else
        cand = if isempty(x0)
            try
                x_est = qr(A, ColumnNorm()) \ b
                2 * maximum(abs.(x_est))
            catch
                1.0
            end
        else
            2 * maximum(x0)
        end
        cand <= 0 ? 1.0 : cand
    end

    # Transition matrix from bits to the continuous spectrum:
    # x_cont[i] = Σ_j (binary[i*n_bits + j] * 2^-(j+1)) * max_value
    n_binary = n_bins * n_bits
    Tmat = zeros(n_bins, n_binary)
    for i in 1:n_bins, j in 1:n_bits
        Tmat[i, (i - 1) * n_bits + j] = 2.0^(-j) * mv
    end

    # Effective response matrix acting on the bit vector: A_scaled = A * Tmat
    A_scaled = A * Tmat

    # QUBO matrix and linear term (as in unfold_qubo.py)
    Q = A_scaled' * A_scaled + Float64(regularization) * Matrix{Float64}(I, n_binary, n_binary)
    l = vec(-2 * (A_scaled' * b))

    # pyqubo Hamiltonian assembly: linear = Q_ii + l_i; each off-diagonal
    # pair term Q_ij x_i x_j added once for j > i, dropped when |Q_ij| <= 1e-10
    L = [Q[i, i] + l[i] for i in 1:n_binary]
    iidx = Int[]
    jidx = Int[]
    Qpair = Float64[]
    @inbounds for i in 1:n_binary, j in (i + 1):n_binary
        if abs(Q[i, j]) > 1e-10
            push!(iidx, i)
            push!(jidx, j)
            push!(Qpair, Q[i, j])
        end
    end

    # Simulated annealing: Ising conversion + default geometric beta schedule
    h, Js = _qubo_ising(L, Qpair, iidx, jidx)
    hot, cold = _dwave_beta_range(h, Js)
    num_betas = Int(annealing_time)
    betas = if num_betas <= 0
        Float64[]
    elseif num_betas == 1
        [cold]
    else
        hot .* (cold / hot) .^ ((0:(num_betas - 1)) / (num_betas - 1))
    end

    # num_reads independent runs, keep the best-energy sample (dimod `first`)
    best_q = zeros(Float64, n_binary)
    best_E = Inf
    for _read in 1:num_reads
        q, E = _dwave_sa(h, Js, L, Qpair, iidx, jidx, betas, rng)
        if E < best_E
            best_E = E
            best_q = q
        end
    end

    # Decode the binary solution into the continuous spectrum
    spectrum = binary_to_spectrum(best_q, n_bins; n_bits=Int(n_bits), max_value=mv)
    spectrum = max.(spectrum, 0.0)

    residual = b .- A * spectrum
    return UnfoldResult(
        Vector{T}(spectrum), Int(max_iterations), isfinite(best_E), T(norm(residual)),
        Dict{String,Any}(
            "n_bits" => Int(n_bits),
            "energy" => best_E,
            "num_reads" => Int(num_reads),
            "annealing_time" => Int(annealing_time),
            "max_value" => mv,
            "beta_range" => (hot, cold),
        ))
end
