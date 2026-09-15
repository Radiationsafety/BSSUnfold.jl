"""
QUBO-based neutron spectrum unfolding using quantum-inspired annealing
(port of unfold_qubo.py, pyqubo + dwave-neal → native simulated annealing).

The module implements the QUBO (Quadratic Unconstrained Binary Optimization)
formulation of the neutron spectrum unfolding problem, solved by
quantum-inspired simulated annealing (in the Python original —
D-Wave Neal) or other QUBO solvers.

The approach discretizes the spectrum into binary variables and formulates
the unfolding as:

    min_x ||A x - b||^2 + λ * R(x)      subject to x >= 0,

where the spectrum is represented in binary encoding for QUBO compatibility.

Each energy bin is encoded by `n_bits` binary variables
(fractional binary expansion): `x[i] = max_value * Σ_j b_ij * 2^-(j+1)`.

The problem reduces to the QUBO Hamiltonian

    E(q) = q' Q q + l' q,
    Q = (A T)' (A T) + regularization * I,   l = -2 (A T)' b,

where T is the transition matrix from bits to the continuous spectrum.

The Python original used `pyqubo` (Hamiltonian compilation) and
`dwave-neal` (simulated annealing). The Julia port implements both stages
**natively** (without external dependencies): the Hamiltonian is assembled
matrix-wise, and the sampler is classical simulated annealing with a
Metropolis criterion and a geometric temperature schedule adapted to the
energy scale of the problem (analogous to SimulatedAnnealingSampler from dwave-neal).
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

# ─── Simulated annealing ───────────────────────────────────────────────────

"""
    _simulated_annealing_qubo(Q, l; num_reads=10, num_sweeps=1000, rng)

Simulated annealing for the QUBO `E(q) = q'Qq + l'q`, q ∈ {0,1}^N.

Metropolis criterion with a geometric temperature schedule adapted to
the problem scale: T0 is estimated from the spread of energies of random
states, T1 = 1e-4 * T0.  Returns (best q, best energy).
"""
function _simulated_annealing_qubo(Q::AbstractMatrix{Float64},
                                  l::Vector{Float64};
                                  num_reads::Integer=10,
                                  num_sweeps::Integer=1000,
                                  rng::AbstractRNG=MersenneTwister())
    N = length(l)

    # Initial temperature: spread of energies of random states
    sample_energies = Float64[]
    for _ in 1:min(50, max(10, num_reads))
        q = rand(rng, 0:1, N)
        E = dot(q, Q * q) + dot(l, q)
        push!(sample_energies, E)
    end
    E_std = isempty(sample_energies) ? 1.0 : std(sample_energies)
    T0 = max(E_std * 1.5, 1e-3)
    T1 = T0 * 1e-4
    rate = (T1 / T0)^(1 / max(num_sweeps - 1, 1))

    best_q = zeros(Int, N)
    best_E = Inf

    for _read in 1:num_reads
        q = rand(rng, 0:1, N)
        h = Q * q .+ l          # h_i = (Qq)_i + l_i
        E = dot(q, Q * q) + dot(l, q)
        T = T0
        for _sweep in 1:num_sweeps
            for i in 1:N
                # ΔE of flipping bit i: (1-2q_i) * (2h_i - l_i + Q_ii)
                dE = (1 - 2 * q[i]) * (2 * h[i] - l[i] + Q[i, i])
                if dE <= 0 || rand(rng) < exp(-dE / T)
                    flip = 1 - 2 * q[i]
                    q[i] = 1 - q[i]
                    E += dE
                    # Incremental update of h: h += Q[:, i] * flip
                    axpy!(flip, view(Q, :, i), h)
                end
            end
            T *= rate
        end
        if E < best_E
            best_E = E
            best_q = copy(q)
        end
    end
    return best_q, best_E
end

# ─── Main solver ────────────────────────────────────────────────────────

"""
    solve_qubo(A, b, x0; n_bits=6, max_value=nothing, regularization=0.01,
               max_iterations=1000, annealing_time=1000, num_reads=10,
               random_state=nothing) -> UnfoldResult

Solve the unfolding problem via a QUBO formulation with simulated annealing.

# Arguments
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: measurements (m,)
- `x0::AbstractVector{T}`: initial guess for scaling
  (used to estimate `max_value` if it is not given)
- `n_bits`: bits per energy bin (default 6)
- `max_value`: maximum value of the spectrum for scaling;
  `nothing` — estimated from the data (2 * max(x0) or pseudoinverse)
- `regularization`: regularization parameter (default 0.01)
- `max_iterations`: maximum number of iterations (returned as
  `iterations`, default 1000)
- `annealing_time`: number of annealing sweeps (default 1000)
- `num_reads`: number of independent reads (default 10)
- `random_state`: seed for reproducibility

# Returns
`UnfoldResult` with the spectrum; `extra` contains `n_bits`, `energy`, `num_reads`,
`annealing_time`.
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

    # Estimate max_value if not given
    mv = if max_value !== nothing
        Float64(max_value)
    elseif any(>(0), x0)
        2 * maximum(x0)
    else
        # Rough estimate from the pseudoinverse
        try
            x_est = qr(A, ColumnNorm()) \ b
            2 * maximum(abs.(x_est))
        catch
            1.0
        end
    end
    mv <= 0 && (mv = 1.0)

    # Transition matrix from bits to the continuous spectrum:
    # x_cont[i] = Σ_j (binary[i*n_bits + j] * 2^-(j+1)) * max_value
    n_binary = n_bins * n_bits
    Tmat = zeros(n_bins, n_binary)
    for i in 1:n_bins, j in 1:n_bits
        Tmat[i, (i - 1) * n_bits + j] = 2.0^(-j) * mv
    end

    # Effective response matrix acting on the bit vector: A_scaled = A * Tmat
    A_scaled = A * Tmat

    # QUBO Hamiltonian: ||A_scaled q - b||^2 + reg * ||q||^2
    # = q'(A'A)q - 2 b'A q + b'b  +  reg * q'q
    Q = A_scaled' * A_scaled + Float64(regularization) * Matrix{Float64}(I, n_binary, n_binary)
    l = vec(-2 * (A_scaled' * b))

    # Simulated annealing
    best_q, best_E = _simulated_annealing_qubo(Q, l;
                                               num_reads=num_reads,
                                               num_sweeps=annealing_time,
                                               rng=rng)

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
        ))
end
