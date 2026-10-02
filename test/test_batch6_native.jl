# Batch 6 — Bucket-D: native ports without external deps (lavrentiev,
# mirror_descent, osem_anlm, tikhonov_sobolev_dp, bayesian_parametric).
# No optional backend required: these run in both base env and env/jump.

using Test, LinearAlgebra, Random
using BSSUnfold

function _make_problem6(m::Int, n::Int, seed::Int=7)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.1
    x_true = range(0.15, 1.0; length=n) .+ 0.05 .* rand(rng, n)
    b = A * x_true .+ 0.005 * randn(rng, m)
    x0 = fill(0.5, n)
    (A, b, x0, x_true)
end

@testset "Batch6 — solve_lavrentiev" begin
    A, b, x0, xt = _make_problem6(10, 15)
    # :gram is the default; equivalent to zeroth-order Tikhonov.
    r_gram = solve_lavrentiev(A, b, x0; alpha=1e-3, form=:gram)
    @test r_gram isa UnfoldResult
    @test length(r_gram.spectrum) == 15
    @test all(r_gram.spectrum .>= 0)
    @test norm(A * r_gram.spectrum - b) < norm(b)

    # :padded (rectangular allowed): last (n-m) components forced to ~0.
    r_pad = solve_lavrentiev(A, b, x0; alpha=1e-2, form=:padded)
    @test length(r_pad.spectrum) == 15

    # :iterated reduces to :gram at n_iterations=1 and decreases residual with
    # more iterations on a well-behaved problem.
    r_it1 = solve_lavrentiev(A, b, x0; alpha=1e-2, form=:iterated, n_iterations=1)
    r_it5 = solve_lavrentiev(A, b, x0; alpha=1e-2, form=:iterated, n_iterations=5)
    @test r_it5.iterations == 5
    @test norm(A * r_it5.spectrum - b) <= norm(A * r_it1.spectrum - b) + 1e-9

    # :direct requires square A.
    @test_throws ArgumentError solve_lavrentiev(A, b, x0; form=:direct)

    Asq = rand(MersenneTwister(2), 8, 8)
    bsq = Asq * ones(8) .+ 0.01 * randn(MersenneTwister(3), 8)
    rd = solve_lavrentiev(Asq, bsq, fill(0.5, 8); alpha=0.05, form=:direct)
    @test length(rd.spectrum) == 8

    # Validation: negative alpha rejected; iterated rejects q ∉ (0,1].
    @test_throws ArgumentError solve_lavrentiev(A, b, x0; alpha=-0.1)
    @test_throws ArgumentError solve_lavrentiev(A, b, x0; form=:iterated, q=0.0)
    @test_throws ArgumentError solve_lavrentiev(A, b, x0; form=:iterated, q=1.5)
    @test_throws ArgumentError solve_lavrentiev(A, b, x0; form=:iterated, n_iterations=0)
    @test_throws ArgumentError solve_lavrentiev(A, b, x0; form=:unknown)
end

@testset "Batch6 — solve_mirror_descent" begin
    A, b, x0, xt = _make_problem6(10, 15)
    F = sum(xt)
    # Entropy map recovers a simplex-normalised spectrum.
    r_ent = solve_mirror_descent(A, b, x0; mirror_map=:entropy,
                                  total_fluence=F, max_iterations=500, tolerance=1e-5)
    @test r_ent isa UnfoldResult
    @test all(r_ent.spectrum .>= -1e-12)
    @test abs(sum(r_ent.spectrum) - F) < 1e-6

    for mm in (:log, :l2, :pnorm)
        r = solve_mirror_descent(A, b, x0; mirror_map=mm, max_iterations=300, tolerance=1e-4)
        @test length(r.spectrum) == 15
        @test all(r.spectrum .>= -1e-6)
    end
    # pnorm rejects p ≤ 1; entropy requires F > 0; unknown map rejected.
    @test_throws ArgumentError solve_mirror_descent(A, b, x0; mirror_map=:pnorm, p=0.5)
    @test_throws ArgumentError solve_mirror_descent(A, b, x0; mirror_map=:entropy, total_fluence=0.0)
    @test_throws ArgumentError solve_mirror_descent(A, b, x0; mirror_map=:bogus)
    # Fixed step_size path must also run.
    r = solve_mirror_descent(A, b, x0; mirror_map=:l2, step_size=1e-3, max_iterations=200)
    @test length(r.spectrum) == 15
end

@testset "Batch6 — estimate_noise_1d / anlm_filter_1d" begin
    # Pure linear signal + iid noise: MAD-of-second-difference ≈ σ
    # (second difference annihilates linear trends, exactly the point of
    # Immerkaer's estimator — a sine would leak its own curvature).
    rng = MersenneTwister(123)
    t = range(0, 20; length=400)
    linear = 0.3 .* t
    noisy = linear .+ 0.05 .* randn(rng, length(t))
    @test 0.02 < estimate_noise_1d(noisy) < 0.10
    @test estimate_noise_1d(linear) < 1e-10   # exactly linear → MAD = 0
    @test estimate_noise_1d([1.0, 2.0]) == 0.0

    # ANLM filter smooths noise but preserves a piecewise-constant plateau.
    step = vcat(fill(1.0, 30), fill(2.0, 30))
    noisy_step = step .+ 0.05 .* randn(MersenneTwister(9), length(step))
    filtered = anlm_filter_1d(max.(noisy_step, 1e-6); search_window=9,
                                       similarity_window=3, log_space=true)
    @test length(filtered) == 60
    @test all(filtered .> 0)
    # Variance reduction: MAD-of-second-diff must drop.
    @test estimate_noise_1d(filtered) < estimate_noise_1d(noisy_step)
    # search_window=1 → identity.
    x0 = collect(1.0:5.0)
    @test anlm_filter_1d(x0; search_window=1) == x0
    @test_throws ArgumentError anlm_filter_1d(Float64[])
    @test_throws ArgumentError anlm_filter_1d(x0; alpha=0.0)
end

@testset "Batch6 — solve_osem_anlm" begin
    A, b, x0, xt = _make_problem6(10, 15)
    # n_subsets=1 is MLEM+ANLM per iteration; must be non-negative and fit data.
    r = solve_osem_anlm(A, b, x0; max_iterations=15, n_subsets=1,
                        search_window=7, similarity_window=3)
    @test r isa UnfoldResult
    @test all(r.spectrum .>= 0)
    @test length(r.spectrum) == 15

    r_post = solve_osem_anlm(A, b, x0; max_iterations=15, n_subsets=2,
                             anlm_mode=:post)
    @test r_post isa UnfoldResult

    @test_throws ArgumentError solve_osem_anlm(A, b, x0; n_subsets=0)
    @test_throws ArgumentError solve_osem_anlm(A, b, x0; n_subsets=size(A,1)+1)
    @test_throws ArgumentError solve_osem_anlm(A, b, x0; anlm_mode=:bogus)
    @test_throws ArgumentError solve_osem_anlm(A, b, x0; h=-1.0)
end

@testset "Batch6 — solve_tikhonov_sobolev_dp" begin
    A, b, x0, xt = _make_problem6(12, 20)
    # Discrepancy principle: residual² ≈ δ² when the bracket covers the root.
    r = solve_tikhonov_sobolev_dp(A, b, x0; noise_level=0.05, penalty=:sobolev)
    @test r isa UnfoldResult
    @test length(r.spectrum) == 20
    @test r.extra["discrepancy_status"] in (0, 1, 2)
    if r.extra["discrepancy_status"] == 0
        δ = r.extra["delta"]
        @test abs(sqrt(r.extra["residual_sq"]) - δ) / δ < 1e-3
        @test r.converged
    end

    # Curvature and identity penalties must produce solutions too.
    for pen in (:curvature, :identity)
        rp = solve_tikhonov_sobolev_dp(A, b, x0; noise_level=0.05, penalty=pen)
        @test length(rp.spectrum) == 20
    end

    # Validation errors.
    @test_throws ArgumentError solve_tikhonov_sobolev_dp(A, b, x0; delta=-1.0)
    @test_throws ArgumentError solve_tikhonov_sobolev_dp(A, b, x0; penalty=:bogus)
    @test_throws ArgumentError solve_tikhonov_sobolev_dp(A, b, x0; alpha_range=(0.0, 1.0))
    @test_throws ArgumentError solve_tikhonov_sobolev_dp(A, b, x0; method=:bogus)

    # Newton-Kantorovich path.
    rn = solve_tikhonov_sobolev_dp(A, b, x0; noise_level=0.05, method=:newton_kantorovich)
    @test length(rn.spectrum) == 20
end

@testset "Batch6 — parametric_model_fp / solve_bayesian_parametric" begin
    n = 25
    E = exp10.(range(-9, 1.3; length=n))
    log_steps = BSSUnfold.compute_log_steps(E) .* log(10)
    # FRUIT-like model has three regimes; check mask boundaries.
    spec = parametric_model_fp(E, 1e-7, 0.025e-6, 1e-7, 1e-7, 2.0)
    @test length(spec) == n
    @test all(spec[findall(<(0.4e-6), E)] .>= 0)
    @test all(spec[findall(>=(0.1), E)] .>= 0)

    # Build a tiny response matrix over the grid and run MH.
    m = 6
    rng = MersenneTwister(4)
    A = rand(rng, m, n) .+ 0.01
    b = A * (spec .* log_steps) .+ 1e-6 * randn(rng, m)

    r1 = solve_bayesian_parametric(A, b, E, log_steps; n_samples=50, burn_in=10,
                                                   random_state=11)
    @test r1 isa UnfoldResult
    @test r1.converged
    @test length(r1.spectrum) == n
    @test all(r1.spectrum .>= 0)
    # Determinism: same seed reproduces the same posterior mean.
    r2 = solve_bayesian_parametric(A, b, E, log_steps; n_samples=50, burn_in=10,
                                                   random_state=11)
    @test r2.spectrum == r1.spectrum
    @test r2.extra["mean_params"] == r1.extra["mean_params"]
end
