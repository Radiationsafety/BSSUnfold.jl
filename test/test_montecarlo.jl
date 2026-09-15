# Monte-Carlo tests (port of tests/test_new_ensemble_refinement.py)
using Test
using BSSUnfold
using LinearAlgebra
using Random
using Statistics

@testset "Monte-Carlo uncertainty — basic" begin
    rng = MersenneTwister(42)
    n = 100
    A = rand(rng, 14, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = abs.(randn(rng, n)) .+ 0.05
    b = A * x_true .+ 0.005 .* randn(rng, 14)
    x0 = ones(n) .* 0.1

    mc = monte_carlo_uncertainty(solve_mlem, A, b, x0, 0.01, 20,
                                 random_state=42, max_iterations=100)

    @test size(mc.all) == (20, n)
    @test length(mc.mean) == n
    @test length(mc.std) == n
    @test length(mc.median) == n
    @test length(mc.p5) == n
    @test length(mc.p95) == n
    @test length(mc.min) == n
    @test length(mc.max) == n
    @test all(mc.std .≥ 0)
end

@testset "Monte-Carlo uncertainty — reproducibility" begin
    rng = MersenneTwister(123)
    A = rand(rng, 10, 30) .+ 0.3
    A ./= sum(A, dims=2)
    b = A * (abs.(randn(rng, 30)) .+ 0.05) .+ 0.01 .* randn(rng, 10)
    x0 = ones(30) .* 0.1

    mc1 = monte_carlo_uncertainty(solve_mlem, A, b, x0, 0.01, 10,
                                  random_state=42, max_iterations=50)
    mc2 = monte_carlo_uncertainty(solve_mlem, A, b, x0, 0.01, 10,
                                  random_state=42, max_iterations=50)
    # With the same seed the statistics must be identical
    @test mc1.mean ≈ mc2.mean
    @test mc1.std ≈ mc2.std
    @test mc1.all ≈ mc2.all
end

@testset "Monte-Carlo uncertainty — noise level effect" begin
    rng = MersenneTwister(7)
    n = 50
    A = rand(rng, 14, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = abs.(randn(rng, n)) .+ 0.05
    b = A * x_true
    x0 = ones(n) .* 0.1

    mc_low = monte_carlo_uncertainty(solve_mlem, A, b, x0, 0.001, 30,
                                     random_state=42, max_iterations=200)
    mc_high = monte_carlo_uncertainty(solve_mlem, A, b, x0, 0.05, 30,
                                       random_state=42, max_iterations=200)
    # Higher noise → larger std
    @test mean(mc_high.std) > mean(mc_low.std)
end

@testset "Monte-Carlo uncertainty — multiple algorithms" begin
    rng = MersenneTwister(99)
    n = 80
    A = rand(rng, 14, n) .+ 0.3
    A ./= sum(A, dims=2)
    b = A * (abs.(randn(rng, n)) .+ 0.05) .+ 0.005 .* randn(rng, 14)
    x0 = ones(n) .* 0.1

    for solve_fn in [solve_mlem, solve_gravel, solve_cgls]
        mc = monte_carlo_uncertainty(solve_fn, A, b, x0, 0.01, 10,
                                     random_state=42, max_iterations=200)
        @test size(mc.all) == (10, n)
        @test all(isfinite.(mc.mean))
    end
end

@testset "Monte-Carlo — add_noise helper" begin
    readings = Dict("a" => 1.0, "b" => 2.0, "c" => 3.0)
    rng = MersenneTwister(42)
    noisy = add_noise(readings, 0.1, rng)
    # All keys preserved
    @test Set(keys(noisy)) == Set(keys(readings))
    # Values close to the originals (within 3σ)
    for (k, v) in readings
        @test abs(noisy[k] - v) < 3 * 0.1 * v
    end
end
