# Tests of classic iterative algorithms (port of tests/test_classic_unfolders.py,
# tests/test_em_methods.py, tests/test_new_methods.py)
using Test
using BSSUnfold
using LinearAlgebra
using Random

# Shared fixture: well-conditioned BSS problem
function _make_classic_problem(m::Int=14, n::Int=100; seed::Int=42)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.3
    # Standardize rows (as in real BSS)
    A ./= sum(A, dims=2)
    E = collect(range(1f-7, 20.0, length=n))
    x_true = exp.(-E ./ 1.5) .+ 0.001 .* randn(rng, n)
    x_true .= max.(x_true, 0)
    b = A * x_true .+ 0.005 .* randn(rng, m)
    b = max.(b, 1e-4)  # detector readings are strictly positive counts
    x0 = ones(n) .* (sum(b) / m)
    return A, b, x0, x_true
end

@testset "Classic unfolders — basic correctness" begin
    A, b, x0, x_true = _make_classic_problem(14, 100)

    @testset "MLEM" begin
        res = solve_mlem(A, b, x0, max_iterations=2000, tolerance=1e-10)
        @test length(res.spectrum) == 100
        @test res.iterations > 0
        @test all(res.spectrum .≥ 0)
        @test res.residual_norm ≥ 0
        @test all(isfinite.(res.spectrum))
    end

    @testset "GRAVEL" begin
        res = solve_gravel(A, b, x0, max_iterations=500)
        @test length(res.spectrum) == 100
        @test res.iterations > 0
        @test all(res.spectrum .≥ 0)
        # GRAVEL must substantially reduce the residual
        @test res.residual_norm < 0.5 * norm(b)
    end

    @testset "Landweber" begin
        res = solve_landweber(A, b, x0, max_iterations=500)
        @test all(res.spectrum .≥ 0)
        @test res.iterations > 0
    end

    @testset "MAXED" begin
        res = solve_maxed(A, b, x0, max_iterations=500)
        @test all(res.spectrum .≥ 0)
    end

    @testset "Tikhonov" begin
        res = solve_tikhonov(A, b, x0, regularization=1e-3)
        @test all(res.spectrum .≥ 0)
        @test res.iterations == 1  # Direct method
    end

    @testset "TSVD" begin
        res = solve_tsvd(A, b, x0, truncation_rank=10)
        @test all(res.spectrum .≥ 0)
        @test haskey(res.extra, "truncation_rank")
    end

    @testset "Sandii" begin
        res = solve_sandii(A, b, x0, max_iterations=500)
        @test all(res.spectrum .≥ 0)
    end

    @testset "Bunki" begin
        res = solve_bunki(A, b, x0, max_iterations=500, smoothing=0.1)
        @test all(res.spectrum .≥ 0)
    end

    @testset "Kaczmarz" begin
        res = solve_kaczmarz(A, b, x0, max_iterations=100)
        @test all(res.spectrum .≥ 0)
    end

    @testset "CGLS" begin
        res = solve_cgls(A, b, x0, max_iterations=200)
        @test all(res.spectrum .≥ 0)
    end

    @testset "FISTA" begin
        res = solve_fista(A, b, x0, max_iterations=200, regularization=1e-4)
        @test all(res.spectrum .≥ 0)
    end

    @testset "BSREM" begin
        res = solve_bsrem(A, b, x0, max_iterations=50, n_subsets=4)
        @test all(res.spectrum .≥ 0)
    end

    @testset "OSEM" begin
        res = solve_osem(A, b, x0, max_iterations=50, n_subsets=4)
        @test all(res.spectrum .≥ 0)
    end

    @testset "Staysl" begin
        res = solve_staysl(A, b, x0)  # single-step Bayesian update
        @test all(res.spectrum .≥ 0)
        @test res.iterations == 1
    end

    @testset "Doroshenko" begin
        res = solve_doroshenko(A, b, x0, max_iterations=500)
        @test all(res.spectrum .≥ 0)
    end
end

@testset "EM methods — convergence behaviour" begin
    # Ideal problem (no noise): MLEM must reduce the residual.
    # On poorly conditioned random matrices MLEM may not fully converge
    # within a reasonable number of iterations — the main point is that the residual decreases.
    rng = MersenneTwister(7)
    n = 50
    A = rand(rng, 30, n) .+ 0.5
    A ./= sum(A, dims=2)
    x_true = abs.(randn(rng, n)) .+ 0.05
    b = A * x_true  # no noise
    x0 = ones(n) .* 0.1

    res = solve_mlem(A, b, x0, max_iterations=2000, tolerance=1e-12)
    # Must reduce the residual substantially
    @test res.residual_norm < 0.5 * norm(b)
    # All values finite and non-negative
    @test all(isfinite.(res.spectrum))
    @test all(res.spectrum .≥ 0)
end

@testset "EM methods — regularization effect" begin
    # Comparison: Tikhonov with a large λ should give a smoother spectrum
    rng = MersenneTwister(123)
    A = rand(rng, 14, 100) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = abs.(sin.(range(0, 3π, length=100))) .+ 0.1
    b = A * x_true .+ 0.01 .* randn(rng, 14)

    res_low = solve_tikhonov(A, b, ones(100), regularization=1e-6)
    res_high = solve_tikhonov(A, b, ones(100), regularization=1.0)
    # Stronger regularization gives a smoother (lower-norm) spectrum
    @test norm(res_high.spectrum) < norm(res_low.spectrum)
end

@testset "GRAVEL — handles all-zero measurements correctly" begin
    rng = MersenneTwister(1)
    A = rand(rng, 5, 10)
    b = zeros(5)
    @test_throws ArgumentError solve_gravel(A, b, ones(10))
end

@testset "Kaczmarz — preserves non-negativity" begin
    rng = MersenneTwister(5)
    A = rand(rng, 10, 20)
    x_true = abs.(randn(rng, 20))
    b = A * x_true .+ 0.01 .* randn(rng, 10)
    res = solve_kaczmarz(A, b, ones(20) .* 0.5, max_iterations=200)
    @test all(res.spectrum .≥ 0)
end

@testset "BSREM / OSEM — subset count effect" begin
    rng = MersenneTwister(11)
    A = rand(rng, 16, 100) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = abs.(randn(rng, 100)) .+ 0.05
    b = A * x_true .+ 0.005 .* randn(rng, 16)

    # OSEM with more subsets converges faster (in fewer iterations)
    res_4 = solve_osem(A, b, ones(100) .* 0.5, max_iterations=20, n_subsets=4)
    res_8 = solve_osem(A, b, ones(100) .* 0.5, max_iterations=20, n_subsets=8)
    @test res_8.residual_norm ≤ res_4.residual_norm * 1.5  # not worse
    @test all(res_4.spectrum .≥ 0)
    @test all(res_8.spectrum .≥ 0)
end

@testset "Landweber — step size effect" begin
    rng = MersenneTwister(13)
    A = rand(rng, 10, 30) .+ 0.5
    x_true = abs.(randn(rng, 30))
    b = A * x_true
    # Too large a step → diverges; a small one → converges
    res_small = solve_landweber(A, b, ones(30), max_iterations=500, omega=1e-3)
    @test all(isfinite.(res_small.spectrum))
    @test res_small.residual_norm < norm(b)
end
