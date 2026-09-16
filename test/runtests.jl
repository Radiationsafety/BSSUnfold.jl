# Main entry point for BSSUnfold.jl tests
# Run: julia --project=. -e 'using Pkg; Pkg.test()'

using Test
using BSSUnfold
using LinearAlgebra
using Random
using Statistics

# Helpers (used across multiple test files)
function make_problem(m::Int=14, n::Int=640; seed::Int=42)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .* 0.99 .+ 0.01
    E = collect(range(1f-9, 20.0; length=n))
    true_spectrum = exp.(-E ./ 1.5) .* (1.0 .+ 0.3 .* sin.(E))
    true_spectrum ./= sum(true_spectrum)
    b = A * true_spectrum .+ 0.01 .* randn(rng, m)
    x0 = ones(n) .* 0.5
    return A, b, x0, true_spectrum
end

cos_sim(a, b) = dot(a, b) / (norm(a) * norm(b) + 1f-30)

# ─── Include sub-test files ───────────────────────────────────────────────────
println("=" ^ 70)
println("BSSUnfold.jl — running test suite")
println("=" ^ 70)

# Basic smoke tests + base algorithms (built-in set)
@testset "BSSUnfold — core types" begin
    A = rand(5, 10)
    b = rand(5)
    x0 = ones(10)
    res = solve_mlem(A, b, x0, max_iterations=10)
    @test typeof(res) == UnfoldResult{Float64}
    @test length(res.spectrum) == 10
    @test res.iterations ≤ 10
    @test res.residual_norm ≥ 0
end

@testset "BSSUnfold — input validation" begin
    A = rand(5, 10)
    b = rand(5)
    x0_bad = ones(8)
    @test_throws DimensionMismatch solve_mlem(A, b, x0_bad)
    @test_throws ArgumentError solve_gravel(A, zeros(5), ones(10))
end

@testset "BSSUnfold — all algorithms smoke" begin
    A, b, x0, _ = make_problem(14, 100)
    # Algorithms that accept max_iterations as a kwarg
    iterative_algos = [solve_mlem, solve_gravel, solve_landweber, solve_maxed,
                      solve_tikhonov, solve_tsvd, solve_sandii, solve_bunki,
                      solve_kaczmarz, solve_cgls, solve_fista, solve_bsrem,
                      solve_osem, solve_staysl, solve_doroshenko,
                      solve_lanczos, solve_randomized_kaczmarz]
    for fn in iterative_algos
        res = fn(A, b, x0, max_iterations=50)
        @test all(res.spectrum .≥ 0)
        @test all(isfinite.(res.spectrum))
    end
    # Iterative refinement has a different signature (without a top-level max_iterations)
    res_ir = solve_iterative_refinement(A, b, x0,
                                       first_pass_kwargs=(max_iterations=50,),
                                       second_pass_kwargs=(max_iterations=30,))
    @test all(res_ir.spectrum .≥ 0)
    @test all(isfinite.(res_ir.spectrum))
end

@testset "BSSUnfold — accuracy on clean problem" begin
    rng = MersenneTwister(123)
    n = 20; m = 50
    A_raw = rand(rng, m, n) .+ 0.5
    A = A_raw ./ sum(A_raw, dims=2)
    x_true = abs.(randn(rng, n)) .+ 0.1
    b = A * x_true .+ 0.001 .* randn(rng, m)
    x0 = ones(n) .* (sum(b) / m)
    res_gravel = solve_gravel(A, b, x0, max_iterations=1000)
    res_cgls = solve_cgls(A, b, x0, max_iterations=200)
    @test cos_sim(res_gravel.spectrum, x_true) > 0.7
    @test cos_sim(res_cgls.spectrum, x_true) > 0.7
end

@testset "BSSUnfold — numerical stability" begin
    A = ones(5, 10)  # rank 1
    b = ones(5)
    x0 = ones(10) * 0.5
    res_mlem = solve_mlem(A, b, x0, max_iterations=10)
    res_landweber = solve_landweber(A, b, x0, max_iterations=10)
    @test all(isfinite.(res_mlem.spectrum))
    @test all(isfinite.(res_landweber.spectrum))
end

# ─── Include separate test files ─────────────────────────────────────────────
include("test_detector.jl")
include("test_classic_unfolders.jl")
include("test_comparison.jl")
include("test_montecarlo.jl")
include("test_regularization.jl")
include("test_iaea_validation.jl")
include("test_iaea_dose.jl")
include("test_iaea_methods.jl")
include("test_new_algorithms.jl")
include("test_dose_interpolation.jl")
include("test_ported_methods.jl")
include("test_batch3_algorithms.jl")
include("test_comparison_metrics.jl")
include("test_dev_methods.jl")

println("\n" * "=" ^ 70)
println("BSSUnfold.jl — all tests finished")
println("=" ^ 70)
