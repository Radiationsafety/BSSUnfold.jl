# Regularization tests (port of tests/test_regularization_new_criteria.py)
using Test
using BSSUnfold
using LinearAlgebra
using Random

@testset "Regularization selection — GCV" begin
    rng = MersenneTwister(42)
    n = 50
    A = rand(rng, 14, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = abs.(randn(rng, n)) .+ 0.05
    b = A * x_true .+ 0.01 .* randn(rng, 14)
    x0 = ones(n) .* 0.1

    result = select_regularization_parameter(A, b, x0, method=:gcv)
    @test result.lambda > 0
    @test result.lambda isa Real
    @test result.method == :gcv
end

@testset "Regularization selection — discrepancy principle" begin
    rng = MersenneTwister(123)
    n = 50
    A = rand(rng, 14, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = abs.(randn(rng, n)) .+ 0.05
    noise_level = 0.01
    b = A * x_true .+ noise_level .* randn(rng, 14)
    x0 = ones(n) .* 0.1

    result = select_regularization_parameter(A, b, x0, method=:discrepancy,
                                             noise_level=noise_level)
    @test result.lambda > 0
    @test result.method == :discrepancy
    @test hasproperty(result.info, :target_residual)
end

@testset "Regularization selection — L-curve" begin
    rng = MersenneTwister(7)
    n = 30
    A = rand(rng, 14, n) .+ 0.3
    A ./= sum(A, dims=2)
    b = A * (abs.(randn(rng, n)) .+ 0.05) .+ 0.01 .* randn(rng, 14)
    x0 = ones(n) .* 0.1

    result = select_regularization_parameter(A, b, x0, method=:lcurve)
    @test result.lambda > 0
    @test result.method == :lcurve
end

@testset "Regularization selection — invalid method" begin
    A = rand(5, 10)
    b = rand(5)
    x0 = ones(10)
    @test_throws ArgumentError select_regularization_parameter(A, b, x0, method=:unknown)
end

@testset "Tikhonov — λ=0 degenerates to least squares" begin
    rng = MersenneTwister(33)
    A = rand(rng, 20, 10) .+ 0.5  # overdetermined
    x_true = abs.(randn(rng, 10))
    b = A * x_true
    x0 = ones(10)

    # As λ → 0, Tikhonov → A\b
    res_tiny = solve_tikhonov(A, b, x0, regularization=1e-10)
    x_lstsq = A \ b
    @test norm(res_tiny.spectrum .- max.(x_lstsq, 0)) < 0.1 * norm(x_true)
end
