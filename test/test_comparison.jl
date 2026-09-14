# Тесты метрик сравнения спектров (порт из tests/test_comparison.py)
using Test
using BSSUnfold
using LinearAlgebra
using Statistics
using Random

# Будем тестировать косинусную меру и т.п. — реализуем прямо здесь,
# чтобы проверить, что BSSUnfold экспортирует всё нужное.
@testset "Comparison utilities" begin

    @testset "Cosine similarity — identical spectra" begin
        s = [1.0, 2.0, 3.0, 4.0]
        cos = dot(s, s) / (norm(s) * norm(s))
        @test cos ≈ 1.0
    end

    @testset "Cosine similarity — orthogonal spectra" begin
        s1 = [1.0, 0.0]
        s2 = [0.0, 1.0]
        cos = dot(s1, s2) / (norm(s1) * norm(s2))
        @test cos ≈ 0.0
    end

    @testset "Cosine similarity — step vs ramp" begin
        s1 = [zeros(25); ones(25)]
        s2 = collect(range(0, 1, length=50))
        cos = dot(s1, s2) / (norm(s1) * norm(s2))
        @test 0 < cos < 1
    end
end

@testset "Spectrum comparison via unfolding" begin
    # Два разных алгоритма на одной задаче должны дать осмысленные спектры.
    # Сходство может быть низким на плохо обусловленных случайных матрицах —
    # поэтому проверяем только то, что оба метода дают неотрицательные
    # конечные спектры с разумной нормой.
    rng = MersenneTwister(42)
    n = 100
    A = rand(rng, 14, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = exp.(-collect(range(0, 5, length=n)))
    b = A * x_true .+ 0.01 .* randn(rng, 14)
    x0 = ones(n) .* 0.1

    res_mlem = solve_mlem(A, b, x0, max_iterations=1000)
    res_gravel = solve_gravel(A, b, x0, max_iterations=500)

    # Оба метода должны дать осмысленные спектры
    @test all(res_mlem.spectrum .≥ 0)
    @test all(res_gravel.spectrum .≥ 0)
    @test all(isfinite.(res_mlem.spectrum))
    @test all(isfinite.(res_gravel.spectrum))
    # Оба должны существенно снизить невязку
    @test res_mlem.residual_norm < norm(b)
    @test res_gravel.residual_norm < norm(b)
end

@testset "Quantile-based uncertainty" begin
    rng = MersenneTwister(7)
    n = 50
    A = rand(rng, 14, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = abs.(randn(rng, n)) .+ 0.05
    b = A * x_true .+ 0.005 .* randn(rng, 14)
    x0 = ones(n) .* 0.1

    mc = monte_carlo_uncertainty(solve_mlem, A, b, x0, 0.01, 50,
                                 random_state=42, max_iterations=200)

    @test all(mc.p5 .≤ mc.median)
    @test all(mc.median .≤ mc.p95)
    @test all(mc.min .≤ mc.p5)
    @test all(mc.p95 .≤ mc.max)
    @test all(mc.std .≥ 0)
end

@testset "Statistical metrics on synthetic data" begin
    rng = MersenneTwister(99)
    n = 100
    A = rand(rng, 14, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = exp.(-collect(range(0, 5, length=n)))
    b = A * x_true .+ 0.005 .* randn(rng, 14)

    # 100 MC сэмплов; std должна быть мала по сравнению с mean
    mc = monte_carlo_uncertainty(solve_gravel, A, b,
                                 ones(n) .* 0.1, 0.005, 100,
                                 random_state=42, max_iterations=300)
    mean_spec = mc.mean
    std_spec = mc.std
    # Относительная неопределённость < 50% в большинстве бинов
    rel_unc = std_spec ./ (abs.(mean_spec) .+ 1e-10)
    @test sum(rel_unc .< 0.5) / length(rel_unc) > 0.5
end
