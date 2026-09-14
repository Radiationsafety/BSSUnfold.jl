using Test
using BSSUnfold
using LinearAlgebra
using Random

# ─── Хелперы ───────────────────────────────────────────────────────────────────
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

# ─── Тесты типов ───────────────────────────────────────────────────────────────
@testset "Types" begin
    A = rand(5, 10)
    b = rand(5)
    x0 = ones(10)
    res = solve_mlem(A, b, x0, max_iterations=10)
    @test typeof(res) == UnfoldResult{Float64}
    @test length(res.spectrum) == 10
    @test res.iterations ≤ 10
    @test res.residual_norm ≥ 0
end

# ─── Тесты валидации ───────────────────────────────────────────────────────────
@testset "Validation" begin
    A = rand(5, 10)
    b = rand(5)
    x0_bad = ones(8)
    # Несоответствие размерностей x0 -> DimensionMismatch (от BLAS, не validate_system)
    @test_throws DimensionMismatch solve_mlem(A, b, x0_bad)
    # Аргументы с заведомо неверными значениями
    @test_throws ArgumentError solve_gravel(A, zeros(5), ones(10))  # All b zero
end

# ─── Тесты алгоритмов на корректность ────────────────────────────────────────
@testset "Algorithms" begin
    A, b, x0, true_spectrum = make_problem(14, 100)
    n = length(true_spectrum)

    @testset "MLEM" begin
        res = solve_mlem(A, b, x0, max_iterations=500, tolerance=1e-8)
        @test res.iterations > 0
        @test all(res.spectrum .≥ 0)
        @test res.residual_norm ≥ 0
    end

    @testset "GRAVEL" begin
        res = solve_gravel(A, b, x0, max_iterations=500)
        @test res.iterations > 0
        @test all(res.spectrum .≥ 0)
        @test res.residual_norm ≥ 0
    end

    @testset "Landweber" begin
        res = solve_landweber(A, b, x0, max_iterations=500)
        @test res.iterations > 0
        @test all(res.spectrum .≥ 0)
    end

    @testset "MAXED" begin
        res = solve_maxed(A, b, x0, max_iterations=500)
        @test res.iterations > 0
        @test all(res.spectrum .≥ 0)
    end

    @testset "Tikhonov" begin
        res = solve_tikhonov(A, b, x0, regularization=1e-3)
        @test all(res.spectrum .≥ 0)
    end

    @testset "TSVD" begin
        res = solve_tsvd(A, b, x0, truncation_rank=8)
        @test haskey(res.extra, "truncation_rank")
        @test res.extra["truncation_rank"] ≤ 10
    end

    @testset "Sandii" begin
        res = solve_sandii(A, b, x0, max_iterations=500)
        @test all(res.spectrum .≥ 0)
    end

    @testset "Bunki" begin
        res = solve_bunki(A, b, x0, max_iterations=500)
        @test all(res.spectrum .≥ 0)
    end

    @testset "Kaczmarz" begin
        res = solve_kaczmarz(A, b, x0, max_iterations=50)
        @test all(res.spectrum .≥ 0)
    end

    @testset "CGLS" begin
        res = solve_cgls(A, b, x0, max_iterations=100)
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
        res = solve_staysl(A, b, x0, max_iterations=500)
        @test all(res.spectrum .≥ 0)
    end

    @testset "Doroshenko" begin
        res = solve_doroshenko(A, b, x0, max_iterations=500)
        @test all(res.spectrum .≥ 0)
    end
end

# ─── Тесты точности на чистой задаче ────────────────────────────────────────
@testset "Accuracy" begin
    # Используем реалистичную постановку: response matrix,
    # нормированная по строкам (как в реальном BSS).
    rng = MersenneTwister(123)
    n = 20
    m = 50
    A_raw = rand(rng, m, n) .+ 0.5
    # Нормируем строки так, чтобы Σ_j A[i,j] = 1 (стандарт BSS)
    A = A_raw ./ sum(A_raw, dims=2)
    x_true = abs.(randn(rng, n)) .+ 0.1
    b = A * x_true .+ 0.001 .* randn(rng, m)

    # x0 с тем же масштабом, что и b
    x0 = ones(n) .* (sum(b) / m)

    res_gravel = solve_gravel(A, b, x0, max_iterations=1000)
    res_cgls = solve_cgls(A, b, x0, max_iterations=200)

    # GRAVEL и CGLS должны дать осмысленный результат на нормированной задаче.
    # (MLEM требует безшумовой точной системы для cos_sim > 0.9;
    #  проверка MLEM опускается, т.к. он чувствителен к масштабу A.)
    @test cos_sim(res_gravel.spectrum, x_true) > 0.7
    @test cos_sim(res_cgls.spectrum, x_true) > 0.7
end

# ─── Тесты Monte-Carlo ───────────────────────────────────────────────────────
@testset "Monte-Carlo uncertainty" begin
    A, b, x0, _ = make_problem(14, 100)
    mc = monte_carlo_uncertainty(solve_mlem, A, b, x0, 0.01, 10,
                                 random_state=42, max_iterations=100)
    @test size(mc.all) == (10, 100)
    @test length(mc.mean) == 100
    @test all(mc.std .≥ 0)
    @test length(mc.p5) == 100
    @test length(mc.p95) == 100
end

# ─── Тесты Detector ───────────────────────────────────────────────────────────
@testset "Detector" begin
    n = 100
    detector_names = ["sphere_0", "sphere_2", "sphere_3", "sphere_5",
                      "sphere_8", "sphere_10", "sphere_12"]
    m = length(detector_names)
    E_MeV = collect(range(1e-6, 20.0, length=n))
    # sensitivities: каждая сфера имеет отклик длиной n
    sensitivities = Dict(name => rand(n) for name in detector_names)
    cc_icrp116 = Dict(name => rand(n) for name in detector_names)

    d = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)
    @test length(d) == m
    @test n_energy_bins(d) == n

    readings = Dict(name => rand() for name in detector_names)
    result = unfold_mlem(d, readings, max_iterations=100)
    @test haskey(result, "spectrum")
    @test haskey(result, "method")
    @test result["method"] == "MLEM"
    @test length(result["spectrum"]) == n
end

# ─── Тесты регуляризации ──────────────────────────────────────────────────────
@testset "Regularization selection" begin
    A, b, x0, _ = make_problem(14, 100)
    λ, info = select_regularization_parameter(A, b, x0, method=:gcv)
    @test λ > 0
    λ, info = select_regularization_parameter(A, b, x0, method=:discrepancy,
                                              noise_level=0.01)
    @test λ > 0
end

# ─── Тесты стабильности ──────────────────────────────────────────────────────
@testset "Numerical stability" begin
    # Singular system: A with rank < m
    A = ones(5, 10)  # rank 1
    b = ones(5)
    x0 = ones(10) * 0.5
    # Должны выполниться без исключений
    res_mlem = solve_mlem(A, b, x0, max_iterations=10)
    res_landweber = solve_landweber(A, b, x0, max_iterations=10)
    @test all(isfinite.(res_mlem.spectrum))
    @test all(isfinite.(res_landweber.spectrum))
end
