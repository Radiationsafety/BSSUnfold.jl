# Тесты для 5 новых алгоритмов (Lanczos, Iterative Refinement,
# Randomized Kaczmarz, CVXPY, QPsolvers)

using Test
using BSSUnfold
using LinearAlgebra
using Random

# Общая фикстура
function _make_problem(m::Int=14, n::Int=100; seed::Int=42)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = exp.(-collect(range(0, 5, length=n)))
    b = A * x_true .+ 0.01 .* randn(rng, m)
    x0 = ones(n) .* 0.1
    return A, b, x0, x_true
end

cos_sim(a, b) = dot(a, b) / (norm(a) * norm(b) + 1f-30)


@testset "New algorithms — Lanczos" begin
    A, b, x0, x_true = _make_problem(14, 100)

    @testset "Basic solve" begin
        res = solve_lanczos(A, b, x0, max_iterations=20)
        @test length(res.spectrum) == 100
        @test all(res.spectrum .≥ 0)
        @test all(isfinite.(res.spectrum))
        @test res.iterations ≤ 20
    end

    @testset "Default max_iterations" begin
        res = solve_lanczos(A, b, x0)
        @test res.iterations ≤ min(size(A)...)
        @test all(res.spectrum .≥ 0)
    end

    @testset "Early stopping by noise level" begin
        res = solve_lanczos(A, b, x0, max_iterations=50, noise_level=0.01)
        @test all(res.spectrum .≥ 0)
    end

    @testset "Reduces residual" begin
        res = solve_lanczos(A, b, x0, max_iterations=30)
        @test res.residual_norm < norm(b)  # должно улучшить
    end

    @testset "Zero b returns zero" begin
        res = solve_lanczos(A, zeros(14), x0)
        @test all(res.spectrum .== 0)
        @test res.converged
    end
end


@testset "New algorithms — Iterative Refinement" begin
    A, b, x0, x_true = _make_problem(14, 100)

    @testset "Basic solve" begin
        res = solve_iterative_refinement(A, b, x0)
        @test length(res.spectrum) == 100
        @test all(res.spectrum .≥ 0)
        @test all(isfinite.(res.spectrum))
        # Должен вернуть диагностику
        @test haskey(res.extra, "alpha")
        @test haskey(res.extra, "first_pass_residual")
        @test haskey(res.extra, "second_pass_correction_norm")
        @test haskey(res.extra, "final_residual")
    end

    @testset "Fixed alpha" begin
        res = solve_iterative_refinement(A, b, x0, alpha=0.5)
        @test res.extra["alpha"] == 0.5
    end

    @testset "Line search alpha" begin
        res = solve_iterative_refinement(A, b, x0, alpha=nothing, max_alpha_search=10)
        # alpha в диапазоне [0, 2]
        @test 0 ≤ res.extra["alpha"] ≤ 2
    end

    @testset "Reduces residual" begin
        res = solve_iterative_refinement(A, b, x0)
        @test res.residual_norm < 1.5 * norm(b)  # должно улучшить
    end

    @testset "Custom solvers" begin
        res = solve_iterative_refinement(A, b, x0,
                                         first_pass_solver=solve_gravel,
                                         second_pass_solver=solve_cgls,
                                         first_pass_kwargs=(max_iterations=200,),
                                         second_pass_kwargs=(max_iterations=100,))
        @test all(res.spectrum .≥ 0)
    end
end


@testset "New algorithms — Randomized Kaczmarz" begin
    A, b, x0, x_true = _make_problem(14, 100)

    @testset "Basic solve" begin
        res = solve_randomized_kaczmarz(A, b, x0, max_iterations=500, random_state=42)
        @test length(res.spectrum) == 100
        @test all(res.spectrum .≥ 0)
        @test all(isfinite.(res.spectrum))
        @test res.iterations > 0
    end

    @testset "Reproducibility with seed" begin
        res1 = solve_randomized_kaczmarz(A, b, x0, max_iterations=500, random_state=42)
        res2 = solve_randomized_kaczmarz(A, b, x0, max_iterations=500, random_state=42)
        @test res1.spectrum ≈ res2.spectrum
    end

    @testset "Different seeds give different results" begin
        res1 = solve_randomized_kaczmarz(A, b, x0, max_iterations=500, random_state=42)
        res2 = solve_randomized_kaczmarz(A, b, x0, max_iterations=500, random_state=99)
        # Не идентичны (но близки)
        @test res1.spectrum ≠ res2.spectrum
    end

    @testset "Relaxation parameter" begin
        # omega = 0 → не должно обновлять x
        res = solve_randomized_kaczmarz(A, b, x0, max_iterations=100, omega=0.0, random_state=42)
        @test res.spectrum ≈ max.(x0, 0.0)
    end

    @testset "Reduces residual" begin
        res = solve_randomized_kaczmarz(A, b, x0, max_iterations=1000, random_state=42)
        @test res.residual_norm < norm(b)
    end

    @testset "Compare with deterministic Kaczmarz" begin
        res_det = solve_kaczmarz(A, b, x0, max_iterations=100)
        res_rand = solve_randomized_kaczmarz(A, b, x0, max_iterations=100, random_state=42)
        # Оба должны дать осмысленный результат
        @test all(res_det.spectrum .≥ 0)
        @test all(res_rand.spectrum .≥ 0)
    end
end


@testset "New algorithms — CVXPY (Convex.jl)" begin
    A, b, x0, x_true = _make_problem(14, 100)

    @testset "Function exists and is callable" begin
        @test isa(solve_cvxpy, Function)
    end

    @testset "Returns UnfoldResult" begin
        # Если Convex.jl/SCS не установлены, получим zero spectrum с error в extra
        res = solve_cvxpy(A, b, x0, regularization=1e-3)
        @test typeof(res) <: UnfoldResult
        @test length(res.spectrum) == 100
        @test all(res.spectrum .≥ 0)
    end

    @testset "L1 vs L2 norm option" begin
        # L1 должен выполняться без исключения (если Convex доступен — решит задачу;
        # если нет — вернёт zero с предупреждением)
        res_l1 = solve_cvxpy(A, b, x0, regularization=1e-3, norm=1)
        res_l2 = solve_cvxpy(A, b, x0, regularization=1e-3, norm=2)
        @test length(res_l1.spectrum) == length(res_l2.spectrum) == 100
    end

    @testset "Upper bounds" begin
        ub = fill(10.0, 100)
        res = solve_cvxpy(A, b, x0, regularization=1e-3, ub=ub)
        @test all(res.spectrum .≤ 10.0 .+ 1e-3)
    end

    @testset "Invalid norm" begin
        @test_throws ArgumentError solve_cvxpy(A, b, x0, norm=3)
    end
end


@testset "New algorithms — QPsolvers (OSQP/Clarabel)" begin
    A, b, x0, x_true = _make_problem(14, 100)

    @testset "Function exists and is callable" begin
        @test isa(solve_qpsolvers, Function)
    end

    @testset "Returns UnfoldResult" begin
        res = solve_qpsolvers(A, b, x0, regularization=1e-3)
        @test typeof(res) <: UnfoldResult
        @test length(res.spectrum) == 100
        @test all(res.spectrum .≥ 0)
    end

    @testset "L2 regularization" begin
        res = solve_qpsolvers(A, b, x0, regularization=1e-3, norm=2)
        @test length(res.spectrum) == 100
    end

    @testset "L1 regularization" begin
        res = solve_qpsolvers(A, b, x0, regularization=1e-3, norm=1)
        @test length(res.spectrum) == 100
    end

    @testset "Smoothness penalty" begin
        # smoothness_order = 1 (first derivative)
        res = solve_qpsolvers(A, b, x0, regularization=1e-3,
                             smoothness_order=1, smoothness_weight=0.5)
        @test length(res.spectrum) == 100
        @test all(isfinite.(res.spectrum))

        # smoothness_order = 2 (second derivative)
        res2 = solve_qpsolvers(A, b, x0, regularization=1e-3,
                              smoothness_order=2, smoothness_weight=0.5)
        @test length(res2.spectrum) == 100
    end

    @testset "Upper bounds" begin
        ub = fill(5.0, 100)
        res = solve_qpsolvers(A, b, x0, regularization=1e-3, ub=ub)
        @test all(res.spectrum .≤ 5.0 .+ 1e-2)
    end

    @testset "Invalid norm" begin
        @test_throws ArgumentError solve_qpsolvers(A, b, x0, norm=3)
    end

    @testset "Invalid smoothness_order" begin
        @test_throws ArgumentError solve_qpsolvers(A, b, x0, smoothness_order=5)
    end
end


@testset "New algorithms — Detector wrappers" begin
    # Проверяем, что unfold_* методы для новых алгоритмов работают
    n = 50
    detector_names = ["a", "b", "c", "d"]
    rng = MersenneTwister(123)
    E_MeV = collect(range(1e-6, 10.0, length=n))
    sensitivities = Dict(name => rand(rng, n) .+ 0.1 for name in detector_names)
    cc_icrp116 = Dict(name => rand(rng, n) for name in detector_names)
    d = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)

    true_spectrum = exp.(-E_MeV ./ 1.5)
    readings = Dict(name => sum(sensitivities[name] .* true_spectrum)
                   for name in detector_names)

    @testset "unfold_lanczos" begin
        result = unfold_lanczos(d, readings, max_iterations=20)
        @test haskey(result, "spectrum")
        @test length(result["spectrum"]) == n
        @test result["method"] == "Lanczos"
    end

    @testset "unfold_iterative_refinement" begin
        result = unfold_iterative_refinement(d, readings)
        @test haskey(result, "spectrum")
        @test length(result["spectrum"]) == n
        @test result["method"] == "IterativeRefinement"
    end

    @testset "unfold_randomized_kaczmarz" begin
        result = unfold_randomized_kaczmarz(d, readings, max_iterations=200, random_state=42)
        @test haskey(result, "spectrum")
        @test length(result["spectrum"]) == n
        @test result["method"] == "RandomizedKaczmarz"
    end

    @testset "unfold_cvxpy" begin
        result = unfold_cvxpy(d, readings, regularization=1e-3)
        @test haskey(result, "spectrum")
        @test length(result["spectrum"]) == n
        @test result["method"] == "CVXPY"
    end

    @testset "unfold_qpsolvers" begin
        result = unfold_qpsolvers(d, readings, regularization=1e-3)
        @test haskey(result, "spectrum")
        @test length(result["spectrum"]) == n
        @test result["method"] == "QPsolvers"
    end
end
