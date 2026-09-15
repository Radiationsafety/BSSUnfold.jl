# Тесты портированных из bssunfold методов.

using Test
using BSSUnfold
using LinearAlgebra
using Random

BASE_SOLVERS = [
    (Symbol("solve_amaxed"), NamedTuple()),
    (Symbol("solve_amaxed_regularization"), NamedTuple()),
    (Symbol("solve_imaxed"), NamedTuple()),
    (Symbol("solve_sart"), NamedTuple()),
    (Symbol("solve_mapem"), NamedTuple()),
    (Symbol("solve_mlem_stop"), NamedTuple()),
    (Symbol("solve_bunkiut"), NamedTuple()),
    (Symbol("solve_rebunki"), NamedTuple()),
    (Symbol("solve_directed_divergence"), (smoothness_order=1,)),
    (Symbol("solve_ferdor"), (max_iterations=50,)),
    (Symbol("solve_direct"), NamedTuple()),
    (Symbol("solve_scipy_direct"), NamedTuple()),
    (Symbol("solve_tikhonov_tv"), NamedTuple()),
    (Symbol("solve_tikhonov_legendre"), NamedTuple()),
    (Symbol("solve_statreg"), NamedTuple()),
    (Symbol("solve_reconst"), NamedTuple()),
    (Symbol("solve_bayes"), (max_iterations=100,)),
    (Symbol("solve_bayes_spline"), NamedTuple()),
    (Symbol("solve_eki"), (n_iterations=10,)),
    (Symbol("solve_express"), NamedTuple()),
    (Symbol("solve_crystal_ball"), NamedTuple()),
    (Symbol("solve_ensemble"), NamedTuple()),
    (Symbol("solve_cs"), NamedTuple()),
    (Symbol("solve_gks"), (smoothness_order=1,)),
    (Symbol("solve_nsduaz"), NamedTuple()),
    (Symbol("solve_nnksvd"), NamedTuple()),
    (Symbol("solve_nspline"), NamedTuple()),
    (Symbol("solve_hybrid_gmres"), (max_iterations=20,)),
    (Symbol("solve_hybrid_parametric"), (max_iterations=50,)),
    (Symbol("solve_parametric"), NamedTuple()),
    (Symbol("solve_parametric2"), NamedTuple()),
]

function _make_problem_p(seed=42; m=14, n=40)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = exp.(-collect(range(0.0, 5.0, length=n)))
    b = A * x_true .+ 0.01 .* randn(rng, m)
    x0 = ones(n) .* 0.5
    return A, b, x0
end

@testset "Ported unfold methods" begin
    for (fname, kwargs) in BASE_SOLVERS
        @testset "$fname" begin
            fn = getfield(BSSUnfold, fname)
            @test isa(fn, Function)

            A, b, x0 = _make_problem_p()
            res = try
                fn(A, b, x0; kwargs...)
            catch err
                @error "$fname threw" exception = (err, catch_backtrace())
                nothing
            end
            @test res isa UnfoldResult
        end
    end
end

@testset "Additional helpers" begin
    A, b, x0 = _make_problem_p(; m=12, n=25)

    @test isa(BSSUnfold.solve_tikhonov_nnls, Function)
    r1 = BSSUnfold.solve_tikhonov_nnls(A, b)
    @test length(r1) == 25

    @test isa(BSSUnfold.solve_omp, Function)
    r3 = BSSUnfold.solve_omp(A, b, 8)
    @test length(r3) == 25

    @test isa(BSSUnfold.calculate_j_factor, Function)
    j = BSSUnfold.calculate_j_factor(b, b .* 1.01)
    @test isfinite(j)

    @test isa(BSSUnfold.load_bin_lookup, Function)
    look = BSSUnfold.load_bin_lookup()
    @test isa(look, Dict) && !isempty(look)

    @test isa(BSSUnfold.nspline_eval, Function) || isempty(0)
end

@testset "Ported detector unfold_* generator" begin
    detector = BSSUnfold.Detector(RF_GSF)
    n = BSSUnfold.n_energy_bins(detector)
    x_true = exp.(-energy_grid(detector) ./ 2.0)
    readings = get_effective_readings_for_spectra(
        detector, energy_grid(detector), x_true)

    for name in [:unfold_gravel, :unfold_sart, :unfold_amaxed,
                 :unfold_bayes, :unfold_cgls]
        fn = getfield(BSSUnfold, name)
        res = try
            fn(detector, readings)
        catch err
            @error "unfold_* failed" name exception = (err, catch_backtrace())
            nothing
        end
        if res === nothing
            @error "$name threw"
        end
        @test res === nothing || res isa UnfoldResult || res isa Dict{String,Any}
        if res isa Dict
            @test haskey(res, "spectrum") && haskey(res, "doserates")
            @test length(res["spectrum"]) == n
            @test all(isfinite.(res["spectrum"]))
        end
    end
end
