# Tests for the dev-branch methods: RFSP-JUL, AMG, Uno, SSR (sisireg),
# MLEM-BS and P-spline REML

using Test
using BSSUnfold
using LinearAlgebra
using Random
using Statistics

# Well-conditioned synthetic unfolding problem (small, exact-ish)
function _make_dev_problem(m::Int=12, n::Int=40; seed::Int=42)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = exp.(-collect(range(0, 4, length=n)))
    b = A * x_true .+ 0.005 .* randn(rng, m)
    x0 = ones(n) .* 0.1
    return A, b, x0, x_true
end


@testset "Dev — RFSP-JUL" begin
    A, b, x0, x_true = _make_dev_problem()

    @testset "Basic solve" begin
        res = solve_rfsp_jul(A, b, x0; max_iterations=500, tolerance=1e-8)
        @test length(res.spectrum) == 40
        @test all(res.spectrum .≥ 0)
        @test all(isfinite.(res.spectrum))
        @test res.iterations ≤ 500
        # data fidelity of the damped least-squares solution
        @test norm(A * res.spectrum .- b) < 0.5 * norm(b)
    end

    @testset "Extra diagnostics" begin
        res = solve_rfsp_jul(A, b, x0)
        @test haskey(res.extra, "tolerance")
        @test res.extra["tolerance"] == 1e-4
    end

    @testset "Custom weights (non-finite entries clamped to zero)" begin
        w = ones(12)
        res = solve_rfsp_jul(A, b, x0; weights=w)
        @test all(isfinite.(res.spectrum))
        # negative weights are clamped to zero by design (mirrors the
        # Python max(0, W) behaviour): the zero-weighted detectors are
        # simply excluded from the fit
        w_neg = copy(w); w_neg[1:3] .= -1.0
        res2 = solve_rfsp_jul(A, b, x0; weights=w_neg)
        @test all(isfinite.(res2.spectrum))
    end

    @testset "Validation errors" begin
        @test_throws ArgumentError solve_rfsp_jul(A, zeros(12), x0)
        @test_throws ArgumentError solve_rfsp_jul(A, b[1:5], x0)
        @test_throws ArgumentError solve_rfsp_jul(A, b, x0[1:5])
    end
end


@testset "Dev — AMG preconditioned Krylov" begin
    A, b, x0, x_true = _make_dev_problem()

    @testset "cg + jacobi (exact, no projection)" begin
        res = solve_amg(A, b, nothing; method="cg", preconditioner="jacobi",
                        max_iterations=200, tolerance=1e-10,
                        nonnegativity=false, regularization=1e-6)
        @test res.converged
        @test all(isfinite.(res.spectrum))
        # damped normal equations must be solved to tight tolerance
        N = A' * A + 1e-6 * Matrix{Float64}(I, 40, 40)
        @test norm(N * res.spectrum .- A' * b) ≤ 1e-6 * norm(A' * b)
    end

    @testset "gmres + ssor (exact, no projection)" begin
        res = solve_amg(A, b, nothing; method="gmres", preconditioner="ssor",
                        max_iterations=200, tolerance=1e-10,
                        nonnegativity=false, regularization=1e-6)
        @test res.converged
        N = A' * A + 1e-6 * Matrix{Float64}(I, 40, 40)
        @test norm(N * res.spectrum .- A' * b) ≤ 1e-6 * norm(A' * b)
    end

    @testset "bicgstab variants" begin
        # BiCGSTAB typically terminates on a Lanczos breakdown before the
        # strict rtol; assert a (looser but still meaningful) data fidelity.
        res = solve_amg(A, b, nothing; method="bicgstab", preconditioner="jacobi",
                        max_iterations=200, tolerance=1e-6,
                        nonnegativity=false, regularization=1e-6)
        @test all(isfinite.(res.spectrum))
        @test norm(A * res.spectrum .- b) < 1e-3 * norm(b)
        # nonsymmetric preconditioner + non-negativity projection
        res2 = solve_amg(A, b, nothing; method="bicgstab", preconditioner="sor",
                         max_iterations=200, tolerance=1e-8)
        @test all(res2.spectrum .≥ 0)
        @test norm(A * res2.spectrum .- b) < 0.2 * norm(b)
    end

    @testset "amg falls back to jacobi with warning" begin
        res = @test_logs (:warn, r"falls back to Jacobi") solve_amg(
            A, b, nothing; preconditioner="amg", max_iterations=100,
            tolerance=1e-8)
        @test all(isfinite.(res.spectrum))
        @test res.extra["preconditioner"] == "amg"
    end

    @testset "x0 as warm start and non-negativity clamp" begin
        res = solve_amg(A, b, x0; max_iterations=200, tolerance=1e-12)
        @test all(res.spectrum .≥ 0)
    end

    @testset "Auto regularization (nothing)" begin
        res = solve_amg(A, b, nothing; max_iterations=200, tolerance=1e-8)
        @test res.extra["regularization"] > 0
    end

    @testset "Validation" begin
        @test_throws ArgumentError solve_amg(A, b, nothing; method="sor")
        @test_throws ArgumentError solve_amg(A, b, nothing; preconditioner="ilu")
        @test_throws ArgumentError solve_amg(A, b, nothing; outer_iterations=0)
        @test_throws ArgumentError solve_amg(A, b, nothing; regularization=-1.0)
    end
end


@testset "Dev — Uno NLP presets" begin
    A, b, x0, x_true = _make_dev_problem()

    @testset "filter_sqp preset (NNLS subproblem)" begin
        d = solve_uno_full(A, b, x0; preset="filter_sqp", tolerance=1e-10)
        @test d["preset"] == "filter_sqp"
        @test d["hessian"] == "exact"
        @test d["constraint_violation"] == 0.0
        @test all(d["spectrum"] .≥ 0)
        @test d["converged"]
        # data fidelity of the NNLS solution
        x = d["spectrum"]
        @test norm(A * x .- b) < 0.5 * norm(b)
    end

    @testset "ipopt_like preset, exact Hessian" begin
        d = solve_uno_full(A, b, x0; preset="ipopt_like", hessian="exact",
                           tolerance=1e-8, max_iterations=300)
        @test d["preset"] == "ipopt_like"
        @test all(d["spectrum"] .≥ 0)
        @test d["constraint_violation"] < 1e-12
        @test norm(A * d["spectrum"] .- b) < 0.5 * norm(b)
    end

    @testset "ipopt_like preset, BFGS Hessian" begin
        d = solve_uno_full(A, b, x0; preset="ipopt_like", hessian="bfgs",
                           tolerance=1e-8, max_iterations=300)
        @test all(d["spectrum"] .≥ 0)
        @test norm(A * d["spectrum"] .- b) < 0.5 * norm(b)
    end

    @testset "Poisson weights" begin
        d = solve_uno_full(A, b, x0; weights="poisson")
        @test all(isfinite.(d["spectrum"]))
        wa = collect(range(0.5, 2.0, length=12))
        d2 = solve_uno_full(A, b, x0; weights=wa)
        @test all(isfinite.(d2["spectrum"]))
        @test_throws ArgumentError solve_uno_full(A, b, x0; weights=-wa)
        @test_throws ArgumentError solve_uno_full(A, b, x0; weights="bogus")
    end

    @testset "solve_uno UnfoldResult wrapper" begin
        res = solve_uno(A, b, x0)
        @test length(res.spectrum) == 40
        @test res.extra["preset"] == "filter_sqp"
        @test haskey(res.extra, "dual_infeasibility")
    end

    @testset "Filter utilities" begin
        f = [(1.0, 1.0)]
        @test uno_filter(f, 0.5, 1.0)          # better objective
        @test uno_filter(f, 1.0, 0.5)          # better violation
        @test !uno_filter(f, 1.0, 1.0)         # dominated
        @test !uno_filter(f, 1.5, 1.5)         # worse in both
        f2 = uno_augment_filter([(1.0, 1.0)], 0.5, 0.5)
        @test f2 == [(1.0, 1.0), (0.5, 0.5)]
    end

    @testset "Objective / gradient consistency" begin
        x = max.(x0, 1e-3)
        cache = Dict{String,Any}()
        f0 = uno_objective(A, b, ones(12), 1e-2, x; cache=cache)
        g = uno_gradient(A, b, ones(12), 1e-2, x; cache=cache)
        h = 1e-6
        xp = copy(x); xp[7] += h
        f1 = uno_objective(A, b, ones(12), 1e-2, xp; cache=cache)
        # central element of the gradient vs finite difference
        @test g[7] ≈ (f1 - f0) / h rtol = 1e-3
    end

    @testset "Validation" begin
        @test_throws ArgumentError solve_uno_full(A, b, x0; preset="sqp")
        @test_throws ArgumentError solve_uno_full(A, b, x0; regularization=-1.0)
    end
end


@testset "Dev — MLEM-BS" begin
    A, b, x0, x_true = _make_dev_problem(12, 40)

    @testset "B-spline basis properties" begin
        E = collect(range(0.5, 8.0, length=40))
        B = build_bspline_basis(E, 8, 4, "uniform")
        @test size(B) == (40, 8)
        @test all(B .≥ 0)
        # partition of unity on the clamped interval
        @test vec(sum(B, dims=2)) ≈ ones(40) atol = 1e-10
        Blog = build_bspline_basis(E, 8, 4, "log")
        @test vec(sum(Blog, dims=2)) ≈ ones(40) atol = 1e-10
        @test_throws ArgumentError build_bspline_basis(E, 3, 4, "uniform")
        @test_throws ArgumentError build_bspline_basis(-E, 8, 4, "uniform")
    end

    @testset "second_difference_matrix" begin
        D2 = second_difference_matrix(5)
        @test size(D2) == (3, 5)
        @test D2[1, :] == [1.0, -2.0, 1.0, 0.0, 0.0]
        @test D2[2, :] == [0.0, 1.0, -2.0, 1.0, 0.0]
        @test D2[3, :] == [0.0, 0.0, 1.0, -2.0, 1.0]
    end

    @testset "ks_statistic" begin
        model = collect(range(1.0, 5.0, length=10))
        # zero residuals: K_S = |0/sum(model) - 1| = 1
        @test ks_statistic(model, model) ≈ 1.0 atol = 1e-12
        big = model .+ 10.0
        @test ks_statistic(big, model) > 2.0
    end

    @testset "solve_mlem_bs basic" begin
        res = solve_mlem_bs(A, b, x0; max_iterations=200, n_basis=10,
                            beta_relative=1e-3)
        @test length(res.spectrum) == 40
        @test all(res.spectrum .≥ 0)
        @test res.iterations ≤ 200
        @test norm(A * res.spectrum .- b) < 0.5 * norm(b)
    end

    @testset "Rich diagnostics via solve_mlem_bs_full" begin
        d = solve_mlem_bs_full(A, b, x0; max_iterations=100, n_basis=8)
        @test haskey(d.extra, "ks_history")
        @test haskey(d.extra, "coefficients")
        @test haskey(d.extra, "n_basis")
        @test d.extra["n_basis"] == 8
        @test haskey(d.extra, "beta_effective")
        @test length(d.extra["ks_history"]) ≥ d.iterations
    end

    @testset "Auto parameter selection" begin
        d = solve_mlem_bs_full(A, b, x0; max_iterations=100, auto_params=true,
                               spline_order=3)
        @test haskey(d.extra, "auto_selection")
        sel = d.extra["auto_selection"]
        @test haskey(sel, "chosen")
        @test haskey(sel, "candidates")
        @test !isempty(sel["candidates"])
    end

    @testset "Absolute beta is scale aware" begin
        # relative and absolute beta must agree when beta = beta_relative * s_bar
        B = build_bspline_basis(collect(range(1.0, 40.0, length=40)), 8, 4, "uniform")
        colsums = vec(sum(A * B; dims=1))
        s_bar = sum(colsums) / length(colsums)
        d1 = solve_mlem_bs_full(A, b, x0; max_iterations=50, n_basis=8,
                                beta_relative=1e-2)
        d2 = solve_mlem_bs_full(A, b, x0; max_iterations=50, n_basis=8,
                                beta=1e-2 * s_bar)
        @test d1.spectrum ≈ d2.spectrum rtol = 1e-8
    end

    @testset "Validation" begin
        @test_throws ArgumentError solve_mlem_bs(A, b, x0; beta=1.0,
                                                 beta_relative=0.1)
        @test_throws ArgumentError solve_mlem_bs(A, b[1:3], x0)
    end
end


@testset "Dev — P-spline REML" begin
    A, b, x0, x_true = _make_dev_problem(12, 40)

    @testset "difference matrix and mixed-model split" begin
        D1 = second_difference_matrix(5)  # public MLEM-BS helper
        D = Matrix{Float64}(I, 5, 5)
        for _ in 1:2
            D = D[2:end, :] .- D[1:end-1, :]
        end
        @test D ≈ D1
        U_fixed, U_random, g_random =
            BSSUnfold._psreml_mixed_model_split(8, 2)
        @test size(U_fixed, 2) == 2          # null space of D2 (linear trend)
        @test size(U_random, 2) == 6
        @test all(g_random .> 0)
        @test issorted(g_random)
    end

    @testset "solve_pspline_reml basic" begin
        res = solve_pspline_reml(A, b; x0=x0, n_basis=10, spline_order=3)
        @test length(res.spectrum) == 40
        @test all(res.spectrum .≥ 0)
        @test res.converged
        @test norm(A * res.spectrum .- b) < 0.5 * norm(b)
    end

    @testset "Positional x0 convenience" begin
        res = solve_pspline_reml(A, b, x0; n_basis=10, spline_order=3)
        @test all(res.spectrum .≥ 0)
    end

    @testset "REML selection (auto lambda)" begin
        d = solve_pspline_reml_full(A, b; x0=x0, n_basis=10, spline_order=3)
        ex = d.extra
        @test ex["lam"] > 0
        @test isfinite(ex["reml_loglik"])
        @test ex["n_iterations"] > 0
        @test ex["ed"] > 0
        @test 0 < ex["ed_norm"] ≤ 1
        @test ex["reml_converged"]
    end

    @testset "Fixed relative lambda skips optimisation" begin
        d = solve_pspline_reml_full(A, b; x0=x0, n_basis=10, spline_order=3,
                                    lam_relative=1.0)
        @test d.extra["n_iterations"] == 0
        @test d.extra["lam_relative"] == 1.0
        @test all(isfinite.(d.spectrum))
    end

    @testset "Poisson weights path" begin
        d = solve_pspline_reml_full(A, b; x0=x0, n_basis=10, weights="poisson")
        @test all(isfinite.(d.spectrum))
        wa = collect(range(0.8, 1.5, length=12))
        d2 = solve_pspline_reml_full(A, b; x0=x0, n_basis=10, weights=wa)
        @test all(isfinite.(d2.spectrum))
        @test_throws ArgumentError solve_pspline_reml_full(A, b; x0=x0,
                                                           n_basis=10,
                                                           weights="bogus")
    end

    @testset "Energy grid" begin
        E = collect(range(0.5, 15.0, length=40))
        d = solve_pspline_reml_full(A, b; x0=x0, n_basis=10, E_MeV=E,
                                    knot_spacing="log")
        @test all(isfinite.(d.spectrum))
        @test_throws ArgumentError solve_pspline_reml_full(A, b; x0=x0,
                                                           E_MeV=E[1:10])
    end

    @testset "Validation" begin
        @test_throws ArgumentError solve_pspline_reml_full(A, b; x0=x0,
                                                           n_basis=50)
        @test_throws ArgumentError solve_pspline_reml_full(A, b; x0=x0,
                                                           n_basis=1)
        Asmall, bsmall, _, _ = _make_dev_problem(3, 20)
        @test_throws ArgumentError solve_pspline_reml_full(Asmall, bsmall)
    end
end


@testset "Dev — SSR (sisireg)" begin
    # SSR needs >= 8 energy bins and >= 3 detectors
    A, b, x0, x_true = _make_dev_problem(10, 24)

    @testset "Statistical helper functions" begin
        @test max_run_quantile(10) == trunc(Int, 3.3 + 1.44 * log(10))
        @test max_run_quantile(1) == 3   # R: as.integer(3.3 + 1.44 log(1))
        @test_throws ArgumentError max_run_quantile(0)
        # partial-sum quantile: F(n, k) = min(sqrt((1 + 2.33 ln n) k), k)
        n, k = 100, 5
        F = min(sqrt((1 + 2.33 * log(n)) * k), k)
        @test partial_sum_quantile(n, k) ≈ F atol = 1e-12
        @test partial_sum_quantile(n, [k, 2k]) isa Vector
        # runs / partial sums on a residual-sign sequence
        dat = collect(range(1.0, 10.0, length=50))
        @test number_of_extrema(dat) ≤ 2
        @test number_of_extrema(sin.(range(0, 6pi, length=200))) ≥ 5
        # adequacy tests take (data, model); a perfect model is valid
        @test partial_sum_valid(dat, dat)
        @test run_valid(dat, dat)
    end

    @testset "solve_ssr_full basic" begin
        d = solve_ssr_full(A, b; x0=x0, max_iterations=100)
        @test length(d["spectrum"]) == 24
        @test all(d["spectrum"] .≥ 0)
        @test d["n_iterations"] > 0
        @test haskey(d, "fn")
        @test haskey(d, "fn_start")
        @test haskey(d, "k_run")
        @test haskey(d, "n_extrema")
        @test haskey(d, "fn_ladder")
        @test !isempty(d["fn_ladder"])
    end

    @testset "solve_ssr UnfoldResult wrapper" begin
        res = solve_ssr(A, b, x0; max_iterations=100)
        @test length(res.spectrum) == 24
        @test all(res.spectrum .≥ 0)
        @test res.extra["k_run"] == max_run_quantile(24)
        @test haskey(res.extra, "ps_valid_data")
        @test haskey(res.extra, "run_valid_data")
    end

    @testset "Fixed threshold fn" begin
        d = solve_ssr_full(A, b; x0=x0, fn=4, max_iterations=100)
        @test d["fn"] == 4
        @test length(d["fn_ladder"]) == 1
    end

    @testset "Energy grid permutation" begin
        E = collect(range(10.0, 0.5, length=24))   # descending on purpose
        d = solve_ssr_full(A, b; x0=x0, E_MeV=E, max_iterations=100)
        @test all(isfinite.(d["spectrum"]))
    end

    @testset "ssr_predict" begin
        mu = collect(range(0.0, 10.0, length=25))
        xq = collect(range(0.5, 9.5, length=19))
        mu_hat = ssr_predict(mu, mu, xq)   # model == identity ramp
        @test length(mu_hat) == length(xq)
        # prediction of a linear ramp stays close to the ramp itself
        @test norm(mu_hat .- xq) / norm(xq) < 0.25
    end

    @testset "Validation" begin
        A2, b2, _, _ = _make_dev_problem(10, 6)     # only 6 energy bins
        @test_throws ArgumentError solve_ssr_full(A2, b2; x0=ones(6))
        A3, b3, _, _ = _make_dev_problem(2, 24)     # only 2 detectors
        @test_throws ArgumentError solve_ssr_full(A3, b3; x0=ones(24))
        @test_throws ArgumentError solve_ssr_full(A, b; x0=x0, fn="bogus")
        @test_throws ArgumentError solve_ssr_full(A, b; x0=x0, fn=0)
    end
end


@testset "Dev — Detector wrappers" begin
    n = 40
    names = ["0_in", "2_in", "5_in", "20.32_cm"]
    rng = MersenneTwister(123)
    E_MeV = collect(range(1e-7, 10.0, length=n))
    sens = Dict(nm => rand(rng, n) .+ 0.1 for nm in names)
    cc = Dict(nm => rand(rng, n) for nm in names)
    d = Detector(names, E_MeV, sens, cc)
    true_spec = exp.(-E_MeV ./ 1.5)
    readings = Dict(nm => sum(sens[nm] .* true_spec) for nm in names)

    @testset "unfold_rfsp_jul" begin
        result = unfold_rfsp_jul(d, readings, max_iterations=300)
        @test result["method"] == "RFSP-JUL"
        @test length(result["spectrum"]) == n
        @test all(result["spectrum"] .≥ 0)
    end

    @testset "unfold_amg" begin
        result = unfold_amg(d, readings, max_iterations=200, tolerance=1e-8)
        @test result["method"] == "AMG_Krylov"
        @test length(result["spectrum"]) == n
    end

    @testset "unfold_uno" begin
        result = unfold_uno(d, readings)
        @test result["method"] == "Uno_NLP"
        @test length(result["spectrum"]) == n
        @test all(result["spectrum"] .≥ 0)
    end

    @testset "unfold_ssr" begin
        result = unfold_ssr(d, readings, max_iterations=150)
        @test result["method"] == "SSR_sisireg"
        @test length(result["spectrum"]) == n
    end

    @testset "unfold_mlem_bs" begin
        result = unfold_mlem_bs(d, readings, max_iterations=150, n_basis=10)
        @test result["method"] == "MLEM_BS"
        @test length(result["spectrum"]) == n
        @test all(result["spectrum"] .≥ 0)
    end

    @testset "unfold_pspline_reml" begin
        result = unfold_pspline_reml(d, readings, n_basis=10)
        @test result["method"] == "P-spline_REML"
        @test length(result["spectrum"]) == n
        @test all(result["spectrum"] .≥ 0)
    end
end
