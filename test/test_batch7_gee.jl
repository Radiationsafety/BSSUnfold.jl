# Batch 7 — GEE (generalized estimating equations, port of unfold_gee.py):
# Liang-Zeger quasi-score IRLS with an exchangeable/AR(1)/independence working
# correlation, a gaussian/poisson/gamma variance function and a scale-normalised
# second-difference ridge, plus model-robust (sandwich) standard errors.

using Test, BSSUnfold, LinearAlgebra, Statistics

# Deterministic (arithmetic-only) over-determined problem with structured,
# positively lag-correlated residuals. Julia does not guarantee rand/randn
# stream stability across versions: with the previous MersenneTwister fixture
# the AR(1) moment estimate landed on the clamp floor (-0.95/(m-1)) on some
# versions, collapsing all family-dependent spectra to one solution. Integer
# arithmetic is bit-stable everywhere, and this data keeps every alpha
# strictly inside the clamp so the families genuinely separate.
function _make_gee_problem(m::Int=20, n::Int=8; noise::Float64=0.12)
    c = 10
    x_true = [(n - j + 1) / n for j in 1:n]
    A = [0.25 + 0.15 * mod((i - 1) * (j + c - 1) + (i - 1), 5) for i in 1:m, j in 1:n]
    b = [sum(A[i, j] * x_true[j] for j in 1:n) +
         noise * (mod(i + 2c, 3) - 1) +
         noise * (mod(i * 3 + c, 4) - 1.5) for i in 1:m]
    x0 = fill(0.5, n)
    return A, b, x0, x_true
end


@testset "Batch7 — GEE" begin
    A, b, x0, x_true = _make_gee_problem()
    m, n = size(A)

    @testset "Basic solve" begin
        res = solve_gee(A, b, x0)
        @test res isa UnfoldResult
        @test length(res.spectrum) == n
        @test all(isfinite.(res.spectrum))
        @test all(res.spectrum .>= 0)
        @test res.iterations >= 1
        @test res.iterations <= 100
        @test res.converged isa Bool
        @test res.residual_norm ≈ norm(b - A * res.spectrum)
        @test norm(A * res.spectrum - b) < 0.2 * norm(b)
        # defaults mirror the Python signature
        @test res.extra["family"] == "gaussian"
        @test res.extra["corstr"] == "exchangeable"
        @test res.extra["regularization"] == 1e-4
        @test res.extra["max_iterations"] == 100
        @test res.extra["tolerance"] == 1e-6
        @test res.extra["diff_order"] == 2
    end

    @testset "Extra diagnostics" begin
        res = solve_gee(A, b, x0; family="gamma", corstr="ar1")
        for key in ("alpha", "phi", "cov_robust", "cov_naive", "robust_se", "naive_se",
                    "residuals", "pearson_residuals", "pearson_chi2", "df",
                    "iterations", "converged", "family", "corstr", "regularization",
                    "max_iterations", "tolerance", "diff_order",
                    "spectrum_uncert_robust", "gee_converged")
            @test haskey(res.extra, key)
        end
        @test res.extra["gee_converged"] == res.converged
        @test res.extra["spectrum_uncert_robust"] == res.extra["robust_se"]
        @test res.extra["df"] == m - n
        @test length(res.extra["robust_se"]) == n
        @test length(res.extra["naive_se"]) == n
        @test length(res.extra["residuals"]) == m
        @test length(res.extra["pearson_residuals"]) == m
        @test size(res.extra["cov_robust"]) == (n, n)
        @test size(res.extra["cov_naive"]) == (n, n)
        @test all(isfinite.(res.extra["robust_se"]))
        @test all(res.extra["robust_se"] .>= 0)
        @test all(isfinite.(res.extra["naive_se"]))
        @test all(res.extra["naive_se"] .>= 0)
        @test res.extra["cov_robust"] ≈ transpose(res.extra["cov_robust"])
        @test res.extra["cov_naive"] ≈ transpose(res.extra["cov_naive"])
        @test res.extra["pearson_chi2"] ≈ sum(res.extra["pearson_residuals"] .^ 2)
        @test res.extra["phi"] > 0
        # alpha is moment-estimated and then clipped into the admissible range
        lo = -0.95 / max(m - 1, 1)
        @test lo - 1e-12 <= res.extra["alpha"] <= 0.95 + 1e-12
    end

    @testset "Working correlation matrices" begin
        Ri = working_correlation(0.5, 5, "independence")
        @test Ri == Matrix{Float64}(I, 5, 5)

        Re = working_correlation(0.3, 6, "exchangeable")
        @test size(Re) == (6, 6)
        @test diag(Re) == ones(6)
        @test Re[1, 2] == 0.3 && Re[6, 1] == 0.3
        @test Re == transpose(Re)

        Ra = working_correlation(0.5, 4, "ar1")
        @test diag(Ra) == ones(4)
        @test Ra[1, 4] ≈ 0.5^3
        @test Ra[2, 3] ≈ 0.5
        @test Ra == transpose(Ra)
        Rn = working_correlation(-0.5, 3, "ar1")
        @test Rn[1, 2] ≈ -0.5 && Rn[1, 3] ≈ 0.25

        @test_throws ArgumentError working_correlation(1.0, 5, "exchangeable")
        @test_throws ArgumentError working_correlation(-0.25, 5, "exchangeable")
        @test_throws ArgumentError working_correlation(1.0, 4, "ar1")
        @test_throws ArgumentError working_correlation(-1.0, 4, "ar1")
        @test_throws ArgumentError working_correlation(0.5, 0)
        @test_throws ArgumentError working_correlation(0.5, 3, "ma2")
    end

    @testset "Alpha moment estimators" begin
        @test estimate_alpha(ones(6), "independence") == (0.0, 1.0)
        @test estimate_alpha(ones(6), "exchangeable") == (0.95, 1.0)
        @test estimate_alpha(ones(6), "ar1") == (0.95, 1.0)

        alternating = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        @test estimate_alpha(alternating, "independence") == (0.0, 1.0)
        @test estimate_alpha(alternating, "exchangeable")[1] ≈ -0.19
        @test estimate_alpha(alternating, "ar1")[1] ≈ -0.19
        @test estimate_alpha(alternating, "exchangeable")[2] ≈ 1.0

        # a single observation carries no correlation information
        @test estimate_alpha([0.5], "exchangeable") == (0.0, 1.0)
        @test estimate_alpha(zeros(5), "independence") == (0.0, 0.0)
        @test estimate_alpha(zeros(5), "ar1") == (0.0, 0.0)

        r = [0.4, -0.2, 0.7, 0.1, -0.5]
        @test estimate_alpha(r, "ar1")[1] < estimate_alpha(r, "exchangeable")[1]
        @test_throws ArgumentError estimate_alpha(ones(4), "free")
    end

    @testset "Family and corstr give different answers" begin
        @test FAMILIES == ("gaussian", "poisson", "gamma")
        @test CORSTRINGS == ("independence", "exchangeable", "ar1")

        sp = Dict{Tuple{String, String}, Vector{Float64}}()
        for fam in FAMILIES, cor in CORSTRINGS
            sp[(fam, cor)] = solve_gee(A, b, x0; family=fam, corstr=cor).spectrum
        end
        # AR(1) weights the residuals differently per variance function
        @test norm(sp[("poisson", "ar1")] - sp[("gaussian", "ar1")]) /
              norm(sp[("gaussian", "ar1")]) > 1e-4
        @test norm(sp[("gamma", "ar1")] - sp[("poisson", "ar1")]) /
              norm(sp[("poisson", "ar1")]) > 1e-4
        @test norm(sp[("gamma", "ar1")] - sp[("gaussian", "ar1")]) /
              norm(sp[("gaussian", "ar1")]) > 1e-4
        @test norm(sp[("poisson", "exchangeable")] - sp[("gaussian", "exchangeable")]) > 0.0
        # the correlation structure itself moves the solution much further
        @test norm(sp[("gaussian", "ar1")] - sp[("gaussian", "independence")]) /
              norm(sp[("gaussian", "independence")]) > 1e-3
        @test norm(sp[("gamma", "ar1")] - sp[("gamma", "exchangeable")]) /
              norm(sp[("gamma", "exchangeable")]) > 1e-3
        @test sp[("gaussian", "ar1")] != sp[("gaussian", "exchangeable")]
        # independence forces alpha to zero, AR(1) here finds positive correlation
        @test solve_gee(A, b, x0; corstr="independence").extra["alpha"] == 0.0
        @test solve_gee(A, b, x0; corstr="ar1").extra["alpha"] > 0.0
    end

    @testset "Independence ignores the variance function" begin
        # with R = I the quasi-score never sees v(mu), so the families coincide
        # exactly while the dispersion estimates stay family-specific
        sg = solve_gee(A, b, x0; family="gaussian", corstr="independence")
        sp = solve_gee(A, b, x0; family="poisson", corstr="independence")
        sy = solve_gee(A, b, x0; family="gamma", corstr="independence")
        @test sg.spectrum == sp.spectrum == sy.spectrum
        @test sg.extra["phi"] != sp.extra["phi"] != sy.extra["phi"]
        @test sg.extra["pearson_chi2"] != sy.extra["pearson_chi2"]
    end

    @testset "Gaussian/independence is projected ridge least squares" begin
        for reg in (1e-4, 1e-3, 1e-1)
            d = gee_fit(A, b, x0; family="gaussian", corstr="independence",
                        regularization=reg)
            D = create_derivative_matrix(Float64, n, 2)
            Gr = D' * D
            Gr ./= max(mean(diag(Gr)), 1.0)
            H = A' * A
            lhs = H + reg * max(mean(diag(H)), 1.0) * Gr
            x_fp = max.(lhs \ (A' * b), 0.0)
            @test norm(x_fp - d["spectrum"]) / norm(x_fp) < 1e-8
        end
    end

    @testset "Non-negativity is projected at every iteration" begin
        res = solve_gee(A, b, fill(-2.0, n); family="gaussian", corstr="ar1")
        @test all(res.spectrum .>= 0)
        @test all(isfinite.(res.spectrum))

        Au = copy(A')
        res_u = solve_gee(Au, b[1:n], zeros(m); family="poisson",
                          corstr="ar1", regularization=0.0)
        @test length(res_u.spectrum) == m
        @test all(res_u.spectrum .>= 0)
        @test all(isfinite.(res_u.spectrum))
        @test res_u.extra["df"] == 1
    end

    @testset "Iteration and regularization knobs" begin
        stopped = solve_gee(A, b, x0; family="gaussian", corstr="ar1",
                            max_iterations=1)
        @test stopped.iterations == 1
        @test stopped.converged === false
        @test all(stopped.spectrum .>= 0)

        # gaussian/independence converges immediately (the iterate is fixed),
        # gaussian/ar1 has to chase alpha, so it needs more steps
        for (fam, cor) in (("gaussian", "independence"), ("poisson", "ar1"),
                           ("gamma", "ar1"))
            loose = solve_gee(A, b, x0; family=fam, corstr=cor, tolerance=1e-3)
            tight = solve_gee(A, b, x0; family=fam, corstr=cor, tolerance=1e-10)
            @test tight.iterations >= loose.iterations
            @test tight.converged
            @test length(tight.spectrum) == n
        end

        no_pen = solve_gee(A, b, x0; corstr="ar1", regularization=0.0)
        strong = solve_gee(A, b, x0; corstr="ar1", regularization=1e-1)
        @test no_pen.extra["regularization"] == 0.0
        @test norm(strong.spectrum - no_pen.spectrum) / norm(no_pen.spectrum) > 1e-4
        # smoothing pulls the solution towards a straight line in bin index
        @test norm(create_derivative_matrix(Float64, n, 2) * strong.spectrum) <
              norm(create_derivative_matrix(Float64, n, 2) * no_pen.spectrum)
        # diff_order is only used when the penalty is active
        @test all(isfinite.(solve_gee(A, b, x0; regularization=0.0,
                                     diff_order=3).spectrum))
        @test_throws ArgumentError solve_gee(A, b, x0; diff_order=3)
    end

    @testset "String and Symbol options are equivalent" begin
        for (fam, cor) in (("gaussian", "independence"), ("poisson", "exchangeable"),
                           ("gamma", "ar1"))
            r1 = solve_gee(A, b, x0; family=fam, corstr=cor)
            r2 = solve_gee(A, b, x0; family=Symbol(uppercase(fam)),
                           corstr=Symbol(uppercase(cor)))
            @test r1.spectrum == r2.spectrum
            @test r1.extra["family"] == fam
            @test r2.extra["corstr"] == cor
        end
    end

    @testset "Nothing initial guess uses the family default" begin
        for fam in FAMILIES
            res = solve_gee(A, b, nothing; family=fam, corstr="ar1")
            @test length(res.spectrum) == n
            @test all(isfinite.(res.spectrum))
            @test all(res.spectrum .>= 0)
        end
        # the multiplicative families start from ones, the gaussian from zeros,
        # and the projected IRLS path therefore ends in different places
        @test solve_gee(A, b, nothing; family="gaussian",
                        corstr="ar1").spectrum !=
              solve_gee(A, b, ones(n); family="poisson", corstr="ar1").spectrum
    end

    @testset "gee_fit dictionary" begin
        d = gee_fit(A, b, x0; family="gamma", corstr="ar1")
        @test d isa Dict{String, Any}
        for key in ("spectrum", "cov_robust", "cov_naive", "robust_se", "naive_se",
                    "alpha", "phi", "residuals", "pearson_residuals", "pearson_chi2",
                    "df", "iterations", "converged", "family", "corstr",
                    "regularization", "max_iterations", "tolerance", "diff_order")
            @test haskey(d, key)
        end
        @test d["family"] == "gamma" && d["corstr"] == "ar1"
        @test d["spectrum"] == solve_gee(A, b, x0; family="gamma", corstr="ar1").spectrum
        @test d["robust_se"] ≈ sqrt.(max.(diag(d["cov_robust"]), 0.0))
        @test norm(b - A * d["spectrum"]) ≈ solve_gee(A, b, x0; family="gamma",
                                                      corstr="ar1").residual_norm
        @test solve_gee_full(A, b, x0; family="gamma", corstr="ar1")["spectrum"] ==
              d["spectrum"]
    end

    @testset "Validation errors" begin
        @test_throws ArgumentError solve_gee(A, zeros(m + 1), x0)
        @test_throws ArgumentError solve_gee(A, b, zeros(n + 1))
        @test_throws ArgumentError solve_gee(A, b, x0; max_iterations=0)
        @test_throws ArgumentError solve_gee(A, b, x0; tolerance=-1.0)
        @test_throws ArgumentError solve_gee(A, b, x0; regularization=-1e-3)
        @test_throws ArgumentError solve_gee(A, b, x0; family="binomial")
        @test_throws ArgumentError solve_gee(A, b, x0; corstr="unstructured")
        @test_throws ArgumentError solve_gee(zeros(m, n), b, x0; corstr="ar1")
        err = try
            solve_gee(A, b, x0; family="binomial")
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("gaussian", err.msg) && occursin("binomial", err.msg)
        err2 = try
            solve_gee(A, b, x0; corstr="unstructured")
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("corstr", err2.msg) && occursin("unstructured", err2.msg)
    end
end
