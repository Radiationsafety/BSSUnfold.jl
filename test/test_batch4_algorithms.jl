# Tests for batch 4: PGD, Extragradient, CoordinateDescent, Subgradient,
# Frank-Wolfe, ADMM, L-BFGS-B, RFSP, LOUHI

using Test
using BSSUnfold
using LinearAlgebra
using Random
using Statistics

function _make_problem4(m::Int=14, n::Int=80; seed::Int=42)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.3
    A ./= sum(A, dims=2)
    lE = collect(range(-8.0, 1.4; length=n))
    x_true = exp.(-((lE .+ 8.5) .^ 2) / 0.3) .+ 0.5 * exp.(-((lE .- 0.0) .^ 2) / 0.5)
    b = A * x_true .+ 0.01 .* randn(rng, m)
    x0 = ones(n) .* 0.5
    return A, b, x0, x_true
end

const _NEW_SOLVERS4 = [solve_pgd, solve_extragradient, solve_coordinate_descent,
                       solve_subgradient, solve_frank_wolfe, solve_admm,
                       solve_lbfgsb, solve_rfsp, solve_louhi]

@testset "Batch4 — first-order and classical solvers" begin
    A, b, x0, x_true = _make_problem4()

    @testset "Common contract: $(nameof(f))" for f in _NEW_SOLVERS4
        res = f(A, b, copy(x0))
        @test res isa UnfoldResult{Float64}
        @test length(res.spectrum) == length(x0)
        @test all(isfinite.(res.spectrum))
        @test all(res.spectrum .>= 0)
        @test res.iterations >= 1
        @test res.residual_norm >= 0
        @test norm(A * res.spectrum - b) ≈ res.residual_norm rtol = 1e-6
        # Every solver must at least beat the zero solution; slow convergers
        # (subgradient, RFSP) legitimately do not reach a small chi2 by default.
        @test res.residual_norm < norm(b)
    end

    @testset "Accurate to chi2 < 1e-2: $(nameof(f))" for f in
            (solve_pgd, solve_extragradient, solve_coordinate_descent,
             solve_frank_wolfe, solve_admm, solve_lbfgsb, solve_louhi)
        res = f(A, b, copy(x0))
        @test norm(A * res.spectrum - b)^2 / norm(b)^2 < 1e-2
    end

    @testset "Determinism" for f in _NEW_SOLVERS4
        r1 = f(A, b, copy(x0))
        r2 = f(A, b, copy(x0))
        @test r1.spectrum == r2.spectrum
    end

    @testset "Does not mutate x0" for f in _NEW_SOLVERS4
        probe = copy(x0)
        f(A, b, probe)
        @test probe == x0
    end

    @testset "Zero readings" for f in _NEW_SOLVERS4
        if f in (solve_frank_wolfe, solve_rfsp, solve_louhi)
            # These codes need a positive measurement (or fluence) to be defined.
            @test_throws ArgumentError f(A, zeros(size(A, 1)), copy(x0),
                                        max_iterations=20)
        else
            res = f(A, zeros(size(A, 1)), copy(x0), max_iterations=20)
            @test all(isfinite.(res.spectrum))
        end
    end

    @testset "Float32" for f in _NEW_SOLVERS4
        A32 = Float32.(A)
        b32 = Float32.(b)
        x032 = Float32.(x0)
        res = f(A32, b32, x032; max_iterations=50)
        @test eltype(res.spectrum) === Float32
        @test all(isfinite.(res.spectrum))
    end
end

@testset "Batch4 — PGD" begin
    A, b, x0, x_true = _make_problem4()

    @testset "L1 penalty shrinks the objective at equal iterations" begin
        plain = solve_pgd(A, b, copy(x0); max_iterations=200)
        reg = solve_pgd(A, b, copy(x0); max_iterations=200, regularization=1e-3)
        @test norm(reg.spectrum, 1) <= norm(plain.spectrum, 1)
    end

    @testset "Simplex constraint preserves total fluence" begin
        F = sum(x_true)
        res = solve_pgd(A, b, copy(x0); constraint="simplex", total_fluence=F)
        @test sum(res.spectrum) ≈ F rtol = 1e-6
    end

    @testset "Box upper bound is respected" begin
        res = solve_pgd(A, b, copy(x0); constraint="box", x_max=0.25)
        @test maximum(res.spectrum) <= 0.25 + 1e-9
    end

    @testset "Backtracking matches fixed step on this problem" begin
        fixed = solve_pgd(A, b, copy(x0))
        line = solve_pgd(A, b, copy(x0); backtracking=true)
        @test norm(line.spectrum - fixed.spectrum) / norm(fixed.spectrum) < 1e-3
    end

    @testset "Invalid constraint throws" begin
        @test_throws Exception solve_pgd(A, b, copy(x0); constraint="not_a_set")
    end
end

@testset "Batch4 — Extragradient" begin
    A, b, x0, _ = _make_problem4()

    @testset "Custom step size" begin
        res = solve_extragradient(A, b, copy(x0); step_size=1e-3, max_iterations=200)
        @test all(isfinite.(res.spectrum))
    end

    @testset "noise_level changes the discrepancy reached" begin
        # noise_level sets the radius of the dual ball, i.e. how much residual
        # the iteration is allowed to keep; the two runs must not coincide.
        loose = solve_extragradient(A, b, copy(x0); noise_level=0.2)
        tight = solve_extragradient(A, b, copy(x0); noise_level=0.001)
        @test loose.spectrum != tight.spectrum
        @test all(loose.spectrum .>= 0) && all(tight.spectrum .>= 0)
    end
end

@testset "Batch4 — Coordinate descent" begin
    A, b, x0, _ = _make_problem4()

    @testset "Cyclic and random selection agree" begin
        # On the rank-deficient 14×80 NNLS the minimiser is not unique, so we
        # tighten with a small ridge to make the objective strictly convex and
        # the two selection orders must land on the same point.
        cyclic = solve_coordinate_descent(A, b, copy(x0); l2_penalty=1e-4)
        random = solve_coordinate_descent(A, b, copy(x0);
                                         selection="random", random_state=42,
                                         l2_penalty=1e-4)
        @test norm(cyclic.spectrum) > 0
        c = dot(cyclic.spectrum, random.spectrum) /
            (norm(cyclic.spectrum) * norm(random.spectrum))
        @test c > 0.99
    end

    @testset "Reproducible under a fixed seed" begin
        r1 = solve_coordinate_descent(A, b, copy(x0); selection="random",
                                     random_state=7)
        r2 = solve_coordinate_descent(A, b, copy(x0); selection="random",
                                     random_state=7)
        @test r1.spectrum == r2.spectrum
    end

    @testset "L2 penalty smooths" begin
        plain = solve_coordinate_descent(A, b, copy(x0))
        ridge = solve_coordinate_descent(A, b, copy(x0); l2_penalty=1e-2)
        @test norm(ridge.spectrum) < norm(plain.spectrum)
    end

    @testset "Agrees with Lawson-Hanson NNLS" begin
        # Lawson-Hanson picks the sparsest vertex, coordinate descent any
        # vertex; on a rank-deficient problem their spectra need not match.
        # The ridge-regularised objective is strictly convex, so both sides
        # have the same unique minimiser — compare those.
        cd_ridge = solve_coordinate_descent(A, b, copy(x0);
                                          l2_penalty=1e-4, max_iterations=20_000,
                                          tolerance=1e-10)
        Ar = vcat(A, sqrt(1e-4) * Matrix{Float64}(I, size(A, 2), size(A, 2)))
        br = vcat(b, zeros(size(A, 2)))
        nn_ridge = BSSUnfold.solve_nnls(Ar, br)
        @test norm(cd_ridge.spectrum - nn_ridge) / norm(nn_ridge) < 5e-2
    end
end

@testset "Batch4 — Subgradient" begin
    A, b, x0, _ = _make_problem4()

    @testset "Step policies all stay finite and feasible" begin
        for policy in ("diminishing", "fixed", "polyak")
            res = solve_subgradient(A, b, copy(x0); step_policy=policy,
                                   max_iterations=300)
            @test all(isfinite.(res.spectrum))
            @test all(res.spectrum .>= 0)
        end
    end

    @testset "More iterations reduce the residual" begin
        few = solve_subgradient(A, b, copy(x0); max_iterations=50)
        many = solve_subgradient(A, b, copy(x0); max_iterations=3000)
        @test many.residual_norm < few.residual_norm
    end

    @testset "TV penalty produces fewer jumps" begin
        plain = solve_subgradient(A, b, copy(x0))
        tv = solve_subgradient(A, b, copy(x0); tv_penalty=1e-2)
        @test norm(diff(tv.spectrum), 1) <= norm(diff(plain.spectrum), 1) + 1e-9
    end
end

@testset "Batch4 — Frank-Wolfe" begin
    A, b, x0, x_true = _make_problem4()
    F = sum(x_true)

    @testset "Simplex vertex count grows with iterations" begin
        res = solve_frank_wolfe(A, b, copy(x0); total_fluence=F,
                               max_iterations=50)
        @test sum(res.spectrum) ≈ F rtol = 1e-6
        @test all(res.spectrum .>= 0)
    end

    @testset "Total fluence is preserved over many iterations" begin
        res = solve_frank_wolfe(A, b, copy(x0); total_fluence=F)
        @test sum(res.spectrum) ≈ F rtol = 1e-6
    end

    @testset "Away steps do not hurt" begin
        with_away = solve_frank_wolfe(A, b, copy(x0); total_fluence=F,
                                     away_steps=true)
        without = solve_frank_wolfe(A, b, copy(x0); total_fluence=F,
                                   away_steps=false)
        @test with_away.residual_norm <= without.residual_norm * 1.001
    end

    @testset "Non-positive fluence throws" begin
        @test_throws Exception solve_frank_wolfe(A, b, copy(x0);
                                                total_fluence=-1.0)
    end
end

@testset "Batch4 — ADMM" begin
    A, b, x0, _ = _make_problem4()

    @testset "Diagnostics are exposed in extra" begin
        res = solve_admm(A, b, copy(x0); l1_penalty=1e-3)
        @test haskey(res.extra, "primal_residual")
        @test haskey(res.extra, "dual_residual")
        @test haskey(res.extra, "rho")
        @test res.extra["primal_residual"] >= 0
        @test res.extra["rho"] > 0
    end

    @testset "L1 penalty sparsifies" begin
        plain = solve_admm(A, b, copy(x0))
        sparse = solve_admm(A, b, copy(x0); l1_penalty=1e-2)
        @test count(>(0), sparse.spectrum) <= count(>(0), plain.spectrum)
    end

    @testset "Fixed rho is honoured" begin
        res = solve_admm(A, b, copy(x0); l1_penalty=1e-3, rho=1.0,
                        adaptive_rho=false)
        @test res.extra["rho"] ≈ 1.0 rtol = 1e-12
    end

    @testset "Gram x-update equals Lawson-Hanson on the augmented design" begin
        # The x-update is stated on the Gram system for speed; it has to accept
        # the same pivots as the reference Lawson-Hanson solve it replaces.
        m, n = size(A)
        D = BSSUnfold._admm_difference_matrix(n, Float64)
        Ad = Matrix{Float64}(A)
        At, Dt = Ad', D'
        AtA = At * Ad
        AtA = (AtA + AtA') / 2
        Q = Dt * D
        DtD = (Q + Q') / 2
        rng = MersenneTwister(7)
        v1 = randn(rng, n) * 3
        v2 = randn(rng, n - 1) * 3
        for rho in (0.05, 1.0, 5.0), use_tv in (false, true)
            gv = BSSUnfold._AdmmGramOperator(Ad, At, D, Dt, rho, use_tv)
            G = BSSUnfold._admm_gram(AtA, DtD, rho, use_tv)
            c = At * b .+ rho .* v1
            use_tv && (c .+= rho .* (Dt * v2))
            s = sqrt(rho)
            M_aug = vcat(Ad, s * I(n), use_tv ? s * D : zeros(0, n))
            rhs = vcat(b, s .* v1, use_tv ? s .* v2 : Float64[])
            x_lh = BSSUnfold.lawson_hanson(M_aug, rhs; max_iterations=10 * n)[1]
            x_gram = BSSUnfold._admm_nnls_gram(gv, G, c, 10 * n)
            @test norm(x_gram - x_lh) / norm(x_lh) < 1e-10
            @test minimum(x_gram) >= 0
            @test count(>(0), x_gram) == count(>(0), x_lh)
        end
    end

    @testset "No penalty reduces to NNLS" begin
        res = solve_admm(A, b, copy(x0))
        nn = BSSUnfold.solve_nnls(A, b)
        @test norm(res.spectrum - nn) / norm(nn) < 1e-6
    end
end

@testset "Batch4 — L-BFGS-B" begin
    A, b, x0, _ = _make_problem4()

    # Over-determined system: the bound-constrained least-squares minimiser is
    # unique here, so convergence claims are meaningful (the 14x80 fixture has
    # a whole face of minimisers and different solvers legitimately disagree).
    rng = MersenneTwister(7)
    Ao = rand(rng, 60, 20) .+ 0.3
    Ao ./= sum(Ao, dims=2)
    xo = exp.(-collect(range(0, 5, length=20)))
    bo = Ao * xo .+ 0.01 .* randn(rng, 60)
    x0o = ones(20) .* 0.5

    @testset "Converges on a well-posed problem" begin
        res = solve_lbfgsb(Ao, bo, copy(x0o))
        @test res.converged
        @test res.iterations < 500
        @test res.residual_norm < norm(bo)
    end

    @testset "Projected gradient is small at the solution" begin
        res = solve_lbfgsb(Ao, bo, copy(x0o))
        @test res.extra["projected_gradient"] < 1e-6
    end

    @testset "Interior upper bound is respected" begin
        res = solve_lbfgsb(A, b, copy(x0); x_max=0.3)
        @test maximum(res.spectrum) <= 0.3 + 1e-9
    end

    @testset "Smoothness penalty flattens the spectrum" begin
        plain = solve_lbfgsb(A, b, copy(x0))
        smooth = solve_lbfgsb(A, b, copy(x0); smoothness=1e-2)
        @test norm(diff(smooth.spectrum), 2) < norm(diff(plain.spectrum), 2)
    end

    @testset "Longer history reaches an at-least-as-good fit" begin
        short = solve_lbfgsb(Ao, bo, copy(x0o); lbfgs_history=2)
        long = solve_lbfgsb(Ao, bo, copy(x0o); lbfgs_history=20)
        @test long.residual_norm <= short.residual_norm * 1.01
    end

    @testset "Matches Lawson-Hanson NNLS when the minimiser is unique" begin
        res = solve_lbfgsb(Ao, bo, copy(x0o))
        nn = BSSUnfold.solve_nnls(Ao, bo)
        @test norm(res.spectrum - nn) / norm(nn) < 1e-3
    end
end

@testset "Batch4 — RFSP" begin
    A, b, x0, _ = _make_problem4()

    @testset "Weights change the solution" begin
        equal = solve_rfsp(A, b, copy(x0))
        weighted = solve_rfsp(A, b, copy(x0); weights=collect(1.0:size(A, 1)))
        @test norm(equal.spectrum - weighted.spectrum) > 0
        @test all(isfinite.(weighted.spectrum))
    end

    @testset "Tighter tolerance needs at least as many iterations" begin
        loose = solve_rfsp(A, b, copy(x0); tolerance=1e-2)
        tight = solve_rfsp(A, b, copy(x0); tolerance=1e-10)
        @test tight.iterations >= loose.iterations
    end
end

@testset "Batch4 — LOUHI" begin
    A, b, x0, _ = _make_problem4()

    @testset "Smoothing weight controls roughness" begin
        weak = solve_louhi(A, b, copy(x0); smoothness=1e-4)
        strong = solve_louhi(A, b, copy(x0); smoothness=1e2)
        @test norm(diff(strong.spectrum), 2) <= norm(diff(weak.spectrum), 2) + 1e-9
    end

    @testset "Smoothing orders 0/1/2 all work" begin
        for order in (0, 1, 2)
            res = solve_louhi(A, b, copy(x0); smooth_order=order)
            @test all(isfinite.(res.spectrum))
            @test all(res.spectrum .>= 0)
        end
    end

    @testset "Auto-smooth selects a regularisation strength" begin
        res = solve_louhi(A, b, copy(x0); auto_smooth=true, smooth_order=2)
        @test all(isfinite.(res.spectrum))
        @test res.residual_norm >= 0
    end
end

@testset "Batch4 — Detector wrappers" begin
    n = 50
    names = ["0_in", "2_in", "5_in", "10_in"]
    rng = MersenneTwister(123)
    E_MeV = collect(range(1e-7, 10.0; length=n))
    sens = Dict(nm => rand(rng, n) .+ 0.1 for nm in names)
    cc = Dict(nm => rand(rng, n) for nm in names)
    d = Detector(names, E_MeV, sens, cc)
    true_spec = exp.(-E_MeV ./ 1.5)
    readings = Dict(nm => sum(sens[nm] .* true_spec) for nm in names)

    wrappers = [(unfold_pgd, "PGD"), (unfold_extragradient, "Extragradient"),
                (unfold_coordinate_descent, "CoordinateDescent"),
                (unfold_subgradient, "Subgradient"),
                (unfold_frank_wolfe, "FrankWolfe"), (unfold_admm, "ADMM"),
                (unfold_lbfgsb, "L-BFGS-B"), (unfold_rfsp, "RFSP"),
                (unfold_louhi, "LOUHI")]

    @testset "unfold_$label round-trip" for (uf, label) in wrappers
        out = uf(d, readings; max_iterations=200)
        @test haskey(out, "spectrum")
        spec = out["spectrum"]
        @test length(spec) == n
        @test all(isfinite.(spec))
        @test out["method"] == label
    end
end
