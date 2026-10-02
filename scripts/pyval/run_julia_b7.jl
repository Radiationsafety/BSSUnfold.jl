# Julia side of batch 7 (first-order / convex / classical BSS ports).
# Usage: julia --project=<pkgroot> run_julia_b7.jl <problem.json> <outdir>

using BSSUnfold
using JSON
using LinearAlgebra
using Printf

const problem = JSON.parse(read(open(ARGS[1]), String))
const DIR = ARGS[2]
const A = hcat((Float64.(r) for r in problem["A"])...)'
const b = Float64.(problem["b"])
const x0 = Float64.(problem["x0"])
const x_true = Float64.(problem["x_true"])
const fluence = sum(x_true)

mkpath(DIR)

function run(name::String, f; kwargs...)
    payload = try
        r = f(A, b, copy(x0); kwargs...)
        chi2 = norm(A * r.spectrum - b)^2 / norm(b)^2
        @printf("%-22s iters=%5d converged=%-5s chi2=%.6g\n",
                name, r.iterations, string(r.converged), chi2)
        Dict("spectrum" => Vector{Float64}(r.spectrum), "chi2" => chi2,
             "iterations" => r.iterations, "converged" => r.converged)
    catch err
        @printf("%-22s ERROR %s\n", name, err)
        Dict("error" => sprint(showerror, err))
    end
    JSON.print(open(joinpath(DIR, name * ".json"), "w"), payload)
end

run("pgd", solve_pgd)
run("pgd_l1", solve_pgd; regularization=1e-3)
run("pgd_simplex", solve_pgd; constraint="simplex", total_fluence=fluence)
run("pgd_backtrack", solve_pgd; backtracking=true)
run("extragradient", solve_extragradient)
run("coordinate_descent", solve_coordinate_descent)
run("coordinate_descent_l1", solve_coordinate_descent; l1_penalty=1e-3)
run("subgradient", solve_subgradient)
run("subgradient_l1", solve_subgradient; l1_penalty=1e-3, step_policy="polyak")
run("frank_wolfe", solve_frank_wolfe; total_fluence=fluence)
run("frank_wolfe_away", solve_frank_wolfe; total_fluence=fluence, away_steps=false)
run("admm_l1", solve_admm; l1_penalty=1e-3)
run("admm_tv", solve_admm; tv_penalty=1e-3)
run("lbfgsb", solve_lbfgsb)
run("lbfgsb_smooth", solve_lbfgsb; regularization=1e-3, smoothness=1e-3)
run("rfsp", solve_rfsp)
run("louhi", solve_louhi)
run("louhi_order2", solve_louhi; smooth_order=2, auto_smooth=true)

println("julia b7 done")
