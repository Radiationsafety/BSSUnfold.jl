# Julia reference run, NN-KSVD only.
# Usage: julia --project=. run_julia_b5.jl <problem.json> <outdir>

using BSSUnfold
using JSON
using LinearAlgebra

const DIR = ARGS[2]

function main()
    problem = JSON.parse(String(read(ARGS[1])))
    A = hcat((Float64.(r) for r in problem["A"])...)'
    b = Float64.(problem["b"])
    x0 = Float64.(problem["x0"])

    mkpath(DIR)
    res = solve_nnksvd(A, b, x0)
    println("nnksvd: res=", res.residual_norm, " n_iter=", res.iterations,
            " converged=", res.converged)
    open(joinpath(DIR, "nnksvd.json"), "w") do io
        JSON.print(io, Dict("spectrum" => Vector{Float64}(res.spectrum)))
    end
end

main()
