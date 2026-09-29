# Julia reference run, FRUIT parametric only.
# Usage: julia --project=. run_julia_b3.jl <problem.json> <outdir>

using BSSUnfold
using JSON
using LinearAlgebra

const DIR = ARGS[2]

function main()
    problem = JSON.parse(String(read(ARGS[1])))
    A = hcat((Float64.(r) for r in problem["A"])...)'
    b = Float64.(problem["b"])
    x0 = Float64.(problem["x0"])
    E_MeV = Float64.(problem["E_MeV"])

    mkpath(DIR)
    res = solve_parametric(A, b, x0; E_MeV=E_MeV)
    println("parametric: res=", res.residual_norm, " nfev=", res.iterations,
            " success=", res.converged, " params=", res.extra["params"])
    open(joinpath(DIR, "parametric.json"), "w") do io
        JSON.print(io, Dict("spectrum" => Vector{Float64}(res.spectrum)))
    end
end

main()
