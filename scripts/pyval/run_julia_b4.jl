# Julia reference run, BON95 parametric2 only.
# Usage: julia --project=. run_julia_b4.jl <problem.json> <outdir>

using BSSUnfold
using JSON
using LinearAlgebra

const DIR = ARGS[2]

function main()
    problem = JSON.parse(String(read(ARGS[1])))
    A = hcat((Float64.(r) for r in problem["A"])...)'
    b = Float64.(problem["b"])
    E_MeV = Float64.(problem["E_MeV"])
    n = length(E_MeV)
    le = log10.(E_MeV)
    log_steps = similar(le)
    log_steps[1] = le[2] - le[1]
    log_steps[n] = le[n] - le[n-1]
    log_steps[2:(n-1)] .= (le[3:end] .- le[1:(n-2)]) ./ 2
    ln_steps = log_steps .* log(10.0)

    mkpath(DIR)
    res = solve_parametric2(A, b; E=E_MeV, ln_steps=ln_steps)
    println("parametric2: res=", res.residual_norm, " nfev=", res.iterations,
            " success=", res.converged, " msg=", res.extra["message"])
    open(joinpath(DIR, "parametric2.json"), "w") do io
        JSON.print(io, Dict("spectrum" => Vector{Float64}(res.spectrum)))
    end
end

main()
