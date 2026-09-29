# Julia reference for MAEO cross-check (batch b2).
# Usage: julia --project=. run_julia_b2.jl <problem.json> <outdir> [seed]
# Writes <outdir>/maeo.json and <outdir>/maeo_ensemble.json.

using BSSUnfold
using JSON
using LinearAlgebra

function main(problem_path, outdir, seed=11)
    mkpath(outdir)
    prob = JSON.parse(read(problem_path, String))
    A = hcat((Float64.(r) for r in prob["A"])...)'
    b = Float64.(prob["b"])
    E = Float64.(prob["E_MeV"])

    cfgs = [
        "maeo" => () -> solve_maeo(A, b, E; E_MeV=E, seed=seed),
        "maeo_ensemble" => () -> solve_maeo_ensemble(A, b, E; E_MeV=E, seed=seed),
    ]
    for (name, f) in cfgs
        res = f()
        x = res.spectrum
        chi2 = sum((A * x .- b) .^ 2) / dot(b, b)
        open(joinpath(outdir, name * ".json"), "w") do io
            JSON.print(io, Dict("spectrum" => Vector{Float64}(x), "chi2" => chi2))
        end
        println("jl ok $name chi2=$chi2")
    end
end

main(ARGS[1], ARGS[2], length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 11)
