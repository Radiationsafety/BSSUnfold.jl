using BSSUnfold, LinearAlgebra, Random, Printf
QUO = '"'
function load()
    spectra = Vector{Vector{Float64}}()
    for line in eachline(joinpath(@__DIR__, "..", "..", "test", "data",
                                 "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv"))
        parts = split(strip(line), ',')
        vals = Float64[]
        for p in parts[2:end]
            v = tryparse(Float64, strip(p, QUO))
            v !== nothing && isfinite(v) && push!(vals, v)
        end
        !isempty(vals) && push!(spectra, vals)
    end
    return spectra
end
spectra = load()
x_true = spectra[1]; n = length(x_true)
rng = MersenneTwister(777)
A = rand(rng, 14, n) .+ 0.3; A ./= sum(A, dims=2)
b = A * x_true .+ 0.005 .* randn(rng, 14)
x0 = ones(n) .* (sum(b) / 14)
results = Dict(
    "MLEM"      => solve_mlem(A, b, x0, max_iterations=1000).spectrum,
    "GRAVEL"    => solve_gravel(A, b, x0, max_iterations=500).spectrum,
    "Landweber" => solve_landweber(A, b, x0, max_iterations=500).spectrum,
    "OSEM"      => solve_osem(A, b, x0, max_iterations=50, n_subsets=4).spectrum)
ms = sort(collect(keys(results)))
for i in 1:length(ms), j in i+1:length(ms)
    s1, s2 = results[ms[i]], results[ms[j]]
    @printf("%-10s vs %-10s cos=%.3f\n", ms[i], ms[j], dot(s1, s2) / (norm(s1) * norm(s2)))
end
for m in ms
    s = results[m]
    @printf("%-10s cos_true=%.3f chi2=%.3e sum=%.3f\n",
           m, dot(s, x_true) / (norm(s) * norm(x_true)), norm(A * s - b)^2 / norm(b)^2, sum(s))
end
