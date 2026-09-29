using BSSUnfold, LinearAlgebra, Random, Printf
QUO = '"'

function load_spectra()
    path = joinpath(@__DIR__, "..", "..", "test", "data",
                    "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv")
    out = Vector{Vector{Float64}}()
    for line in eachline(path)
        parts = split(strip(line), ',')
        vals = Float64[]
        for p in parts[2:end]
            v = tryparse(Float64, strip(p, QUO))
            v !== nothing && isfinite(v) && push!(vals, v)
        end
        !isempty(vals) && push!(out, vals)
    end
    return out
end

# OSEM exactly as in HEAD (pre-port): sensitivity-normalised subset MLEM
function osem_old(A, b, x0; max_iterations=50, n_subsets=4, tol=1e-6)
    m, n = size(A)
    eps = 1e-10
    x = max.(copy(x0), eps)
    size_sub = max(cld(m, n_subsets), 1)
    subs = [collect((i - 1) * size_sub + 1:min(i * size_sub, m)) for i in 1:n_subsets]
    sens = [vec(sum(A[s, :], dims=1)) for s in subs]
    for k in 1:max_iterations
        xp = copy(x)
        for (i, st) in enumerate(subs)
            As = view(A, st, :)
            bs = view(b, st)
            ax = max.(As * x, eps)
            x = max.(x .* (As' * (bs ./ ax)) ./ max.(sens[i], eps), 0)
        end
        norm(x .- xp) / (norm(xp) + eps) < tol && break
    end
    return x
end

spectra = load_spectra()
x_true = spectra[1]
n = length(x_true)
rng = MersenneTwister(777)
A = rand(rng, 14, n) .+ 0.3
A ./= sum(A, dims=2)
b = A * x_true .+ 0.005 .* randn(rng, 14)
x0 = ones(n) .* (sum(b) / 14)

mlem = solve_mlem(A, b, x0, max_iterations=1000).spectrum
new = solve_osem(A, b, x0, max_iterations=50, n_subsets=4).spectrum
old = osem_old(A, b, x0, max_iterations=50, n_subsets=4)
gravel = solve_gravel(A, b, x0, max_iterations=500).spectrum
land = solve_landweber(A, b, x0, max_iterations=500).spectrum
cos(a, c) = dot(a, c) / (norm(a) * norm(c) + 1e-30)

for (tag, o) in (("HEAD-osem", old), ("new-osem", new))
    rs = Dict("MLEM" => mlem, "GRAVEL" => gravel, "Landweber" => land, "OSEM" => o)
    ms = sort(collect(keys(rs)))
    npass = 0
    ntot = 0
    for i in 1:length(ms), j in i+1:length(ms)
        c = cos(rs[ms[i]], rs[ms[j]])
        ntot += 1
        npass += c > 0.3
        @printf("%-10s %-9s vs %-10s cos=%.3f\n", tag, ms[i], ms[j], c)
    end
    @printf("%s pass rate = %d/%d = %.3f\n\n", tag, npass, ntot, npass / ntot)
end
@printf("osem_old vs MLEM  cos=%.3f  cos_true=%.3f\n", cos(old, mlem), cos(old, x_true))
@printf("osem_new vs MLEM  cos=%.3f  cos_true=%.3f\n", cos(new, mlem), cos(new, x_true))
@printf("osem_old vs new   cos=%.3f\n", cos(old, new))
