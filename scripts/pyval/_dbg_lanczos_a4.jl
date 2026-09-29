using BSSUnfold, JSON, LinearAlgebra
d = JSON.parse(read(open("/tmp/bsscmp/problem.json"), String))
A = hcat((Float64.(r) for r in d["A"])...)'
b = Float64.(d["b"])
m, n = size(A)
ref = Float64.(JSON.parse(read(open("/tmp/bsscmp/py/lanczos.json"), String))["spectrum"])

function trace(A, b; layout::Bool)
    m, n = size(A)
    Bm = layout ? copy(permutedims(A)) : A
    beta = norm(b)
    U = [b / beta]
    Vcols = Vector{Float64}[]
    alphas = Float64[]; betas = Float64[]
    local x
    for k in 1:min(m, n)
        u = U[k]
        v = k == 1 ? (layout ? Bm * u : A' * u) : (layout ? Bm * u : A' * u) - betas[end] * Vcols[end]
        alpha = norm(v); v = v / alpha; push!(Vcols, v)
        w = layout ? transpose(Bm) * v : A * v
        u2 = w - alpha * u
        nb = norm(u2)
        if nb > 1e-14 push!(U, u2 / nb) end
        push!(alphas, alpha); push!(betas, nb)
        B = zeros(k + 1, k)
        for i in 1:k B[i, i] = alphas[i]; i < k && (B[i + 1, i] = betas[i]) end
        bhat = zeros(k + 1); bhat[1] = beta
        lam = BSSUnfold._projected_gcv(B, bhat, m)
        (lam <= 0 || !isfinite(lam)) && (lam = 1e-8)
        F = svd(B; full = false)
        c = F.U' * bhat; s = F.S
        y = F.Vt * (s .* c ./ (s .^ 2 .+ lam))
        x = reduce(hcat, Vcols) * y
    end
    return x
end
for layout in (false, true)
    x = trace(A, b; layout)
    println("layout=$layout relL2=", norm(x - ref) / norm(ref), " cos=", dot(x, ref) / (norm(x) * norm(ref)))
end
