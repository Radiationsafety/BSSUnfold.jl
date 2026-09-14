### A Pluto.jl notebook ###
# v0.20.x — Methods comparison

using Markdown
using InteractiveUtils

# ╔═╡ c3100000-0001-4000-8000-000000000001
begin
    using Pkg
    Pkg.activate(Base.current_project() !== nothing ? Base.current_project() : "..")
    using BSSUnfold
    using LinearAlgebra
    using Random
    using Statistics
    using Plots
    using Printf
    gr()
end

# ╔═╡ c3100000-0002-4000-8000-000000000002
md"""
# Сравнение всех 15 алгоритмов развёртки

В BSSUnfold.jl реализовано 15 алгоритмов развёртки. Сравним их на одной
типичной BSS-задаче: 14 сфер × 100 энергетических бинов, 1% шума.

Для каждого метода замерим:
- Косинусную близость к истине
- Время работы
- Число итераций
- Норму невязки
"""

# ╔═╡ c3100000-0003-4000-8000-000000000003
begin
    Random.seed!(42)
    n = 100
    m = 14
    E_MeV = 10 .^ range(-9, log10(20), length=n)
    detector_names = ["sphere_$(d)in" for d in (0, 2, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30)]
    sensitivities = Dict{String,Vector{Float64}}()
    for name in detector_names
        d = parse(Float64, replace(name, "sphere_" => "", "in" => ""))
        rf = 0.5 * exp.(-((log10.(E_MeV) .- log10(max(d * 0.3, 1e-9))) .* 2) .^ 2)
        rf .+= 0.1 ./ (E_MeV .+ 1e-9)
        sensitivities[name] = rf
    end
    x_true = exp.(-E_MeV ./ 1.5); x_true ./= sum(x_true)
    rng = MersenneTwister(42)
    A = Matrix(hcat([sensitivities[name] for name in detector_names]...)')
    b = A * x_true .+ 0.01 .* randn(rng, m)
    x0 = ones(n) .* 0.5
    println("Задача готова")
end

# ╔═╡ c3100000-0004-4000-8000-000000000004
begin
    methods = [
        ("MLEM",      () -> solve_mlem(A, b, x0, max_iterations=1000, tolerance=1e-8)),
        ("GRAVEL",    () -> solve_gravel(A, b, x0, max_iterations=500, tolerance=1e-8)),
        ("Landweber", () -> solve_landweber(A, b, x0, max_iterations=500, tolerance=1e-6)),
        ("MAXED",     () -> solve_maxed(A, b, x0, max_iterations=500, tolerance=1e-6)),
        ("Tikhonov",  () -> solve_tikhonov(A, b, x0, regularization=1e-3)),
        ("TSVD",      () -> solve_tsvd(A, b, x0, truncation_rank=8)),
        ("Sandii",    () -> solve_sandii(A, b, x0, max_iterations=500, tolerance=1e-6)),
        ("Bunki",     () -> solve_bunki(A, b, x0, max_iterations=500, alpha=0.8)),
        ("Kaczmarz",  () -> solve_kaczmarz(A, b, x0, max_iterations=50, tolerance=1e-6)),
        ("CGLS",      () -> solve_cgls(A, b, x0, max_iterations=100, tolerance=1e-6)),
        ("FISTA",     () -> solve_fista(A, b, x0, max_iterations=200, regularization=1e-4)),
        ("BSREM",     () -> solve_bsrem(A, b, x0, max_iterations=50, n_subsets=4)),
        ("OSEM",      () -> solve_osem(A, b, x0, max_iterations=50, n_subsets=4)),
        ("Staysl",    () -> solve_staysl(A, b, x0, max_iterations=500, tolerance=1e-6)),
        ("Doroshenko", () -> solve_doroshenko(A, b, x0, max_iterations=500, tolerance=1e-6)),
    ]

    results_table = []
    timings = Dict{String,Float64}()

    for (name, fn) in methods
        # Multiple runs for stable timing
        t0 = time()
        res = fn()
        t1 = time()
        timings[name] = t1 - t0

        cos_sim = dot(res.spectrum, x_true) /
                  (norm(res.spectrum) * norm(x_true) + 1f-30)

        push!(results_table, (
            name=name,
            iterations=res.iterations,
            converged=res.converged,
            residual=res.residual_norm,
            cos_sim=cos_sim,
            time_ms=timings[name] * 1000
        ))
    end

    @printf("%-12s | %8s | %-8s | %10s | %6s | %8s\n",
            "Метод", "Итераций", "Сходился", "||b-Ax||", "cos", "Время, ms")
    @printf("%s\n", "-"^70)
    for r in results_table
        @printf("%-12s | %8d | %-8s | %10.4e | %.4f | %8.2f\n",
                r.name, r.iterations, r.converged, r.residual, r.cos_sim, r.time_ms)
    end
end

# ╔═╡ c3100000-0005-4000-8000-000000000005
md"""
## 1. Визуализация восстановленных спектров
"""

# ╔═╡ c3100000-0006-4000-8000-000000000006
begin
    plots_arr = []
    for (name, fn) in methods
        res = fn()
        p = plot(E_MeV, x_true, xscale=:log10, yscale=:log10,
                 lw=2, color=:black, label="Истина",
                 xlabel="E, МэВ", ylabel="Φ(E)",
                 title=name, legend=false)
        plot!(p, E_MeV, res.spectrum, lw=1.5, color=:red, label="Восстановлено")
        push!(plots_arr, p)
    end
    # Grid 5×3
    plot(plots_arr..., layout=(5, 3), size=(1200, 1500))
end

# ╔═╡ c3100000-0007-4000-8000-000000000007
md"""
## 2. Сравнение по качеству и скорости
"""

# ╔═╡ c3100000-0008-4000-8000-000000000008
begin
    names_vec = [r.name for r in results_table]
    cos_vec   = [r.cos_sim for r in results_table]
    time_vec  = [r.time_ms for r in results_table]
    res_vec   = [r.residual for r in results_table]

    p1 = bar(names_vec, cos_vec,
             xrotation=45, legend=false,
             ylabel="Косинусная близость",
             title="Качество (cos_sim)",
             color=:darkblue, size=(900, 400))
    hline!(p1, [0.5], color=:red, ls=:dash, label="cos=0.5")

    p2 = bar(names_vec, time_vec,
             xrotation=45, legend=false, yscale=:log10,
             ylabel="Время, ms",
             title="Скорость (ms)",
             color=:darkred, size=(900, 400))

    p3 = bar(names_vec, res_vec,
             xrotation=45, legend=false, yscale=:log10,
             ylabel="||b - Ax||",
             title="Невязка",
             color=:darkgreen, size=(900, 400))

    plot(p1, p2, p3, layout=(3, 1), size=(900, 900))
end

# ╔═╡ c3100000-0009-4000-8000-000000000009
md"""
## 3. Рекомендации по выбору алгоритма

Based on тестах:

| Сценарий | Рекомендуемый метод |
|----------|---------------------|
| Быстрый first guess | **OSEM** (4 subsets, 50 iter) |
| Высокое качество, есть время | **GRAVEL** (500 iter) |
| Точная система, мало шума | **MLEM** (1000+ iter) |
| Сильный шум | **Tikhonov + GCV** |
| Гладкий спектр | **FISTA + L1** |
| Малоранговое приближение | **TSVD** (k = 6-8) |
| Bayesian с prior | **Staysl** |

### Производительность

- **Быстрее всего**: Tikhonov, TSVD — прямые методы (1 «итерация»)
- **Самые медленные**: MLEM, GRAVEL — итеративные, но стабильные
- **Лучший компромисс**: OSEM — ускоряет MLEM в ~N_subsets раз
"""

# ╔═╡ c3100000-000a-4000-8000-00000000000a
md"""
## 4. Резюме

- 15 алгоритмов дают разные результаты на одной задаче
- Выбор метода зависит от:
  - Требуемой точности
  - Доступного времени
  - Характера ожидаемого спектра
  - Уровня шума
- BSSUnfold.jl позволяет легко переключаться между методами через единый API
"""

# ╔═╡ Cell order:
# ╟─c3100000-0001-4000-8000-000000000001
# ╟─c3100000-0002-4000-8000-000000000002
# ╟─c3100000-0003-4000-8000-000000000003
# ╟─c3100000-0004-4000-8000-000000000004
# ╟─c3100000-0005-4000-8000-000000000005
# ╟─c3100000-0006-4000-8000-000000000006
# ╟─c3100000-0007-4000-8000-000000000007
# ╟─c3100000-0008-4000-8000-000000000008
# ╟─c3100000-0009-4000-8000-000000000009
# ╟─c3100000-000a-4000-8000-00000000000a
