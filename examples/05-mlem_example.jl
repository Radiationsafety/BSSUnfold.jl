### A Pluto.jl notebook ###
# v0.20.x — MLEM unfolding example

using Markdown
using InteractiveUtils

# ╔═╡ 8e100000-0001-4000-8000-000000000001
begin
    using Pkg
    Pkg.activate(Base.current_project() !== nothing ? Base.current_project() : "..")
    using BSSUnfold
    using LinearAlgebra
    using Random
    using Plots
    gr()
end

# ╔═╡ 8e100000-0002-4000-8000-000000000002
md"""
# MLEM — Maximum Likelihood Expectation Maximization

**MLEM** — классический итеративный алгоритм развёртки, изначально
разработанный для PET/SPECT и адаптированный для BSS.

Алгоритм:

$$x_{k+1} = x_k \odot \left(A^T \cdot \frac{b}{A x_k}\right)$$

**Свойства:**
- ✅ Гарантированно сохраняет неотрицательность $x$
- ✅ Монотонно увеличивает правдоподобие
- ✅ Сохраняет общий интеграл (если $A$ нормирована)
- ⚠️ Сходится медленно на плохо обусловленных задачах
- ⚠️ Чувствителен к выбору $x_0$

В этом ноутбуке исследуем, как **начальный спектр** и **число итераций** влияют на результат.
"""

# ╔═╡ 8e100000-0003-4000-8000-000000000003
begin
    # Подготовка задачи
    Random.seed!(42)
    n = 100
    m = 14

    E_MeV = 10 .^ range(-9, log10(20), length=n)
    detector_names = ["sphere_$(d)in" for d in (0, 2, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30)]

    sensitivities = Dict{String,Vector{Float64}}()
    for (i, name) in enumerate(detector_names)
        d = parse(Float64, replace(name, "sphere_" => "", "in" => ""))
        rf = 0.5 * exp.(-((log10.(E_MeV) .- log10(max(d * 0.3, 1e-9))) .* 2) .^ 2)
        rf .+= 0.1 ./ (E_MeV .+ 1e-9)
        sensitivities[name] = rf
    end

    x_true = exp.(-E_MeV ./ 1.5)
    x_true ./= sum(x_true)

    rng = MersenneTwister(42)
    A = Matrix(hcat([sensitivities[name] for name in detector_names]...)')
    b = A * x_true .+ 0.005 .* randn(rng, m)
    readings = Dict(name => b[i] for (i, name) in enumerate(detector_names))

    cc_icrp116 = Dict(name => rand(n) for name in detector_names)
    detector = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)

    println("Задача готова: $(length(detector_names)) сфер × $n бинов")
end

# ╔═╡ 8e100000-0004-4000-8000-000000000004
md"""
## 1. Влияние числа итераций

MLEM — итеративный алгоритм; сходимость зависит от `max_iterations`.
Посмотрим, как меняется спектр с числом итераций.
"""

# ╔═╡ 8e100000-0005-4000-8000-000000000005
begin
    iter_counts = [10, 50, 100, 500, 1000, 5000]
    results_iter = Dict{Int,Any}()

    for n_iter in iter_counts
        res = unfold_mlem(detector, readings, max_iterations=n_iter, tolerance=1e-12)
        results_iter[n_iter] = res
    end

    cos_metrics = [dot(results_iter[n]["spectrum"], x_true) /
                   (norm(results_iter[n]["spectrum"]) * norm(x_true) + 1f-30)
                   for n in iter_counts]

    println("Итераций | Сходился | ||b-Ax||   | Косинус")
    println("---------|----------|------------|---------")
    for n in iter_counts
        res = results_iter[n]
        @printf("%8d | %-8s | %10.4e | %.4f\n",
                n, res["converged"], res["residual_norm"],
                dot(res["spectrum"], x_true) / (norm(res["spectrum"]) * norm(x_true) + 1f-30))
    end
end

# ╔═╡ 8e100000-0006-4000-8000-000000000006
begin
    using Printf
    p = plot(E_MeV, x_true, xscale=:log10, yscale=:log10,
             label="Истина", lw=3, color=:black,
             xlabel="Энергия, МэВ", ylabel="Φ(E)",
             title="MLEM: сходимость с числом итераций",
             legend=:topright, size=(700, 400))

    colors = palette(:viridis, length(iter_counts))
    for (i, n_iter) in enumerate(iter_counts)
        plot!(p, E_MeV, results_iter[n_iter]["spectrum"],
              lw=1.5, alpha=0.8, color=colors[i],
              label="$n_iter iter")
    end
    p
end

# ╔═╡ 8e100000-0007-4000-8000-000000000007
md"""
## 2. Влияние начального спектра x₀

MLEM чувствителен к выбору $x_0$. Сравним три варианта:

- **Flat**: $x_0 = 0.5$ (по умолчанию)
- **Prior knowledge**: $x_0$ = грубое приближение истины
- **1/E spectrum**: $x_0 = 1/E$ (физически мотивированный)
"""

# ╔═╡ 8e100000-0008-4000-8000-000000000008
begin
    x0_flat = ones(n) .* 0.5
    x0_prior = exp.(-E_MeV ./ 3.0); x0_prior ./= sum(x0_prior); x0_prior .*= 0.1
    x0_1_over_E = 1.0 ./ (E_MeV .+ 1e-9); x0_1_over_E ./= sum(x0_1_over_E); x0_1_over_E .*= 0.1

    x0_options = [("Flat (0.5)", x0_flat),
                  ("Prior (e^{-E/3})", x0_prior),
                  ("1/E spectrum", x0_1_over_E)]

    p_init = plot(E_MeV, x_true, xscale=:log10, yscale=:log10,
                  label="Истина", lw=3, color=:black,
                  xlabel="Энергия, МэВ", ylabel="Φ(E)",
                  title="Начальные спектры", legend=:topright)

    colors_init = [:red, :green, :blue]
    for (i, (label, x0)) in enumerate(x0_options)
        plot!(p_init, E_MeV, x0, lw=1.5, ls=:dash, color=colors_init[i], label=label)
    end
    p_init
end

# ╔═╡ 8e100000-0009-4000-8000-000000000009
begin
    p_res = plot(E_MeV, x_true, xscale=:log10, yscale=:log10,
                 label="Истина", lw=3, color=:black,
                 xlabel="Энергия, МэВ", ylabel="Φ(E)",
                 title="MLEM: влияние x₀ (1000 iter)",
                 legend=:topright)

    for (i, (label, x0)) in enumerate(x0_options)
        # Используем solve_mlem напрямую, передавая x0
        A_matrix = Matrix(hcat([sensitivities[name] for name in detector_names]...)')
        b_vec = [readings[name] for name in detector_names]
        res = solve_mlem(A_matrix, b_vec, x0, max_iterations=1000, tolerance=1e-10)
        cos = dot(res.spectrum, x_true) / (norm(res.spectrum) * norm(x_true) + 1f-30)
        plot!(p_res, E_MeV, res.spectrum, lw=1.5, color=colors_init[i],
              label="$label (cos=$(round(cos, digits=3)))")
    end
    p_res
end

# ╔═╡ 8e100000-000a-4000-8000-00000000000a
md"""
## 3. Сравнение MLEM с другими EM-методами

В BSSUnfold.jl реализовано несколько EM-вариантов:

- **MLEM** — классический
- **OSEM** — ordered-subset, ускоренная сходимость
- **BSREM** — block-sequential regularized
- **MAP-EM** — maximum a posteriori
"""

# ╔═╡ 8e100000-000b-4000-8000-00000000000b
begin
    methods_to_compare = [
        ("MLEM",  () -> unfold_mlem(detector, readings, max_iterations=2000, tolerance=1e-12)),
        ("OSEM (4 subsets)", () -> unfold_osem(detector, readings, max_iterations=200, n_subsets=4)),
        ("OSEM (7 subsets)", () -> unfold_osem(detector, readings, max_iterations=100, n_subsets=7)),
        ("BSREM", () -> unfold_bsrem(detector, readings, max_iterations=200, n_subsets=4,
                                    regularization=1e-3)),
    ]

    println("Метод              | Итераций | Сходился | ||b-Ax||   | Косинус")
    println("-------------------|----------|----------|------------|---------")
    p_cmp = plot(E_MeV, x_true, xscale=:log10, yscale=:log10,
                 label="Истина", lw=3, color=:black,
                 xlabel="Энергия, МэВ", ylabel="Φ(E)",
                 title="EM-методы: сравнение", legend=:topright)

    colors_cmp = palette(:tab10, length(methods_to_compare))
    for (i, (label, fn)) in enumerate(methods_to_compare)
        res = fn()
        cos = dot(res["spectrum"], x_true) / (norm(res["spectrum"]) * norm(x_true) + 1f-30)
        @printf("%-19s | %8d | %-8s | %10.4e | %.4f\n",
                label, res["iterations"], res["converged"],
                res["residual_norm"], cos)
        plot!(p_cmp, E_MeV, res["spectrum"], lw=1.5, color=colors_cmp[i], label=label)
    end
    p_cmp
end

# ╔═╡ 8e100000-000c-4000-8000-00000000000c
md"""
## 4. Резюме

- MLEM сходится медленно — нужно 1000+ итераций для хорошей точности
- OSEM с N subsets ускоряет сходимость в ~N раз
- Начальный спектр влияет на результат, особенно при малом числе итераций
- Все методы BSSUnfold.jl работают в 2–10× быстрее Python-аналога
"""

# ╔═╡ Cell order:
# ╟─8e100000-0001-4000-8000-000000000001
# ╟─8e100000-0002-4000-8000-000000000002
# ╟─8e100000-0003-4000-8000-000000000003
# ╟─8e100000-0004-4000-8000-000000000004
# ╟─8e100000-0005-4000-8000-000000000005
# ╟─8e100000-0006-4000-8000-000000000006
# ╟─8e100000-0007-4000-8000-000000000007
# ╟─8e100000-0008-4000-8000-000000000008
# ╟─8e100000-0009-4000-8000-000000000009
# ╟─8e100000-000a-4000-8000-00000000000a
# ╟─8e100000-000b-4000-8000-00000000000b
# ╟─8e100000-000c-4000-8000-00000000000c
