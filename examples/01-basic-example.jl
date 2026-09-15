### A Pluto.jl notebook ###
# v0.20.x

using Markdown
using InteractiveUtils

# ╔═╡ 005c4e30-3c8a-4a90-b3df-7e15e08a32e0
begin
    using Pkg
    Pkg.activate(Base.current_project() !== nothing ? Base.current_project() : "..")
    using BSSUnfold
    using LinearAlgebra
    using Random
    using Statistics
    using Plots
    gr()
end

# ╔═╡ c1000000-0001-4000-8000-000000000001
md"""
# BSSUnfold.jl — Базовый пример развёртки

В этом ноутбуке показан классический сценарий использования BSSUnfold:

1. Создаём синтетическую **response matrix** (имитация спектрометра Боннера)
2. Генерируем «истинный» спектр нейтронов
3. Получаем синтетические показания детекторов
4. Разворачиваем спектр алгоритмом **GRAVEL**
5. Сравниваем восстановленный спектр с истиной

> **Установка:** `using Pkg; Pkg.add("BSSUnfold")`
"""

# ╔═╡ c1000000-0002-4000-8000-000000000002
md"""
## 1. Подготовка данных

Спектрометр Боннера (BSS) состоит из 14 сфер разного диаметра (0", 2", 3", 5",
8", 10", 12" и т.д.). Каждая сфера имеет свою **response function** —
чувствительность к нейтронам разных энергий.

Энергетическая сетка типично покрывает диапазон от тепловых нейтронов (10⁻⁹ МэВ)
до быстрых (20 МэВ), с логарифмическим шагом — обычно 100–640 бинов.
"""

# ╔═╡ c1000000-0003-4000-8000-000000000003
begin
    Random.seed!(42)

    # Конфигурация: 14 сфер, 100 энергетических бинов
    m = 14  # число сфер
    n = 100 # число бинов

    # Энергетическая сетка: логарифмическая от 1e-9 до 20 МэВ
    E_MeV = 10 .^ range(-9, log10(20), length=n)

    # Имена детекторов
    detector_names = ["sphere_$(d)in" for d in (0, 2, 3, 4.2, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18)]

    # Имитация response functions: пик смещается с ростом диаметра сферы
    sensitivities = Dict{String,Vector{Float64}}()
    for (i, name) in enumerate(detector_names)
        d = parse(Float64, replace(name, "sphere_" => "", "in" => ""))
        # Тепловая компонента + пик при ~d МэВ
        rf = 0.5 * exp.(-((log10.(E_MeV) .- log10(max(d * 0.3, 1e-9))) .* 2) .^ 2)
        rf .+= 0.1 ./ (E_MeV .+ 1e-9)  # 1/E компонента (тепловая)
        sensitivities[name] = rf
    end

    # Коэффициенты ICRP-116 для дозы (упрощённо)
    cc_icrp116 = interpolate_coefficients(get_coefficients("ICRP116"), E_MeV)

    println("Готово: $(length(detector_names)) сфер, $n энергетических бинов")
    println("Диапазон энергий: $(round(E_MeV[1], digits=2)) – $(round(E_MeV[end], digits=2)) МэВ")
end

# ╔═╡ c1000000-0004-4000-8000-000000000004
begin
    # Истинный спектр: типичный fission-спектр + тепловая компонента
    x_true = exp.(-E_MeV ./ 1.5) .* (1.0 .+ 0.3 .* sin.(E_MeV .* 2))
    x_true ./= sum(x_true)  # нормируем

    # Синтетические показания: b = A * x_true + noise
    rng = MersenneTwister(42)
    A = Matrix(hcat([sensitivities[name] for name in detector_names]...)')
    b_true = A * x_true
    b_noisy = b_true .+ 0.01 .* randn(rng, m)

    readings = Dict(name => b_noisy[i] for (i, name) in enumerate(detector_names))

    # Визуализация
    p1 = plot(E_MeV, x_true, xscale=:log10, yscale=:log10,
              label="Истинный спектр", lw=2, color=:darkblue,
              xlabel="Энергия, МэВ", ylabel="Φ(E), отн. ед.",
              title="Истинный спектр нейтронов",
              legend=:topright)
    p2 = bar(detector_names, b_noisy,
             label="Показания детекторов",
             color=:darkred, alpha=0.7,
             xlabel="Детектор", ylabel="Счёт",
             title="Синтетические показания",
             xrotation=45, legend=false)
    plot(p1, p2, layout=(1, 2), size=(900, 400))
end

# ╔═╡ c1000000-0005-4000-8000-000000000005
md"""
## 2. Развёртка методом GRAVEL

Создаём объект `Detector` и вызываем `unfold_gravel`.

GRAVEL — итеративный алгоритм, основанный на взвешенном лог-правдоподобии:

$$x_{k+1}[j] = x_k[j] \cdot \exp\left(\frac{\sum_i W_{ij} \ln(b_i / (Ax_k)_i)}{\sum_i W_{ij}}\right)$$

где $W_{ij} = b_i \cdot A_{ij} \cdot x_k[j] / (Ax_k)_i$.
"""

# ╔═╡ c1000000-0006-4000-8000-000000000006
begin
    detector = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)

    result = unfold_gravel(detector, readings,
                         max_iterations=500,
                         tolerance=1e-8)

    println("Метод:      $(result["method"])")
    println("Итераций:   $(result["iterations"])")
    println("Сходился:   $(result["converged"])")
    println("||b - Ax||: $(round(result["residual_norm"], digits=6))")
end

# ╔═╡ c1000000-0007-4000-8000-000000000007
begin
    # Сравнение восстановленного спектра с истиной
    cos_sim = dot(result["spectrum"], x_true) /
              (norm(result["spectrum"]) * norm(x_true) + 1e-30)
    rel_err = norm(result["spectrum"] .- x_true) / (norm(x_true) + 1e-30)

    println("Косинусная близость: $(round(cos_sim, digits=4))")
    println("Относительная ошибка: $(round(rel_err, digits=4))")

    plot(E_MeV, x_true, xscale=:log10, yscale=:log10,
         label="Истина", lw=3, color=:darkblue,
         xlabel="Энергия, МэВ", ylabel="Φ(E)",
         title="Сравнение спектров",
         legend=:topright, size=(700, 400))
    plot!(E_MeV, result["spectrum"], lw=2, color=:red,
          label="GRAVEL восстановленный")
end

# ╔═╡ c1000000-0008-4000-8000-000000000008
md"""
## 3. Дозовые характеристики

BSSUnfold автоматически вычисляет **дозовые коэффициенты** по ICRP-116:
эффективная доза и операционные величины (H*(10), H_p(10) и т.д.).
"""

# ╔═╡ c1000000-0009-4000-8000-000000000009
begin
    if haskey(result, "doserates")
        println("Дозовые коэффициенты (по каждой сфере, как демонстрация):")
        for (name, dr) in sort(collect(result["doserates"]), by=x->x[1])[1:5]
            println("  $name: $(round(dr, digits=6))")
        end
    end
end

# ╔═╡ c1000000-000a-4000-8000-00000000000a
md"""
## 4. Резюме

В этом ноутбуке мы:

- ✅ Создали синтетическую BSS-задачу (14 сфер × 100 бинов энергии)
- ✅ Запустили GRAVEL-развёртку через одну команду `unfold_gravel`
- ✅ Получили спектр, дозовые коэффициенты и метрики качества
- ✅ Визуально сравнили восстановленный спектр с истиной

### Что дальше?

- Ноутбук **03-uncertainty** — оценка неопределённости Monte-Carlo
- Ноутбук **05-mlem_example** — MLEM-алгоритм
- Ноутбук **33-methods_comparison** — сравнение всех 15 алгоритмов
"""

# ╔═╡ Cell order:
# ╟─005c4e30-3c8a-4a90-b3df-7e15e08a32e0
# ╟─c1000000-0001-4000-8000-000000000001
# ╟─c1000000-0002-4000-8000-000000000002
# ╟─c1000000-0003-4000-8000-000000000003
# ╟─c1000000-0004-4000-8000-000000000004
# ╟─c1000000-0005-4000-8000-000000000005
# ╟─c1000000-0006-4000-8000-000000000006
# ╟─c1000000-0007-4000-8000-000000000007
# ╟─c1000000-0008-4000-8000-000000000008
# ╟─c1000000-0009-4000-8000-000000000009
# ╟─c1000000-000a-4000-8000-00000000000a
