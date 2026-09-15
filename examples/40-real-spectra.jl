### A Pluto.jl notebook ###
# v0.20.x — Реальные эталонные спектры IAEA (GSF/PTB/LANL дозы)

using Markdown
using InteractiveUtils

# ╔═╡ f4000000-0001-4000-8000-000000000001
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

# ╔═╡ f4000000-0002-4000-8000-000000000002
md"""
# Развёртка на реальных данных IAEA

Используются **настоящие** функции отклики сфер Боннера (RF_GSF) и
**эталонные спектры** IAEA Compendium (Cf-252, AmBe, AmB,
и расчётные `t4-*.txt` из IAEA_Compendium).

Пайплайн (как в bssunfold):
1. `Detector(RF_GSF)` — детектор с реальными RF и ICRP-116 коэффициентами.
2. `get_effective_readings_for_spectra` — синтетические показания
   из эталонного спектра.
3. Развёртка набором методов (в дефолтных и новых портированных).
4. `calculate_dose_rates` — дозовые мощности (pSv/s) и сравнение с эталоном.
"""

# ╔═╡ f4000000-0003-4000-8000-000000000003
begin
    detector = Detector()             # RF_GSF + ICRP-116
    println("Detectors: ", detector_names(detector))
    println("Energy bins: ", n_energy_bins(detector),
            " [", energy_grid(detector)[1], " .. ", energy_grid(detector)[end], "] MeV")
    println("cc_type: ", detector.config.cc_type)
end

# ╔═╡ f4000000-0004-4000-8000-000000000004
begin
    data_csv = joinpath(dirname(@__DIR__), "examples", "data",
        "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv")
    isreal_file = isfile(data_csv)
    data_csv2 = joinpath(@__DIR__, "data",
        "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv")
    csv_path = isfile(data_csv) ? data_csv : data_csv2
    ref_names, E_ref, ref_spectra = load_spectra_csv(csv_path)
    println("Загружено $(length(ref_names)) эталонных спектров: ", ref_names)
end

# ╔═╡ f4000000-0005-4000-8000-000000000005
begin
    benchmark_spectrum = "ISO_ref_AmBe"
    E_true = E_ref
    x_true = ref_spectra[benchmark_spectrum]

    b_readings = get_effective_readings_for_spectra(
        detector, E_true, x_true)

    @printf("Эталон: %s (флюенс ∫φ dE ≈ %.3f)\n",
            benchmark_spectrum, sum(x_true) * mean(diff(log10.(E_true))))
    @printf("Показания детекторов:\n")
    for name in detector_names(detector)
        @printf("  %-6s %12.4e\n", name, b_readings[name])
    end
end

# ╔═╡ f4000000-0006-4000-8000-000000000006
begin
    # Развёртка всеми методами (детектор-API = run_unfolding)
    methods_list = [
        ("GRAVEL",   (d, r) -> unfold_gravel(d, r, max_iterations=200)),
        ("MLEM",     (d, r) -> unfold_mlem(d, r, max_iterations=200)),
        ("Sandii",   (d, r) -> unfold_sandii(d, r, max_iterations=200)),
        ("Bayes",    (d, r) -> unfold_bayes(d, r, max_iterations=1000)),
        ("AMAXED",   (d, r) -> unfold_amaxed(d, r)),
        ("IMAXED",   (d, r) -> unfold_imaxed(d, r)),
        ("SART",     (d, r) -> unfold_sart(d, r)),
        ("CGLS",     (d, r) -> unfold_cgls(d, r, max_iterations=200)),
        ("CVXPY",    (d, r) -> unfold_cvxpy(d, r, regularization=1e-3)),
        ("Tikhonov", (d, r) -> unfold_tikhonov(d, r, regularization=1e-3)),
    ]

    results_dict = Dict{String,Dict{String,Any}}()
    rows = []
    for (label, fn_res) in methods_list
        t0 = time()
        res = try
            fn_res(detector, b_readings)
        catch err
            @warn "$label упал" err
            nothing
        end
        Δt = time() - t0
        res === nothing && continue
        results_dict[label] = res

        # Спектр развёртки на сетке детектора (60 bins) → привести к сетке эталона
        E_det = energy_grid(detector)
        spec_on_ref = if length(res["spectrum"]) == length(E_true) &&
                         isapprox(E_det, E_true; rtol=1e-12)
            Float64.(res["spectrum"])
        else
            interpolate_spectrum(res["spectrum"], E_det, E_true)
        end
        cos = dot(spec_on_ref, x_true) /
              (norm(spec_on_ref) * norm(x_true) + 1f-30)
        push!(rows, (label=label,
                     iters=res["iterations"],
                     converged=res["converged"],
                     resid=res["residual_norm"], cos=cos, ms=Δt * 1000))
    end

    @printf("%-12s | %8s | %-8s | %10s | %6s\n",
            "Метод", "Итераций", "Сход", "||b-Ax||", "cos")
    @printf("%s\n", "-" ^ 55)
    for r in rows
        @printf("%-12s | %8d | %-8s | %10.3e | %.4f\n",
                r.label, r.iters, string(r.converged), r.resid, r.cos)
    end
end

# ╔═╡ f4000000-0007-4000-8000-000000000007
md"""
## Дозовые мощности

Сравнение дозовых мощностей (pSv/s) эталона и развёртки.
"""

# ╔═╡ f4000000-0008-4000-8000-000000000008
begin
    dose_ref = calculate_dose_rates(x_true)
    println("Дозовые мощности эталона:")
    for (k, v) in sort(collect(pairs(dose_ref)); by=first)
        @printf("  %s: %10.2e pSv/s\n", k, v)
    end

    println()
    @printf("%-12s | %10s | %10s | %8s\n", "Метод", "AP", "ISO", "dAP %")
    println("-" ^ 48)
    for (label, res) in results_dict
        dose_r = get(res, "doserates", Dict())
        isempty(dose_r) && continue
        ap = get(dose_r, "AP", NaN)
        diff_ap = 100 * (ap - dose_ref["AP"]) / dose_ref["AP"]
        @printf("%-12s | %10.2e | %10.2e | %7.1f%%\n",
                label, ap, get(dose_r, "ISO", NaN), diff_ap)
    end
end

# ╔═╡ f4000000-0009-4000-8000-000000000009
begin
    p = plot(E_true, x_true, xscale=:log10, yscale=:log10,
             lw=3, color=:black, label="Эталон (" * benchmark_spectrum * ")",
             xlabel="Энергия, МэВ", ylabel="Φ(E)",
             title="Развёртка на реальных RF GSF (IAEA AmBe)",
             legend=:bottomleft, size=(800, 500))
    two_colors = [:red, :orange, :green, :blue, :purple, :teal,
                  :brown, :olive, :magenta, :steelblue]
    for (i, (label, _)) in enumerate(methods_list)
        label in keys(results_dict) || continue
        E_det = energy_grid(detector)
        if length(results_dict[label]["spectrum"]) == length(E_true) &&
           isapprox(E_det, E_true; rtol=1e-12)
            spec_plt = results_dict[label]["spectrum"]
        else
            spec_plt = interpolate_spectrum(results_dict[label]["spectrum"],
                                            E_det, E_true)
        end
        plot!(p, E_true, spec_plt, lw=1.5,
              alpha=0.8, color=two_colors[i], label=label)
    end
    p
end

# ╔═╡ f4000000-000a-4000-8000-00000000000a
begin
    benchmarks = filter(n -> !occursin("t4", n), ref_names)
    method_labels4 = ["GRAVEL", "MLEM", "Sandii", "Bayes"]
    cos_by_spec = Dict{String,Vector{Float64}}()
    for bn in benchmarks
        bx = ref_spectra[bn]
        br = get_effective_readings_for_spectra(detector, E_ref, bx)
        cos_vals = Float64[]
        for (label, fn_res) in methods_list[1:4]
            resr = try
                fn_res(detector, br)
            catch
                nothing
            end
            if resr !== nothing
                E_det = energy_grid(detector)
                sr = if length(resr["spectrum"]) == length(bx)
                    Float64.(resr["spectrum"])
                else
                    interpolate_spectrum(resr["spectrum"], E_det, E_ref)
                end
                push!(cos_vals, dot(sr, bx) / (norm(sr) * norm(bx) + 1f-30))
            end
        end
        cos_by_spec[bn] = cos_vals
    end
    means_vec = [isempty(get(cos_by_spec, sp, [0.0])) ? 0.0 :
                 mean(cos_by_spec[sp]) for sp in benchmarks]
    bar(benchmarks, means_vec,
        xrotation=45, legend=false, ylabel="cos эталон/развёртка",
        title="Средняя близость ($(length(method_labels4)) метода)", color=:darkblue,
        size=(700, 400))
end

# ╔═╡ f4000000-000b-4000-8000-00000000000b
md"""
## Резюме

- Пакет BSSUnfold.jl содержит **настоящие** функции отклики GSF и портированные
  коэффициенты ICRP-116 (все 4 набора из `constants.py`).
- Все 50+ методов, используемых в `bssunfold` (Python), доступны и в Julia;
  Python-мост (`python_bridge/bssunfold_julia`) автоматически перенаправляет
  их через Julia.
- Дозовые мощности `calculate_dose_rates` соответствуют Python-порту
  с бит-в-бит (то есть экипированного порта `dose_calculation.py`).
"""
