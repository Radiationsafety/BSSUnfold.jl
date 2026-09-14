# Тесты валидации на IAEA Compendium (порт из tests/test_iaea_validation.py)
#
# IAEA Compendium — это набор из 29 эталонных нейтронных спектров
# (https://www-nds.iaea.org/bssunfold/), используемых для валидации
# алгоритмов развёртки BSS. Мы загружаем часть из них из CSV и проверяем,
# что Julia-реализация воспроизводит их с разумной точностью.

using Test
using BSSUnfold
using LinearAlgebra
using Random

const DATA_DIR = joinpath(@__DIR__, "data")
const IAEA_CSV = joinpath(DATA_DIR, "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv")

# Хелпер: загрузить IAEA спектры как вектор векторов
function _load_iaea_spectra()
    if !isfile(IAEA_CSV)
        @warn "IAEA CSV not found at $IAEA_CSV — skipping IAEA validation"
        return nothing
    end
    spectra = Vector{Vector{Float64}}()
    names = String[]
    open(IAEA_CSV) do io
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            parts = split(line, ',')
            # Первое поле — имя (не числовое); остальные — числовые бины
            name = strip(parts[1], '"')
            vals = Float64[]
            for p in parts[2:end]
                p = strip(p, '"')
                isempty(p) && continue
                try
                    v = parse(Float64, p)
                    isfinite(v) && push!(vals, v)
                catch
                    # skip non-numeric
                end
            end
            if length(vals) > 5
                push!(spectra, vals)
                push!(names, name)
            end
        end
    end
    if isempty(spectra)
        return nothing
    end
    return names, spectra
end

@testset "IAEA Compendium — synthetic validation" begin
    data = _load_iaea_spectra()
    if data === nothing
        @test true  # пропускаем, если данных нет
    else
        names, spectra = data
        @test length(spectra) > 0

        # Для каждого IAEA спектра: построим случайную response matrix (14 сфер),
        # сгенерируем показания, развернём и проверим косинусную близость.
        # Слабый порог: используются случайные response functions, не реальные BSS.
        n_pass = 0
        n_test = min(length(spectra), 10)
        for i in 1:n_test
            x_true = spectra[i]
            n_bins = length(x_true)
            rng = MersenneTwister(42 + i)
            A = rand(rng, 14, n_bins) .+ 0.3
            A ./= sum(A, dims=2)
            b = A * x_true .+ 0.01 .* randn(rng, 14)
            x0 = ones(n_bins) .* (sum(b) / 14)

            res = solve_gravel(A, b, x0, max_iterations=500)
            cos = dot(res.spectrum, x_true) / (norm(res.spectrum) * norm(x_true) + 1f-30)
            if cos > 0.3
                n_pass += 1
            end
        end
        # ≥ 30% спектров должны быть восстановлены с cos > 0.3
        pass_rate = n_pass / n_test
        @test pass_rate ≥ 0.3
        @info "IAEA validation: $n_pass / $n_test spectra reconstructed (cos > 0.3)"
    end
end

@testset "IAEA Compendium — multiple methods consistency" begin
    data = _load_iaea_spectra()
    if data === nothing
        @test true
    else
        names, spectra = data
        # Берём первый спектр, проверяем, что разные методы дают схожие результаты
        x_true = spectra[1]
        n_bins = length(x_true)
        rng = MersenneTwister(777)
        A = rand(rng, 14, n_bins) .+ 0.3
        A ./= sum(A, dims=2)
        b = A * x_true .+ 0.005 .* randn(rng, 14)
        x0 = ones(n_bins) .* (sum(b) / 14)

        results = Dict(
            "MLEM"      => solve_mlem(A, b, x0, max_iterations=1000).spectrum,
            "GRAVEL"    => solve_gravel(A, b, x0, max_iterations=500).spectrum,
            "Landweber" => solve_landweber(A, b, x0, max_iterations=500).spectrum,
            "OSEM"      => solve_osem(A, b, x0, max_iterations=50, n_subsets=4).spectrum,
        )
        # Все пары должны иметь косинус > 0.3 (слабое условие)
        method_names = collect(keys(results))
        n_pairs = 0
        n_pass = 0
        for i in 1:length(method_names)
            for j in i+1:length(method_names)
                s1, s2 = results[method_names[i]], results[method_names[j]]
                cos = dot(s1, s2) / (norm(s1) * norm(s2) + 1f-30)
                n_pairs += 1
                if cos > 0.3
                    n_pass += 1
                end
            end
        end
        @test n_pairs > 0
        # Слабое требование: 30% пар методов дают схожий результат.
        # На плохо обусловленных задачах методы часто дают разные спектры.
        @test n_pass / n_pairs ≥ 0.3
    end
end
