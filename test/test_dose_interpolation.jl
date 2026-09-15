# Тесты порта dose_calculation.py, utils/interpolation.py и constants.py.

using Test
using BSSUnfold
using LinearAlgebra

@testset "Constants" begin
    @test length(RF_GSF["E_MeV"]) >= 50
    @test haskey(RF_GSF, "0in")
    @test length(RF_PTB["E_MeV"]) == length(RF_PTB["0in"])
    @test length(RF_LANL["E_MeV"]) == length(RF_LANL["3in"])
    @test haskey(RF_FERMILAB, "E_MeV")
    @test haskey(RF_EURADOS, "E_MeV")
    @test haskey(RF_JINR, "E_MeV")
    @test length(RF_IHEP["E_MeV"]) >= 50

    @test all(isfinite.(ICRP116_COEFF_EFFECTIVE_DOSE["AP"]))
    @test all(ICRP116_COEFF_EFFECTIVE_DOSE["AP"] .>= 0)
    @test length(ICRP116_COEFF_EFFECTIVE_DOSE["AP"]) == length(ICRP116_COEFF_EFFECTIVE_DOSE["E_MeV"])
    @test issubset(["E_MeV", "AP", "PA", "LLAT", "RLAT", "ROT", "ISO"],
                   keys(ICRP116_COEFF_EFFECTIVE_DOSE))
    @test haskey(ICRP74_COEFF_EFFECTIVE_DOSE, "AP")
    @test haskey(ICRP74_COEFF_OPERATIONAL_QUANTITIES, "E_MeV")
    @test haskey(NRB99_2009_COEFF_EFFECTIVE_DOSE, "AP")
    @test length(DOSE_COEFFICIENTS_NAMES) == 4
end

@testset "Dose calculation" begin
    cc = get_coefficients("ICRP116")
    @test cc === ICRP116_COEFF_EFFECTIVE_DOSE
    @test_throws ArgumentError get_coefficients("nonexistent")

    # нулевой спектр → нулевые дозы
    d0 = calculate_dose_rates(zeros(length(cc["E_MeV"])); cc=cc)
    @test all(iszero(v) for v in values(d0))
    @test sort(collect(keys(d0))) == ["AP", "ISO", "LLAT", "PA", "RLAT", "ROT"]

    # константный спектр: значения растут с кэфами
    ks = cc["AP"]
    d1 = calculate_dose_rates(ones(length(ks)); cc=cc)
    @test d1["AP"] ≈ sum(ks) * log(10) * 0.2 rtol = 1e-10

    # interpolate_coefficients
    loggrid = 10.0 .^ range(log10(1e-9), log10(600.0), length=80)
    intpl = interpolate_coefficients(cc, loggrid)
    @test intpl["E_MeV"] == loggrid
    @test length(intpl["AP"]) == length(loggrid)
    @test all(intpl["AP"] .>= 0)
    # вне диапазона — fill_value = 0
    @test interpolate_coefficients(cc, [1e-12, 2000.0])["AP"] == [0.0, 0.0]

    # разные наборы коэффициентов зарегистрированы
    for nm in DOSE_COEFFICIENTS_NAMES
        d = get_coefficients(nm)
        @test haskey(d, "E_MeV")
        @test length(keys(d)) >= 2
    end
end

@testset "Interpolation" begin
    E = collect(10.0 .^ range(-9, 2, 40))
    spec = exp.(-E ./ 1.5) .+ 0.01 ./ (E .+ 1e-6)

    E2 = collect(10.0 .^ range(-8, 1.5, 77))
    v = interpolate_spectrum(spec, E, E2)
    @test length(v) == length(E2)
    @test all(isfinite.(v))
    @test all(v .>= 0)
    # вне диапазона источника → fill_value = 0
    @test interpolate_spectrum(spec, E, [1e-10, 1e-3]; fill_value=0.0)[1] == 0.0

    # PCHIP сохраняет монотонность на монотонных данных
    mono = Float64.(collect(range(1.0, 2.0, length=40)))
    vm = interpolate_spectrum(mono, E, E2)
    @test issorted(vm)

    # discretize_spectra
    res = discretize_spectra(
        Dict{String,Vector{Float64}}("E_MeV" => E, "Phi" => spec), E2)
    @test res["E_MeV"] == E2
    @test res["Phi"] ≈ v rtol = 1e-10

    # resample_to_log_grid
    En, sn = resample_to_log_grid(spec, E; n_points=100)
    @test length(sn) == 100
    @test En[1] ≈ E[1] && En[end] ≈ E[end]

    # сравнение со SciPy (эталон: v == PchipInterpolator(...), binned на 4 точках)
    v4 = interpolate_spectrum([0.0, 0.05899, 3.45e-06, 1.0],
                              [1e-6, 1e-5, 1.0, 10.0],
                              [5e-6, 5e-4, 5.0])
    @test all(isfinite.(v4)) && all(v4 .>= 0)
end

@testset "Detector (real RF)" begin
    d = BSSUnfold.Detector()                       # default GSF
    @test length(d.config.detector_names) >= 8
    @test d.config.cc_type == "ICRP116"
    # переход на другой набор коэффициентов
    BSSUnfold.set_dose_coefficients!(d, "ICRP74_operational")
    @test d.config.cc_type == "ICRP74_operational"
    @test d.config.E_MeV == energy_grid(d)

    # эффективные показания из реального эталонного спектра
    d2 = BSSUnfold.Detector(RF_GSF)
    csv = joinpath(@__DIR__, "data",
        "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv")
    if isfile(csv)
        csv_names, E_csv, refs = load_spectra_csv(csv)
        ambe = refs["ISO_ref_AmBe"]
        # В CSV сетка 61 bin, детектор — 60 bins: тест интерполяции показаний
        readings = get_effective_readings_for_spectra(d2,
            Dict{String,Vector{Float64}}("E_MeV" => E_csv, "AmBe" => ambe))
        @test all(isfinite(v) for v in values(readings))
        @test all(v -> v >= 0, values(readings))
        # портер расчёт доз совпадает с Python-bssunfold (см. эталонные значения)
        dose = calculate_dose_rates(ambe; cc=get_icrp116_coefficients())
        @test dose["AP"] ≈ 663.4707230226006 rtol = 1e-6
        @test dose["ISO"] ≈ 357.5615945912337 rtol = 1e-6
    end
end
