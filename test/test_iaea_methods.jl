# IAEA-spectrum tests for the special (batch-3) and dev-branch methods, plus
# physics-level consistency checks that do not exist in the Python suite:
#   * NSDUAZ catalogue selection against real ISO reference spectra,
#   * batch-3 methods (Genetic / MCMC / QUBO / NSpline) on Cf-252,
#   * dev-branch methods (v0.5.0) on several IAEA spectra,
#   * cross-detector consistency (GSF vs PTB vs LANL),
#   * response-matrix reading reconstruction and noise robustness,
#   * Detector utilities on real data (subdetector, dose-coefficient sets).

using Test
using BSSUnfold
using LinearAlgebra
using Random
using Statistics
using Printf

const EX_DATA_DIR = joinpath(@__DIR__, "data")
const EX_IAEA_CSV = joinpath(EX_DATA_DIR,
    "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv")

function _load_ex_reference()
    isfile(EX_IAEA_CSV) || return nothing
    _, E_ref, spectra = load_spectra_csv(EX_IAEA_CSV)
    (isempty(E_ref) || isempty(spectra)) && return nothing
    return E_ref, spectra
end

"Discretized reference + synthetic readings for one named spectrum on `d`."
function _ex_prepare(d, E_ref, spectra, spec_name)
    ref_dict = Dict{String,Vector{Float64}}("E_MeV" => E_ref,
                                            spec_name => spectra[spec_name])
    interp = discretize_spectra(ref_dict, energy_grid(d))[spec_name]
    readings = get_effective_readings_for_spectra(d, ref_dict)
    return interp, readings
end

"Response matrix A (detectors × bins) of a detector in canonical sphere order."
function _ex_system(d)
    A = Matrix(hcat([Float64.(d.config.sensitivities[n]) for n in detector_names(d)]...)')
    return A
end

# ─── 1. NSDUAZ catalogue selection (physics test) ────────────────────────────

@testset "IAEA — NSDUAZ catalogue selection" begin
    data = _load_ex_reference()
    if data === nothing
        @test true
    else
        E_ref, spectra = data
        d = Detector()
        # The χ² catalogue test must select the same catalogue entry as the
        # Python reference implementation (verified against the live Python
        # package: `d.unfold_nsduaz` picks "ambe" for both spectra — the
        # analytic Watt shape fits the soft ISO Cf-252 fluence worse than
        # the AmBe continuum on the GSF reading ratios).
        for (spec_name, expected) in (("ISO_ref_Cf252", "ambe"),
                                      ("ISO_ref_AmBe", "ambe"))
            _, readings = _ex_prepare(d, E_ref, spectra, spec_name)
            res = unfold_nsduaz(d, readings)
            @test haskey(res, "spectrum")
            @test haskey(res, "catalogue")
            @test res["catalogue"] == expected
            spec = res["spectrum"]
            @test all(isfinite.(spec)) && all(spec .≥ 0)
        end
    end
end

# ─── 2. Batch-3 methods on the Cf-252 reference ──────────────────────────────

@testset "IAEA — batch3 methods on Cf252 (GSF)" begin
    data = _load_ex_reference()
    if data === nothing
        @test true
    else
        E_ref, spectra = data
        d = Detector()
        interp, readings = _ex_prepare(d, E_ref, spectra, "ISO_ref_Cf252")

        @testset "Genetic (PSO, small budget)" begin
            res = unfold_genetic(d, readings; solver=:pso, epoch=30,
                                 pop_size=20, random_state=42)
            spec = res["spectrum"]
            @test all(isfinite.(spec)) && all(spec .≥ 0)
            c = cosine_similarity(interp, spec)
            @test c > 0.5
            println("      genetic  cos=", round(c; digits=3))
        end

        @testset "MCMC (Turing-aware)" begin
            # Without Turing.jl the documented graceful degradation returns a
            # zero spectrum; with Turing installed the posterior mean must
            # reconstruct the reference shape.
            turing_available = try
                Base.require(Main, :Turing); true
            catch
                false
            end
            res = unfold_mcmc(d, readings; n_samples=200, tune=50,
                              chains=1, random_state=42)
            spec = res["spectrum"]
            @test length(spec) == n_energy_bins(d)
            if turing_available
                @test all(isfinite.(spec)) && all(spec .≥ 0)
                c = cosine_similarity(interp, spec)
                @test c > 0.25
                println("      mcmc (Turing) cos=", round(c; digits=3))
            else
                @test all(==(0.0), spec)
                println("      mcmc: Turing absent — documented zero-spectrum fallback")
            end
        end

        @testset "QUBO (binary encoding)" begin
            res = unfold_qubo(d, readings; n_bits=6, random_state=42)
            spec = res["spectrum"]
            @test all(isfinite.(spec)) && all(spec .≥ 0)
            c = cosine_similarity(interp, spec)
            @test c > 0.2
            println("      qubo     cos=", round(c; digits=3))
        end

        @testset "NSpline" begin
            res = unfold_nspline(d, readings)
            spec = res["spectrum"]
            @test all(isfinite.(spec))
            c = cosine_similarity(interp, spec)
            @test c > 0.5
            println("      nspline  cos=", round(c; digits=3))
        end
    end
end

# ─── 3. Dev-branch methods (v0.5.0) on IAEA spectra ──────────────────────────

@testset "IAEA — dev methods on reference spectra (GSF)" begin
    data = _load_ex_reference()
    if data === nothing
        @test true
    else
        E_ref, spectra = data
        d = Detector()
        dev_methods = [
            ("rfsp_jul",     (dd, r) -> unfold_rfsp_jul(dd, r)),
            ("ssr",          (dd, r) -> unfold_ssr(dd, r)),
            ("uno",          (dd, r) -> unfold_uno(dd, r)),
            ("mlem_bs",      (dd, r) -> unfold_mlem_bs(dd, r)),
            ("pspline_reml", (dd, r) -> unfold_pspline_reml(dd, r)),
            ("amg",          (dd, r) -> unfold_amg(dd, r)),
        ]
        for spec_name in ("ISO_ref_Cf252", "ISO_ref_AmBe", "t4-16-s.txt_2")
            interp, readings = _ex_prepare(d, E_ref, spectra, spec_name)
            # Minimum cosine gates. rfsp_jul is intrinsically weak on the
            # compendium — its median cosine (≈0.17) matches the Python
            # reference exactly — so it gets a looser shape gate.
            min_cos = Dict("rfsp_jul" => 0.1,
                           "ssr" => 0.5, "uno" => 0.5, "mlem_bs" => 0.5,
                           "pspline_reml" => 0.5, "amg" => 0.5)
            @testset "$m on $spec_name" for (m, mfn) in dev_methods
                res = mfn(d, readings)
                spec = res["spectrum"]
                @test length(spec) == n_energy_bins(d)
                @test all(isfinite.(spec))
                @test all(spec .≥ 0)
                c = cosine_similarity(interp, spec)
                @test c > min_cos[m]
                fr = total_flux_ratio(interp, spec)
                if m == "rfsp_jul"
                    @test 0.0 ≤ fr ≤ 10.0
                else
                    @test 0.3 ≤ fr ≤ 3.0
                end
            end
        end
    end
end

# ─── 4. Cross-detector consistency (same source, different instruments) ──────

@testset "IAEA — cross-detector consistency" begin
    data = _load_ex_reference()
    if data === nothing
        @test true
    else
        E_ref, spectra = data
        d_gsf  = Detector()
        d_ptb  = Detector(RF_PTB)
        d_lanl = Detector(RF_LANL)
        for spec_name in ("ISO_ref_Cf252", "ISO_ref_AmBe")
            results = Dict{String,Tuple{Detector,Vector{Float64},Vector{Float64},Float64}}()
            for (label, dd) in (("GSF", d_gsf), ("PTB", d_ptb), ("LANL", d_lanl))
                interp, readings = _ex_prepare(dd, E_ref, spectra, spec_name)
                spec = unfold_landweber(dd, readings; max_iterations=500)["spectrum"]
                dose = ambient_dose_equivalent_rate(spec, energy_grid(dd),
                                                    dd.config.cc_icrp116)
                results[label] = (dd, spec, interp, dose)
                # integral fluence of the unfolding is consistent with the
                # discretized reference on the same grid
                fr = total_flux_ratio(interp, spec)
                @test 0.5 ≤ fr ≤ 2.0
                # shape recovered on each instrument
                @test cosine_similarity(interp, spec) > 0.8
            end
            # integral dose rate is (approximately) instrument-independent:
            # the same source must give the same H*(10) on all three sets
            dose_gsf = results["GSF"][4]
            for label in ("PTB", "LANL")
                @test 0.5 ≤ results[label][4] / dose_gsf ≤ 2.0
            end
            # spectral shapes agree between instruments (all three grids are
            # the 60-bin ICRP-116 log grid, so the comparison is direct)
            spec_gsf = results["GSF"][2]
            for label in ("PTB", "LANL")
                c = cosine_similarity(spec_gsf, results[label][2])
                @test c > 0.9
                println("      $spec_name GSF/$label cos=", round(c; digits=3),
                        "  dose ratio=", round(results[label][4] / dose_gsf; digits=3))
            end
        end
    end
end

# ─── 5. Reading reconstruction (response-matrix consistency) ─────────────────

@testset "IAEA — reading reconstruction" begin
    data = _load_ex_reference()
    if data === nothing
        @test true
    else
        E_ref, spectra = data
        d = Detector()
        A = _ex_system(d)
        for spec_name in ("ISO_ref_Cf252", "ISO_ref_AmBe")
            interp, readings = _ex_prepare(d, E_ref, spectra, spec_name)
            b = [readings[n] for n in detector_names(d)]
            norm_b = norm(b)
            norm_b > 0 || continue
            # MLEM fits the Poisson (KL) likelihood, not the L2 residual, so
            # its raw L2 residual stays bounded but not tiny; the L2-based
            # methods (GRAVEL / Landweber) must fit the readings to <5%.
            max_res = Dict("MLEM" => 0.5, "GRAVEL" => 0.05, "Landweber" => 0.05)
            for (mname, mfn) in (("MLEM",  (dd, r) -> unfold_mlem(dd, r; max_iterations=1000)),
                                 ("GRAVEL", (dd, r) -> unfold_gravel(dd, r; max_iterations=1000)),
                                 ("Landweber", (dd, r) -> unfold_landweber(dd, r; max_iterations=1000)))
                spec = mfn(d, readings)["spectrum"]
                rel_res = norm(A * spec - b) / norm_b
                @test rel_res < max_res[mname]
                println("      $spec_name $mname rel residual=", round(rel_res; digits=4))
            end
        end
    end
end

# ─── 6. Robustness to measurement noise ──────────────────────────────────────

@testset "IAEA — robustness to noisy readings" begin
    data = _load_ex_reference()
    if data === nothing
        @test true
    else
        E_ref, spectra = data
        d = Detector()
        A = _ex_system(d)
        interp, readings = _ex_prepare(d, E_ref, spectra, "ISO_ref_Cf252")
        b = [readings[n] for n in detector_names(d)]
        x0 = zeros(size(A, 2))  # Python-parity default initial (zeros)
        cosines = Float64[]
        for noise in (0.0, 0.02, 0.05, 0.10)
            rng = MersenneTwister(2024)
            b_noisy = noise == 0.0 ? Float64.(b) :
                      b .* (1.0 .+ noise .* randn(rng, length(b)))
            res = solve_landweber(A, b_noisy, x0; max_iterations=500)
            push!(cosines, cosine_similarity(interp, res.spectrum))
            @test all(isfinite.(res.spectrum))
            @test all(res.spectrum .≥ 0)
        end
        println("      noise sweep cos: ", round.(cosines; digits=3))
        # noise-free reconstruction is excellent
        @test cosines[1] > 0.95
        # accuracy degrades monotonically with noise level
        @test issorted(cosines; rev=true)
        # even 10% noise keeps the gross shape
        @test cosines[4] > 0.9
        # 2% noise stays close to noise-free quality
        @test cosines[2] > 0.95
    end
end

# ─── 7. Detector utilities on real data ──────────────────────────────────────

@testset "IAEA — detector utilities on real spectra" begin
    data = _load_ex_reference()
    if data === nothing
        @test true
    else
        E_ref, spectra = data
        d = Detector()
        interp, readings = _ex_prepare(d, E_ref, spectra, "ISO_ref_Cf252")

        # subdetector: energy-restricted detector still unfolds
        mask = energy_grid(d) .≤ 10.0
        dsub = subdetector(d, mask)
        @test n_energy_bins(dsub) == count(mask)
        @test length(dsub) == length(d)
        readings_sub = get_effective_readings_for_spectra(
            dsub, Dict{String,Vector{Float64}}("E_MeV" => energy_grid(d), "Phi" => interp))
        res_sub = unfold_mlem(dsub, readings_sub; max_iterations=200)
        @test length(res_sub["spectrum"]) == count(mask)
        @test all(res_sub["spectrum"] .≥ 0)

        # switching the dose-coefficient set changes the dose rate
        dose116 = ambient_dose_equivalent_rate(interp, energy_grid(d),
                                               d.config.cc_icrp116)
        set_dose_coefficients!(d, "ICRP74_effective")
        dose74 = ambient_dose_equivalent_rate(interp, energy_grid(d),
                                              d.config.cc_icrp116)
        @test d.config.cc_type == "ICRP74_effective"
        @test dose74 > 0 && dose116 > 0
        @test !isapprox(dose74, dose116; rtol=1e-6)
        set_dose_coefficients!(d, "ICRP116")  # restore
        @test d.config.cc_type == "ICRP116"

        # upper bounds helper restricts the active energy range
        ub = detector_upper_bounds(d, 10.0)
        @test all(ub[energy_grid(d) .≤ 10.0] .== Inf)
        @test all(ub[energy_grid(d) .> 10.0] .== 0.0)
        mask10 = detector_max_energy_mask(d, 10.0)
        @test mask10 == (energy_grid(d) .≤ 10.0)
    end
end
