# Validation tests on the IAEA Compendium spectra (port of tests/test_iaea_validation.py).
#
# For each detector type (GSF, PTB, LANL) and each reference spectrum:
#   1. Interpolate the reference spectrum onto the detector energy grid
#   2. Compute synthetic detector readings (`get_effective_readings_for_spectra`)
#   3. Run each unfolding method via the `Detector` wrappers
#   4. Compare the unfolded spectrum with the reference using `compare_spectra`
#   5. Flag any method/spectrum pair exceeding the warning thresholds
#      (cosine_similarity < 0.85, r2_score < 0.7, mape > 100,
#       total_flux_ratio outside [0.5, 2.0]) — same limits as the Python suite.
#
# Unlike synthetic-problem tests elsewhere in the suite, these tests exercise
# the full high-level pipeline (Detector + run_unfolding + compare_spectra)
# on physically meaningful input: real BSS response functions (RF_GSF /
# RF_PTB / RF_LANL) and the 20 IAEA reference spectra from
# `MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv`.

using Test
using BSSUnfold
using LinearAlgebra
using Random
using Statistics
using Printf

const IAEA_DATA_DIR = joinpath(@__DIR__, "data")
const IAEA_CSV = joinpath(IAEA_DATA_DIR,
    "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv")

# ─── Reference data ──────────────────────────────────────────────────────────

"""
    load_iaea_reference() -> Union{Nothing,Tuple{Vector{Float64},Dict{String,Vector{Float64}}}}

Load the IAEA compendium dataset (column-oriented CSV: `E_MeV` column + 20
named spectra). Returns `(E_ref, spectra)` or `nothing` when the file is
missing.
"""
function load_iaea_reference()
    isfile(IAEA_CSV) || return nothing
    _, E_ref, spectra = load_spectra_csv(IAEA_CSV)
    (isempty(E_ref) || isempty(spectra)) && return nothing
    return E_ref, spectra
end

# ─── Warning thresholds (port of WARNING_THRESHOLDS from Python) ─────────────

const IAEA_WARNING_THRESHOLDS = [
    ("cosine_similarity", :lt, 0.85),
    ("r2_score",          :lt, 0.7),
    ("mape",              :gt, 100.0),
    ("total_flux_ratio",  :out_of_range, (0.5, 2.0)),
]

"Port of `_check_warnings`: human-readable threshold violations."
function check_iaea_warnings(metrics::Dict{String,Float64})
    msgs = String[]
    for (m, op, thr) in IAEA_WARNING_THRESHOLDS
        val = get(metrics, m, NaN)
        isnan(val) && continue
        if op === :lt && val < thr
            push!(msgs, @sprintf("%s=%.4f < %g", m, val, thr))
        elseif op === :gt && val > thr
            push!(msgs, @sprintf("%s=%.1f > %g", m, val, thr))
        elseif op === :out_of_range
            lo, hi = thr
            (val < lo || val > hi) &&
                push!(msgs, @sprintf("%s=%.4f not in [%g, %g]", m, val, lo, hi))
        end
    end
    return msgs
end

# ─── Python reference medians (parity targets) ─────────────────────────────
#
# Median cosine similarity per (detector, method), extracted from the
# committed Python-suite result tables (bssunfold/tests/iaea_validation_*.csv).
# These are the golden regression targets: the Julia port must reproduce them.
#
# `PARITY_METHODS` — methods asserted against the reference (|Δmed cos| ≤ 0.15).
# Methods whose Julia implementation is known to deviate (different stopping
# rules / smoothing details — see worklog) are kept in `PY_REF_COS` for the
# report but NOT asserted.

const PY_REF_COS = Dict{String,Dict{String,Float64}}(
    "GSF" => Dict(
        "mlem" => 0.211, "gravel" => 0.288, "landweber" => 0.922,
        "maxed" => 0.165, "doroshenko" => 0.166, "kaczmarz" => 0.919,
        "randomized_kaczmarz" => 0.921, "bayes" => 0.364, "sandii" => 0.156,
        "osem" => 0.816, "mapem" => 0.832, "bsrem" => 0.164, "sart" => 0.167,
        "cgls" => 0.943, "fista" => 0.075, "tsvd" => 0.922,
        "tikhonov_legendre" => 0.116, "bunki" => 0.908, "bunkiut" => 0.919,
        "staysl" => 0.145, "amaxed" => 0.834, "imaxed" => 0.157,
        "mlem_stop" => 0.211, "ferdor" => 0.370, "scipy_direct" => 0.943,
        "cvxpy" => 0.950, "qpsolvers" => 0.923, "nsduaz" => 0.876,
        "eki" => 0.655, "ensemble" => 0.349, "hybrid_parametric" => 0.526,
        "rfsp_jul" => 0.174,
    ),
    "PTB" => Dict(
        "mlem" => 0.197, "gravel" => 0.228, "landweber" => 0.872,
        "cgls" => 0.974, "bayes" => 0.400, "bunki" => 0.906, "tsvd" => 0.903,
        "nsduaz" => 0.839,
    ),
    "LANL" => Dict(
        "mlem" => 0.193, "gravel" => 0.798, "landweber" => 0.837,
        "cgls" => 0.856, "bayes" => 0.563, "bunki" => 0.826, "tsvd" => 0.848,
        "nsduaz" => 0.613,
    ),
)

const PARITY_METHODS = Dict{String,Set{String}}(
    "GSF" => Set(["mlem", "mlem_stop", "gravel", "landweber", "kaczmarz",
                  "randomized_kaczmarz", "tsvd", "bunkiut", "amaxed", "mapem",
                  "osem", "sart", "doroshenko", "ferdor", "bayes", "imaxed",
                  "maxed", "cvxpy", "qpsolvers", "scipy_direct",
                  "tikhonov_legendre", "eki", "ensemble", "rfsp_jul",
                  # reconciled deviant methods (exact Python ports now):
                  "bunki", "sandii", "bsrem", "cgls", "fista", "staysl",
                  "nsduaz", "hybrid_parametric"]),
    "PTB" => Set(["mlem", "gravel", "landweber", "tsvd", "bayes",
                  "bunki", "cgls", "nsduaz"]),
    "LANL" => Set(["mlem", "gravel", "landweber", "tsvd", "bayes",
                   "bunki", "cgls", "nsduaz"]),
)

const PARITY_TOL = 0.15

# ─── Method registry ─────────────────────────────────────────────────────────
#
# (name, closure(detector, readings) -> output-dict) pairs. Mirrors the
# METHODS list of the Python suite (Julia API kwargs).

iaea_all_methods() = [
    ("mlem",               (d, r) -> unfold_mlem(d, r; max_iterations=500)),
    ("gravel",             (d, r) -> unfold_gravel(d, r; max_iterations=500)),
    ("landweber",          (d, r) -> unfold_landweber(d, r; max_iterations=500)),
    ("maxed",              (d, r) -> unfold_maxed(d, r; max_iterations=500)),
    ("doroshenko",         (d, r) -> unfold_doroshenko(d, r; max_iterations=500)),
    ("kaczmarz",           (d, r) -> unfold_kaczmarz(d, r; max_iterations=500)),
    ("randomized_kaczmarz",(d, r) -> unfold_randomized_kaczmarz(d, r; max_iterations=500)),
    ("bayes",              (d, r) -> unfold_bayes(d, r; max_iterations=500)),
    ("sandii",             (d, r) -> unfold_sandii(d, r; max_iterations=50)),
    ("osem",               (d, r) -> unfold_osem(d, r; max_iterations=50, n_subsets=4)),
    ("mapem",              (d, r) -> unfold_mapem(d, r; max_iterations=50)),
    ("bsrem",              (d, r) -> unfold_bsrem(d, r; max_iterations=50)),
    ("sart",               (d, r) -> unfold_sart(d, r; max_iterations=50)),
    ("cgls",               (d, r) -> unfold_cgls(d, r; max_iterations=100)),
    ("fista",              (d, r) -> unfold_fista(d, r; max_iterations=500)),
    ("tsvd",               (d, r) -> unfold_tsvd(d, r)),
    ("tikhonov_legendre",  (d, r) -> unfold_tikhonov_legendre(d, r; delta=0.05)),
    ("bunki",              (d, r) -> unfold_bunki(d, r; max_iterations=200)),
    ("bunkiut",            (d, r) -> unfold_bunkiut(d, r; max_iterations=200)),
    ("staysl",             (d, r) -> unfold_staysl(d, r)),
    ("amaxed",             (d, r) -> unfold_amaxed(d, r; max_iterations=500)),
    ("imaxed",             (d, r) -> unfold_imaxed(d, r; max_iterations=500)),
    ("mlem_stop",          (d, r) -> unfold_mlem_stop(d, r; max_iterations=500)),
    ("ferdor",             (d, r) -> unfold_ferdor(d, r)),
    ("scipy_direct",       (d, r) -> unfold_scipy_direct(d, r; method="cg", max_iterations=500)),
    ("cvxpy",              (d, r) -> unfold_cvxpy(d, r; regularization=1e-3)),
    ("qpsolvers",          (d, r) -> unfold_qpsolvers(d, r; regularization=1e-3)),
    ("nsduaz",             (d, r) -> unfold_nsduaz(d, r)),
    ("eki",                (d, r) -> unfold_eki(d, r; n_ensemble=30, n_iterations=30)),
    ("ensemble",           (d, r) -> unfold_ensemble(d, r; combination="trimmed_mean")),
    ("hybrid_parametric",  (d, r) -> unfold_hybrid_parametric(d, r)),
    # dev-branch methods (v0.5.0)
    ("rfsp_jul",           (d, r) -> unfold_rfsp_jul(d, r)),
    ("ssr",                (d, r) -> unfold_ssr(d, r)),
    ("uno",                (d, r) -> unfold_uno(d, r)),
    ("mlem_bs",            (d, r) -> unfold_mlem_bs(d, r)),
    ("pspline_reml",       (d, r) -> unfold_pspline_reml(d, r)),
    ("amg",                (d, r) -> unfold_amg(d, r)),
]

"Fast subset used for the secondary detectors (keeps suite runtime bounded)."
iaea_core_methods() = [
    ("mlem",      (d, r) -> unfold_mlem(d, r; max_iterations=500)),
    ("gravel",    (d, r) -> unfold_gravel(d, r; max_iterations=500)),
    ("landweber", (d, r) -> unfold_landweber(d, r; max_iterations=500)),
    ("cgls",      (d, r) -> unfold_cgls(d, r; max_iterations=100)),
    ("bayes",     (d, r) -> unfold_bayes(d, r; max_iterations=500)),
    ("bunki",     (d, r) -> unfold_bunki(d, r; max_iterations=200)),
    ("tsvd",      (d, r) -> unfold_tsvd(d, r)),
    ("nsduaz",    (d, r) -> unfold_nsduaz(d, r)),
]

# ─── Sweep machinery ─────────────────────────────────────────────────────────

"""
    iaea_sweep(detector, methods, E_ref, spectra) -> Vector{Dict{String,Any}}

For every reference spectrum: discretize onto the detector grid, synthesize
readings, run every method and score against the discretized reference.
Returns one row per (spectrum, method) with `status`, the four key metrics
and the raw `error` string.
"""
function iaea_sweep(detector, methods, E_ref, spectra)
    E_det = energy_grid(detector)
    rows = Vector{Dict{String,Any}}()
    for spec_name in sort(collect(keys(spectra)))
        φ_src = spectra[spec_name]
        ref_dict = Dict{String,Vector{Float64}}("E_MeV" => E_ref, spec_name => φ_src)
        interp = discretize_spectra(ref_dict, E_det)[spec_name]
        readings = get_effective_readings_for_spectra(detector, ref_dict)
        for (mname, mfn) in methods
            row = Dict{String,Any}("spectrum" => spec_name, "method" => mname,
                                   "status" => "OK", "error" => "",
                                   "cosine_similarity" => NaN, "r2_score" => NaN,
                                   "mape" => NaN, "total_flux_ratio" => NaN)
            try
                res = mfn(detector, readings)
                spec = res["spectrum"]
                length(spec) == length(interp) ||
                    error("spectrum length $(length(spec)) != grid $(length(interp))")
                metrics = compare_spectra(interp, spec; energy=E_det)
                for k in ("cosine_similarity", "r2_score", "mape", "total_flux_ratio")
                    row[k] = get(metrics, k, NaN)
                end
                msgs = check_iaea_warnings(metrics)
                isempty(msgs) || (row["status"] = "WARN: " * join(msgs, "; "))
            catch err
                row["status"] = "ERROR: " * sprint(showerror, err)
                row["error"] = sprint(showerror, err)
            end
            push!(rows, row)
        end
    end
    return rows
end

"""
    iaea_report_and_assert(rows, label; pyref=Dict{String,Float64}(), parity=Set{String}())

Print a per-method summary and assert:
  * no method/spectrum pair ended in ERROR (the Python suite's hard gate),
  * for methods listed in `parity`: median cosine within `PARITY_TOL` of the
    Python reference median (golden regression targets from
    `bssunfold/tests/iaea_validation_*.csv`),
  * every median flux ratio stays within a sane envelope (no explosions).
"""
function iaea_report_and_assert(rows, label; pyref=Dict{String,Float64}(),
                                parity=Set{String}())
    n_total = length(rows)
    n_err  = count(r -> startswith(r["status"], "ERROR"), rows)
    n_warn = count(r -> startswith(r["status"], "WARN"), rows)
    n_ok   = count(r -> r["status"] == "OK", rows)

    println("\n", "="^78)
    println("IAEA Validation Report: $label")
    println("Total: $n_total | OK: $n_ok | WARN: $n_warn | ERR: $n_err")
    println("-"^78)
    @printf("%-22s %9s %9s %11s %6s %5s\n", "method", "med cos", "med r2",
            "med fluxrat", "warn", "err")
    for m in sort(unique(r["method"] for r in rows))
        mrows = filter(r -> r["method"] == m, rows)
        cosv  = [r["cosine_similarity"] for r in mrows if isfinite(r["cosine_similarity"])]
        r2v   = [r["r2_score"] for r in mrows if isfinite(r["r2_score"])]
        frv   = [r["total_flux_ratio"] for r in mrows if isfinite(r["total_flux_ratio"])]
        mcos  = isempty(cosv) ? NaN : median(cosv)
        mr2   = isempty(r2v) ? NaN : median(r2v)
        mfr   = isempty(frv) ? NaN : median(frv)
        merr  = count(r -> startswith(r["status"], "ERROR"), mrows)
        mwarn = count(r -> startswith(r["status"], "WARN"), mrows)
        @printf("%-22s %9.3f %9.3f %11.3f %4d/%d %4d\n", m, mcos, mr2, mfr,
                mwarn, length(mrows), merr)
        # Show a few distinct error messages to ease debugging
        if merr > 0
            errs = unique(String(first(r["error"], 90))
                          for r in mrows if startswith(r["status"], "ERROR"))
            println("      errors: ", join(collect(errs), " | "))
        end
        @test merr == 0
        # Python-parity gate for curated methods
        if m in parity && haskey(pyref, m)
            ref = pyref[m]
            @test isfinite(mcos) && abs(mcos - ref) ≤ PARITY_TOL
            if !isfinite(mcos) || abs(mcos - ref) > PARITY_TOL
                println("      PARITY FAIL: med cos=$mcos vs python=$ref")
            end
        elseif haskey(pyref, m) && isfinite(mcos)
            dev = abs(mcos - pyref[m])
            dev > PARITY_TOL &&
                println("      known deviation vs python: med cos=$mcos vs ",
                        "$(pyref[m]) (not asserted)")
        end
        # Sanity envelope for the integral fluence of the median unfolding
        # (asserted for parity methods only — known deviants may legitimately
        # blow up, mirroring the Python reference behaviour)
        if isfinite(mfr) && m in parity
            @test 0.05 ≤ mfr ≤ 5.0
        end
    end
    println("="^78, "\n")
    @test n_err == 0
    return (n_total=n_total, n_ok=n_ok, n_warn=n_warn, n_err=n_err)
end

# ─── 1. Dataset integrity ────────────────────────────────────────────────────

@testset "IAEA reference data integrity" begin
    data = load_iaea_reference()
    if data === nothing
        @test true  # skip: dataset not shipped
    else
        E_ref, spectra = data
        @test length(spectra) == 20
        @test length(E_ref) ≥ 60
        @test issorted(E_ref) && E_ref[1] > 0
        for (name, φ) in spectra
            @test length(φ) == length(E_ref)
            @test all(isfinite.(φ))
            @test all(φ .≥ 0)
            @test sum(φ) > 0
        end
        expected_names = ["ISO_ref_Cf252", "ISO_ref_Cf252_2", "ISO_ref_AmBe",
                          "ISO_ref_AmB"]
        for n in expected_names
            @test haskey(spectra, n)
        end
    end
end

# ─── 2. Full sweep on the default GSF detector ───────────────────────────────

@testset "IAEA validation — GSF detector (full sweep)" begin
    data = load_iaea_reference()
    if data === nothing
        @test true
    else
        E_ref, spectra = data
        d = Detector()  # default GSF response functions
        rows = iaea_sweep(d, iaea_all_methods(), E_ref, spectra)
        iaea_report_and_assert(rows, "GSF"; pyref=PY_REF_COS["GSF"],
                               parity=PARITY_METHODS["GSF"])
    end
end

# ─── 3. Secondary detectors: PTB and LANL response sets ──────────────────────

@testset "IAEA validation — PTB detector" begin
    data = load_iaea_reference()
    if data === nothing
        @test true
    else
        E_ref, spectra = data
        d = Detector(RF_PTB)
        @test length(d) > 0 && n_energy_bins(d) > 0
        rows = iaea_sweep(d, iaea_core_methods(), E_ref, spectra)
        iaea_report_and_assert(rows, "PTB"; pyref=PY_REF_COS["PTB"],
                               parity=PARITY_METHODS["PTB"])
    end
end

@testset "IAEA validation — LANL detector" begin
    data = load_iaea_reference()
    if data === nothing
        @test true
    else
        E_ref, spectra = data
        d = Detector(RF_LANL)
        @test length(d) > 0 && n_energy_bins(d) > 0
        rows = iaea_sweep(d, iaea_core_methods(), E_ref, spectra)
        iaea_report_and_assert(rows, "LANL"; pyref=PY_REF_COS["LANL"],
                               parity=PARITY_METHODS["LANL"])
    end
end

# ─── 4. Benchmark harness on the IAEA compendium ─────────────────────────────

@testset "IAEA benchmark harness (GSF)" begin
    data = load_iaea_reference()
    if data === nothing
        @test true
    else
        E_ref, spectra = data
        d = Detector()
        bench = benchmark_unfold_methods(d, Float64.(E_ref), spectra;
                                         progress=false)
        @test bench isa BenchmarkResult
        @test !isempty(bench.results)
        @test !isempty(bench.ranking)
        @test !isempty(bench.report)
        # Some (method, params) pairs legitimately fail on this ill-posed
        # problem (e.g. CGLS at aggressive iteration counts); the harness
        # itself must succeed for the overwhelming majority of runs.
        ok_rate = count(r -> r["success"], bench.results) / length(bench.results)
        @test ok_rate ≥ 0.85
        failed = unique(String(first(r["error"], 60))
                        for r in bench.results if !r["success"])
        isempty(failed) ||
            println("      benchmark failures ($(round(100(1 - ok_rate); digits=1))%): ",
                    join(collect(failed), " | "))
        # ranking must be sorted by mean r2 descending (default rank metric)
        r2means = [r["r2_score_mean"] for r in bench.ranking
                   if isfinite(get(r, "r2_score_mean", NaN))]
        @test issorted(r2means; rev=true)
        ok_row = first(filter(r -> r["success"], bench.results))
        @test haskey(ok_row, "cosine_similarity")
        @test haskey(ok_row, "spectrum") && haskey(ok_row, "method")
        # every registry method produced results
        methods_seen = unique(r["method"] for r in bench.results)
        @test length(methods_seen) ==
              length(keys(BSSUnfold.default_unfold_benchmark_methods()))
        println(bench.report)
    end
end

# ─── 5. Method agreement on a single spectrum (port of "multiple methods") ───

@testset "IAEA — multiple methods agreement (Cf252, GSF)" begin
    data = load_iaea_reference()
    if data === nothing
        @test true
    else
        E_ref, spectra = data
        d = Detector()
        E_det = energy_grid(d)
        ref_dict = Dict{String,Vector{Float64}}("E_MeV" => E_ref,
                                                "ISO_ref_Cf252" => spectra["ISO_ref_Cf252"])
        interp = discretize_spectra(ref_dict, E_det)["ISO_ref_Cf252"]
        readings = get_effective_readings_for_spectra(d, ref_dict)

        unfolded = Dict{String,Vector{Float64}}(
            "Landweber"  => unfold_landweber(d, readings; max_iterations=1000)["spectrum"],
            "Kaczmarz"   => unfold_kaczmarz(d, readings; max_iterations=1000)["spectrum"],
            "CGLS"       => unfold_cgls(d, readings; max_iterations=100)["spectrum"],
            "TSVD"       => unfold_tsvd(d, readings)["spectrum"],
            "BunkiUT"    => unfold_bunkiut(d, readings; max_iterations=200)["spectrum"],
            "cvxpy"      => unfold_cvxpy(d, readings; regularization=1e-3)["spectrum"],
        )
        # Every method must reproduce the reference reasonably well on
        # noise-free readings with a real BSS response matrix.
        for (mname, spec) in unfolded
            c = cosine_similarity(interp, spec)
            @test isfinite(c) && c > 0.7
        end
        # Pairwise agreement of independently unfolded spectra
        names = collect(keys(unfolded))
        n_pairs, n_good = 0, 0
        for i in 1:length(names), j in i+1:length(names)
            c = cosine_similarity(unfolded[names[i]], unfolded[names[j]])
            n_pairs += 1
            n_good += (c > 0.7)
        end
        @test n_good / n_pairs ≥ 0.8
        # compare_multiple must accept the unfolded set
        cmp = compare_multiple(vcat([interp], collect(values(unfolded)));
                               labels=vcat(["Reference"], names))
        @test length(cmp) == length(names)
        @test haskey(cmp, "Reference vs Landweber")
        @test isfinite(cmp["Reference vs Landweber"]["cosine_similarity"])
    end
end
