"""
Optional JuMP / Optim backend (bucket-C unlock).

Lazy-loaded JuMP layer mirroring the Turing pattern in
`src/algorithms/mcmc.jl`: the package does NOT declare JuMP as a dependency
(see `Project.toml`); users opt-in via a separate environment (`env/jump`).

Kernels exposed:
- `_bucket_c_qp(...)`  — shared QP builder for docplex/scip/commercial
- `_solve_qp_jump(P, q, lb, ub; engine, time_limit, x0)` — the low-level QP
- `_solve_boxqp_jump(P, q, lb, ub; engine, time_limit, x0)` — nnqp/qpmad path
- `_solve_lp_jump(c; A_rows, lhs, rhs, x_lb, x_ub, engine, sense)` — interval LP
- `_has_jump()` / `has_jump()` (exported) — availability probe

Engines registered in `JUMP_ENGINES`: `:highs` covers LP+QP+MIP;
`:clarabel`/`:osqp`/`:scs` are conic QP; `:glpk`/`:cbc` are LP-only. GLPK
and Cbc must not be reached with a quadratic objective (MOI does not
bridge it). Commercial names (`:gurobi`, `:mosek`, `:cplex`, `:copt`,
`:xpress`) mirror Python's `COMMERCIAL_SOLVER_ALIASES`; they resolve to
JuMP extensions when installed and warn-and-return-zeros otherwise.

`BSSUNFOLD_JL_BACKEND=0` in the environment disables the whole layer —
useful from `python_bridge/` when the host must not pay precompile cost.
"""

const _JUMP_LOADED    = Ref(false)
const _JUMP_AVAILABLE = Ref(false)
const _HI_GHS_CTOR    = Ref{Any}(nothing)

function _try_load_jump()
    _JUMP_LOADED[] && return _JUMP_AVAILABLE[]
    _JUMP_LOADED[] = true
    if get(ENV, "BSSUNFOLD_JL_BACKEND", "1") == "0"
        _JUMP_AVAILABLE[] = false
        return false
    end
    try
        Base.eval(Main, :(using JuMP))
    catch err
        _JUMP_AVAILABLE[] = false
        return false
    end
    _JUMP_AVAILABLE[] = true
    return true
end

"""
    has_jump() -> Bool

Whether JuMP is loadable and the optional backend is usable in this session.
"""
has_jump() = _try_load_jump()

const JUMP_ENGINES = (
    (:highs,    (:lp, :qp, :mip), :HiGHS),
    (:clarabel, (:lp, :qp),       :Clarabel),
    (:osqp,     (:lp, :qp),       :OSQP),
    (:cosmo,    (:lp, :qp),       :COSMO),
    (:scs,      (:lp, :qp),       :SCS),
    (:glpk,     (:lp, :mip),      :GLPK),
    (:cbc,      (:lp, :mip),      :Cbc),
    (:gurobi,   (:lp, :qp, :mip), :Gurobi),
    (:mosek,    (:lp, :qp, :mip), :Mosek),
    (:cplex,    (:lp, :qp, :mip), :Cplex),
    (:copt,     (:lp, :qp, :mip), :Copt),
    (:xpress,   (:lp, :qp, :mip), :Xpress),
)

const _ENGINE_CACHE = Dict{Symbol,Vector{Symbol}}()
function _available_engines(cap::Symbol)
    haskey(_ENGINE_CACHE, cap) && return _ENGINE_CACHE[cap]
    out = Symbol[]
    for (name, caps, pkg) in JUMP_ENGINES
        cap in caps || continue
        if isdefined(Main, pkg) || _can_import(pkg)
            push!(out, name)
        end
    end
    _ENGINE_CACHE[cap] = out
    return out
end

function _can_import(pkg::Symbol)
    try
        Base.require(Base, pkg)
        return true
    catch
        return false
    end
end

function _import_pkg(pkg::Symbol)
    if isdefined(Main, pkg)
        mod = getfield(Main, pkg)
        mod isa Module && return mod
    end
    try
        m = Base.require(Base, pkg)
        m isa Module || error("not a module")
        return m
    catch err
        error("package $(pkg) unavailable in current environment: $err")
    end
end

function _engine_ctor(name::Symbol)
    _try_load_jump() || error("JuMP not loaded")
    if name === :highs
        _HI_GHS_CTOR[] !== nothing && return _HI_GHS_CTOR[]
        HiGHS = _import_pkg(:HiGHS)
        ctor = if isdefined(HiGHS, :Optimizer)
            HiGHS.Optimizer
        elseif isdefined(HiGHS, :Optimiser)
            HiGHS.Optimiser
        else
            error("HiGHS has neither Optimizer nor Optimiser")
        end
        _HI_GHS_CTOR[] = ctor
        return ctor
    elseif name === :clarabel
        return _import_pkg(:Clarabel).Optimizer
    elseif name === :osqp
        return _import_pkg(:OSQP).Optimizer
    elseif name === :cosmo
        return _import_pkg(:COSMO).Optimizer
    elseif name === :scs
        return _import_pkg(:SCS).Optimizer
    elseif name === :glpk
        return _import_pkg(:GLPK).Optimizer
    elseif name === :cbc
        return _import_pkg(:Cbc).Optimizer
    elseif name === :gurobi
        return _import_pkg(:Gurobi).Optimizer
    elseif name === :mosek
        return _import_pkg(:Mosek).Optimizer
    elseif name === :cplex
        return _import_pkg(:Cplex).Optimizer
    elseif name === :copt
        return _import_pkg(:Copt).Optimizer
    elseif name === :xpress
        return _import_pkg(:Xpress).Optimizer
    else
        error("unknown engine: $name")
    end
end

_default_engine(cap::Symbol) = first(_available_engines(cap))

_jump_variable_info(lb::Real, ub::Real) = begin
    JuMP = Main.JuMP
    has_lb = lb > typemin(Float64)
    has_ub = ub < typemax(Float64)
    JuMP.VariableInfo(has_lb, has_lb ? Float64(lb) : NaN,
                      has_ub, has_ub ? Float64(ub) : NaN,
                      false, NaN, false, NaN, false, false)
end

function _jump_build_and_solve(build_fn, ctor; time_limit::Real=0.0)
    JuMP = Main.JuMP
    MOI = JuMP.MOI
    model = JuMP.Model(ctor)
    JuMP.set_silent(model)
    time_limit > 0 && JuMP.set_time_limit_sec(model, Float64(time_limit))
    x = build_fn(model)
    JuMP.optimize!(model)
    term = JuMP.termination_status(model)
    primal = JuMP.primal_status(model)
    ok = (term == MOI.OPTIMAL) &&
         (primal == MOI.FEASIBLE_POINT || primal == MOI.GRADIENT_POINT)
    sol = ok ? Float64[JuMP.value(x[i]) for i in eachindex(x)] : nothing
    objv = ok ? Float64(JuMP.objective_value(model)) : NaN
    iters = 0
    for attr in (MOI.SimplexIterations(), MOI.BarrierIterations())
        iters = max(iters, try
            Int(JuMP.get(model, attr))
        catch
            0
        end)
    end
    return (sol=sol, ok=ok, term=term, primal=primal, obj=objv, iters=iters)
end

# ─── QP kernel ───────────────────────────────────────────────────────────────
"""
    _solve_qp_jump(P, q, lb, ub; engine, time_limit, x0)

`min ½ x'Px + q'x  s.t. lb ≤ x ≤ ub`. Returns a NamedTuple
`(x, ok, term, primal, obj, iters, engine)`. Never throws; if the backend
is absent or the solve fails, `ok=false` and `x` is a zero vector.
"""
function _solve_qp_jump(P::AbstractMatrix{T}, q::AbstractVector{T},
                        lb::AbstractVector{T}, ub::AbstractVector{T};
                        engine::Symbol=:default,
                        time_limit::Real=0.0,
                        x0::Union{Nothing,AbstractVector}=nothing) where T<:AbstractFloat
    n = length(q)
    zero_result = (x=zeros(Float64, n), ok=false, term=:unavailable, primal=:unavailable,
                   obj=NaN, iters=0, engine=engine)
    _try_load_jump() || return zero_result
    eng = (engine === :default) ? _default_engine(:qp) : engine
    ctor = try
        _engine_ctor(eng)
    catch err
        if get(ENV, "BSSUNFOLD_DEBUG", "0") == "1"
            println(stderr, "solve_qp ctor error: "); showerror(stderr, err, catch_backtrace())
        end
        return (x=zeros(Float64, n), ok=false, term=:ctor_error, primal=:ctor_error,
                obj=NaN, iters=0, engine=eng)
    end
    JuMP = Main.JuMP
    MOI = JuMP.MOI
    Pmat = Matrix{Float64}(P); qv = Vector{Float64}(q)
    lbv = Vector{Float64}(lb); ubv = Vector{Float64}(ub)
    x0v = x0 === nothing ? nothing : Vector{Float64}(x0)
    function build(m)
        vars = [JuMP.add_variable(m,
                    JuMP.ScalarVariable(_jump_variable_info(lbv[i], ubv[i])),
                    "x$i") for i in 1:n]
        if x0v !== nothing
            for i in 1:n
                try
                    JuMP.set_start_value(vars[i], x0v[i])
                catch
                    break
                end
            end
        end
        obj = JuMP.AffExpr(0.0)
        for i in 1:n
            obj += qv[i] * vars[i]
        end
        for i in 1:n
            for j in i:n
                coef = (i == j) ? 0.5 * Pmat[i, i] : Pmat[i, j]
                coef != 0.0 && (obj += coef * vars[i] * vars[j])
            end
        end
        JuMP.set_objective(m, MOI.MIN_SENSE, obj)
        return vars
    end
    r = try
        Base.invokelatest(_jump_build_and_solve, build, ctor; time_limit=time_limit)
    catch err
        if get(ENV, "BSSUNFOLD_DEBUG", "0") == "1"
            println(stderr, "solve_qp build error: "); showerror(stderr, err, catch_backtrace())
        end
        return (x=zeros(Float64, n), ok=false, term=:exception, primal=:exception,
                obj=NaN, iters=0, engine=eng)
    end
    x = r.sol === nothing ? zeros(Float64, n) : r.sol
    return (x=x, ok=r.ok, term=r.term, primal=r.primal, obj=r.obj,
            iters=r.iters, engine=eng)
end

_solve_boxqp_jump(P::AbstractMatrix{T}, q::AbstractVector{T},
                  lb::AbstractVector{T}, ub::AbstractVector{T};
                  kwargs...) where T<:AbstractFloat =
    _solve_qp_jump(P, q, lb, ub; kwargs...).x

# ─── LP kernel ───────────────────────────────────────────────────────────────
"""
    _solve_lp_jump(c; A_rows, lhs, rhs, x_lb, x_ub, engine, time_limit, sense)

`min (or :max) c'x  s.t. lhs ≤ A_rows·x ≤ rhs, x_lb ≤ x ≤ x_ub`.
`A_rows` is a matrix; `lhs`/`rhs` may contain ±Inf for one-sided rows.
"""
function _solve_lp_jump(c::AbstractVector{T};
                        A_rows::AbstractMatrix=Matrix{T}(undef, 0, 0),
                        lhs::AbstractVector=T[],
                        rhs::AbstractVector=T[],
                        x_lb::Union{Nothing,AbstractVector}=nothing,
                        x_ub::Union{Nothing,AbstractVector}=nothing,
                        engine::Symbol=:default,
                        time_limit::Real=0.0,
                        sense::Symbol=:min) where T<:AbstractFloat
    n = length(c)
    zero_result = (x=zeros(Float64, n), ok=false, term=:unavailable, primal=:unavailable,
                   obj=NaN, iters=0, engine=engine)
    _try_load_jump() || return zero_result
    eng = (engine === :default) ? _default_engine(:lp) : engine
    ctor = try
        _engine_ctor(eng)
    catch err
        if get(ENV, "BSSUNFOLD_DEBUG", "0") == "1"
            println(stderr, "solve_lp ctor error: "); showerror(stderr, err, catch_backtrace())
        end
        return (x=zeros(Float64, n), ok=false, term=:ctor_error, primal=:ctor_error,
                obj=NaN, iters=0, engine=eng)
    end
    JuMP = Main.JuMP
    MOI = JuMP.MOI
    cv = Vector{Float64}(c)
    Am = Matrix{Float64}(A_rows)
    lo = Vector{Float64}(lhs); hi = Vector{Float64}(rhs)
    lbv = x_lb === nothing ? fill(-Inf, n) : Vector{Float64}(x_lb)
    ubv = x_ub === nothing ? fill( Inf, n) : Vector{Float64}(x_ub)
    function build(m)
        vars = [JuMP.add_variable(m,
                    JuMP.ScalarVariable(_jump_variable_info(lbv[i], ubv[i])),
                    "x$i") for i in 1:n]
        for k in 1:size(Am, 1)
            expr = JuMP.AffExpr(0.0)
            for j in 1:n
                v = Am[k, j]
                v != 0.0 && (expr += v * vars[j])
            end
            l = lo[k]; u = hi[k]
            set = if l == -Inf && u == Inf
                nothing
            elseif l == -Inf
                MOI.LessThan(u)
            elseif u == Inf
                MOI.GreaterThan(l)
            elseif l == u
                MOI.EqualTo(l)
            else
                MOI.Interval(l, u)
            end
            if set !== nothing
                con = JuMP.build_constraint(err -> error("c$k: $err"), expr, set)
                JuMP.add_constraint(m, con, "c$k")
            end
        end
        obj = JuMP.AffExpr(0.0)
        for j in 1:n
            cv[j] != 0.0 && (obj += cv[j] * vars[j])
        end
        JuMP.set_objective(m, sense === :max ? MOI.MAX_SENSE : MOI.MIN_SENSE, obj)
        return vars
    end
    r = try
        Base.invokelatest(_jump_build_and_solve, build, ctor; time_limit=time_limit)
    catch err
        if get(ENV, "BSSUNFOLD_DEBUG", "0") == "1"
            println(stderr, "solve_lp build error: "); showerror(stderr, err, catch_backtrace())
        end
        return (x=zeros(Float64, n), ok=false, term=:exception, primal=:exception,
                obj=NaN, iters=0, engine=eng)
    end
    x = r.sol === nothing ? zeros(Float64, n) : r.sol
    return (x=x, ok=r.ok, term=r.term, primal=r.primal, obj=r.obj,
            iters=r.iters, engine=eng)
end

# ─── shared QP builder for docplex/scip/commercial ───────────────────────────
"""
    _bucket_c_qp(A, b, x0; regularization, norm, timeout, smoothness_order,
                 smoothness_weight, nonneg, ub, engine) -> UnfoldResult

Canonical Tikhonov QP that docplex, scip and commercial all describe:
    min ½‖Ax−b‖² + penalty  s.t.  lb ≤ x ≤ ub
where `penalty` =
- L2, sm=0:              α‖x‖²
- L2, sm∈{1,2}:          α·w‖Lx‖²   (derivative *replaces* the identity)
- L1 (only under x≥0):   α·Σx       (exact; shifts `q` by α)
- L1 + smoothness:       α·Σx + α·w‖Lx‖²

Post-hoc `x[ub .== 0] .= 0`. Returns an UnfoldResult with `extra["engine"]`.
"""
function _bucket_c_qp(A::AbstractMatrix{T}, b::AbstractVector{T},
                      x0::Union{Nothing,AbstractVector{T}};
                      regularization::T=T(1e-4),
                      norm::Integer=2,
                      timeout::Real=10.0,
                      smoothness_order::Integer=0,
                      smoothness_weight::T=T(1.0),
                      nonneg::Bool=true,
                      ub::Union{Nothing,AbstractVector}=nothing,
                      engine::Symbol=:default) where T<:AbstractFloat
    (norm != 1 && norm != 2) && throw(ArgumentError("Unsupported norm type: $norm"))
    smoothness_order in (0, 1, 2) ||
        throw(ArgumentError("smoothness_order must be 0, 1, or 2; got $smoothness_order"))
    norm == 1 && !nonneg && throw(ArgumentError(
        "L1 penalty equals alpha * sum(x) only under the non-negativity constraint (pass nonneg=true or use norm=2)"))
    m, n = size(A)
    if !_try_load_jump()
        @warn "solve: JuMP backend unavailable; returning zero spectrum."
        return UnfoldResult(zeros(T, n), 0, false, T(0),
                            Dict{String,Any}("error" => "JuMP not available"))
    end
    P = Matrix{T}(A' * A)
    q = -(A' * b)
    if norm == 2
        if smoothness_order in (1, 2)
            L = create_derivative_matrix(T, n, smoothness_order)
            P = P .+ (regularization * smoothness_weight) .* (L' * L)
        else
            P = P .+ regularization .* Matrix{T}(I, n, n)
        end
    else
        q = q .+ regularization
        if smoothness_order in (1, 2)
            L = create_derivative_matrix(T, n, smoothness_order)
            P = P .+ (regularization * smoothness_weight) .* (L' * L)
        end
    end
    P = T(0.5) .* (P .+ P')
    lb = nonneg ? zeros(T, n) : fill(-typemax(T), n)
    ubv = ub === nothing ? fill(typemax(T), n) : Vector{T}(ub)
    r = _solve_qp_jump(P, q, lb, ubv; engine=engine, time_limit=Float64(timeout),
                       x0=x0 === nothing ? nothing : Vector{T}(x0))
    x = Vector{T}(r.x)
    if ub !== nothing
        x[ub .== T(0)] .= T(0)
    end
    nonneg && (x = max.(x, T(0)))
    residual = b .- A * x
    return UnfoldResult(x, r.iters, r.ok, sqrt(sum(abs2, residual)),
                        Dict{String,Any}(
                            "norm" => Int(norm),
                            "regularization" => regularization,
                            "smoothness_order" => Int(smoothness_order),
                            "smoothness_weight" => smoothness_weight,
                            "timeout" => Float64(timeout),
                            "nonneg" => nonneg,
                            "engine" => string(r.engine),
                        ))
end

# ─── shared Detector-level wrapper ───────────────────────────────────────────
"""
    _bucket_c_unfold(fn_sym, method_label, d::Detector, readings; kwargs...)

Common prelude for `unfold_docplex`, `unfold_scip`, `unfold_commercial`
(and their per-engine aliases): resolves α via `resolve_regularization_parameter`,
computes `ub` via `upper_bounds(cfg.E_MeV, max_neutron_energy)`, then calls
`run_unfolding` with a closure that injects the pre-resolved values.
"""
function _bucket_c_unfold(fn_sym::Symbol, method_label::AbstractString,
                          d::Detector, readings::Dict{String,T};
                          regularization::T=T(1e-4),
                          norm::Integer=2,
                          initial_spectrum::Union{Nothing,Vector{T}}=nothing,
                          regularization_method::AbstractString="manual",
                          noise_var::Union{Nothing,Real}=nothing,
                          max_neutron_energy::Union{Nothing,Real}=nothing,
                          kwargs...) where T<:AbstractFloat
    cfg = d.config
    A, b, _ = build_system(readings, cfg.detector_names, cfg.sensitivities)
    alpha = resolve_regularization_parameter(A, b, regularization_method,
                                             regularization, cfg.n_energy_bins;
                                             initial_spectrum=initial_spectrum,
                                             norm=norm,
                                             noise_var=noise_var === nothing ? nothing : Float64(noise_var),
                                             verbose=false)
    ub = upper_bounds(cfg.E_MeV, max_neutron_energy)
    ubT = ub === nothing ? nothing : Vector{T}(ub)
    fn = getfield(BSSUnfold, fn_sym)
    wrapped = (A_, b_, x0_; kws...) -> fn(A_, b_, x0_;
                                          regularization=T(alpha), norm=norm,
                                          ub=ubT, kws...)
    run_unfolding(wrapped, cfg.detector_names, cfg.n_energy_bins, cfg.E_MeV,
                  cfg.sensitivities, cfg.cc_icrp116, readings;
                  method_name=method_label,
                  initial_spectrum=initial_spectrum,
                  solve_kwargs=NamedTuple(kwargs))
end
