"""
FRUIT-based parametric unfolding (Bedogni et al., NIM A 580, 1301-1309,
2007; Pyshkina et al., 2021).

The spectrum is a weighted superposition of three components:

    Thermal    (E < 1e-7 MeV):  (E/T0^2) * exp(-E/T0)
    Epithermal (1e-7 < E < 0.1):[1 - exp(-(E/Ed)^2)] * E^(b-1) * exp(-E/beta')
    Fast       (E > 0.1 MeV):   E^alpha * exp(-E/beta)

with the constraint `P_th + P_epi + P_f = 1` (`P_f = 1 - P_th - P_epi`).
FRUIT constants: `_T0 = 2.53e-8`, `_Ed = 7.07e-8` MeV.

Optimizers (lmfit / qpsolvers are replaced by in-house implementations):
`solve_parametric` — multi-start Levenberg-Marquardt with parameter
bounds; `solve_parametric_cvxpy` and `solve_parametric_qpsolvers` —
SQP iterations: linearization by a numerical Jacobian and a projectional
Tikhonov substep solved by regularized normal equations
(regularized Newton) with clamping to the bounds; `solve_parametric_combined`
— first a leastsq fit, then a QP refinement of the spectrum via NNLS
on the augmented matrix (`solve_nnls` of the BSSUnfold package).
"""
const PARAMETRIC_T0 = 2.53e-8
const PARAMETRIC_ED = 7.07e-8
const PARAMETRIC_THERMAL_MAX = 1e-7
const PARAMETRIC_FAST_MIN = 0.1

const PARAM_NAMES = ["b", "beta_prime", "alpha", "beta", "P_th", "P_epi"]
const PARAM_DEFAULTS = [(1.0, 0.5, 2.0), (0.01, 1e-4, 1.0), (0.5, 0.0, 5.0),
                        (2.0, 0.1, 20.0), (1.0, 0.0, 1.0), (1.0, 0.0, 1.0)]

"""
    compute_log_steps(E_MeV) -> Vector{Float64}

Logarithmic steps d log10 E over the energy grid: edge bins use a
one-sided difference, interior bins a central difference.  For d ln E
multiply by `log(10)` (convention of the python package).
"""
function compute_log_steps(E::AbstractVector{<:Real})
    E_f = Float64.(collect(E))
    n = length(E_f)
    log_steps = zeros(n)
    log_e = log10.(E_f .+ 1e-15)
    if n > 1
        log_steps[1] = log_e[2] - log_e[1]
        log_steps[end] = log_e[end] - log_e[end-1]
    else
        log_steps[1] = 1.0
    end
    if n > 2
        for i in 2:(n-1)
            log_steps[i] = (log_e[i+1] - log_e[i-1]) / 2.0
        end
    end
    return log_steps
end

function _param_th(E_f::Vector{Float64})
    out = zeros(length(E_f))
    m = findall(<(PARAMETRIC_THERMAL_MAX), E_f)
    for j in m
        out[j] = (E_f[j] / PARAMETRIC_T0^2) * exp(-E_f[j] / PARAMETRIC_T0)
    end
    return out
end

function _param_epi(E_f::Vector{Float64}, b::Float64, beta_prime::Float64)
    out = zeros(length(E_f))
    m = findall(x -> x >= PARAMETRIC_THERMAL_MAX && x < PARAMETRIC_FAST_MIN, E_f)
    for j in m
        out[j] = (1.0 - exp(-((E_f[j] / PARAMETRIC_ED)^2))) *
                 E_f[j]^(b - 1.0) * exp(-E_f[j] / beta_prime)
    end
    return out
end

function _param_fast(E_f::Vector{Float64}, alpha::Float64, beta::Float64)
    out = zeros(length(E_f))
    m = findall(>=(PARAMETRIC_FAST_MIN), E_f)
    for j in m
        out[j] = E_f[j]^alpha * exp(-E_f[j] / beta)
    end
    return out
end

"""
    parametric_model(E, b, beta_prime, alpha, beta, P_th, P_epi) -> Vector{Float64}

Three-component parametric FRUIT model of the neutron spectrum
(fl per energy bin).  `P_f = max(0, 1 - P_th - P_epi)`.
"""
function parametric_model(E::AbstractVector{<:Real}, b::Real, beta_prime::Real,
                          alpha::Real, beta::Real, P_th::Real, P_epi::Real)
    E_f = Float64.(collect(E))
    P_f = max(0.0, 1.0 - Float64(P_th) - Float64(P_epi))
    return Float64(P_th) .* _param_th(E_f) .+
           Float64(P_epi) .* _param_epi(E_f, Float64(b), Float64(beta_prime)) .+
           P_f .* _param_fast(E_f, Float64(alpha), Float64(beta))
end

function _param_default_vec()
    return Float64[d[1] for d in PARAM_DEFAULTS]
end

function _param_lo_vec()
    return Float64[d[2] for d in PARAM_DEFAULTS]
end

function _param_hi_vec()
    return Float64[d[3] for d in PARAM_DEFAULTS]
end

function _param_clamp!(p::Vector{Float64})
    lo = _param_lo_vec()
    hi = _param_hi_vec()
    for i in eachindex(p)
        p[i] = clamp(p[i], lo[i], hi[i])
    end
    return p
end

function _fruit_internal_from_external(p::Vector{Float64})
    lo = _param_lo_vec()
    hi = _param_hi_vec()
    u = similar(p)
    @inbounds for i in eachindex(p)
        t = 2.0 * (p[i] - lo[i]) / (hi[i] - lo[i]) - 1.0
        u[i] = asin(clamp(t, -1.0, 1.0))
    end
    return u
end

function _fruit_external_from_internal(u::Vector{Float64})
    lo = _param_lo_vec()
    hi = _param_hi_vec()
    p = similar(u)
    @inbounds for i in eachindex(u)
        p[i] = lo[i] + (sin(u[i]) + 1.0) * (hi[i] - lo[i]) / 2.0
    end
    return p
end

function _fruit_residual(u::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64},
                         E_f::Vector{Float64}, ls::Vector{Float64},
                         reg_alpha::Float64, p_ref::Union{Nothing,Vector{Float64}})
    p = _fruit_external_from_internal(u)
    r = A * (parametric_model(E_f, p[1], p[2], p[3], p[4], p[5], p[6]) .* ls) .- b
    if reg_alpha > 0 && p_ref !== nothing
        r = vcat(r, sqrt(reg_alpha) .* (p .- p_ref))
    end
    return r, p
end

"""
    _param_lm_fit(p_start, A, b, E_f, ls; reg_alpha=0.0, ...)

Levenberg-Marquardt in the lmfit/Minuit internal (bounds-transformed) space
`p_i = lo_i + (sin(u_i)+1)*(hi_i-lo_i)/2`.  Faithful port of the MINPACK
`lmdif`/`lmpar`/`qrfac`/`qrsolv`/`fdjac2` chain used by
`lmfit method="leastsq"` (ftol=xtol=1.5e-8, gtol=0, epsfcn=1e-10, factor=100,
maxfev=28000, plus the lmfit-side abort at 14000 residual evaluations).
"""
struct _FruitAbortError <: Exception end

function _fruit_qrfac!(a::Matrix{Float64}, ipvt::Vector{Int},
                       rdiag::Vector{Float64}, acnorm::Vector{Float64},
                       wa_c::Vector{Float64})
    m, n = size(a)
    epsmch = eps(Float64)
    @inbounds for j in 1:n
        acnorm[j] = norm(a[:, j])
        rdiag[j] = acnorm[j]
        wa_c[j] = rdiag[j]
        ipvt[j] = j
    end
    minmn = min(m, n)
    @inbounds for j in 1:minmn
        kmax = j
        for k in j:n
            if rdiag[k] > rdiag[kmax]
                kmax = k
            end
        end
        if kmax != j
            for i in 1:m
                temp = a[i, j]
                a[i, j] = a[i, kmax]
                a[i, kmax] = temp
            end
            rdiag[kmax] = rdiag[j]
            wa_c[kmax] = wa_c[j]
            k = ipvt[j]
            ipvt[j] = ipvt[kmax]
            ipvt[kmax] = k
        end
        ajnorm = norm(a[j:m, j])
        if ajnorm != 0.0
            if a[j, j] < 0.0
                ajnorm = -ajnorm
            end
            for i in j:m
                a[i, j] /= ajnorm
            end
            a[j, j] += 1.0
            if n > j
                for k in (j+1):n
                    sum = 0.0
                    for i in j:m
                        sum += a[i, j] * a[i, k]
                    end
                    temp = sum / a[j, j]
                    for i in j:m
                        a[i, k] -= temp * a[i, j]
                    end
                    if rdiag[k] != 0.0
                        temp = a[j, k] / rdiag[k]
                        rdiag[k] *= sqrt(max(0.0, 1.0 - temp * temp))
                        temp = rdiag[k] / wa_c[k]
                        if 0.05 * (temp * temp) <= epsmch
                            rdiag[k] = norm(a[(j+1):m, k])
                            wa_c[k] = rdiag[k]
                        end
                    end
                end
            end
        end
        rdiag[j] = -ajnorm
    end
    return nothing
end

function _fruit_qrsolv!(r::Matrix{Float64}, n::Int, ipvt::Vector{Int},
                        diagd::Vector{Float64}, qtb::Vector{Float64},
                        x::Vector{Float64}, sdiag::Vector{Float64},
                        wa::Vector{Float64})
    @inbounds for j in 1:n
        for i in j:n
            r[i, j] = r[j, i]
        end
        x[j] = r[j, j]
        wa[j] = qtb[j]
    end
    @inbounds for j in 1:n
        l = ipvt[j]
        if diagd[l] != 0.0
            for k in j:n
                sdiag[k] = 0.0
            end
            sdiag[j] = diagd[l]
            qtbpj = 0.0
            for k in j:n
                if sdiag[k] != 0.0
                    if abs(r[k, k]) < abs(sdiag[k])
                        cotan = r[k, k] / sdiag[k]
                        sv = 0.5 / sqrt(0.25 + 0.25 * (cotan * cotan))
                        cv = sv * cotan
                    else
                        tnv = sdiag[k] / r[k, k]
                        cv = 0.5 / sqrt(0.25 + 0.25 * (tnv * tnv))
                        sv = cv * tnv
                    end
                    temp = cv * wa[k] + sv * qtbpj
                    qtbpj = -sv * wa[k] + cv * qtbpj
                    wa[k] = temp
                    r[k, k] = cv * r[k, k] + sv * sdiag[k]
                    if n > k
                        for i in (k+1):n
                            temp = cv * r[i, k] + sv * sdiag[i]
                            sdiag[i] = -sv * r[i, k] + cv * sdiag[i]
                            r[i, k] = temp
                        end
                    end
                end
            end
        end
        sdiag[j] = r[j, j]
        r[j, j] = x[j]
    end
    nsing = n
    @inbounds for j in 1:n
        if sdiag[j] == 0.0 && nsing == n
            nsing = j - 1
        end
        if nsing < n
            wa[j] = 0.0
        end
    end
    if nsing >= 1
        @inbounds for k in 1:nsing
            j = nsing - k + 1
            sum = 0.0
            if nsing > j
                for i in (j+1):nsing
                    sum += r[i, j] * wa[i]
                end
            end
            wa[j] = (wa[j] - sum) / sdiag[j]
        end
    end
    @inbounds for j in 1:n
        l = ipvt[j]
        x[l] = wa[j]
    end
    return nothing
end

function _fruit_lmpar(r::Matrix{Float64}, n::Int, ipvt::Vector{Int},
                      diag::Vector{Float64}, qtb::Vector{Float64}, delta::Float64,
                      par::Float64, x::Vector{Float64}, sdiag::Vector{Float64},
                      wa1::Vector{Float64}, wa2::Vector{Float64})
    dwarf = 2.2250738585072014e-308
    nsing = n
    @inbounds for j in 1:n
        wa1[j] = qtb[j]
        if r[j, j] == 0.0 && nsing == n
            nsing = j - 1
        end
        if nsing < n
            wa1[j] = 0.0
        end
    end
    if nsing >= 1
        @inbounds for k in 1:nsing
            j = nsing - k + 1
            wa1[j] /= r[j, j]
            temp = wa1[j]
            for i in 1:(j-1)
                wa1[i] -= r[i, j] * temp
            end
        end
    end
    @inbounds for j in 1:n
        l = ipvt[j]
        x[l] = wa1[j]
    end
    iter = 0
    @inbounds for j in 1:n
        wa2[j] = diag[j] * x[j]
    end
    dxnorm = norm(view(wa2, 1:n))
    fp = dxnorm - delta
    if fp <= 0.1 * delta
        # Gauss-Newton step accepted; par stays 0
    else
        parl = 0.0
        if nsing >= n
            @inbounds for j in 1:n
                l = ipvt[j]
                wa1[j] = diag[l] * (wa2[l] / dxnorm)
            end
            @inbounds for j in 1:n
                sum = 0.0
                for i in 1:(j-1)
                    sum += r[i, j] * wa1[i]
                end
                wa1[j] = (wa1[j] - sum) / r[j, j]
            end
            temp = norm(wa1)
            parl = fp / delta / temp / temp
        end
        @inbounds for j in 1:n
            sum = 0.0
            for i in 1:j
                sum += r[i, j] * qtb[i]
            end
            l = ipvt[j]
            wa1[j] = sum / diag[l]
        end
        gnorm = norm(wa1)
        paru = gnorm / delta
        if paru == 0.0
            paru = dwarf / min(delta, 0.1)
        end
        par = max(par, parl)
        par = min(par, paru)
        if par == 0.0
            par = gnorm / dxnorm
        end
        while true
            iter += 1
            if par == 0.0
                par = max(dwarf, 0.001 * paru)
            end
            temp = sqrt(par)
            @inbounds for j in 1:n
                wa1[j] = temp * diag[j]
            end
            _fruit_qrsolv!(r, n, ipvt, wa1, qtb, x, sdiag, wa2)
            @inbounds for j in 1:n
                wa2[j] = diag[j] * x[j]
            end
            dxnorm = norm(view(wa2, 1:n))
            temp = fp
            fp = dxnorm - delta
            if abs(fp) <= 0.1 * delta ||
               (parl == 0.0 && fp <= temp && temp < 0.0) || iter == 10
                break
            end
            @inbounds for j in 1:n
                l = ipvt[j]
                wa1[j] = diag[l] * (wa2[l] / dxnorm)
            end
            @inbounds for j in 1:n
                wa1[j] /= sdiag[j]
                temp = wa1[j]
                if n > j
                    for i in (j+1):n
                        wa1[i] -= r[i, j] * temp
                    end
                end
            end
            temp = norm(wa1)
            parc = fp / delta / temp / temp
            if fp > 0.0
                parl = max(parl, par)
            end
            if fp < 0.0
                paru = min(paru, par)
            end
            par = max(parl, par + parc)
        end
    end
    if iter == 0
        par = 0.0
    end
    return x, par
end

function _fruit_lmdif(resid::Function, u0::Vector{Float64};
                      ftol::Float64=1.5e-8, xtol::Float64=1.5e-8,
                      gtol::Float64=0.0, maxfev::Integer=28000,
                      epsfcn::Float64=1e-10, factor::Float64=100.0)
    x = copy(u0)
    n = length(x)
    fvec = resid(x)
    m = length(fvec)
    nfev = 1
    epsmch = eps(Float64)
    info = 0
    fnorm = norm(fvec)
    iter = 1
    xnorm = 0.0
    delta = 0.0
    par = 0.0
    diag = ones(Float64, n)
    fjac = Matrix{Float64}(undef, m, n)
    ipvt = Vector{Int}(undef, n)
    qtf = Vector{Float64}(undef, n)
    wa1 = Vector{Float64}(undef, n)
    wa2 = Vector{Float64}(undef, n)
    wa3 = Vector{Float64}(undef, n)
    wa4 = Vector{Float64}(undef, m)

    done = false
    while !done
        eps_fd = sqrt(max(epsfcn, epsmch))
        @inbounds for j in 1:n
            temp = x[j]
            h = eps_fd * abs(temp)
            if h == 0.0
                h = eps_fd
            end
            x[j] = temp + h
            wa4 .= resid(x)
            x[j] = temp
            for i in 1:m
                fjac[i, j] = (wa4[i] - fvec[i]) / h
            end
        end
        nfev += n

        _fruit_qrfac!(fjac, ipvt, wa1, wa2, wa3)

        if iter == 1
            @inbounds for j in 1:n
                diag[j] = wa2[j]
                if wa2[j] == 0.0
                    diag[j] = 1.0
                end
            end
            @inbounds for j in 1:n
                wa3[j] = diag[j] * x[j]
            end
            xnorm = norm(wa3)
            delta = factor * xnorm
            if delta == 0.0
                delta = factor
            end
        end

        wa4 .= fvec
        @inbounds for j in 1:n
            if fjac[j, j] != 0.0
                sum = 0.0
                for i in j:m
                    sum += fjac[i, j] * wa4[i]
                end
                temp = -sum / fjac[j, j]
                for i in j:m
                    wa4[i] += fjac[i, j] * temp
                end
            end
            fjac[j, j] = wa1[j]
            qtf[j] = wa4[j]
        end

        gnorm = 0.0
        if fnorm != 0.0
            @inbounds for j in 1:n
                l = ipvt[j]
                if wa2[l] != 0.0
                    sum = 0.0
                    for i in 1:j
                        sum += fjac[i, j] * (qtf[i] / fnorm)
                    end
                    gnorm = max(gnorm, abs(sum / wa2[l]))
                end
            end
        end

        if gnorm <= gtol
            info = 4
        end
        if info != 0
            break
        end

        @inbounds for j in 1:n
            diag[j] = max(diag[j], wa2[j])
        end

        inner = true
        while inner
            _, par = _fruit_lmpar(fjac, n, ipvt, diag, qtf, delta, par,
                                  wa1, wa2, wa3, wa4)
            @inbounds for j in 1:n
                wa1[j] = -wa1[j]
                wa2[j] = x[j] + wa1[j]
                wa3[j] = diag[j] * wa1[j]
            end
            pnorm = norm(wa3)
            if iter == 1
                delta = min(delta, pnorm)
            end
            wa4 .= resid(wa2)
            nfev += 1
            fnorm1 = norm(wa4)
            actred = -1.0
            if 0.1 * fnorm1 < fnorm
                t1 = fnorm1 / fnorm
                actred = 1.0 - t1 * t1
            end
            @inbounds for j in 1:n
                wa3[j] = 0.0
                l = ipvt[j]
                temp = wa1[l]
                for i in 1:j
                    wa3[i] += fjac[i, j] * temp
                end
            end
            t1e = norm(wa3) / fnorm
            t2e = (sqrt(par) * pnorm) / fnorm
            prered = t1e * t1e + t2e * t2e / 0.5
            dirder = -(t1e * t1e + t2e * t2e)
            ratio = 0.0
            if prered != 0.0
                ratio = actred / prered
            end
            if ratio <= 0.25
                if actred >= 0.0
                    temp = 0.5
                else
                    temp = 0.5 * dirder / (dirder + 0.5 * actred)
                end
                if 0.1 * fnorm1 >= fnorm || temp < 0.1
                    temp = 0.1
                end
                delta = temp * min(delta, pnorm / 0.1)
                par /= temp
            else
                if par == 0.0 || ratio >= 0.75
                    delta = pnorm / 0.5
                    par = 0.5 * par
                end
            end
            accepted = ratio >= 1e-4
            if accepted
                @inbounds for j in 1:n
                    x[j] = wa2[j]
                    wa2[j] = diag[j] * x[j]
                end
                @inbounds for i in 1:m
                    fvec[i] = wa4[i]
                end
                xnorm = norm(wa2)
                fnorm = fnorm1
                iter += 1
            end
            if abs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1.0
                info = 1
            end
            if delta <= xtol * xnorm
                info = 2
            end
            if abs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1.0 && info == 2
                info = 3
            end
            if info != 0
                done = true
                break
            end
            if nfev >= maxfev
                info = 5
            end
            if abs(actred) <= epsmch && prered <= epsmch && 0.5 * ratio <= 1.0
                info = 6
            end
            if delta <= epsmch * xnorm
                info = 7
            end
            if gnorm <= epsmch
                info = 8
            end
            if info != 0
                done = true
                break
            end
            if accepted
                inner = false
            end
        end
    end
    return x, fvec, info, nfev
end

function _param_lm_fit(p_start::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64},
                       E_f::Vector{Float64}, ls::Vector{Float64};
                       reg_alpha::Float64=0.0, max_iter::Integer=28000,
                       ftol::Float64=1.5e-8, xtol::Float64=1.5e-8,
                       gtol::Float64=0.0, factor::Float64=100.0,
                       epsfcn::Float64=1e-10, max_nfev::Integer=14000)
    p_ref = copy(p_start)
    u = _fruit_internal_from_external(p_start)
    counter = Ref(0)
    abort_u = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    function resid(uu::Vector{Float64})
        counter[] += 1
        if counter[] > max_nfev
            abort_u[] = copy(uu)
            throw(_FruitAbortError())
        end
        r, _ = _fruit_residual(uu, A, b, E_f, ls, reg_alpha, p_ref)
        return r
    end
    x_final = copy(u)
    info = 0
    try
        x_final, _, info, _ = _fruit_lmdif(resid, u; ftol=ftol, xtol=xtol, gtol=gtol,
                                           maxfev=max_iter, epsfcn=epsfcn, factor=factor)
    catch e
        if e isa _FruitAbortError
            x_final = abort_u[]
            info = -1
        else
            rethrow(e)
        end
    end
    try
        resid(x_final)
    catch e
        if !(e isa _FruitAbortError)
            rethrow(e)
        end
    end
    p = _fruit_external_from_internal(x_final)
    lo = _param_lo_vec()
    hi = _param_hi_vec()
    for i in eachindex(p)
        p[i] = clamp(p[i], lo[i], hi[i])
    end
    nfev = counter[]
    success = info in (1, 2, 3, 4)
    message = if info in (1, 2, 3)
        "Fit succeeded."
    elseif info == 4
        "Fit succeeded (gtol reached)."
    elseif info == 5
        "Number of evaluations exceeded the maximum."
    elseif info == -1
        "Fit aborted."
    elseif info in (6, 7, 8)
        "Tolerance seems to be too small."
    else
        "Number of evaluations exceeded the maximum."
    end
    return p, success, message, nfev
end

function _param_merge_user(initial_params)
    p = _param_default_vec()
    if initial_params !== nothing
        if initial_params isa AbstractDict
            for (i, name) in enumerate(PARAM_NAMES)
                if haskey(initial_params, name)
                    p[i] = Float64(initial_params[name])
                end
            end
        elseif initial_params isa AbstractVector
            for i in 1:min(length(initial_params), length(p))
                p[i] = Float64(initial_params[i])
            end
        end
    end
    return _param_clamp!(p)
end

function _param_dict(p::Vector{Float64})
    return Dict{String,Float64}(n => p[i] for (i, n) in enumerate(PARAM_NAMES))
end

"""
    find_initial_params(A, b, E, log_steps; n_grid=5, n_restarts=1)

Coarse grid search over `P_th x P_epi` (the remaining parameters — FRUIT defaults);
candidates are sorted by the residual norm.  With `n_restarts == 1` a single best
`Dict{String,Float64}` is returned, otherwise — a list of the best by residual.
"""
function find_initial_params(A::AbstractMatrix, b::AbstractVector, E::AbstractVector,
                             log_steps::AbstractVector; n_grid::Integer=5, n_restarts::Integer=1)
    candidates = Tuple{Float64,Vector{Float64}}[]
    p0 = _param_default_vec()
    idx_th = findfirst(==("P_th"), PARAM_NAMES)
    idx_epi = findfirst(==("P_epi"), PARAM_NAMES)
    for kk in 1:n_grid
        p_th = (kk - 1) / max(n_grid - 1, 1)
        for m in 1:n_grid
            p_epi = (m - 1) / max(n_grid - 1, 1)
            p_th + p_epi > 1.0 && continue
            p = copy(p0)
            p[idx_th] = p_th
            p[idx_epi] = p_epi
            spectrum = parametric_model(E, p[1], p[2], p[3], p[4], p[5], p[6]) .* log_steps
            residual = A * spectrum .- b
            push!(candidates, (norm(residual), p))
        end
    end
    isempty(candidates) && return n_restarts > 1 ? Vector{Vector{Float64}}[] : [[p0]]
    sort!(candidates, by = c -> c[1])
    tops = [c[2] for c in candidates[1:min(length(candidates), n_restarts)]]
    return n_restarts > 1 ? tops : tops[1]
end

"""
    compute_parametric_jacobian(E, log_steps, p::Vector{Float64}; delta=1e-8)

Numerical Jacobian of `(parametric_model .* log_steps)` with respect to the 6 parameters
with clamping of perturbations to the bounds; at the boundary — a backward difference.
"""
function compute_parametric_jacobian(E::AbstractVector, log_steps::AbstractVector,
                                     p::Vector{Float64}; delta::Float64=1e-8)
    E_f = Float64.(collect(E))
    ls = Float64.(collect(log_steps))
    lo = _param_lo_vec()
    hi = _param_hi_vec()
    np = length(p)
    J = zeros(length(E_f), np)
    s0 = parametric_model(E_f, p[1], p[2], p[3], p[4], p[5], p[6]) .* ls

    for i in 1:np
        d = delta
        if p[i] + d > hi[i]
            d = max(0.0, hi[i] - p[i]) * 0.5
        end
        p[i] + d < lo[i] && (d = 0.0)
        if d < 1e-15
            d = delta
            if lo[i] >= 0.0 && p[i] - d >= lo[i]
                p_pert = copy(p)
                p_pert[i] = p[i] - d
                s_pert = parametric_model(E_f, p_pert[1], p_pert[2], p_pert[3], p_pert[4],
                                          p_pert[5], p_pert[6]) .* ls
                J[:, i] .= (s0 .- s_pert) ./ d
            else
                J[:, i] .= 0.0
            end
            continue
        end
        p_plus = copy(p)
        p_plus[i] = p[i] + d
        s_plus = parametric_model(E_f, p_plus[1], p_plus[2], p_plus[3], p_plus[4],
                                  p_plus[5], p_plus[6]) .* ls
        J[:, i] .= (s_plus .- s0) ./ d
    end
    return J, s0
end

"""
    solve_parametric(A, b, x0=nothing; E_MeV=nothing, initial_params=nothing,
                     method="leastsq", alpha=0.0, alpha_auto=false, n_restarts=5)
      -> UnfoldResult

Nonlinear LS fit of the FRUIT model parameters — faithful port of
`bssunfold.core._fruit.solve_parametric`: multi-start Levenberg-Marquardt
(lmfit `leastsq` bounds transformation and undamped counts-domain
residuals) with restarts from the top-`n_restarts` points of a 7x7 grid
scan over `P_th x P_epi` and FRUIT defaults elsewhere.  `alpha > 0`
adds a Tikhonov penalty `sqrt(alpha)*(p - p_start)` to the residuals;
`x0` — kept for API compatibility (optional).  `method` is kept as
a label (only the LM algorithm is implemented in the port).
"""
function solve_parametric(A::AbstractMatrix, b::AbstractVector, x0::Union{Nothing,AbstractVector}=nothing;
                          E_MeV::Union{Nothing,AbstractVector}=nothing,
                          initial_params=nothing,
                          method::String="leastsq",
                          alpha::Real=0.0,
                          alpha_auto::Bool=false,
                          n_restarts::Integer=5)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    n_energy = size(AF, 2)
    E_f = E_MeV === nothing ? collect(10.0 .^ range(-9, 2, length=n_energy)) : Float64.(collect(E_MeV))
    ls = compute_log_steps(E_f)

    if initial_params === nothing
        tops = find_initial_params(AF, bf, E_f, ls; n_grid=7, n_restarts=Int(n_restarts))
        starts_vec = Int(n_restarts) > 1 ? tops : [tops]
    else
        starts_vec = [_param_merge_user(initial_params)]
    end

    reg_alpha = Float64(alpha)
    if alpha_auto
        reg_alpha = _param_gcv_select_alpha(AF, bf, E_f, ls, starts_vec[1])
    end

    best_spectrum = nothing
    best_residual = Inf
    best_success = false
    best_message = ""
    total_nfev = 0
    best_params = nothing

    for sp in starts_vec
        p_start = _param_clamp!(copy(sp))
        p_opt, success, message, nfev = _param_lm_fit(p_start, AF, bf, E_f, ls;
                                                      reg_alpha=reg_alpha)
        total_nfev += nfev
        spectrum = parametric_model(E_f, p_opt[1], p_opt[2], p_opt[3], p_opt[4],
                                    p_opt[5], p_opt[6]) .* ls
        res = norm(AF * spectrum .- bf)
        if res < best_residual
            best_residual = res
            best_spectrum = spectrum
            best_success = success
            best_message = message
            best_params = _param_dict(p_opt)
        end
    end

    extra = Dict{String,Any}(
        "params" => best_params,
        "message" => best_message,
        "method" => "parametric",
        "T0" => PARAMETRIC_T0,
        "Ed" => PARAMETRIC_ED,
    )
    return UnfoldResult(max.(best_spectrum, 0.0), total_nfev, best_success,
                        best_residual, extra)
end

"""
    _param_gcv_select_alpha(A, b, E_f, ls, p_start; n_coarse=50, n_refine=20)

SVD-based GCV selection of the Tikhonov weight after linearizing the model
around `p_start` (`A_eff = A * J`), port of `_gcv_select_alpha`.
"""
function _param_gcv_select_alpha(A::Matrix{Float64}, b::Vector{Float64},
                                 E_f::Vector{Float64}, ls::Vector{Float64},
                                 p_start::Vector{Float64};
                                 n_coarse::Integer=50, n_refine::Integer=20)
    J_s, _ = compute_parametric_jacobian(E_f, ls, p_start)
    A_eff = A * J_s
    m, npar = size(A_eff)
    (m < 2 || npar < 2) && return 1e-4
    F = svd(A_eff)
    s_sq = F.S .^ 2
    UTb = F.U' * b
    function gcv_value(a::Float64)
        filt = s_sq ./ (s_sq .+ a)
        resid_coeff = a ./ (s_sq .+ a)
        residual_sq = sum((resid_coeff .* UTb) .^ 2)
        denom = (m - sum(filt))^2
        denom < 1e-30 && return Inf
        return residual_sq / denom
    end
    alphas_coarse = 10.0 .^ range(-8.0, 2.0, length=Int(n_coarse))
    gcv_coarse = [gcv_value(a) for a in alphas_coarse]
    alpha_best = alphas_coarse[argmin(gcv_coarse)]
    alphas_refine = range(max(alpha_best / 10.0, 1e-10), alpha_best * 10.0,
                          length=Int(n_refine))
    gcv_refine = [gcv_value(a) for a in alphas_refine]
    return Float64(alphas_refine[argmin(gcv_refine)])
end

function _param_sqp_core(A::Matrix{Float64}, b::Vector{Float64}, E_f::Vector{Float64},
                         ln_steps::Vector{Float64}, p::Vector{Float64};
                         alpha::Float64, max_iter::Integer, tol::Float64)
    message = ""
    nfev = 0
    for k in 0:max_iter
        J_s, s_k = compute_parametric_jacobian(E_f, ln_steps, p)
        nfev += 1
        residual = A * s_k .- b
        if norm(residual) < tol
            return p, true, "Converged in $k iterations", nfev
        end

        A_eff = A * J_s
        n_p = length(p)
        P = A_eff' * A_eff .+ (max(alpha, 1e-300)) .* Matrix{Float64}(I, n_p, n_p)
        q = A_eff' * residual

        delta_val = -(P \ q)
        lo = _param_lo_vec()
        hi = _param_hi_vec()
        for i in 1:n_p
            delta_val[i] = clamp(p[i] + delta_val[i], lo[i], hi[i]) - p[i]
        end

        p = p .+ delta_val

        if norm(delta_val) < tol
            return p, true, "Converged in $(k+1) iterations", nfev
        end
    end
    isempty(message) && (message = "Max iterations ($max_iter) reached")
    return p, false, message, nfev
end

"""
    solve_parametric_cvxpy(A, b, E, log_steps; initial_params=nothing, alpha=1e-4,
                           max_iter=50, tol=1e-6) -> UnfoldResult

SQP unfolding: at each iteration linearization `A_eff = A @ J`
(where J is the Jacobian of the spectrum with respect to the parameters), the subproblem

    min ||A_eff*delta + residual||^2 + alpha*||delta||^2,  bounded delta

is solved by regularized normal equations (Newton) with clamping
to the bounds (in the python port the substep was solved via cvxpy; here — the same
math directly).  `E` — energy grid in MeV; `log_steps` — d(log10 E)
or d ln E steps.
"""
function solve_parametric_cvxpy(A::AbstractMatrix, b::AbstractVector, E::AbstractVector,
                                log_steps::AbstractVector; initial_params=nothing,
                                alpha::Real=1e-4, max_iter::Integer=50, tol::Real=1e-6)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    E_f = Float64.(collect(E))
    ls = Float64.(collect(log_steps))
    p = _param_merge_user(initial_params)
    p, success, message, nfev = _param_sqp_core(AF, bf, E_f, ls, p;
                                                alpha=Float64(alpha), max_iter=Int(max_iter),
                                                tol=Float64(tol))
    spectrum = parametric_model(E_f, p[1], p[2], p[3], p[4], p[5], p[6]) .* ls
    res = norm(AF * spectrum .- bf)
    extra = Dict{String,Any}(
        "params" => _param_dict(p),
        "message" => message,
        "optimizer" => "cvxpy-analytic-QP",
    )
    return UnfoldResult(max.(spectrum, 0.0), nfev, success, res, extra)
end

"""
    solve_parametric_qpsolvers(A, b, E, log_steps; initial_params=nothing,
                               alpha=1e-4, max_iter=50, tol=1e-6) -> UnfoldResult

SQP unfolding equivalent to `solve_parametric_cvxpy`, but the subproblem
is written in the standard QP form of the project (`P = A_effᵀ A_eff +
alpha*I`, `q = A_effᵀ residual`, minimizing `0.5 dᵀ P d + qᵀ d` under
bound constraints) and solved directly (`d = -P\\q`) — qpsolvers is replaced
by an in-house solution via regularized Newton.
"""
function solve_parametric_qpsolvers(A::AbstractMatrix, b::AbstractVector, E::AbstractVector,
                                    log_steps::AbstractVector; initial_params=nothing,
                                    alpha::Real=1e-4, max_iter::Integer=50, tol::Real=1e-6)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    E_f = Float64.(collect(E))
    ls = Float64.(collect(log_steps))
    p = _param_merge_user(initial_params)
    message = ""
    nfev = 0
    success = false
    for k in 0:max_iter
        J_s, s_k = compute_parametric_jacobian(E_f, ls, p)
        nfev += 1
        residual = AF * s_k .- bf
        if norm(residual) < tol
            success = true
            message = "Converged in $k iterations"
            break
        end
        A_eff = AF * J_s
        n_p = length(p)
        P = A_eff' * A_eff .+ (max(Float64(alpha), 1e-300)) .* Matrix{Float64}(I, n_p, n_p)
        q = A_eff' * residual
        delta_val = -(P \ q)
        lo = _param_lo_vec()
        hi = _param_hi_vec()
        for i in 1:n_p
            delta_val[i] = clamp(p[i] + delta_val[i], lo[i], hi[i]) - p[i]
        end
        p = p .+ delta_val
        if norm(delta_val) < tol
            success = true
            message = "Converged in $(k+1) iterations"
            break
        end
    end
    isempty(message) && (message = "Max iterations ($max_iter) reached")
    spectrum = parametric_model(E_f, p[1], p[2], p[3], p[4], p[5], p[6]) .* ls
    res = norm(AF * spectrum .- bf)
    extra = Dict{String,Any}(
        "params" => _param_dict(p),
        "message" => message,
        "optimizer" => "qpsolvers-analytic-QP",
    )
    return UnfoldResult(max.(spectrum, 0.0), nfev, success, res, extra)
end

"""
    solve_parametric_combined(A, b, E, log_steps; initial_params=nothing,
                              method="leastsq", alpha=1e-4, solver_backend="auto",max_iter=50, tol=1e-6) -> UnfoldResult

Combined pipeline: (1) leastsq fit (LM) of the FRUIT model parameters,
(2) QP refinement of the spectrum: `min ||A x - b||^2 + alpha||x - x_init||^2,
x >= 0` via the augmented matrix and `solve_nnls` of the BSSUnfold package
(in the python port the refinement went through cvxpy/qpsolvers).  The resulting spectrum
is returned multiplied by `log_steps` (log convention of the port).
"""
function solve_parametric_combined(A::AbstractMatrix, b::AbstractVector, E::AbstractVector,
                                   log_steps::AbstractVector; initial_params=nothing,
                                   method::String="leastsq", alpha::Real=1e-4,
                                   solver_backend="auto", max_iter::Integer=50, tol::Real=1e-6)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    E_f = Float64.(collect(E))
    ls = Float64.(collect(log_steps))

    lm_res = solve_parametric(AF, bf, nothing; E_MeV=E_f, initial_params=initial_params,
                              method=method, alpha=0.0)
    spectrum_lmfit = lm_res.spectrum

    np_bins = size(AF, 2)
    p0 = max.(spectrum_lmfit ./ max.(ls, 1e-300), 0.0)
    xa = solve_nnls(
        vcat(AF, sqrt(max(Float64(alpha), 0.0)) .* Matrix{Float64}(I, np_bins, np_bins)),
        vcat(bf, sqrt(max(Float64(alpha), 0.0)) .* p0))
    refined = xa .* ls

    msg = "leastsq + QP refinement OK"
    extra = Dict{String,Any}(
        "message" => msg,
        "optimizer" => "combined",
        "solver_backend" => string(solver_backend),
        "params" => getfield(lm_res, :extra)["params"],
    )
    return UnfoldResult(max.(refined, 0.0), lm_res.iterations, lm_res.converged,
                        norm(AF * max.(refined, 0.0) .- bf), extra)
end
