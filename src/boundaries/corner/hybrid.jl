abstract type AbstractDoFPCM end
abstract type AbstractBooleanDoFPCM <: AbstractDoFPCM end

struct Never <: AbstractDoFPCM end
struct Always <: AbstractDoFPCM end

dofpcm(::CTMRG) = Never()
dofpcm(::FPCM) = Always()

struct AtFrequency <: AbstractBooleanDoFPCM
    val::Int64
end

struct NotAtFrequency <: AbstractBooleanDoFPCM
    val::Int64
end

struct IfBelowTolerance <: AbstractBooleanDoFPCM
    val::Float64
end
struct IfAboveTolerance <: AbstractBooleanDoFPCM
    val::Float64
end

function (dofpcm::AtFrequency)(state)
    freq = dofpcm.val
    if mod(state.info.iterations, 0:(freq - 1)) == 0
        return true
    else
        return false
    end
end

function (dofpcm::NotAtFrequency)(state)
    freq = dofpcm.val
    if mod(state.info.iterations, 0:(freq - 1)) == 0
        return false
    else
        return true
    end
end

function (dofpcm::IfBelowTolerance)(state)
    tol = dofpcm.val
    if state.info.error < tol
        return true
    else
        return false
    end
end

function (dofpcm::IfAboveTolerance)(state)
    tol = dofpcm.val
    if state.info.error > tol
        return true
    else
        return false
    end
end

@kwdef struct HybridCornerMethod{C,F,D<:AbstractDoFPCM} <: AbstractCornerMethod
    ctmrg::C = CTMRG()
    fpcm::F = FPCM()
    bonddim::Int
    dofpcm::D = AtFrequency(10)
    randinit::Bool = false
    maxiter::Int = 100
    verbose::Bool = true
    tol::Float64 = 1e-12
    function HybridCornerMethod(
        ctmrg, fpcm, bonddim, dofpcm, randinit, maxiter, verbose, tol
    )
        c = CTMRG(;
            bonddim=bonddim,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
            ptol=ctmrg.ptol,
            svdalg=ctmrg.svdalg,
            randinit=randinit,
        )
        f = FPCM(;
            bonddim=bonddim,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
            randinit=randinit,
            docorners=fpcm.docorners,
        )

        return new{typeof(c),typeof(f),typeof(dofpcm)}(
            c, f, bonddim, dofpcm, randinit, maxiter, verbose, tol
        )
    end
end

dofpcm(alg::HybridCornerMethod) = alg.dofpcm

step!(::Union{Never,Always}, problem) = step!(problem.runtime, problem.algorithm)

function step!(dofpcm::AbstractBooleanDoFPCM, problem)
    if dofpcm(problem)
        @info "Doing FPCM..."
        error = step!(problem.runtime, problem.algorithm.fpcm)
    else
        @info "Doing CTMRG..."
        error = step!(problem.runtime, problem.algorithm.ctmrg)
    end
    return error
end
