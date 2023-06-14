abstract type AbstractBoundaryTensors <: AbstractContractionTensors end
abstract type AbstractBoundaryAlgorithm <: AbstractContractionAlgorithm end
abstract type AbstractBoundaryState{B<:AbstractBoundaryAlgorithm} <:
              AbstractContractionState{B} end

function similarboundary(
    ::Type{Alg}, alg::AbstractBoundaryAlgorithm; kwargs...
) where {Alg<:AbstractBoundaryAlgorithm}
    return Alg(;
        bonddim=alg.bonddim,
        verbosity=alg.verbosity,
        maxiter=alg.maxiter,
        tol=alg.tol,
        kwargs...,
    )
end

"""
    BoundaryState{Alg, BuType, BoType} <: AbstractBoundaryState{B}

A struct representing the state of a boundary algorithm of type `Alg` used to contract a network of 
type `BuType` using boundary tensors of type `BoType`. Note, avoid constructing this object directly,
instead use function `initialize`.

# Fields
- `tensors::BoType <: AbstractBoundaryTensors`: the tensors that compose the boundary
- `network::BuType <: AbstractNetwork`: the network of tensors to be contracted
- `alg::Alg`: the contraction algorithm to be used
- `info::ConvergenceInfo`: information about the covergence progress of the algorithm
- `outfile::String`: file to write data to
"""
struct BoundaryState{
    Alg<:AbstractBoundaryAlgorithm,
    BuType<:AbstractContractableTensors,
    BoType<:AbstractBoundaryTensors,
} <: AbstractBoundaryState{Alg}
    tensors::BoType
    bulk::BuType
    alg::Alg
    info::ConvergenceInfo
    outfile::String
    prestep::Function
    poststep::Function
    initial_tensors::BoType
    function BoundaryState(
        tensors::BoType,
        bulk::BuType,
        alg::Alg,
        info::ConvergenceInfo,
        outfile::String,
        prestep::Function,
        poststep::Function,
        initial_tensors::BoType,
    ) where {BoType,BuType,Alg}
        boundary_verify(tensors, alg)
        return new{Alg,BuType,BoType}(
            tensors, bulk, alg, info, outfile, prestep, poststep, initial_tensors
        )
    end
    function BoundaryState(
        tensors::BoType,
        bulk::BuType,
        alg::Alg,
        info::ConvergenceInfo,
        outfile::String,
        prestep::Function,
        poststep::Function,
    ) where {BoType,BuType,Alg}
        boundary_verify(tensors, alg)
        return new{Alg,BuType,BoType}(tensors, bulk, alg, info, outfile, prestep, poststep)
    end
end

function boundary_verify(tensors, alg)
    btype = contraction_boundary_type(tensors)
    if !(btype == typeof(alg))
        throw(ArgumentError("Incompatible type of boundary for algorithm provided"))
    end
    return nothing
end

function initialize(
    bulk,
    alg,
    initial_tensors=inittensors(rand, bulk, alg);
    store_initial=true,
    outfile="",
    prestep=identity,
    poststep=identity,
)
    info = ConvergenceInfo()
    if store_initial
        initial_copy = deepcopy(initial_tensors)
        return BoundaryState(
            initial_tensors, bulk, alg, info, outfile, prestep, poststep, initial_copy
        )
    else
        return BoundaryState(initial_tensors, bulk, alg, info, outfile, prestep, poststep)
    end
end

function run!(state::AbstractBoundaryState)
    prestep = state.prestep
    poststep = state.poststep

    alg = state.alg
    info = state.info

    alg.verbose && @info "Running algorithm $alg"

    # Immutable paramaters
    maxiter = alg.maxiter
    tol = alg.tol

    # Remove any wrappers, converting tensors to appropriate forms.
    bulk = detrace(state.bulk)

    args = start(state)

    while info.iterations < maxiter && info.error ≥ tol
        prestep(state, args...)

        info.error = step!(state.tensors, bulk, alg, args...)

        info.iterations += 1

        alg.verbose && @info "Error ≈ $(info.error) \t after $(info.iterations) iterations"

        poststep(state, args...)
    end

    info.error > tol ? info.converged = false : info.converged = true

    info.finished = true

    return state
end

Base.identity(x::AbstractBoundaryState, args...; kwargs...) = x

detrace(x) = x

function similarboundary(
    ::Type{Alg}, state::AbstractBoundaryState
) where {Alg<:AbstractBoundaryTensors}
    bulk = state.bulk
    new_alg = similarboundary(Alg, state.alg)
    new_state = initialize(bulk, new_alg)
    return new_state
end

function boundaryerror!(S_old::AbstractMatrix, C_new::AbstractMatrix)
    S_new = boundaryerror.(C_new)
    err = @. norm(S_old - S_new)
    S_old .= S_new
    return err
end

function boundaryerror(c_new::AbstractTensorMap)
    _, s_new, _ = tsvd(c_new, (1,), (2,))
    normalize!(s_new)
    return s_new
end
