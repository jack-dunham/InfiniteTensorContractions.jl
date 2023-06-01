abstract type AbstractBoundaryTensors <: AbstractContractionTensors end
abstract type AbstractBoundaryAlgorithm <: AbstractContractionAlgorithm end
abstract type AbstractBoundaryState{B} <: AbstractContractionState{B<:AbstractBoundaryAlgorithm}


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

@doc raw"""
    BoundaryState{B<:AbstractBoundary} <: AbstractBoundaryState{B}

A struct representing the state of boundary algorithm of type `B`. The paramater `B`
can be one of `VUMPS`, `CTMRG`, `FPCM`, `FPCM_CTMRG`, or a user-defined algorithm. 

# Fields
- `data::B`: the tensors that compose the boundary, in addition to algorithm specific 
    data
- `mpo::InfMPO`: the matrix product operator whos boundary tensors are stored in `data`
- `initial::Alg`: the tensors used as the starting point of the algorithm
- `info::ConvergenceInfo`: information about the covergence progress of the algorithm
- `param::BoundaryParameters`: the paramaters used in the algorithm
"""
struct BoundaryState{
    Alg<:AbstractBoundaryAlgorithm,
    BuType<:AbstractUnitCell,
    BoType<:AbstractBoundaryTenors,
} <: AbstractBoundaryState{Alg}
    tensors::BoType
    initial_tensors::BoType
    bulk::BulkType
    alg::Alg
    info::ConvergenceInfo
    function BoundaryState(
        tensors::BoType,
        initial_tensors::BoType,
        bulk::BuType,
        alg::Alg,
        info::ConvergenceInfo,
    ) where {
        BoundaryType<:AbstractBoundary,
        BulkType<:AbstractUnitCell,
        Alg<:AbstractBoundaryAlgorithm,
    }
        btype = contraction_boundary_type(tensors)
        if !(btype == typeof(alg))
            throw(ArgumentError("Incompatible type of boundary for algorithm provided"))
        else
            return new{Alg,BulkType,BoundaryType}(tensors, initial_tensors, bulk, alg, info)
        end
    end
end

function BoundaryState(tensors, bulk, alg; info=ConvergenceInfo(), store_initial=false)
    if store_initial
        initial = deepcopy(tensors)
    else
        initial = similar(tensors)
    end
    return BoundaryState(tensors, initial, bulk, alg, info)
end

@doc raw"""
    inittensors(f, T, alg::BoundaryAlgorithm)

Initialise the boundary tensors compatible with lattice `T` for use in boundary algorithm
`alg`.
"""
function inittensors(f, bulk, alg::AbstractBoundaryAlgorithm) end

# @doc raw"""
# Initilise a new `alg` boundary algorithm state for contracting the infinite lattice
# generated by `T` using boundary tensors generated from the function `f`. The callable 
# `f` should be a callable compatible with the `TensorMap` object. Alternatively, initialise 
# a new algorithm state with initial boundary given by `B`.
# """
# function initstate(f, T, alg::BoundaryAlgorithm)
#     return boundary = inittensors(f, T, alg)
# end

function initialize(bulk, alg::AbstractBoundaryAlgorithm; kwargs...)
    tensors = inittensors(detrace(bulk), alg; kwargs...)
    return initialize(tensors, bulk, alg; kwargs...)
end
function initialize(tensors, bulk, alg; kwargs...)
    return BoundaryState(tensors, bulk, alg; kwargs...)
end

@doc raw"""
    contract(b::AbstractBoundaryState)

Perform the contraction represented by algorithm state `b`.
"""
function contract!(state::AbstractBoundaryState)
    alg = state.alg
    # Immutable parameters
    verbosity = alg.verbosity
    maxiter = alg.maxiter
    tol = alg.tol
    bonddim = alg.bonddim
    # Mutating parameters
    info = state.info
    error = info.error
    iterations = info.iterations

    # Remove any wrappers, converting tensors to appropraite forms.
    bulk = detrace(state.bulk)

    info.error, info.iterations = calculate!(
        state.tensors, bulk; bonddim=bonddim, verbosity=verbosity, tol=tol, maxiter=maxiter
    )

    info.error > tol ? info.converged = false : info.converged = true

    info.finished = true

    return state
end

function similarboundary(
    ::Type{Alg}, state::AbstractBoundaryState
) where {Alg<:AbstractBoundary}
    bulk = state.bulk
    new_alg = similarboundary(Alg, state.alg)
    new_state = new_alg(bulk)
    return new_state
end