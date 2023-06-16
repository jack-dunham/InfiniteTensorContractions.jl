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
    BoundaryState{Alg, NType, T} <: AbstractBoundaryState{B}

A struct representing the state of a boundary algorithm of type `Alg` used to contract a network of 
type `NType` using boundary tensors of type `T`. Note, avoid constructing this object directly,
instead use function `initialize`.

# Fields
- `tensors::T <: AbstractBoundaryTensors`: the tensors that compose the boundary
- `network::NType <: AbstractNetwork`: the network of tensors to be contracted
- `alg::Alg`: the contraction algorithm to be used
- `info::ConvergenceInfo`: information about the covergence progress of the algorithm
- `outfile::String`: file to write data to
"""
struct BoundaryState{
    Alg<:AbstractBoundaryAlgorithm,NType<:AbstractNetwork,T<:AbstractBoundaryTensors
} <: AbstractBoundaryState{Alg}
    tensors::T
    network::NType
    alg::Alg
    info::ConvergenceInfo
    outfile::String
    prestep::Function
    poststep::Function
    initial_tensors::T
    function BoundaryState(
        tensors::T,
        network::NType,
        alg::Alg,
        info::ConvergenceInfo,
        outfile::String,
        prestep::Function,
        poststep::Function,
        initial_tensors::T,
    ) where {T,NType,Alg}
        boundary_verify(tensors, alg)
        return new{Alg,NType,T}(
            tensors, network, alg, info, outfile, prestep, poststep, initial_tensors
        )
    end
    function BoundaryState(
        tensors::T,
        network::NType,
        alg::Alg,
        info::ConvergenceInfo,
        outfile::String,
        prestep::Function,
        poststep::Function,
    ) where {T,NType,Alg}
        boundary_verify(tensors, alg)
        return new{Alg,NType,T}(tensors, network, alg, info, outfile, prestep, poststep)
    end
end

function boundary_verify(tensors, alg)
    btype = contraction_boundary_type(tensors)
    if !(typeof(alg) <: btype)
        throw(ArgumentError("Incompatible algorithm for type of boundary"))
    end
    return nothing
end

function initialize(
    network,
    alg,
    initial_tensors=inittensors(rand, network, alg);
    store_initial=true,
    outfile="",
    prestep=identity,
    poststep=identity,
)
    info = ConvergenceInfo()
    if store_initial
        initial_copy = deepcopy(initial_tensors)
        return BoundaryState(
            initial_tensors, network, alg, info, outfile, prestep, poststep, initial_copy
        )
    else
        return BoundaryState(
            initial_tensors, network, alg, info, outfile, prestep, poststep
        )
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
    network = detrace(state.network)

    args = start(state)

    while info.iterations < maxiter && info.error ≥ tol
        prestep(state, args...)

        info.error = step!(state.tensors, network, alg, args...)

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
    network = state.network
    new_alg = similarboundary(Alg, state.alg)
    new_state = initialize(network, new_alg)
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
