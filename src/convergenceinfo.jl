abstract type AbstractConvergenceInfo end

@doc raw"""
    ConvergenceInfo

A mutable struct that holds information about the convergence progress of an algorithm
# Fields
- `converged::Bool`: a boolean stating if an algorithm has converged or not
- `error::Float64`: stores the current error
- `iterations::Int`: stores the number of interations so far
- `finished::Bool`: whether or not the algorithm has finished running
"""
@with_kw mutable struct ConvergenceInfo <: AbstractConvergenceInfo
    converged::Bool = false
    error::Float64 = Inf
    iterations::Int = 0
    finished::Bool = false
end

isconverged(conv::ConvergenceInfo) = conv.converged
geterror(conv::ConvergenceInfo) = conv.error
numiter(conv::ConvergenceInfo) = conv.iterations
isfinished(conv::ConvergenceInfo) = conv.finished

function reset!(conv::ConvergenceInfo)
    conv.finished = false
    return conv
end
