abstract type AbstractAlgorithm end
abstract type AbstractRuntime end

"""
    InfiniteContraction{Alg, Net, Run, Out} <: AbstractProblemState

Concrete struct representing the state of a contraction algorithm of type `Alg` used to 
contract a network of type `Net` with runtime tensors of type `Run`. If a callback is 
provided, any returned data is stored in an instance of type `Out`. Note, avoid 
constructing this object directly, instead use the function `newproblem` to construct a 
a new instance of `InfiniteContraction`.

# Fields
- `algorithm::Alg`: the contraction algorithm to be used
- `runtime::Run`: runtime tensors required for the contraction
- `network::Net <: AbstractNetwork`: the network of tensors to be contracted
- `info::ConvergenceInfo`: information about the covergence progress of the algorithm
- `callback::Callback{Out}`: represents a function to be executed at the end of each step
- `verbose::Bool`: if `false`, will surpress top-level information about algorithm progress
- `initialruntime::Run`: initial state of the runtime tensors if stored
"""
struct InfiniteContraction{Alg,Net,Run,Out}
    algorithm::Alg
    network::Net
    runtime::Run
    info::ConvergenceInfo
    callback::Callback{Out}
    verbose::Bool
    initialruntime::Run
    function InfiniteContraction(
        algorithm::Alg,
        network::Net,
        runtime::Run,
        info::ConvergenceInfo,
        callback::Callback{Out},
        verbose::Bool,
        initialruntime::Run,
    ) where {Alg,Run,Net,Out}
        # verify(runtime, algorithm)
        return new{Alg,Net,Run,Out}(
            algorithm, network, runtime, info, callback, verbose, initialruntime
        )
    end
    function InfiniteContraction(
        algorithm::Alg,
        network::Net,
        runtime::Run,
        info::ConvergenceInfo,
        callback::Callback{Out},
        verbose::Bool,
    ) where {Alg,Net,Run,Out}
        # verify(runtime, algorithm)
        return new{Alg,Net,Run,Out}(algorithm, network, runtime, info, callback, verbose)
    end
end

"""
    newcontraction(algorithm, network, [initial_tensors]; kwargs...)

Initialize an instance of a `InfiniteContraction` to contract network of tensors 
`network` using algorithm `algorithm`. 

# Arguments
- `network::AbstractNetwork`: the unit cell of tensors to be contracted
- `alg::AbstractAlgorithm`: the algorithm to contract `network` with
- `initial_tensors::AbstractRuntime`: initial runtime tensors to use (optional)

# Keywords
- `store_initial::Bool = true`: if true, store a deep copy of the initial tensors
- `verbose::Bool = true`: if `false`, will surpress top-level information about 
    algorithm progress
- `callback::Callback{Out} = Callback(identity, nothing)`: represents a function to be 
    executed at the end of each step

# Returns
- `ProblemState{Alg,...}`: problem state instancecorresponding to the supplied tensors 
    and parameters
"""
function newcontraction(network; alg, kwargs...)
    initial_runtime = initialize(network, alg)
    return newcontraction(network, initial_runtime; alg=alg, kwargs...)
end

function newcontraction(
    network,
    initial_runtime;
    alg,
    store_initial=true,
    verbose=true,
    callback=Callback(identity, nothing),
)
    info = ConvergenceInfo()

    if store_initial
        initial_copy = deepcopy(initial_runtime)
        return InfiniteContraction(
            alg, network, initial_runtime, info, callback, verbose, initial_copy
        )
    else
        return InfiniteContraction(alg, network, initial_runtime, info, callback, verbose)
    end
end

function _run!(problem::InfiniteContraction)
    callback = problem.callback
    info = problem.info
    alg = problem.algorithm

    problem.verbose && @info "Running algorithm:" algorithm = alg

    while info.iterations < alg.maxiter && info.error ≥ alg.tol
        info.error = step!(problem)

        info.iterations += 1

        problem.verbose &&
            @info "Convergence ≈ $(info.error) after $(info.iterations) iterations."

        callback(problem)
    end
    info.error > alg.tol ? info.converged = false : info.converged = true

    info.finished = true

    problem.verbose && begin
        @info "Convergence: $(info.error)"
        if info.converged
            @info "Algorithm convergenced to within tolerance $(alg.tol) after $(info.iterations) iterations"
        else
            @warn "Algorithm did not convergence to within $(alg.tol) after $(info.iterations) iterations"
        end
    end

    return info.finished
end

"""
    runcontraction(problem::ProblemState)

Equivalent to `runcontraction!(deepcopy(problem))`.
"""
runcontraction(problem) = runcontraction!(deepcopy(problem))

"""
    runcontraction!(problem::ProblemState)

Calculate the contraction tensors required to contract `problem.network` using 
`problem.algorithm`. Returns mutated `problem`. Use `runcontraction` for a non-mutating 
version of the same function.
"""
function runcontraction!(problem::InfiniteContraction)
    if problem.info.finished == true
        println(
            "Problem has reached termination according to parameters set. Use `forcerun!`, 
                or `continue!` followed by `runcontraction!` to ignore this and continue anyway.",
        )
    else
        _run!(problem)
    end
end

"""
    continue!(problem::ProblemState)

Allow `problem` to continue past the termination criteria.
"""
function continue!(problem::InfiniteContraction)
    problem.info.finished = false
    return problem
end

"""
    reset!(problem::ProblemState)

Reset the convergence info of `problem`.
"""
function reset!(problem::InfiniteContraction)
    continue!(problem)
    problem.info.converged = false
    problem.info.error = Inf
    problem.info.iterations = 0
    reset!(problem.runtime, problem.network)
    return problem
end

"""
    recycle!(problem::ProblemState)

Reset the convergence info of `problem`.
"""
function recycle!(problem::InfiniteContraction, network)
    continue!(problem)
    problem.info.converged = false
    problem.info.error = Inf
    problem.info.iterations = 0
    reset!(problem.runtime, network)
    return problem
end

"""
    restart!(problem::ProblemState)

Restart the algorithm entirely, returning the tensors to their initial state.
"""
function restart!(problem::InfiniteContraction)
    if true #isdefined(problem, :initial_tensors)
        reset!(problem)
        deepcopy!(problem.runtime, problem.initialruntime)
    else
        println(
            "Cannot restart algorithm as initial tensors are undefined. Doing nothing..."
        )
    end
    return problem
end

"""
    forcerun!(problem::ProblemState)

Force the algorithm to continue. Equivalent to calling `continue!` followed by `run!`.
"""
function forcerun!(problem::InfiniteContraction)
    continue!(problem)
    runcontraction!(problem)
    return problem
end
