abstract type AbstractContractionAlgorithm end
abstract type AbstractContractionState{Alg<:AbstractContractionAlgorithm} end
abstract type AbstractContractionTensors end

"""
    initialize(network::AbstractNetwork, alg::AbstractContractionAlgorithm; kwargs...)
    initialize(network::AbstractNetwork, alg::AbstractContractionAlgorithm, initial_tensors::AbstractContractionTensors; kwargs...)

Initialize an algorithm state to contract network of tensors `network` using algorithm `alg`. The keyword argument `store_initial`
determines whether or not the initial state of the contraction tensors is copied and stored in the field
`initial_tensors` of the returned `AbstractContractionState` object.

# Arguments
- `network::AbstractNetwork`: the unitcell of tensors to be contracted
- `alg::AbstractContractionAlgorithm`: the algorithm to contract `network` with
- `initial_tensors::AbstractContractionTensors`: initial tensors to use (optional)

# Keywords
- `store_initial::Bool = true`: perform a deep copy of the initial tensors
- `outfile::String = "outfile.jld2"`: file to write data into
- `prestep::Function = identity`: function to run before every algorithm step
- `poststep::Function = identity`: function to run after every algorithm step

# Returns
- `AbstractContractionState{Alg,...}`: algorithmic state corresponding to the supplied tensors and parameters
"""
function initialize end
const initialise = initialize

"""
    calculate(alg_state::AbstractContractionState)

Equivalent to `calculate!(deepcopy(alg_state))`.
"""
calculate(alg_state::AbstractContractionState) = calculate!(deepcopy(alg_state))

"""
    calculate!(alg_state::AbstractContractionState)

Calculate the contraction tensors required to contract `alg_state.network` using `alg_state.alg`. Returns
mutated `alg_state`. Use `calculate` for a non-mutating version of the same function.
"""
function calculate!(alg_state::AbstractContractionState)
    return _calculate!(alg_state, Val(alg_state.info.finished))
end

function _calculate!(alg_state::AbstractContractionState, ::Val{true})
    println(
        "Algorithm, finished according to parameters set. Use `forcecalculate!`, or `continue!` followed by `contract!` to ignore this and continue anyway. ",
    )
    return alg_state
end

_calculate!(alg_state::AbstractContractionState, ::Val{false}) = run!(alg_state)

"""
    continue!(alg_state::AbstractContractionState)

Allow `alg_state` to continue past the termination criteria.
"""
function continue!(alg_state::AbstractContractionState)
    alg_state.info.finished = false
    return alg_state
end

"""
    reset!(alg_state::AbstractContractionState)

Reset the convergence info of `alg_state`.
"""
function reset!(alg_state::AbstractContractionState)
    continue!(alg_state)
    alg_state.info.converged = false
    alg_state.info.error = Inf
    alg_state.info.iterations = 0
    return alg_state
end

"""
    restart!(alg_state::AbstractContractionState)

Restart the algorithm entirely, returning the tensors to their initial state.
"""
function restart!(alg_state::AbstractContractionState)
    if isdefined(alg_state, :initial_tensors)
        reset!(alg_state)
        deepcopy!(alg_state.tensors, alg_state.initial_tensors)
    else
        println(
            "Cannot restart algorithm as initial tensors are undefined. Doing nothing..."
        )
    end
    return alg_state
end

"""
    forcecalculate!(alg_state::AbstractContractionState)

Froce the algorithm to continue. Equivalent to calling `continue!` followed by`contract!`.
"""
function forcecalculate!(alg_state::AbstractContractionState)
    continue!(alg_state)
    calculate!(alg_state)
    return alg_state
end

@generated function deepcopy!(ten_dst::T, ten_src::T) where {T}
    assignments = [:(deepcopy!(ten_dst.$name, ten_src.$name)) for name in fieldnames(T)]
    quote
        $(assignments...)
    end
end

# DEPREC
# @generated function isinitialised(ten::T) where {T}
#     checks = [
#         :(isinitialised(ten.$name)) for name in fieldnames(T)
#     ]
#     quote
#         out = [$(checks...)]
#         all(out)
#     end
# end

# @generated function testfunc(out, t1::AbstractTensorMap{S,2,N}, t2::AbstractTensorMap{S,N,2}) where {S,N}
#     phys1 = Expr(:row, :k1, :b1)
#     phys2 = Expr(:row, :k2, :b2)
#
#     ex_args = (
#         Symbol("x" * string(i)) for i in 1:N
#     )
#
#     ex = Expr(:row, ex_args...)
#
#     et1 = Expr(:typed_vcat, :t1, phys1, ex)
#     et2 = Expr(:typed_vcat, :t2, ex, phys2)
#     # println(dump(:($et1 * $et2)))
#     quote
#         @tensoropt out[k1 b1; k2 b2] = $et1 * $et2
#     end
# end

function contract(tensors, network)
    inds = collect(CartesianIndices(network))
    return contract.(Ref(tensors), Ref(network), inds)
end

contract(tensors, network, i1::Int, i2::Int) = contract(tensors, network, i1:i1, i2:i2)
contract(tensors, network, inds) = contract(tensors, network, inds[1], inds[2])

function contract(ctmrg, network, i1::UnitRange, i2::UnitRange)
    cs = ctmrg.corners[i1, i2]
    es = map(x -> tuple(x...), ctmrg.edges[i1, i2])
    return _contractall(cs..., es..., network)
end

function get_bond_symbol(i, j, dir)
    if dir == :h
        str = "$(i)$(j)_$(i + 1)$(j)"
    elseif dir == :v
        str = "$(i)$(j)_$(i)$(j + 1)"
    end
    return Symbol(str)
end

@generated function _contractall(
    C1,
    C2,
    C3,
    C4,
    T1::NTuple{Nx,E},
    T2::NTuple{Ny,E},
    T3::NTuple{Nx,E},
    T4::NTuple{Ny,E},
    MS::AbstractMatrix{<:AbsTen{0}},
) where {Nx,Ny,E}
    gh = (i, j) -> get_bond_symbol(i, j, :h)
    gv = (i, j) -> get_bond_symbol(i, j, :v)

    e_T1s = [
        Expr(:ref, Expr(:ref, :T1, i), gv(i + 1, 1), (gh(i + 1, 1), gh(i, 1))...) for
        i in 1:Nx
    ]

    e_T2s = [
        Expr(
            :ref,
            Expr(:ref, :T2, j),
            gh(Nx + 1, j + 1),
            (gv(Nx + 2, j), gv(Nx + 2, j + 1))...,
        ) for j in 1:Ny
    ]
    e_T3s = [
        Expr(
            :ref,
            Expr(:ref, :T3, i),
            gv(i + 1, Ny + 1),
            (gh(i, Ny + 2), gh(i + 1, Ny + 2))...,
        ) for i in 1:Nx
    ]
    e_T4s = [
        Expr(:ref, Expr(:ref, :T4, j), gh(1, j + 1), (gv(1, j + 1), gv(1, j))...) for
        j in 1:Ny
    ]

    e_MS = [
        Expr(
            :ref,
            Expr(:ref, :MS, i, j),
            gh(i + 1, j + 1),
            gv(i + 1, j + 1),
            gh(i, j + 1),
            gv(i + 1, j),
        ) for i in 1:Nx, j in 1:Ny
    ]

    e_C1 = Expr(:ref, :C1, gh(1, 1), gv(1, 1))
    e_C2 = Expr(:ref, :C2, gh(Nx + 1, 1), gv(Nx + 2, 1))
    e_C3 = Expr(:ref, :C3, gh(Nx + 1, Ny + 2), gv(Nx + 2, Ny + 1))
    e_C4 = Expr(:ref, :C4, gh(1, Ny + 2), gv(1, Ny + 1))

    e_einsum = Expr(
        :call, :*, e_C1, e_C2, e_C3, e_C4, e_T1s..., e_T2s..., e_T3s..., e_T4s..., e_MS...
    )

    quote
        @tensoropt rv = $e_einsum
    end
end
