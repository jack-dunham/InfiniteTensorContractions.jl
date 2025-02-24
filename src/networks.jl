abstract type ContractableTrait end
struct IsContractable <: ContractableTrait end
struct NotContractable <: ContractableTrait end

const AbstractNetwork{G,ElType<:AbstractTensorMap,A} = AbstractUnitCell{G,ElType,A}
const AbstractSingleLayerNetwork{G,ElType<:TensorMap,A} = AbstractUnitCell{G,ElType,A}

ContractableTrait(::Type{<:AbsTen{0,4}}) = IsContractable()
ContractableTrait(::Type{<:AbsTen{1,4}}) = IsContractable()
ContractableTrait(::Type{<:AbsTen{2,4}}) = IsContractable()
ContractableTrait(::Type) = NotContractable()

TensorKit.spacetype(uc::AbstractUnitCell) = spacetype(typeof(uc))
TensorKit.spacetype(::Type{<:AbstractNetwork{G,T}}) where {G,T} = spacetype(T)

"""
    CompositeTensor{M,T,S<:IndexSpace,N₁,N₂, ...} <: AbstractTensorMap{T,S,N₁,N₂}

Subtype of `AbstractTensorMap` for representing a M-layered tensor map, e.g. a 
tensor and its conjugate.
"""
struct CompositeTensor{M,T,S,N₁,N₂,A<:TensorMap{T,S,N₁,N₂},F<:Tuple} <:
       AbstractTensorMap{T,S,N₁,N₂}
    layers::NTuple{M,A}
    funcs::F
end

Base.length(ct::CompositeTensor{M}) where {M} = M

Base.map(func, ct::CompositeTensor) = map((ten, af) -> func(af(ten)), ct.layers, ct.funcs)
mapbefore(f, ct::CompositeTensor) = map((t, af) -> af(f(t)), ct.layers, ct.funcs)

function CompositeTensor(ct::NTuple{M,T}) where {M,T<:TensorMap}
    funcs = NTuple{M,typeof(identity)}(fill(identity, M))

    return CompositeTensor(ct, funcs)
end
function CompositeTensor(ct::NTuple{M,T}) where {M,T<:TensorKit.AdjointTensorMap}
    funcs = NTuple{M,typeof(adjoint)}(fill(adjoint, M))
    return CompositeTensor(map(parent, ct), funcs)
end

CompositeTensor(ct::NTuple{M,AbstractTensorMap}) where {M} = CompositeTensor(ct...)

function CompositeTensor(args::Vararg{<:AbstractTensorMap})
    unwrap = map(t -> t isa TensorKit.AdjointTensorMap ? parent(t) : t, args)
    conj = map(t -> t isa TensorKit.AdjointTensorMap ? adjoint : identity, args)
    return CompositeTensor(unwrap, conj)
end

function Base.getindex(
    ct::CompositeTensor{M,T,S,N1,N2,A,F},
    ind::Union{Int128,Int16,Int32,Int64,Int8,UInt128,UInt16,UInt32,UInt64,UInt8},
) where {M,T<:Number,S<:ElementarySpace,N1,N2,A,F}
    return ct.funcs[ind](ct.layers[ind])
end
Base.firstindex(ct::CompositeTensor) = 1
Base.lastindex(ct::CompositeTensor) = length(ct)

function Base.iterate(ct::CompositeTensor, state=1)
    state > length(ct) && nothing
    return getindex(ct, state), state + 1
end

Base.similar(ct::CompositeTensor) = CompositeTensor(mapbefore(similar, ct)...)
Base.copy!(dst::CompositeTensor, src::CompositeTensor) = (foreach(copy!, dst, src); dst)

doublelayer(t::TensorMap) = CompositeTensor((t, t))

Base.:(==)(ct1::CompositeTensor, ct2::CompositeTensor) = all(map(==, ct1, ct2))

tensortype(::Type{<:CompositeTensor{M,T,S,N₁,N₂,A}}) where {M,T,S,N₁,N₂,A} = A
TensorKit.spacetype(t::CompositeTensor) = spacetype(tensortype(t))
Base.eltype(t::CompositeTensor) = eltype(typeof(t))
Base.eltype(T::Type{<:CompositeTensor}) = eltype(tensortype(T))

TensorKit.storagetype(t::CompositeTensor) = storagetype(typeof(t))
TensorKit.storagetype(t::Type{<:CompositeTensor}) = storagetype(tensortype(t))

TensorKit.scalartype(T::Type{<:CompositeTensor}) = scalartype(tensortype(T))

ContractableTrait(t::Type{CompositeTensor}) = ContractableTrait(tensortype(t))

## Implement functions for contractable tensors etc
## ContractableTensors need a notion of an east, south, west, north bonds

virtualspace(network::AbstractUnitCell, dir) = virtualspace.(network, dir)

swapaxes(network::AbstractNetwork) = swapaxes.(network)
invertaxes(network::AbstractNetwork) = invertaxes.(network)

## INTERFACE

"""
    virtualspace(t, [dir])

Return an `NTuple{4,<:VectorSpace}` containing the east, south, west, and north vector spaces associated with the respective
bonds, in that order. Used to initialise appropriate algorithm tensors. For custom data types, this function must be 
specified for use in contraction algorithms.
"""
function virtualspace(t)
    e = virtualspace(t, 1)
    s = virtualspace(t, 2)
    w = virtualspace(t, 3)
    n = virtualspace(t, 4)

    return (e, s, w, n)
end

"""
    swapaxes(t)

Swap the "x" and the "y" bonds of an object `t`. For custom data types, this function must be specified for use in 
contraction algorithms.
"""
function swapaxes end

"""
    invertaxes(t)

Invert the "x" and the "y" bonds of an object `t`, that is, south ↔ north and east ↔ west. For custom data types, 
this function must be specified for use in contraction algorithms.
"""
function invertaxes end

virtualspace(t::AbsTen{0,4}, dir) = domain(t, dir)
virtualspace(ct::CompositeTensor, dir::Int) = prod(mapbefore(t -> domain(t, dir), ct))

swapaxes(t::TenAbs{4}) = permutedom(t, (2, 1, 4, 3))
invertaxes(t::TenAbs{4}) = permutedom(t, (3, 4, 1, 2))

swapaxes(ct::CompositeTensor) = CompositeTensor(mapbefore(swapaxes, ct))
invertaxes(ct::CompositeTensor) = CompositeTensor(mapbefore(invertaxes, ct))

ensure_contractable(x) = x

# function ensure_contractable(
#     x::AbstractUnitCell{<:AbstractUnitCellGeometry,<:TensorMap{<:IndexSpace,0,4}}
# )
#     return x
# end
# function ensure_contractable(
#     x::AbstractUnitCell{<:AbstractUnitCellGeometry,<:TensorMap{<:IndexSpace,N,4}}
# ) where {N}
#     return TensorPair.(x)
# end
#
rotate(ct::CompositeTensor, i) = CompositeTensor(mapbefore(t -> rotate(t, i), ct))

"""
    adjoining_bondspace(network::AbstractUnitCell) -> E, S, W, N

Returns a tuple of the east, south, west, and north bond spaces as unit cells shifted one 
position in the opposite cardinal direction resepctively. For example, `N[x,y]` is the same 
as `north(network)[x,y + 1]`.
"""
function adjoining_bondspace(network)
    # east, south, west, north ← bondspace(network)
    return map(circshift, virtualspace(network), ((1, 0), (0, 1), (-1, 0), (0, -1)))
end
