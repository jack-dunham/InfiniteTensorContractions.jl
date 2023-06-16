abstract type AbstractNetwork{G,ElType,A} <: AbstractUnitCell{G,ElType,A} end

abstract type ContractableTrait end
struct IsContractable <: ContractableTrait end
struct NotContractable <: ContractableTrait end

ContractableTrait(::Type{<:AbsTen{0,4}}) = IsContractable()
ContractableTrait(::Type{<:AbsTen{1,4}}) = IsContractable()
ContractableTrait(::Type{<:AbsTen{2,4}}) = IsContractable()
ContractableTrait(::Type) = NotContractable()

struct TensorPair{S,N₁,N₂,T<:AbstractTensorMap{S,N₁,N₂}} <: AbstractTensorMap{S,N₁,N₂}
    top::T
    bot::T
end

top(tp::TensorPair) = tp.top
bot(tp::TensorPair) = tp.bot

tensortype(tp::TensorPair) = tensortype(typeof(tp))
tensortype(::Type{<:TensorPair{S,N₁,N₂,T}}) where {S,N₁,N₂,T} = T
TensorKit.spacetype(t::TensorPair) = spacetype(tensortype(t))
Base.eltype(t::TensorPair) = eltype(typeof(t))
Base.eltype(T::Type{<:TensorPair}) = eltype(tensortype(T))

numbertype(t::TensorPair) = eltype(t)

ContractableTrait(t::Type{TensorPair}) = ContractableTrait(tensortype(t))

"""
    Network{G,ElType<:AbstractTensorMap,A,...}
"""
struct Network{G,ElType<:AbstractTensorMap,A,U<:AbstractUnitCell{G,ElType,A}} <:
       AbstractNetwork{G,ElType,A}
    data::U
end

@inline getdata(network::AbstractNetwork) = network.data

Network(t1::AbstractMatrix, t2::AbstractMatrix) = Network(TensorPair.(t1, t2))

Network(data) = Network{Square}(data)
Network{G}(data) where {G} = Network(UnitCell{G}(data))

Base.convert(::Type{Network}, uc::AbstractUnitCell) = Network(uc)

struct NetworkStyle{G,A} <: AbstractUnitCellStyle{G,A} end

@inline function Base.BroadcastStyle(
    ::Type{<:AbstractNetwork{G,ElType,A}}
) where {G,ElType,A}
    return NetworkStyle{G,typeof(Base.BroadcastStyle(A))}()
end
@inline function Base.BroadcastStyle(
    ::NetworkStyle{G,A}, ::AbstractUnitCellStyle{G,B}
) where {G,A,B}
    return NetworkStyle{G,typeof(Base.BroadcastStyle(A(), B()))}()
end

@inline function Base.similar(
    bc::Broadcast.Broadcasted{NetworkStyle{G,A}}, ::Type{ElType}
) where {G,A,ElType}
    return _similar(ContractableTrait(ElType), bc, ElType)
end

@inline function _similar(
    ::IsContractable, bc::Broadcast.Broadcasted{NetworkStyle{G,A}}, ::Type{ElType}
) where {G,A,ElType}
    return Network{G}(similar(Base.convert(Broadcast.Broadcasted{A}, bc), ElType))
end
@inline function _similar(
    ::NotContractable, bc::Broadcast.Broadcasted{NetworkStyle{G,A}}, ::Type{ElType}
) where {G,A,ElType}
    return similar(Base.convert(Broadcast.Broadcasted{UnitCellStyle{G,A}}, bc), ElType)
end

# ContractableTensors{G,T,A}(data::A) where {G,T,A<:AbstractMatrix{T}} = ContractableTensors{G}(data)

## Implement functions for contractable tensors etc
## ContractableTensors need a notion of an east, south, west, north bonds

function bondspace(network::AbstractNetwork)
    out = tuple((getindex.(bondspace.(network), i) for i in 1:4)...)
    return out
end

swapaxes(network::AbstractNetwork) = swapaxes.(network)
invertaxes(network::AbstractNetwork) = invertaxes.(network)

## INTERFACE

"""
    bondspace(t)

Return an NTuple{4,<:VectorSpace} containing the east, south, west, and north vector spaces associated with the respective
bonds, in that order. Used to initialise appropriate algorithm tensors. For custom data types, this function must be 
specified for use in contraction algorithms.
"""
function bondspace end

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

bondspace(t::AbsTen{0,4}) = tuple(domain(t)...)
function bondspace(tp::T) where {T<:TensorPair}
    return tuple((domain(tp.top)[i] * domain(tp.bot)[i]' for i in 1:4)...)
end

swapaxes(t::TenAbs{4}) = permutedom(t, (2, 1, 4, 3))
invertaxes(t::TenAbs{4}) = permutedom(t, (3, 4, 1, 2))

function swapaxes(tp::TensorPair)
    return TensorPair(swapaxes(tp.top), swapaxes(tp.bot))
end
function invertaxes(tp::TensorPair)
    return TensorPair(invertaxes(tp.top), invertaxes(tp.bot))
end

east(t) = bondspace(t)[1]
south(t) = bondspace(t)[2]
west(t) = bondspace(t)[3]
north(t) = bondspace(t)[4]


ensure_contractable(x) = x
