abstract type AbstractContractableTensors{G,ElType,A} <: AbstractUnitCell{G,ElType,A} end

abstract type ContractableTrait end
struct IsContractable <: ContractableTrait end
struct NotContractable <: ContractableTrait end

ContractableTrait(::Type{<:AbsTen{0,4}}) = IsContractable()
ContractableTrait(::Type{<:AbsTen{1,4}}) = IsContractable()
ContractableTrait(::Type{<:AbsTen{2,4}}) = IsContractable()
ContractableTrait(::Type) = NotContractable()


struct ContractableTensors{G,ElType,A,U<:AbstractUnitCell{G,ElType,A}} <: AbstractContractableTensors{G,ElType,A}
    data::U
end

ContractableTensors(data) = ContractableTensors{Square}(data)
ContractableTensors{G}(data) where {G} = ContractableTensors(UnitCell{G}(data))

@inline getdata(ct::ContractableTensors) = ct.data

struct ContractableTensorsStyle{G,A} <: AbstractUnitCellStyle{G,A} end

@inline function Base.BroadcastStyle(::Type{<:ContractableTensors{G,ElType, A}}) where {G,ElType, A} 
    ContractableTensorsStyle{G, typeof(Base.BroadcastStyle(A))}()
end
@inline function Base.BroadcastStyle(::ContractableTensorsStyle{G,A}, ::AbstractUnitCellStyle{G,B}) where {G, A, B} 
    ContractableTensorsStyle{G, typeof(Base.BroadcastStyle(A(),B()))}()
end

@inline function Base.similar(bc::Broadcast.Broadcasted{ContractableTensorsStyle{G,A}}, ::Type{ElType}) where {G,A,ElType}
    return _similar(ContractableTrait(ElType), bc, ElType)
end


@inline function _similar(::IsContractable, bc::Broadcast.Broadcasted{ContractableTensorsStyle{G,A}}, ::Type{ElType}) where {G,A,ElType}
    return ContractableTensors{G}(similar(Base.convert(Broadcast.Broadcasted{A}, bc), ElType))
end
@inline function  _similar(::NotContractable, bc::Broadcast.Broadcasted{ContractableTensorsStyle{G,A}}, ::Type{ElType}) where {G,A,ElType} 
    return similar(Base.convert(Broadcast.Broadcasted{UnitCellStyle{G,A}}, bc),   ElType)
end

# ContractableTensors{G,T,A}(data::A) where {G,T,A<:AbstractMatrix{T}} = ContractableTensors{G}(data)

## Implement functions for contractable tensors etc
## ContractableTensors need a notion of an east, south, west, north bonds

function bondspace(ct::ContractableTensors)
    out = tuple((getindex.(bondspace.(ct), i) for i in 1:4)...)
    return out
end

swapaxes(ct::ContractableTensors) = swapaxes.(ct)
invertaxes(ct::ContractableTensors) = invertaxes.(ct)


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
bondspace(t::Union{AbsTen{1,4},AbsTen{2,4}}) = tuple((domain(t)[i] * domain(t)[i]' for i in 1:4)...)

swapaxes(t::TenAbs{4}) = permutedom(t, (2, 1, 4, 3))
invertaxes(t::TenAbs{4}) = permutedom(t, (3, 4, 1, 2))

