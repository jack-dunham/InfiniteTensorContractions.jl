using CircularArrays
using LinearAlgebra

abstract type AbstractUnitCellGeometry end
abstract type AbstractUnitCell{ElType,G<:AbstractUnitCellGeometry} <: AbstractMatrix{ElType} end

struct Square <: AbstractUnitCellGeometry end

struct UnitCell{ElType,G,A<:AbstractMatrix{ElType}} <: AbstractUnitCell{ElType,G}
    data::CircularArray{ElType,2,A}
    UnitCell{G}(data::CircularArray{T,2,A}) where {T,G,A} = new{T,G,A}(data)
end


@inline getdata(uc::UnitCell) = uc.data

## ABSTRACT ARRAY

@inline Base.size(uc::AbstractUnitCell) = size(getdata(uc))
@inline Base.getindex(uc::AbstractUnitCell, i...) = getindex(getdata(uc), i...)
@inline Base.setindex!(uc::AbstractUnitCell, v, i...) = setindex!(getdata(uc), v, i...)

## SIMILAR

@inline Base.similar(uc::UnitCell{T,G}) where {T,G} = UnitCell{G}(similar(getdata(uc)))
@inline function Base.similar(uc::UnitCell{T,G}, ::Type{S}) where {G,S,T}
    return UnitCell{G}(similar(getdata(uc), S))
end
@inline function Base.similar(uc::UnitCell{T,G}, dims::Dims) where {T,G}
    return UnitCell{G}(similar(getdata(uc), dims))
end
@inline function Base.similar(uc::UnitCell{T,G}, ::Type{S}, dims::Dims) where {T,S,G}
    return UnitCell{G}(similar(getdata(uc), S, dims))
end

## BROADCASTING 

@inline function Base.BroadcastStyle(::Type{<:UnitCell{T,G,A}}) where {T,G,A}
    return Broadcast.ArrayStyle{UnitCell{T,G,A}}()
end

@inline function Base.BroadcastStyle(
    UC::Broadcast.ArrayStyle{<:UnitCell}, ::Broadcast.ArrayStyle{<:CircularArray}
)
    return UC
end
@inline function Base.BroadcastStyle(
    UC::Broadcast.ArrayStyle{<:UnitCell{T,G,<:Transpose{T}}},
    ::Broadcast.ArrayStyle{<:UnitCell},
) where {T,G}
    return UC
end

@inline function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{UnitCell{T,G,A}}}, ::Type{ElType}
) where {T,G,A,ElType}
    return UnitCell{G}(
        similar(
            convert(Broadcast.Broadcasted{typeof(Broadcast.BroadcastStyle(A))}, bc), ElType
        ),
    )
end

@inline function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{UnitCell{T,G,Transpose{T,A}}}},
    ::Type{ElType},
) where {T,G,A,ElType}
    arr = permutedims(
        similar(
            convert(Broadcast.Broadcasted{typeof(Broadcast.BroadcastStyle(A))}, bc), ElType
        ),
    )
    return UnitCell{G}(transpose(arr))
end
    
## CONSTRUCTORS

# Default to sqaure lattice
UnitCell(data) = UnitCell{Square}(data)
# Convert normal matrices into `CircularArray`
UnitCell{G}(data::AbstractMatrix) where {G} = UnitCell{G}(CircularArray(data))


## LINEAR ALGEBRA

function LinearAlgebra.transpose(uc::AbstractUnitCell)
    return UnitCell(CircularArray(transpose(getdata(uc).data)))
end
