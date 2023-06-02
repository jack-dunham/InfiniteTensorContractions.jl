abstract type AbstractUnitCellGeometry end
abstract type AbstractUnitCell{G<:AbstractUnitCellGeometry,ElType} <: AbstractMatrix{ElType} end

struct Square <: AbstractUnitCellGeometry end

struct UnitCell{G<:AbstractUnitCellGeometry,ElType,A} <: AbstractUnitCell{G,ElType}
    data::CircularArray{ElType,2,A}
    UnitCell{G,T,A}(data::CircularArray{T,2,A}) where {G,T,A} = new{G,T,A}(data)
end

@inline getdata(uc::UnitCell) = uc.data

Base.convert(U::Type{<:AbstractUnitCell}, data::AbstractMatrix) = U(data)
Base.convert(U::Type{<:AbstractUnitCell}, uc::AbstractUnitCell) = U(getdata(uc))

## ABSTRACT ARRAY

@inline Base.size(uc::AbstractUnitCell) = size(getdata(uc))
@inline Base.getindex(uc::AbstractUnitCell, i...) = getindex(getdata(uc), i...)
@inline Base.setindex!(uc::AbstractUnitCell, v, i...) = setindex!(getdata(uc), v, i...)

## SIMILAR

@inline Base.similar(uc::UnitCell{G,T}) where {G,T} = UnitCell{G}(similar(getdata(uc)))
@inline function Base.similar(uc::UnitCell{G,T}, ::Type{S}) where {G,S,T}
    return UnitCell{G}(similar(getdata(uc), S))
end
@inline function Base.similar(uc::UnitCell{G,T}, dims::Dims) where {T,G}
    return UnitCell{G}(similar(getdata(uc), dims))
end
@inline function Base.similar(uc::UnitCell{G,T}, ::Type{S}, dims::Dims) where {T,S,G}
    return UnitCell{G}(similar(getdata(uc), S, dims))
end

## BROADCASTING
abstract type AbstractUnitCellStyle{G,T} <: Broadcast.AbstractArrayStyle{2} end
struct UnitCellStyle{G,T,A} <: AbstractUnitCellStyle{G,T} end

(T::Type{<:AbstractUnitCellStyle})(::Val{2}) = T()
(T::Type{<:AbstractUnitCellStyle})(::Val{N}) where {N} = Broadcast.DefaultArrayStyle{N}()

# @inline function Base.BroadcastStyle(::Type{<:UnitCell{T,G,A}}) where {T,G,A}
#     return Broadcast.ArrayStyle{UnitCell{T,G,A}}()
# end

@inline function Base.BroadcastStyle(::Type{UnitCell{G,ElType,A}}) where {G,ElType,A}
    return UnitCellStyle{G, ElType, A}()
end

@inline function Base.BroadcastStyle(
    U::UnitCellStyle, ::Broadcast.ArrayStyle{<:CircularArray}
) 
    return U
end

@inline function Base.similar(
    bc::Broadcast.Broadcasted{UnitCellStyle{G,T,A}}, ::Type{ElType}
) where {G,T,A,ElType}
    return UnitCell{G}(
        similar(
            convert(Broadcast.Broadcasted{typeof(Broadcast.BroadcastStyle(A))}, bc), ElType
        ),
    )
end

## CONSTRUCTORS

# Make circular
UnitCell(data) = UnitCell{Square}(data)
UnitCell{G}(data::A) where {G,T,A<:AbstractMatrix{T}} = UnitCell{G,T,A}(data)
UnitCell{G,T,A}(data::A) where {G,T,A<:AbstractMatrix{T}} = UnitCell{G,T,A}(CircularArray(data))
# Convert normal matrices into `CircularArray`

## LINEAR ALGEBRA

function LinearAlgebra.transpose(uc::AbstractUnitCell)
    return UnitCell(CircularArray(transpose(getdata(uc).data)))
end

## VIEW

@inline function Base.view(uc::AbstractUnitCell, i1, i2, inds...)
    new_inds = (int_to_range(i1, i2)..., inds...)
    return UnitCell(view(getdata(uc).data, new_inds...))
end

_unitrange(i::Int) = UnitRange(i, i)
_unitrange(x) = x

int_to_range(i1, i2) = (_unitrange(i1), _unitrange(i2))

## UTILS

size_allequal(ucs...) = allequal(size.(ucs))
function check_size_allequal(ucs...)
    if !size_allequal(ucs)
        throw(DimensionMismatch("Unit cells provided do not have same dimensionality"))
    end
    return nothing
end

## TENSORS

TensorKit.spacetype(::AbstractUnitCell{G,<:AbstractTensorMap{S}}) where {G,S} = S
numbertype(::AbstractUnitCell{G,T}) where {G,T} = eltype(T)

