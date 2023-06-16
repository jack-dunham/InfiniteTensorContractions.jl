abstract type AbstractUnitCellGeometry end
abstract type AbstractUnitCell{G<:AbstractUnitCellGeometry,ElType,A} <:
              AbstractMatrix{ElType} end

const AbUnCe{T,G,A} = AbstractUnitCell{G,T,A}

struct Square <: AbstractUnitCellGeometry end

struct UnitCell{G<:AbstractUnitCellGeometry,ElType,A} <: AbstractUnitCell{G,ElType,A}
    data::CircularArray{ElType,2,A}
    UnitCell{G,T,A}(data::CircularArray{T,2,A}) where {G,T,A} = new{G,T,A}(data)
    UnitCell{G}(data::CircularArray{T,2,A}) where {G,T,A} = new{G,T,A}(data)
end

@inline getdata(uc::UnitCell{G,T,A}) where {G,T,A} = uc.data::CircularArray{T,2,A}

@inline datatype(U::Type{<:AbstractUnitCell{G,ElType,A}}) where {G,ElType,A} = A
@inline datatype(U::Type) = U

## ABSTRACT ARRAY

@inline Base.size(uc::AbstractUnitCell) = size(getdata(uc))
@inline Base.getindex(uc::AbstractUnitCell, i...) = getindex(getdata(uc), i...)
@inline Base.setindex!(uc::AbstractUnitCell, v, i...) = setindex!(getdata(uc), v, i...)

## SIMILAR

@inline function Base.similar(uc::AbstractUnitCell{G,T}) where {G,T}
    return UnitCell{G}(similar(getdata(uc)))
end
@inline function Base.similar(uc::AbstractUnitCell{G,T}, ::Type{S}) where {G,S,T}
    return UnitCell{G}(similar(getdata(uc), S))
end
@inline function Base.similar(uc::AbstractUnitCell{G,T}, dims::Dims) where {T,G}
    return UnitCell{G}(similar(getdata(uc), dims))
end
@inline function Base.similar(
    uc::AbstractUnitCell{G,T}, ::Type{S}, dims::Dims
) where {T,S,G}
    return UnitCell{G}(similar(getdata(uc), S, dims))
end

## BROADCASTING

# Pass in the `BroadcastStyle` such that we can get compute the winning broadcast style
# of the underlying abstract matrix.

abstract type AbstractUnitCellStyle{G,A<:Base.BroadcastStyle} <:
              Broadcast.AbstractArrayStyle{2} end

struct UnitCellStyle{G,A} <: AbstractUnitCellStyle{G,A} end

(T::Type{<:AbstractUnitCellStyle})(::Val{2}) = T()
(T::Type{<:AbstractUnitCellStyle})(::Val{N}) where {N} = Broadcast.DefaultArrayStyle{N}()

@inline function Base.BroadcastStyle(::Type{UnitCell{G,ElType,A}}) where {G,ElType,A}
    return UnitCellStyle{G,typeof(Base.BroadcastStyle(A))}()
end

@inline function Base.BroadcastStyle(
    ::UnitCellStyle{G,A}, ::Broadcast.ArrayStyle{<:CircularArray{ElType,2,B}}
) where {G,ElType,A,B}
    AB = Base.BroadcastStyle(A(), Base.BroadcastStyle(B))
    return UnitCellStyle{G,typeof(AB)}()
end

@inline function Base.similar(
    bc::Broadcast.Broadcasted{UnitCellStyle{G,A}}, ::Type{ElType}
) where {G,A,ElType}
    return UnitCell{G}(similar(Base.convert(Broadcast.Broadcasted{A}, bc), ElType))
end

## CONSTRUCTORS

tocircular(data::AbstractMatrix) = CircularArray(data)
tocircular(data::CircularArray) = data
tocircular(data::AbstractUnitCell) = getdata(data)

# Make circular
UnitCell(data) = UnitCell{Square}(data)
UnitCell{G}(data) where {G} = UnitCell{G}(tocircular(data))

## VIEW (Probably deprec)

@inline function Base.view(uc::AbstractUnitCell, i1, i2, inds...)
    # @debug "Using UnitCell view..."
    new_inds = (int_to_range(i1, i2)..., inds...)
    return UnitCell(view(getdata(uc), new_inds...))
end

const SubUnitCell{G,T,A<:SubArray} = UnitCell{G,T,A}

_unitrange(i::Int) = UnitRange(i, i)
_unitrange(x) = x

int_to_range(i1, i2) = (_unitrange(i1), _unitrange(i2))

## UTILS

size_allequal(ucs...) = allequal(size.(ucs...))
function check_size_allequal(ucs...)
    if !size_allequal(ucs)
        throw(DimensionMismatch("Unit cells provided do not have same dimensionality"))
    end
    return nothing
end

## TENSORS

TensorKit.spacetype(::AbstractUnitCell{G,<:AbstractTensorMap{S}}) where {G,S} = S
numbertype(::AbstractUnitCell{G,T}) where {G,T<:AbstractTensorMap} = eltype(T)
