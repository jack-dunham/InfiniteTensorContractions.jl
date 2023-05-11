# Convenience const. 
const AbsTen{N₁,N₂,S} = AbstractTensorMap{S,N₁,N₂}


# Output the permutation of that results in the k'th element of p moved to the
# n'th position 
function nswap(k::Int, n::Int, p::NTuple{N}) where {N}
    if k == n
        return p
    elseif n > k
        return (p[1:(k - 1)]..., p[(k + 1):n]..., p[k]..., p[(n + 1):N]...)::typeof(p)
    else
        return (p[1:(n - 1)]..., p[k], p[n:(k - 1)]..., p[(k + 1):N]...)::typeof(p)
    end
end

# depreciate
function swap(k::Int, l::Int, N::Int)
    if !(k <= N && l <= N)
        throw(ArgumentError("k and l must be less than or equal to N"))
    end
    p = Vector(1:N)
    p[k] = l
    p[l] = k
    return Tuple(p)
end

@doc raw"""
    swap(k::Int, l::Int, p::NTuple)

Given `p`, return the `Tuple` with elements `k` and `l` swapped. Note, `k` and `l` must both 
be less than or equal to `length(p)`. 
"""
function swap(k::Int, l::Int, p::NTuple{N}) where {N}
    if !(k <= N && l <= N)
        throw(ArgumentError("k and l must be less than or equal to N"))
    end
    v = Vector{Int}([p...])
    v[k] = p[l]
    v[l] = p[k]
    return typeof(p)(v)
end

delta(x, y) = ==(x, y)
function delta(z...)
    for i in z
        !delta(z[1], i) && return false
    end
    return true
end

# type stable
function delta(T::Type{<:Number}, dim1::Int, dim2::Int, dims::Vararg{Int,N}) where {N}
    a = Array{T}(undef, dim1, dim2, dims...)
    for (i, _) in pairs(a)
        a[i] = T(delta(i))
    end
    return a
end

delta(I::CartesianIndex) = delta(Tuple(I)...)

function delta(T::Type{<:Number}, cod::VectorSpace, dom::VectorSpace)
    return TensorMap(s -> delta(T, s...), cod, dom)
end

function slice(ind::NTuple{N,Int}, sp::ProductSpace{S,N₁}) where {N,N₁,S<:IndexSpace}
    return *(one(S), one(S), (sp[i] for i in ind)...)
end

westbond(t::AbstractTensorMap) = codomain(t)[1]
southbond(t::AbstractTensorMap) = codomain(t)[2]
downbond(t::AbstractTensorMap) = codomain(t)[3]

eastbond(t::AbstractTensorMap) = domain(t)[1]
northbond(t::AbstractTensorMap) = domain(t)[2]
upbond(t::AbstractTensorMap) = domain(t)[3]

# Alt permute() 
function _permute(t::AbstractTensorMap{<:IndexSpace,N,M}, p::Tuple) where {N,M}
    if !(length(p) == N + M)
        throw(DimensionMismatch())
    else
        return permute(t, p[1:N], p[(N + 1):(N + M)])
    end
end

# Swap codomain and domain (Base.transpose reverses indices)
function _transpose(tsrc::AbsTen{N,M}) where {N,M}
    return permute(tsrc, Tuple((N+1):(N+M))::NTuple{M::Int}, Tuple(1:N)::NTuple{N,Int})
end
