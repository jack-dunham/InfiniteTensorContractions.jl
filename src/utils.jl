# Alt permute() 
function _permute(t::AbstractTensorMap{<:IndexSpace,N,M}, p::Tuple) where {N,M}
    if !(length(p) == N + M)
        throw(DimensionMismatch())
    else
        return permute(t, p[1:N], p[(N + 1):(N + M)])
    end
end

#depreciate
function nswap(k::Int, n::Int, N::Int)
    if k == n
        return Tuple(1:N)
    elseif n > k
        return (invperm(nswap(n, k, N)))
    else
        return (Tuple(1:(n - 1))..., k, Tuple(n:(k - 1))..., Tuple((k + 1):N)...)
    end
end

# Output the permutation of that results in the k'th element of p moved to the
# n'th position 
function nswap(k::Int, n::Int, p::NTuple{N}) where {N}
    if k == n
        return p
    elseif n > k
        return (p[1:(k - 1)]..., p[(k+1):n]..., p[k]..., p[(n+1):N]...)::typeof(p)
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
function delta(T::Type{<:Number}, dims::NTuple{N,Int}) where {N}
    a = Array{T,N}(undef, dims)
    for (i, x) in pairs(a)
        a[i] = T(delta(i))
    end
    return a
end

delta(T::Type{<:Number}, inddim::Int) = delta(T, inddim, 2)
function delta(T::Type{<:Number}, inddim::Int, indnum::Int)
    δ = Array{T,indnum}(delta(T, ntuple(x -> inddim, indnum)))
    return δ
end

delta(I::CartesianIndex) = delta(Tuple(I)...)

function delta(T::Type{<:Number}, cod::VectorSpace, dom::VectorSpace)
    return TensorMap(delta(T, (dim(cod)..., dim(dom)...)), cod, dom)
end

raise(T::Type{<:Number}, n::Int) = (circshift(delta(T, n), 1))::Matrix{T}
function raise(T::Type{<:Number}, S::Type{<:IndexSpace}, n::Int)
    r = raise(T, n)
    s = oneunit(S)
    sp = ⊕(fill(s, n)...)
    return TensorMap(r, sp, sp)
end
function raise(T::Type{<:Number}, S::IndexSpace)
    r = raise(T, dim(S))
    return TensorMap(r, S, S)
end

lower(T::Type{<:Number}, n::Int) = circshift(delta(T, n), -1)
function lower(T::Type{<:Number}, S::Type{<:IndexSpace}, n::Int)
    l = lower(T, n)
    s = oneunit(S)
    sp = ⊕(fill(s, n)...)
    return TensorMap(l, sp, sp)
end
function lower(T::Type{<:Number}, S::IndexSpace)
    l = lower(T, dim(S))
    return TensorMap(l, S, S)
end

# function vec2tensor(
#     As::TensorMap{
#         <:IndexSpace,
#         N₁,
#         N₂,
#         S,
#         Union{AbstractMatrix{T},TensorKit.SortedVectorDict{S,AbstractMatrix{T}}},
#     },
# ) where {S<:Sector,T<:Number,N₁,N₂}
#     N = length(As)
#     Asa = [convert(Array, a) for a in As]
#     return Ast = Array{T}(undef, 
# end

function _get_sitespace(A::Matrix{<:TensorMap{S}}; conj=false) where {S<:IndexSpace}
    Nx, Ny = size(A)

    if Nx == 1
        spx = oneunit(S)
    else
        spx = ⊕(fill(oneunit(S), Nx)...)
    end

    if Ny == 1
        spy = oneunit(S)
    else
        spy = ⊕(fill(oneunit(S), Ny)...)
    end

    if conj
        return spx' * spy'
    else
        return spx * spy
    end
end

function _tensorize(A::Matrix{<:TensorMap{S}}; conj=false) where {S<:GradedSpace}
    
    sp = _get_sitespace(A; conj=conj)

    t = similar(A[1, 1], codomain(A[1, 1]), domain(A[1, 1]) * sp)

    c = blocks.(A)
    for (s, b) in blocks(t)
        blocks(t)[s] = reduce(hcat, broadcast(x -> getindex(x, s), c))
    end

    return t
end
function _tensorize(A::Matrix{<:TensorMap{S}}; conj=false) where {S<:ComplexSpace}

    sp = _get_sitespace(A; conj=conj)

    c = blocks.(A)
    data = reduce(hcat, broadcast(x -> Base.getindex(x, Trivial()), c))

    t = TensorMap(data, codomain(A[1, 1]), domain(A[1, 1]) * sp)

    return t
end


function Base.convert(::Type{TensorMap}, A::Vector{<:TensorMap})
    return listten2ten(A)
end

function listten2ten(As::Array{<:TensorMap{S}}) where {S}
    N = length(As)
    if N == 1
        sp = oneunit(S)
    else
        sp = ⊕(fill(oneunit(S), N)...)
    end
    data = listten2array(As)
    return TensorMap(data, codomain(As[1]), domain(As[1]) * sp * sp)
end

function listten2array(As::Array{T}) where {T<:TensorMap}
    N = length(As)
    Asa = [Base.convert(Array, a) for a in As]
    dims = size(Asa[1])
    Ast = Array{eltype(T)}(undef, dims..., N, N)#::Array{eltype(T), numind(T)+2}
    return listten2array_fill(Ast, Asa)
end

function listten2array_fill(A::Array{T}, Aa::Vector{B}) where {T<:Number,B<:Array{T}}
    N = length(Aa)
    col = fill(:, ndims(B))
    in = CartesianIndices((1:N, 1:N))
    for i in in
        A[col..., i] = Aa[mod(i[1] + i[2] - 1, 1:N)]
    end
    return A
end

#faster, assumes site index lie at end of domain
function getsite(t::TensorMap{<:IndexSpace,N₁,N₂}, x::Int, y::Int) where {N₁,N₂}
    S = sectortype(t)
    T = eltype(t)
    N = numind(t)

    if S == Trivial
        sec = Trivial()
    else
        sec = S(0)
    end

    xsp = domain(t)[N₂ - 1]
    ysp = domain(t)[N₂]
    δx = Tensor(zeros, T, xsp)
    block(δx, sec)[x] = one(T)
    δy = Tensor(zeros, T, ysp)
    block(δy, sec)[y] = one(T)

    tp = permute(t, NTuple{N - 2,Int64}(1:(N - 2)), (N - 1, N))
    tpp = tp * (δx ⊗ δy)
    return permute(tpp, NTuple{N₁,Int64}(1:N₁), NTuple{N₂ - 2,Int64}((N₁ + 1):(N - 2)))
end

function slice(
    ind::NTuple{N,Int}, sp::ProductSpace{S,N₁}
) where {N,N₁,S<:IndexSpace}
    return *(one(S),one(S),(sp[i] for i in ind)...)
end
