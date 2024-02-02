abstract type AbstractCornerMethod <: AbstractBoundaryAlgorithm end

struct Projectors{A<:AbstractUnitCell}
    UL::A
    VL::A
    UR::A
    VR::A
end

Projectors(t::NTuple{4}) = Projectors(t...)

function Base.getindex(p::Projectors, i::Int64)
    tup = (p.UL, p.VL, p.UR, p.VR)
    return tup[i]
end

struct Corners{C<:AbstractUnitCell}
    data::NTuple{4,C}
end

struct Edges{E<:AbstractUnitCell}
    data::NTuple{4,E}
end
struct CornerSingularValues{S<:AbstractUnitCell}
    data::NTuple{4,S}
end

const FourTupleLike{A} = Union{Corners{A},Edges{A},CornerSingularValues{A},Projectors{A}}

(T::Type{<:FourTupleLike})(args::Vararg{A,4}) where {A} = T(args)

Base.length(::FourTupleLike) = 4
Base.getindex(t::FourTupleLike, i::Int) = getindex(t.data, i)
function Base.iterate(t::FourTupleLike, state=1)
    return state > length(t) ? nothing : (t[state], state + 1)
end

Base.convert(::Type{<:Tuple}, t::FourTupleLike{A}) where {A} = NTuple{4,A}(i for i in t)

Base.BroadcastStyle(::Type{<:FourTupleLike{A}}) where {A} = Base.BroadcastStyle(NTuple{4,A})
Base.broadcastable(t::FourTupleLike) = t.data

function Base.map(f, args::Vararg{Union{Tuple,FourTupleLike}})
    return map(f, map(x -> convert(Tuple, x), args)...)
end

function LinearAlgebra.normalize!(t::FourTupleLike)
    foreach(x -> normalize!.(x), t)
    return t
end

TensorKit.scalartype(::Type{<:FourTupleLike{A}}) where {A} = scalartype(A)

struct CornerMethodTensors{C<:Corners,E<:Edges,P<:Projectors,N<:AbstractNetwork}
    corners::C
    edges::E
    projectors::P
    network::N
end

corners(t::CornerMethodTensors) = t.corners
edges(t::CornerMethodTensors) = t.edges

function TensorKit.scalartype(::Type{<:CornerMethodTensors{C,E}}) where {C,E}
    return promote_type(scalartype(C), scalartype(E))
end

struct CornerMethodRuntime{T<:CornerMethodTensors,S<:CornerSingularValues} <:
       AbstractRuntime
    primary::T
    permuted::T
    svals::S
end

corners(t::CornerMethodRuntime) = corners(t.primary)
edges(t::CornerMethodRuntime) = edges(t.primary)

TensorKit.scalartype(::Type{<:CornerMethodRuntime{T}}) where {T} = scalartype(T)

# Return the bond space of the boundary
chispace(c::Corners) = domain(c[1][1, 1])[1]
chispace(c::CornerMethodTensors) = chispace(c.corners)

function step!(problem::InfiniteContraction{<:AbstractCornerMethod})
    fpcm = dofpcm(problem.algorithm)
    error = step!(fpcm, problem)
    return error
end

function getboundary(corners::Corners, i...)
    C1, C2, C3, C4 = corners

    (x, y) = to_indices(C1.data, i)

    x1 = firstindex(x)
    y1 = firstindex(y)
    x2 = lastindex(x)
    y2 = lastindex(y)

    c1 = C1[x1 - 1, y1 - 1]
    c2 = C2[x2 + 1, y1 - 1]
    c3 = C3[x2 + 1, y2 + 1]
    c4 = C4[x1 - 1, y2 + 1]

    return c1, c2, c3, c4
end

function getboundary(edges::Edges, i...)
    T1, T2, T3, T4 = edges

    (x, y) = to_indices(T1.data, i)

    x1 = firstindex(x)
    y1 = firstindex(y)
    x2 = lastindex(x)
    y2 = lastindex(y)

    @views begin
        t1s = T1[x1:x2, y1 - 1]
        t2s = T2[x2 + 1, y1:y2]
        t3s = T3[x1:x2, y2 + 1]
        t4s = T4[x1 - 1, y1:y2]
    end

    return t1s, t2s, t3s, t4s
end

function updatecorners!(tensors::CornerMethodTensors, tensors_permuted::CornerMethodTensors)
    updatecorners!(tensors.corners, tensors_permuted.corners)
    return tensors
end

function updatecorners!(corners::C, corners_permuted::C) where {C<:Corners}
    C1, C2, C3, C4 = corners_permuted

    # Between primary and permuted corners, C4 and C2 have swapped positions
    foreach(corners, (C1, C4, C3, C2)) do c1, c2
        # The unit cell is also transposed...
        broadcast(c1, permutedims(c2)) do t1, t2
            permutedom!(t1, t2, (2, 1))
        end
    end

    return corners
end

function ctmerror!(runtime::CornerMethodRuntime)
    return ctmerror!(runtime.svals, runtime.primary.corners)
end

function ctmerror!(svals::CornerSingularValues, corners::Corners)
    errors = map(boundaryerror!, svals, corners)

    foreach(enumerate(errors)) do (i, error)
        @debug "Corner matrix convergence: C$(i):" error
    end

    return maximum(maximum.(errors))
end
