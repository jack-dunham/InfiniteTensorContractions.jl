function initialize(algorithm::AbstractCornerMethod, network; kwargs...)
    if size(network) > (1, 1) && isa(algorithm, FPCM)
        println(
            "The FPCM for non-trivial unit cells is experimental and currently not working"
        )
    end

    primary_tensors = inittensors(network, algorithm; kwargs...)
    permuted_tensors = initpermuted(primary_tensors)
    svals = initerror(primary_tensors)
    return CornerMethodRuntime(primary_tensors, permuted_tensors, svals)
end

function inittensors(network, alg::AbstractCornerMethod; randinit=alg.randinit)
    # Convert the bond dimension into an IndexSpace
    chi = dimtospace(spacetype(network), alg.bonddim)

    corners = initcorners(network, chi; randinit=randinit)
    edges = initedges(network, chi; randinit=randinit)
    projectors = initprojectors(network, chi)

    return CornerMethodTensors(corners, edges, projectors, network)
end

function initpermuted(tensors::CornerMethodTensors)
    chi = chispace(tensors)

    # First swap the horizontal and vertical axes:
    transposed_corners = map(permutedims, tensors.corners)
    transposed_edges = map(permutedims, tensors.edges)

    # Then permute the indices of each tensor on the lattice:
    cs = map(transposed_corners) do c
        broadcast(c) do t
            return permute(t, ((), (2, 1)))
        end
    end

    # The order of the tensors needs adjusted such that they appear in the correct place
    corners = Corners(cs[1], cs[4], cs[3], cs[2])
    edges = Edges(reverse(transposed_edges))

    # Need to swap the axes bonds and transpose the network
    network = permutedims(swapaxes(tensors.network))

    # Projectors are easiest to construct assuming a transposed and permuted unit cell 
    projectors = initprojectors(network, chi)

    return CornerMethodTensors(corners, edges, projectors, network)
end

## CORNERS 

function initcorners(network, chi::S; randinit::Bool=false) where {S<:IndexSpace}
    corner_tensors = map((1:4...,)) do i
        broadcast(network) do site
            return init_single_corner(site, chi, i)
        end
    end

    # Randomize in place 
    if randinit
        map(corner_tensors) do Ci
            broadcast(randnt!, Ci)
        end
    end

    corners = Corners(corner_tensors)

    return randomize_if_zero!(corners)
end
function init_single_corner(ten::AbstractTensorMap, chi, i)
    chis = tcircshift((chi, chi', chi, chi'), -i + 1)

    tenp = rotate(ten, -i + 1)

    d = bondspace(tenp)

    u1 = get_embedding_isometry(d[1], chis[1])
    u2 = get_embedding_isometry(d[2], chis[1])
    u3 = get_removal_isometry(d[3])
    u4 = get_removal_isometry(d[4])

    corner = init_single_corner(tenp, u1, u2, u3, u4)

    return corner
end

function init_single_corner(tensor::AbstractTensorMap, ue, us, uw, un)
    s_o1 = domain(ue)
    s_o2 = domain(us)
    c = TensorMap(undef, scalartype(tensor), one(s_o1), s_o1 * s_o2)
    init_single_corner!(c, tensor, ue, us, uw, un)
    return c
end

function init_single_corner!(t_dst, t_src::TensorPair, ue, us, uw, un)
    return __init_single_corner!(t_dst, top(t_src), bot(t_src), ue, us, uw, un)
end
function init_single_corner!(t_dst, t_src::TensorMap, ue, us, uw, un)
    return __init_single_corner!(t_dst, t_src, ue, us, uw, un)
end

function __init_single_corner!(t_dst, t_src::TensorMap{<:IndexSpace,0,4}, ue, us, uw, un)
    @tensoropt t_dst[o1 o2] = t_src[e s w n] * ue[e; o1] * us[s; o2] * uw[w] * un[n]
    return t_dst
end

function __init_single_corner!(
    t_dst, t1::T, t2::T, ue, us, uw, un
) where {T<:TensorMap{<:IndexSpace,1,4}}
    @tensoropt t_dst[o1 o2] =
        t1[k; e1 s1 w1 n1] *
        (t2')[e2 s2 w2 n2; k] *
        ue[e1 e2; o1] *
        us[s1 s2; o2] *
        uw[w1 w2] *
        un[n1 n2]
    return t_dst
end

function __init_single_corner!(
    t_dst, t1::T, t2::T, ue, us, uw, un
) where {T<:TensorMap{<:IndexSpace,2,4}}
    @tensoropt t_dst[o1 o2] =
        t1[k b; e1 s1 w1 n1] *
        (t2')[e2 s2 w2 n2; k b] *
        ue[e1 e2; o1] *
        us[s1 s2; o2] *
        uw[w1 w2] *
        un[n1 n2]
    return t_dst
end

##

## EDGES

function initedges(network, chi::IndexSpace; randinit::Bool=false)
    edge_tensors = map((1:4...,)) do i
        broadcast(network) do site
            return init_single_edge(site, chi, i)
        end
    end

    # Randomize in place 
    if randinit
        map(edge_tensors) do Ei
            broadcast(randnt!, Ei)
        end
    end

    edges = Edges(edge_tensors)

    return randomize_if_zero!(edges)
end

function init_single_edge(tensor::AbstractTensorMap, chi, i)
    tenp = rotate(tensor, -i + 1)

    d = bondspace(tenp)

    u1 = get_embedding_isometry(d[1], chi)
    u2 = isometry(swap(d[2]), swap(d[2]))
    u3 = get_embedding_isometry(d[3], chi')
    u4 = get_removal_isometry(d[4])

    edge = init_single_edge(tenp, u1, u2, u3, u4)

    return edge
end

function init_single_edge(tensor, ue, us, uw, un)
    s_o1 = domain(ue)
    s_o2 = domain(uw)
    d = swap(bondspace(tensor)[2]) # swapped south bond (north bond of tensor below)

    t_dst = TensorMap(undef, scalartype(tensor), d, s_o1 * s_o2)

    return init_single_edge!(t_dst, tensor, ue, us, uw, un)
end

function init_single_edge!(t_dst, t_src::TensorPair, ue, us, uw, un)
    return __init_single_edge!(t_dst, top(t_src), bot(t_src), ue, us, uw, un)
end
function init_single_edge!(t_dst, t_src::TensorMap, ue, us, uw, un)
    return __init_single_edge!(t_dst, t_src, ue, us, uw, un)
end

function __init_single_edge!(t_dst, t_src::TensorMap{<:IndexSpace,0,4}, ue, us, uw, un)
    @tensoropt t_dst[ss; o1 o2] = t_src[e s w n] * ue[e; o1] * us[ss; s] * uw[w; o2] * un[n]
end
function __init_single_edge!(
    t_dst, t1::T, t2::T, ue, us, uw, un
) where {T<:TensorMap{<:IndexSpace,1,4}}
    @tensoropt t_dst[ss1 ss2; o1 o2] =
        t1[k; e1 s1 w1 n1] *
        (t2')[e2 s2 w2 n2; k] *
        ue[e1 e2; o1] *
        us[ss1 ss2; s1 s2] *
        uw[w1 w2; o2] *
        un[n1 n2]
end
function __init_single_edge!(
    t_dst, t1::T, t2::T, ue, us, uw, un
) where {T<:TensorMap{<:IndexSpace,2,4}}
    @tensoropt t_dst[ss1 ss2; o1 o2] =
        t1[k b; e1 s1 w1 n1] *
        (t2')[e2 s2 w2 n2; k b] *
        ue[e1 e2; o1] *
        us[ss1 ss2; s1 s2] *
        uw[w1 w2; o2] *
        un[n1 n2]
end

##

## PROJECTORS

function initprojectors(network, chi::IndexSpace)
    _, south_bonds, _, north_bonds = bondspace(network)

    T = scalartype(network)

    UL = construct_projector(T, adjoint(chi), north_bonds, (-1, -1))
    VL = construct_projector(T, chi, south_bonds, (-1, 0))
    UR = construct_projector(T, chi, north_bonds, (1, -1))
    VR = construct_projector(T, adjoint(chi), south_bonds, (1, 0))

    return Projectors(UL, VL, UR, VR)
end

function construct_projector(T, chi, sitebonds, incr)
    rv = broadcast(circshift(sitebonds, incr)) do bonds
        return TensorMap(undef, T, chi * bonds, chi)
    end
    return rv
end

##

## ERROR

initerror(ctm::CornerMethodTensors) = initerror(ctm.corners)
function initerror(corners::Corners)
    # Permute into some form compatible with tsvd
    svals = map(corners) do corn
        broadcast(corn) do site
            _, rv, _ = tsvd(permute(site, ((1,), (2,))))
            return rv
        end
    end

    return CornerSingularValues(svals)
end

##

## UTILS

randomize_if_zero!(corners::Corners) = randomize_if_zero!(corners, :C)
randomize_if_zero!(edges::Edges) = randomize_if_zero!(edges, :T)

function randomize_if_zero!(corners_or_edges, type::Symbol)
    for (i, c_or_e) in enumerate(corners_or_edges)
        map(x -> randomize_if_zero!(c_or_e, type, i, x), CartesianIndices(c_or_e))
    end

    return corners_or_edges
end
function randomize_if_zero!(uc, type, i, ind)
    ten = uc[ind]
    if ten ≈ zero(ten)
        @info "Initial tensor $type$i ≈ 0 at site $(Tuple(ind)); using a random tensor instead."
        randnt!(ten)
    end
    return uc
end

##
