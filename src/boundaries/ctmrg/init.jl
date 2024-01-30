function inittensors(network, alg::AbstractCornerMethod; randinit=alg.randinit)
    # Convert the bond dimension into an IndexSpace
    chi = dimtospace(spacetype(network), alg.bonddim)

    corners = initcorners(network, chi, Val(randinit))
    edges = initedges(network, chi, Val(randinit))

    return inittensors(corners, edges, alg)
end

inittensors(corners::Corners, edges::Edges, ::CTMRG) = CTMRGTensors(corners, edges)

function initpermuted(ctmrg::CTMRGTensors)
    C1_t, C2_t, C3_t, C4_t = map(permutedims, unpack(ctmrg.corners))

    pm(x) = map(i -> permute(i, ((), (2, 1))), x)

    C1_tp = pm(C1_t)
    C2_tp = pm(C2_t)
    C3_tp = pm(C3_t)
    C4_tp = pm(C4_t)

    corners_p = Corners(C1_tp, C4_tp, C3_tp, C2_tp)

    T1_t, T2_t, T3_t, T4_t = map(permutedims, unpack(ctmrg.edges))

    edges_p = Edges(T4_t, T3_t, T2_t, T1_t)

    return CTMRGTensors(corners_p, edges_p)
end

function initprojectors(network, chi::IndexSpace)
    _, bot_bonds, _, top_bonds = bondspace(network)

    T = eltype(network.data[1, 1])

    projectors = _initprojectors(T, top_bonds, bot_bonds, chi)

    return projectors::Projectors
end

function _initprojectors(T::Type{<:Number}, top_bonds, bot_bonds, chi::IndexSpace)
    chi = Ref(chi)

    UL = @. TensorMap(
        undef, T, adjoint(chi) * $circshift(top_bonds, (-1, -1)), adjoint(chi)
    )
    VL = @. TensorMap(undef, T, chi * $circshift(bot_bonds, (-1, 0)), chi)

    UR = @. TensorMap(undef, T, chi * $circshift(top_bonds, (1, -1)), chi)
    VR = @. TensorMap(undef, T, adjoint(chi) * $circshift(bot_bonds, (1, 0)), adjoint(chi))

    return Projectors(UL, VL, UR, VR)
end

function initcorners(network, chi::S, v::Val) where {S<:IndexSpace}
    C1, C2, C3, C4 = _initcorners(network, chi, v)

    c = Corners(C1, C2, C3, C4)

    randomize_if_zero!(c)

    return c
end

function _initcorners(network, chi::S, ::Val{true}) where {S<:IndexSpace}
    fl = let network = network, chi = chi, T = scalartype(network)
        c -> broadcast(_ -> TensorMap(randn, T, one(chi), c * c), network)
    end

    C1 = fl(chi)
    C2 = fl(adjoint(chi))
    C3 = fl(chi)
    C4 = fl(adjoint(chi))

    return C1, C2, C3, C4
end

function _initcorners(network, chi::S, ::Val{false}) where {S<:IndexSpace}
    C1, C2, C3, C4 = map(1:4) do i
        return broadcast(network) do tensor
            return init_single_corner(tensor, chi, i)
        end
    end
    return C1, C2, C3, C4
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
    _init_single_corner!(c, tensor, ue, us, uw, un)
    return c
end

function _init_single_corner!(t_dst, t_src::TensorMap{<:IndexSpace,0,4}, ue, us, uw, un)
    @tensoropt t_dst[o1 o2] = t_src[e s w n] * ue[e; o1] * us[s; o2] * uw[w] * un[n]
    return t_dst
end

function _init_single_corner!(t_dst, t_src::TensorPair, ue, us, uw, un)
    return _init_single_corner!(t_dst, top(t_src), bot(t_src), ue, us, uw, un)
end

function _init_single_corner!(
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

function _init_single_corner!(
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

function get_embedding_isometry(bond, chi)
    if bond ≾ chi
        iso = transpose(isometry(chi', bond'))
    else
        iso = isometry(bond, chi)
    end
end
get_removal_isometry(bond) = get_embedding_isometry(bond, one(bond))

function initedges(network, chi::IndexSpace, ::Val{true})
    dom = Ref(chi * chi')

    el = numbertype(network)

    east_bonds, south_bonds, west_bonds, north_bonds = bondspace(network)

    T1 = @. TensorMap(randn, el, $circshift(north_bonds, (0, -1)), dom)
    T2 = @. TensorMap(randn, el, $circshift(east_bonds, (-1, 0)), dom)
    T3 = @. TensorMap(randn, el, $circshift(south_bonds, (0, 1)), dom)
    T4 = @. TensorMap(randn, el, $circshift(west_bonds, (1, 0)), dom)

    return Edges(T1, T2, T3, T4)
end

function initedges(network, chi::IndexSpace, _::Val{false}=Val(false))
    edges = map(i -> init_single_edge.(network, Ref(chi), i), 1:4)
    return randomize_if_zero!(Edges(edges...))
end
function init_single_edge(ten::AbstractTensorMap, chi, i)
    tenp = rotate(ten, -i + 1)

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

    return _init_single_edge!(t_dst, tensor, ue, us, uw, un)
end

function _init_single_edge!(t_dst, t_src::TensorMap{<:IndexSpace,0,4}, ue, us, uw, un)
    @tensoropt t_dst[ss; o1 o2] = t_src[e s w n] * ue[e; o1] * us[ss; s] * uw[w; o2] * un[n]
end

function _init_single_edge!(t_dst, t_src::TensorPair, ue, us, uw, un)
    return _init_single_edge!(t_dst, top(t_src), bot(t_src), ue, us, uw, un)
end

function _init_single_edge!(
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
function _init_single_edge!(
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

function initerror(ctmrg::CTMRGTensors)
    tsvd_sing_val = x -> tsvd(x)[2]
    # Permute into some form compatible with tsvd
    S1, S2, S3, S4 = map(
        x -> tsvd_sing_val.(permute.(x, Ref((1,)), Ref((2,)))), unpack(ctmrg.corners)
    )

    # Set these matrices to equal the singular values of the ctms
    # ctmerror!(S1, S2, S3, S4, ctmrg)

    return S1, S2, S3, S4
end

randomize_if_zero!(corners::C) where {C<:Corners} = _randomize_if_zero!(corners, :C)::C
randomize_if_zero!(edges::E) where {E<:Edges} = _randomize_if_zero!(edges, :T)::E

function _randomize_if_zero!(corners_or_edges, type)
    all_tensors = unpack(corners_or_edges)

    for i in 1:4
        @inbounds unit_cell = all_tensors[i]
        map(x -> _randomize_if_zero!(unit_cell, type, i, x), CartesianIndices(unit_cell))
    end

    return corners_or_edges
end
function _randomize_if_zero!(uc, type, i, ind)
    ten = uc[ind]
    if ten ≈ zero(ten)
        @info "Initial tensor $type$i ≈ 0 at site $(Tuple(ind)); using a random tensor instead."
        copy!(ten, TensorMap(randn, eltype(ten), codomain(ten), domain(ten)))
    end
    return uc
end
