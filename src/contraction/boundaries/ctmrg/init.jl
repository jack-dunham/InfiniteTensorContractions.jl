function inittensors(
    f, bulk::ContractableTensors, alg::BoundaryAlgorithm{Alg}
) where {Alg<:AbstractCornerBoundary}
    # Convert the bond dimension into an IndexSpace
    chi = dimtospace(bulk, alg.bonddim)

    return inittensors(f, Alg, bulk, chi)
end
function inittensors(
    f, ::Type{<:CTMRG}, bulk::ContractableTensors, bonddim::IndexSpace; kwargs...
)
    corners = initcorners(f, bulk, bonddim)
    edges = initedges(f, bulk, bonddim)

    return CTMRG(corners, edges; kwargs...)
end

function initpermuted(ctmrg::CTMRG)
    C1_t, C2_t, C3_t, C4_t = transpose.(unpack(ctmrg.corners))
    C1_tp = permute.(C1_t, Ref(()), Ref((2, 1)))
    C3_tp = permute.(C3_t, Ref(()), Ref((2, 1)))

    corners_p = Corners(C1_tp, C4_t, C3_tp, C2_t)

    T1_t, T2_t, T3_t, T4_t = transpose.(unpack(ctmrg.edges))

    edges_p = Edges(T4_t, T3_t, T2_t, T1_t)

    return CTMRG(corners_p, edges_p)
end

function initprojectors(bulk, chi::IndexSpace)
    _, bot_bonds, _, top_bonds = _bondspaces(bulk)

    T = numbertype(bulk)

    projectors = _initprojectors(T, top_bonds, bot_bonds, chi)

    return projectors::Projectors
end

function _initprojectors(T::Type{<:Number}, top_bonds, bot_bonds, chi::IndexSpace)
    chi = Ref(chi)
    LType = latticetype(typeof(top_bonds))

    UL = @. TensorMap(
        undef, T, adjoint(chi) * $circshift(top_bonds, (-1, -1)), adjoint(chi)
    )
    VL = @. TensorMap(undef, T, chi * $circshift(bot_bonds, (-1, 0)), chi)

    UR = @. TensorMap(undef, T, chi * $circshift(top_bonds, (1, -1)), chi)
    VR = @. TensorMap(undef, T, adjoint(chi) * $circshift(bot_bonds, (1, 0)), adjoint(chi))

    return Projectors(UL, VL, UR, VR)::Projectors{LType}
end

function initcorners(f, bulk::ContractableTensors{L}, chi::IndexSpace) where {L}
    nil = one(chi)
    nil = Ref(nil)

    el = numbertype(bulk)

    chi_lat = fill(chi, lattice(bulk))

    C1 = @. TensorMap(f, el, nil, chi_lat * chi_lat)
    C2 = @. TensorMap(f, el, nil, adjoint(chi_lat) * adjoint(chi_lat))
    C3 = @. TensorMap(f, el, nil, chi_lat * chi_lat)
    C4 = @. TensorMap(f, el, nil, adjoint(chi_lat) * adjoint(chi_lat))

    return Corners(C1, C2, C3, C4)
end
function initedges(f, bulk, chi::IndexSpace)
    dom = Ref(chi * chi')

    el = numbertype(bulk)

    east_bonds, south_bonds, west_bonds, north_bonds = _bondspaces(bulk)

    T1 = @. TensorMap(f, el, $circshift(north_bonds, (0, -1)), dom)
    T2 = @. TensorMap(f, el, $circshift(east_bonds, (-1, 0)), dom)
    T3 = @. TensorMap(f, el, $circshift(south_bonds, (0, 1)), dom)
    T4 = @. TensorMap(f, el, $circshift(west_bonds, (1, 0)), dom)

    return Edges(T1, T2, T3, T4)
end

function initerror(ctmrg::CTMRG)
    temp_tsvd = x -> tsvd(x)[2]
    # Permute into some form compatible with tsvd
    S1, S2, S3, S4 = map(
        x -> temp_tsvd.(permute.(x, Ref((1,)), Ref((2,)))), unpack(ctmrg.corners)
    )

    # Set these matrices to equal the singular values of the ctms
    # ctmerror!(S1, S2, S3, S4, ctmrg)

    return S1, S2, S3, S4
end
