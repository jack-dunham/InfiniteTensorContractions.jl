abstract type AbstractCornerMethod <: AbstractBoundaryAlgorithm end

@kwdef struct CTMRG{SVD<:OrthogonalFactorizationAlgorithm} <: AbstractCornerMethod
    bonddim::Int
    verbosity::Int = 0
    maxiter::Int = 100
    tol::Float64 = 1e-12
    breaktol::Float64 = 0.0
    ptol::Float64 = 1e-7
    svd_alg::SVD = TensorKit.SVD()
end

abstract type AbstractCornerBoundary <: AbstractBoundaryTensors end

corners(ctm::AbstractCornerBoundary) = ctm.corners
edges(ctm::AbstractCornerBoundary) = ctm.edges

struct Corners{A<:AbstractUnitCell}
    C1::A
    C2::A
    C3::A
    C4::A
end
struct Edges{A<:AbstractUnitCell}
    T1::A
    T2::A
    T3::A
    T4::A
end
struct Projectors{A<:AbstractUnitCell}
    UL::A
    VL::A
    UR::A
    VR::A
end

unpack(ctm::Corners) = ctm.C1, ctm.C2, ctm.C3, ctm.C4
unpack(ctm::Edges) = ctm.T1, ctm.T2, ctm.T3, ctm.T4
unpack(ctm::Projectors) = ctm.UL, ctm.VL, ctm.UR, ctm.VR

function Base.getindex(corners::Corners, i...)
    C1, C2, C3, C4 = unpack(corners)

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

function Base.getindex(edges::Edges, i...)
    T1, T2, T3, T4 = unpack(edges)

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

function updatecorners!(corners::C, corners_p::C) where {C<:Corners}
    C1, C2, C3, C4 = unpack(corners)
    C1_p, C2_p, C3_p, C4_p = unpack(corners_p)
    permute!.(C1, permutedims(C1_p), Ref(()), Ref((2, 1)))
    permute!.(C2, permutedims(C2_p), Ref(()), Ref((2, 1)))
    permute!.(C3, permutedims(C3_p), Ref(()), Ref((2, 1)))
    permute!.(C4, permutedims(C4_p), Ref(()), Ref((2, 1)))
    return corners
end

struct CTMRGTensors{CType,TType} <: AbstractCornerBoundary
    corners::Corners{CType}
    edges::Edges{TType}
end

contraction_boundary_type(::CTMRGTensors) = CTMRG

## Top level

function run!(ctmrg::CTMRGTensors, bulk, alg::CTMRG; kwargs...)
    return ctmrgloop!(
        ctmrg,
        bulk;
        bonddim=alg.bonddim,
        verbosity=alg.verbosity,
        tol=alg.tol,
        maxiter=alg.maxiter,
    )
end

function ctmrgloop!(
    ctmrg::CTMRGTensors, bulk; bonddim=2, verbosity=1, tol=1e-12, maxiter=100, kwargs...
)
    bondspace = dimtospace(spacetype(bulk), bonddim)

    ctmrg_p = initpermuted(ctmrg)
    bulk_p = swapaxes(bulk)

    x_projectors = initprojectors(bulk, bondspace)
    y_projectors = initprojectors(bulk_p, bondspace)

    S1, S2, S3, S4 = initerror(ctmrg)

    error = Inf
    iterations = 0

    # bulk_tensors = contractphysical_maybe.(bulk)
    # bulk_tensors_p = contractphysical_maybe.(bulk_p)

    while error > tol && iterations < maxiter

        # Sweep along the x axis (left/right)
        ctmrgmove!(ctmrg, x_projectors, bulk, bonddim; kwargs...)

        # Write updated tensors into the permuted placeholders
        updatecorners!(ctmrg_p, ctmrg)

        # Sweep along the y axis (up/down)
        ctmrgmove!(ctmrg_p, y_projectors, bulk_p, bonddim; kwargs...)

        # Update the unpermuted (true) tensors.
        updatecorners!(ctmrg, ctmrg_p)

        error = ctmerror!(S1, S2, S3, S4, ctmrg.corners)

        verbosity > 0 && @info "\t Step $(iterations): error ≈ $(error)"

        iterations += 1
    end

    return error, iterations
end

function start(state)
    bonddim = state.alg.bonddim
    bulk = state.bulk
    ctmrg = state.tensors

    bondspace = dimtospace(spacetype(bulk), bonddim)

    ctmrg_p = initpermuted(ctmrg)
    bulk_p = swapaxes(bulk)

    x_projectors = initprojectors(bulk, bondspace)
    y_projectors = initprojectors(bulk_p, bondspace)

    S1, S2, S3, S4 = initerror(ctmrg)

    return ctmrg_p, bulk_p, x_projectors, y_projectors, S1, S2, S3, S4
end

function step!(
    ctmrg::CTMRGTensors,# mutating 
    bulk,
    alg::CTMRG,
    ctmrg_p,            # mutating
    bulk_p,             # mutating
    x_projectors,       # mutating
    y_projectors,       # mutating
    S1,                 # mutating
    S2,                 # mutating
    S3,                 # mutating
    S4,                 # mutating
)
    bonddim = alg.bonddim

    ctmrgmove!(ctmrg, x_projectors, bulk, bonddim; svd_alg=alg.svd_alg, ptol=alg.ptol)

    # Write updated tensors into the permuted placeholders
    updatecorners!(ctmrg_p, ctmrg)

    # Sweep along the y axis (up/down)
    ctmrgmove!(ctmrg_p, y_projectors, bulk_p, bonddim; svd_alg=alg.svd_alg, ptol=alg.ptol)

    # Update the unpermuted (true) tensors.
    updatecorners!(ctmrg, ctmrg_p)

    error = ctmerror!(S1, S2, S3, S4, ctmrg.corners)

    return error
end

## ERROR CALCULATION

function ctmerror!(S1_old, S2_old, S3_old, S4_old, ctmrg::CTMRGTensors)
    return ctmerror!(S1_old, S2_old, S3_old, S4_old, ctmrg.corners)
end
function ctmerror!(S1_old, S2_old, S3_old, S4_old, corners::Corners)
    err1 = boundaryerror!(S1_old, corners.C1)
    err2 = boundaryerror!(S2_old, corners.C2)
    err3 = boundaryerror!(S3_old, corners.C3)
    err4 = boundaryerror!(S4_old, corners.C4)
    return max(err1..., err2..., err3..., err4...)
end

## GET PROJECTORS

function projectors(
    C1_00,
    C2_30,
    C3_33,
    C4_03,
    T1_10,
    T1_20,
    T2_31,
    T2_32,
    T3_13,
    T3_23,
    T4_01,
    T4_02,
    M_11,
    M_21,
    M_12,
    M_22,
    bonddim;
    kwargs...,
)

    # Top
    top = halfcontract(C1_00, T1_10, T1_20, C2_30, T4_01, M_11, M_21, T2_31)
    U, S, V = tsvd!(top)
    FUL = sqrt(S) * V
    FUR = U * sqrt(S)

    # Bottom (requires permutation)
    MP_12 = invertaxes(M_12)
    MP_22 = invertaxes(M_22)

    bot = halfcontract(C3_33, T3_23, T3_13, C4_03, T2_32, MP_22, MP_12, T4_02)
    U, S, V = tsvd!(bot)
    FDL = U * sqrt(S)
    FDR = sqrt(S) * V

    UL, VL = biorth_truncation(FUL, FDL, bonddim; kwargs...)
    VR, UR = biorth_truncation(FDR, FUR, bonddim; kwargs...)

    return UL, VL, UR, VR
end

function biorth_truncation(U0, V0, xi; ptol=1e-7, svd_alg=TensorKit.SVD())
    Q, S, W = tsvd!(U0 * V0; trunc=truncdim(xi), alg=svd_alg)
    SX = pinv(sqrt(S); rtol=ptol)
    normalize!(SX)
    U = _transpose(SX * Q' * U0)
    V = V0 * W' * SX

    normalize!(U)
    normalize!(V)

    return U, V
end

function ctmrgmove!(ctmrg::CTMRGTensors, proj::Projectors, bulk, bonddim; kwargs...)
    C1, C2, C3, C4 = unpack(ctmrg.corners)
    T1, T2, T3, T4 = unpack(ctmrg.edges)
    UL, VL, UR, VR = unpack(proj)

    for x in axes(bulk, 1)
        for y in axes(bulk, 2)
            # println(x,"",y)
            UL[x + 0, y + 1], VL[x + 0, y + 1], UR[x + 3, y + 1], VR[x + 3, y + 1] = projectors(
                C1[x + 0, y + 0],
                C2[x + 3, y + 0],
                C3[x + 3, y + 3],
                C4[x + 0, y + 3],
                T1[x + 1, y + 0],
                T1[x + 2, y + 0],
                T2[x + 3, y + 1],
                T2[x + 3, y + 2],
                T3[x + 1, y + 3],
                T3[x + 2, y + 3],
                T4[x + 0, y + 1],
                T4[x + 0, y + 2],
                bulk[x + 1, y + 1],
                bulk[x + 2, y + 1],
                bulk[x + 1, y + 2],
                bulk[x + 2, y + 2],
                bonddim;
                kwargs...,
            )
        end
        for y in axes(bulk, 2)
            projectcorner!(
                C1[x + 1, y + 0], C1[x + 0, y + 0], T1[x + 1, y + 0], VL[x + 0, y + 0]
            )
            projectcorner!(
                C2[x + 2, y + 0],
                C2[x + 3, y + 0],
                swapvirtual(T1[x + 2, y + 0]),
                VR[x + 3, y + 0],
            )
            projectcorner!(
                C3[x + 2, y + 3], C3[x + 3, y + 3], T3[x + 2, y + 3], UR[x + 3, y + 2]
            )
            projectcorner!(
                C4[x + 1, y + 3],
                C4[x + 0, y + 3],
                swapvirtual(T3[x + 1, y + 3]),
                UL[x + 0, y + 2],
            )
            projectedge!(
                T4[x + 1, y + 1],
                T4[x + 0, y + 1],
                bulk[x + 1, y + 1],
                UL[x + 0, y + 0],
                VL[x + 0, y + 1],
            )
            projectedge!(
                T2[x + 2, y + 1],
                T2[x + 3, y + 1],
                invertaxes(bulk[x + 2, y + 1]),
                VR[x + 3, y + 1],
                UR[x + 3, y + 0],
            )
        end
    end

    normalize!.(C1)
    normalize!.(C2)
    normalize!.(C3)
    normalize!.(C4)
    normalize!.(T1)
    normalize!.(T2)
    normalize!.(T3)
    normalize!.(T4)

    return ctmrg
end

# swapvirtual(t::AbsTen{1,2}) = permute(t, (1,), (3, 2))
# swapvirtual(t::AbsTen{2,2}) = permute(t, (1, 2), (4, 3))

swapvirtual(t::AbstractTensorMap) = permutedom(t, (2, 1))

function updatecorners!(ctmrg::CTMRGTensors, ctmrg_p::CTMRGTensors)
    updatecorners!(ctmrg.corners, ctmrg_p.corners)
    return ctmrg
end
# function ctmrgmove!(ctmrg::CTMRG, bulk, bonddim)
#     cs = ctmrg.corners
#     ts = ctmrg.edges
#     projectors = ctmrg.xproj
#     for x in axes(M, 1)
#         for y in axes(M, 2)
#             c1, c2, c3, c4 = cs[x + 1, y + 1]
#             # println(x,"",y)
#             UL[x + 0, y + 1], VL[x + 0, y + 1], UR[x + 3, y + 1], VR[x + 3, y + 1] = projectors(
#                 C1[x + 0, y + 0],
#                 C2[x + 3, y + 0],
#                 C3[x + 3, y + 3],
#                 C4[x + 0, y + 3],
#                 T1[x + 1, y + 0],
#                 T1[x + 2, y + 0],
#                 T2[x + 3, y + 1],
#                 T2[x + 3, y + 2],
#                 T3[x + 1, y + 3],
#                 T3[x + 2, y + 3],
#                 T4[x + 0, y + 1],
#                 T4[x + 0, y + 2],
#                 M[x + 1, y + 1],
#                 M[x + 2, y + 1],
#                 M[x + 1, y + 2],
#                 M[x + 2, y + 2],
#                 dim(D),
#             )
#         end
#         for y in axes(M, 2)
#             project_corner!(
#                 C1[x + 1, y + 0], C1[x + 0, y + 0], T1[x + 1, y + 0], VL[x + 0, y + 0]
#             )
#             project_corner!(
#                 C2[x + 2, y + 0],
#                 C2[x + 3, y + 0],
#                 permute(T1[x + 2, y + 0], (1,), (3, 2)),
#                 VR[x + 3, y + 0],
#             )
#             project_corner!(
#                 C3[x + 2, y + 3], C3[x + 3, y + 3], T3[x + 2, y + 3], UR[x + 3, y + 2]
#             )
#             project_corner!(
#                 C4[x + 1, y + 3],
#                 C4[x + 0, y + 3],
#                 permute(T3[x + 1, y + 3], (1,), (3, 2)),
#                 UL[x + 0, y + 2],
#             )
#             project_edge!(
#                 T4[x + 1, y + 1],
#                 T4[x + 0, y + 1],
#                 M[x + 1, y + 1],
#                 UL[x + 0, y + 0],
#                 VL[x + 0, y + 1],
#             )
#             project_edge!(
#                 T2[x + 2, y + 1],
#                 T2[x + 3, y + 1],
#                 permute(M[x + 2, y + 1], (3, 4, 1, 2)),
#                 VR[x + 3, y + 1],
#                 UR[x + 3, y + 0],
#             )
#         end
#     end
#
#     normalize!.(C1)
#     normalize!.(C2)
#     normalize!.(C3)
#     normalize!.(C4)
#     normalize!.(T1)
#     normalize!.(T2)
#     normalize!.(T3)
#     normalize!.(T4)
#
#     return nothing
# end

# function initprojectors(bulk_tensors::ContractableTensor, chi::IndexSpace)
#     east_bonds, south_bonds, west_bonds, north_bonds = bondspaces_onlattice(bulk)
#
#     T = numbertype(bulk_tensors)
#
#     projx = _initprojectors(T, north_bonds, south_bonds, chi)
#     projy = _initprojectors(T, east_bonds, west_bonds, chi)
#
#     return UL, VL, UR, VR
# end

function testctmrg(data_func)
    βc = log(1 + sqrt(2)) / 2

    s = ℂ^2

    bulk =
        x -> ContractableTensors(
            fill(TensorMap((data_func(x)[1]), one(s), s * s * s' * s'), 2, 2)
        )
    bulk_magn =
        x -> ContractableTensors(
            fill(TensorMap(data_func(x)[2], one(s), s * s * s' * s'), 2, 2)
        )

    rv = []
    rv_exact = []

    alg = CTMRG(; bonddim=2, verbosity=1, maxiter=200)

    for x in 1.1:0.001:2
        b1 = bulk(x * βc)
        b2 = bulk_magn(x * βc)

        state = initialize(b1, alg)

        did_converge = false

        calculate!(state)

        Z = contract(state.tensors, b1)
        magn = contract(state.tensors, b2) / Z

        push!(rv, abs(magn))

        M = abs((1 - sinh(2 * x * βc)^(-4)))^(1 / 8)
        push!(rv_exact, M)
    end

    return rv, rv_exact
end
