abstract type AbstractCornerBoundary <: AbstractBoundary end

corners(ctm::AbstractCornerBoundary) = ctm.corners
edges(ctm::AbstractCornerBoundary) = ctm.edges

struct Corners{L,CType}
    C1::OnLattice{L,CType,Matrix{CType}}
    C2::OnLattice{L,CType,Matrix{CType}}
    C3::OnLattice{L,CType,Matrix{CType}}
    C4::OnLattice{L,CType,Matrix{CType}}
end
struct Edges{L,TType}
    T1::OnLattice{L,TType,Matrix{TType}}
    T2::OnLattice{L,TType,Matrix{TType}}
    T3::OnLattice{L,TType,Matrix{TType}}
    T4::OnLattice{L,TType,Matrix{TType}}
end
struct Projectors{L,PType}
    UL::OnLattice{L,PType,Matrix{PType}}
    VL::OnLattice{L,PType,Matrix{PType}}
    UR::OnLattice{L,PType,Matrix{PType}}
    VR::OnLattice{L,PType,Matrix{PType}}
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

function updatecorners!(corners::Corners, corners_p::Corners)
    C1, _, C3, _ = unpack(corners)
    C1_p, _, C3_p, _ = unpack(corners_p)
    permute!.(C1, transpose(C1_p), Ref(()), Ref((2, 1)))
    permute!.(C3, transpose(C3_p), Ref(()), Ref((2, 1)))
    return corners
end

struct CTMRG{L,CType,TType,SVDAlg<:OrthogonalFactorizationAlgorithm} <:
       AbstractCornerBoundary
    corners::Corners{L,CType}
    edges::Edges{L,TType}
    pinvtol::Float64
    svdalg::SVDAlg
    # function CTMRG(
    #     corners::Corners{L,CType},
    #     edges::Edges{L,TType},
    #     pinvtol::Float64,
    #     svdalg::OrthogonalFactorizationAlgorithm,
    # ) where {L,CType,TType}
    #     return new{L,CType,TType,typeof(svdalg)}(corners, edges, pinvtol, svdalg)
    # end
end

function CTMRG(corners, edges; pinvtol=1e-7, svdalg=TensorKit.SVD())
    return CTMRG(corners, edges, pinvtol, svdalg)
end

## Top level

function calculate!(ctmrg::CTMRG, bulk; kwargs...)
    return ctmrgloop!(ctmrg, bulk; kwargs...)
end

function ctmrgloop!(
    ctmrg::CTMRG, bulk::ContractableTensors; bonddim=2, verbosity=1, tol=1e-12, maxiter=100
)
    bondspace = dimtospace(bulk, bonddim)

    ctmrg_p = initpermuted(ctmrg)
    bulk_p = swapaxes.(bulk)

    x_projectors = initprojectors(bulk, bondspace)
    y_projectors = initprojectors(bulk_p, bondspace)

    S1, S2, S3, S4 = initerror(ctmrg)

    error = Inf
    iterations = 0

    # bulk_tensors = contractphysical_maybe.(bulk)
    # bulk_tensors_p = contractphysical_maybe.(bulk_p)

    while error > tol && iterations < maxiter
        # Sweep along the x axis (left/right)
        ctmrgmove!(ctmrg, x_projectors, bulk, bonddim)

        # Write updated tensors into the permuted placeholders
        updatecorners!(ctmrg_p, ctmrg)

        # Sweep along the y axis (up/down)
        ctmrgmove!(ctmrg_p, y_projectors, bulk_p, bonddim)

        # Update the unpermuted (true) tensors.
        updatecorners!(ctmrg, ctmrg_p)

        error = ctmerror!(S1, S2, S3, S4, ctmrg.corners)

        verbosity > 0 && @info "\t Step $(iterations): error ≈ $(error)"

        Z = tracecontract(ctmrg, bulk)
        # println(Z)

        iterations += 1
    end

    return error, iterations
end

## ERROR CALCULATION

function ctmerror!(S1_old, S2_old, S3_old, S4_old, ctmrg::CTMRG)
    return ctmerror!(S1_old, S2_old, S3_old, S4_old, ctmrg.corners)
end
function ctmerror!(S1_old, S2_old, S3_old, S4_old, corners::Corners)
    err1 = ctmerror!(S1_old, corners.C1)
    err2 = ctmerror!(S2_old, corners.C2)
    err3 = ctmerror!(S3_old, corners.C3)
    err4 = ctmerror!(S4_old, corners.C4)
    return max(err1..., err2..., err3..., err4...)
end

function ctmerror!(S_old::AbstractMatrix, C_new::AbstractMatrix)
    S_new = ctmerror.(S_old, C_new)
    err = @. norm(S_old - S_new)
    S_old .= S_new
    return err
end

function ctmerror(s_old::AbsTen{1,1}, c_new::AbstractTensorMap)
    _, s_new, _ = tsvd(c_new, (1,), (2,))
    normalize!(s_new)
    return s_new
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

    # # Biorthogonalization
    # Q, S, W = tsvd!(FUL * FDL; trunc=truncdim(xi), alg=TensorKit.SVD())
    # SX = pinv(sqrt(S); rtol=1e-7)
    # normalize!(SX)
    # UL = _transpose(SX * Q' * FUL)
    # VL = FDL * W' * SX

    # W, S, Q = tsvd!(FDR * FUR; trunc=truncdim(xi), alg=TensorKit.SVD())
    # SX = pinv(sqrt(S); rtol=1e-7)
    # normalize!(SX)
    # VR = _transpose(SX * W' * FDR)
    # UR = FUR * Q' * SX

    UL, VL = biorth_truncation(FUL, FDL, bonddim; kwargs...)
    VR, UR = biorth_truncation(FDR, FUR, bonddim; kwargs...)

    return UL, VL, UR, VR
end

function biorth_truncation(U0, V0, xi; tol=1e-7, alg=TensorKit.SVD())
    Q, S, W = tsvd!(U0 * V0; trunc=truncdim(xi), alg=alg)
    SX = pinv(sqrt(S); rtol=tol)
    normalize!(SX)
    U = _transpose(SX * Q' * U0)
    V = V0 * W' * SX

    normalize!(U)
    normalize!(V)

    return U, V
end

function ctmrgmove!(ctmrg::CTMRG, proj::Projectors, bulk, bonddim)
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
                tol=ctmrg.pinvtol,
                alg=ctmrg.svdalg,
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

swapvirtual(t::AbsTen{1,2}) = permute(t, (1,), (3, 2))
swapvirtual(t::AbsTen{2,2}) = permute(t, (1, 2), (4, 3))

function updatecorners!(ctmrg::CTMRG, ctmrg_p::CTMRG)
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

    L = Lattice{1,1,Infinite}([s;;], true)
    # L = Lattice{2,2,Infinite}([s s; s s], true)

    bulk = x -> fill(TensorMap(ComplexF64.(data_func(x)[1]), one(s), s * s * s' * s'), L)
    bulk_magn = x -> fill(TensorMap(data_func(x)[2], one(s), s * s * s' * s'), L)

    rv = []
    rv_exact = []

    alg = BoundaryAlgorithm(; alg=VUMPS, bonddim=2, verbosity=1)

    for x in 0.01:0.01:2
        b1 = bulk(x * βc)
        b2 = bulk_magn(x * βc)

        state = alg(bulk(x * βc))

        did_converge = false

        numiter = 0
        while !(did_converge) && numiter < 1
            calculate!(state)
            did_converge = state.info.converged
            numiter += 1
        end

        Z = tracecontract(state.tensors, b1)
        magn = tracecontract(state.tensors, b2) ./ Z
        push!(rv, (magn))

        M = abs((1 - sinh(2 * x * βc)^(-4)))^(1 / 8)
        push!(rv_exact, M)
    end

    return rv, rv_exact
end
