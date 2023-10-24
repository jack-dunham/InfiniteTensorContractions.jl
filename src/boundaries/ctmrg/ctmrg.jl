abstract type AbstractCornerMethod <: AbstractBoundaryAlgorithm end

"""
    CTMRG{SVD<:OrthogonalFactorizationAlgorithm}

# Fields
- `bonddim::Int`: the bond dimension of the boundary
- `maxiter::Int = 100`: maximum number of iterations
- `tol::Float64 = 1e-12`: convergence tolerance
- `verbose::Bool = true`: when true, will print algorithm convergence progress
- `ptol::Float64 = 1e-7`: tolerance used in the pseudoinverse
- `svd_alg::SVD = TensorKit.SVD()`: algorithm used for the SVD. Either `TensorKit.SVD()` or `TensorKit.SDD()`
"""
@kwdef struct CTMRG{SVD<:OrthogonalFactorizationAlgorithm} <: AbstractCornerMethod
    bonddim::Int
    maxiter::Int = 100
    tol::Float64 = 1e-12
    verbose::Bool = true
    ptol::Float64 = 5e-8
    svd_alg::SVD = TensorKit.SVD()
    randinit::Bool = false
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

numbertype(::Union{Corners{A},Edges{A},Projectors{A}}) where {A} = numbertype(A)

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
    permute!.(C2, permutedims(C4_p), Ref(()), Ref((2, 1)))
    permute!.(C3, permutedims(C3_p), Ref(()), Ref((2, 1)))
    permute!.(C4, permutedims(C2_p), Ref(()), Ref((2, 1)))
    return corners
end

struct CTMRGTensors{CType,TType} <: AbstractCornerBoundary
    corners::Corners{CType}
    edges::Edges{TType}
end

contraction_boundary_type(::CTMRGTensors) = CTMRG

numbertype(::CTMRGTensors{C,T}) where {C,T} = promote_type(numbertype(C), numbertype(T))

## Top level

function run!(ctmrg::CTMRGTensors, network, alg::CTMRG; kwargs...)
    return ctmrgloop!(
        ctmrg,
        network;
        bonddim=alg.bonddim,
        verbose=alg.verbose,
        tol=alg.tol,
        maxiter=alg.maxiter,
    )
end

function ctmrgloop!(
    ctmrg::CTMRGTensors, network; bonddim=2, verbosity=1, tol=1e-12, maxiter=100, kwargs...
)
    bondspace = dimtospace(spacetype(network), bonddim)

    ctmrg_p = initpermuted(ctmrg)
    network_p = swapaxes(network)

    x_projectors = initprojectors(network, bondspace)
    y_projectors = initprojectors(network_p, bondspace)

    S1, S2, S3, S4 = initerror(ctmrg)

    error = Inf
    iterations = 0

    # network_tensors = contractphysical_maybe.(network)
    # network_tensors_p = contractphysical_maybe.(network_p)

    while error > tol && iterations < maxiter

        # Sweep along the x axis (left/right)
        ctmrgmove!(ctmrg, x_projectors, network, bonddim; kwargs...)

        # Write updated tensors into the permuted placeholders
        updatecorners!(ctmrg_p, ctmrg)

        # Sweep along the y axis (up/down)
        ctmrgmove!(ctmrg_p, y_projectors, network_p, bonddim; kwargs...)

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
    network = state.network
    ctmrg = state.tensors

    bondspace = dimtospace(spacetype(network), bonddim)

    ctmrg_p = initpermuted(ctmrg)
    network_p = permutedims(swapaxes(network))

    x_projectors = initprojectors(network, bondspace)
    y_projectors = initprojectors(network_p, bondspace)

    S1, S2, S3, S4 = initerror(ctmrg)

    return ctmrg_p, network_p, x_projectors, y_projectors, S1, S2, S3, S4, state.info
end

function step!(
    ctmrg::CTMRGTensors,# mutating 
    network,
    alg::CTMRG,
    ctmrg_p,            # mutating
    network_p,             # mutating
    x_projectors,       # mutating
    y_projectors,       # mutating
    S1,                 # mutating
    S2,                 # mutating
    S3,                 # mutating
    S4,                 # mutating
    info,
)
    bonddim = alg.bonddim

    fpcm = true

    freq = 1
    # freq = 2500

    if false #mod(info.iterations, 0:(freq - 1)) == freq - 1
        @info "Doing FPCM"
        for i in 1:1
            fpcmstep!(ctmrg, ctmrg_p, x_projectors, y_projectors, network, network_p)
        end
    else
        ctmrgmove!(
            ctmrg,
            x_projectors,
            network,
            bonddim;
            svd_alg=alg.svd_alg,
            ptol=alg.ptol,
            fpcm=fpcm,
        )

        # Write updated tensors into the permuted placeholders
        updatecorners!(ctmrg_p, ctmrg)

        # println("Half", S1o[1, 1])

        # Sweep along the y axis (up/down)
        ctmrgmove!(
            ctmrg_p,
            y_projectors,
            network_p,
            bonddim;
            svd_alg=alg.svd_alg,
            ptol=alg.ptol,
            fpcm=fpcm,
        )

        # Update the unpermuted (true) tensors.
        updatecorners!(ctmrg, ctmrg_p)
    end
    error = ctmerror!(S1, S2, S3, S4, ctmrg.corners)

    return error
end

function fpcmstep!(
    ctmrg::CTMRGTensors, ctmrg_p, x_projectors, y_projectors, network, network_p
)
    C1, C2, C3, C4 = unpack(ctmrg.corners)
    T1, T2, T3, T4 = unpack(ctmrg.edges)

    fpcmprojectors!(x_projectors, ctmrg, network)

    edgemove!(ctmrg, x_projectors, network)
    # Write updated tensors into the permuted placeholders
    updatecorners!(ctmrg_p, ctmrg)

    fpcmprojectors!(y_projectors, ctmrg_p, network_p)

    edgemove!(ctmrg_p, y_projectors, network_p)

    # Update the unpermuted (true) tensors.
    updatecorners!(ctmrg, ctmrg_p)
    # Write updated tensors into the permuted placeholders

    normalize!.(T1)
    normalize!.(T2)
    normalize!.(T3)
    normalize!.(T4)

    # Get new corners useing fixed point equations
    cornermove!(ctmrg, x_projectors, y_projectors, network)

    # Write updated tensors into the permuted placeholders

    # Update the unpermuted (true) tensors.
    # normalize!.(T1)
    # normalize!.(T2)
    # normalize!.(T3)
    # normalize!.(T4)

    # Sweep along the y axis (up/down)

    # error = ctmerror!(S1, S2, S3, S4, ctmrg.corners)

    normalize!.(C1)
    normalize!.(C2)
    normalize!.(C3)
    normalize!.(C4)

    # updatecorners!(ctmrg_p, ctmrg)

    return ctmrg
end

function cornermove!(ctmrg, xproj, yproj, network)
    C1, C2, C3, C4 = unpack(ctmrg.corners)
    T1, T2, T3, T4 = unpack(ctmrg.edges)
    UL, VL, UR, VR = unpack(xproj)
    UU, VU, UD, VD = permutedims.(unpack(yproj))

    Ms = network

    corner_fixed_point_reverse!(C1, T1, T4, VU, VL, Ms; corner=:C1)
    corner_fixed_point_reverse!(C2, T1, T2, UU, VR, Ms; corner=:C2)
    corner_fixed_point_reverse!(C3, T3, T2, UD, UR, Ms; corner=:C3)
    corner_fixed_point_reverse!(C4, T3, T4, VD, UL, Ms; corner=:C4)

    # normalize!.(C1)
    # normalize!.(C2)
    # normalize!.(C3)
    # normalize!.(C4)

    return ctmrg
end

function corner_fixed_point_reverse!(C, TH, TV, PH, PV, M; corner::Symbol)
    flipx = x -> permutedom(x, (3, 2, 1, 4))
    flipy = x -> permutedom(x, (1, 4, 3, 2))

    if corner === :C1
        increment = (1, 1)
    elseif corner === :C2
        increment = (-1, 1)
        TH = swapvirtual.(TH)
        TV = swapvirtual.(TV)
        PH = circshift(PH, (1, 0))
        M = flipx.(M)
    elseif corner === :C3
        increment = (-1, -1)
        PH = circshift(PH, (1, 0))
        PV = circshift(PV, (0, 1))
        M = invertaxes(M)
    elseif corner === :C4
        increment = (1, -1)
        TH = swapvirtual.(TH)
        TV = swapvirtual.(TV)
        PV = circshift(PV, (0, 1))
        M = flipy.(M)
    end

    return corner_fixed_point_alt_2d!(
        C, TH, TV, PH, PV, M; increment=CartesianIndex(increment)
    )
end

function edgemove!(ctmrg, proj, network)
    _, T2, _, T4 = unpack(ctmrg.edges)

    UL, VL, UR, VR = unpack(proj)

    for y in axes(network, 2)
        edge_fixed_point_alt!(
            T4[:, y + 1], network[:, y + 1], UL[:, y + 0], VL[:, y + 1], 1
        )
        edge_fixed_point_alt!(
            reverse(T2[:, y + 1]),
            invertaxes.(reverse(network[:, y + 1])),
            reverse(VR[:, y + 1]),
            reverse(UR[:, y + 0]),
            1,
        )
    end

    # normalize!.(T2)
    # normalize!.(T4)

    return ctmrg
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

    @debug "Corner matrix convergence:" C1 = err1 C2 = err2
    @debug "Corner matrix convergence:" C3 = err3 C4 = err4
    # @debug "Corner matrix singular values" S1_old[1, 1] - S3_old[2, 2]
    # @debug "Corner matrix singular values" S1_old[1, 1] - S3_old[3, 3]
    # @debug "Corner matrix singular values" S2_old[1, 1] - S4_old[2, 2]
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
    svd_alg=TensorKit.SVD(),
    kwargs...,
)
    # Top
    top = halfcontract(C1_00, T1_10, T1_20, C2_30, T4_01, M_11, M_21, T2_31)
    # println(top)
    U, S, V = tsvd!(top; alg=svd_alg)
    FUL = sqrt(S) * V
    FUR = U * sqrt(S)

    # Bottom (requires permutation)
    MP_12 = invertaxes(M_12)
    MP_22 = invertaxes(M_22)

    bot = halfcontract(C3_33, T3_23, T3_13, C4_03, T2_32, MP_22, MP_12, T4_02)
    U, S, V = tsvd!(bot; alg=svd_alg)
    FDL = U * sqrt(S)
    FDR = sqrt(S) * V

    UL, VL = biorth_truncation(FUL, FDL, bonddim; svd_alg=svd_alg, kwargs...)
    VR, UR = biorth_truncation(FDR, FUR, bonddim; svd_alg=svd_alg, kwargs...)

    return UL, VL, UR, VR
end

function biorth_truncation(U0, V0, xi; ptol=5e-8, svd_alg=TensorKit.SVD())
    Q, S, W = tsvd!(U0 * V0; trunc=truncdim(xi), alg=svd_alg)

    # normalize!(S)

    SX = pinv(sqrt(S); rtol=ptol)

    # normalize!(SX)
    U_temp = SX * Q' * U0
    U = _transpose(U_temp)

    V = V0 * W' * SX

    # biorthness = norm(normalize(U_temp * V) - normalize(one(U_temp * V)))
    #
    # btol = sqrt(eps())
    #
    # if biorthness > btol
    #     @warn "Biorthogonalisation failed!" biorthness
    #     @info "" U_temp * V
    # end
    # normalize!(U)
    # normalize!(V)

    return U, V
end

function ctmrgmove!(
    ctmrg::CTMRGTensors, proj::Projectors, network, bonddim; fpcm=false, kwargs...
)
    C1, C2, C3, C4 = unpack(ctmrg.corners)
    T1, T2, T3, T4 = unpack(ctmrg.edges)
    UL, VL, UR, VR = unpack(proj)

    # cj_f = x -> axpby!(1//2, cj(permutedom(x, (2, 1))), 1//2, x)
    if !fpcm
        for x in axes(network, 1)
            for y in axes(network, 2)
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
                    network[x + 1, y + 1],
                    network[x + 2, y + 1],
                    network[x + 1, y + 2],
                    network[x + 2, y + 2],
                    bonddim;
                    kwargs...,
                )
            end
            for y in axes(network, 2)
                #=
                  1
                C --- T ---
                |2    |
                *--V--*
                   |
                =#
                projectcorner!(
                    C1[x + 1, y + 0], C1[x + 0, y + 0], T1[x + 1, y + 0], VL[x + 0, y + 0]
                )
                #=
                        1
                --- T --- C
                    |    2|
                    *--V--*
                       |
                =#
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
                #=
                   |
                *--U--*
                |2    |
                T --- M ---
                |1    |
                *--V--*
                   |
                =#
                projectedge!(
                    T4[x + 1, y + 1],
                    T4[x + 0, y + 1],
                    network[x + 1, y + 1],
                    UL[x + 0, y + 0],
                    VL[x + 0, y + 1],
                )
                #=
                       |
                    *--U--*
                    |    1|
                --- M --- T
                    |    2|
                    *--V--*
                       |
                =#
                projectedge!(
                    T2[x + 2, y + 1],
                    T2[x + 3, y + 1],
                    invertaxes(network[x + 2, y + 1]),
                    # flipaxis(network[x + 2, y + 1]),
                    VR[x + 3, y + 1],
                    UR[x + 3, y + 0],
                )
            end
        end
    end

    if fpcm
        # left/down move
        for y in axes(network, 2)
            recleftbiorth!(
                UL[:, y], VL[:, y], C1[:, y], C4[:, y + 1], T1[:, y], T3[:, y + 1]; x0=1, incr=1
            )
        end
        for y in axes(network, 2)
            corner_fixed_point_alt!(C1[:, y + 0], T1[:, y + 0], VL[:, y + 0], 1)
            corner_fixed_point_alt!(
                C4[:, y + 1], swapvirtual.(T3[:, y + 1]), UL[:, y + 0], 1
            )

            edge_fixed_point_alt!(
                T4[:, y + 0], network[:, y + 0], UL[:, y - 1], VL[:, y + 0], 1
            )
        end
        # right/up move
        for y in axes(network, 2)
            # recleftbiorth!(
            #     reverse(VR[:, y]),
            #     reverse(UR[:, y]),
            #     reverse(C3[:, y + 1]),
            #     reverse(C2[:, y]),
            #     reverse(T3[:, y + 1]),
            #     reverse(T1[:, y]),
            # )
            recleftbiorth!(
                VR[:, y], UR[:, y], C3[:, y + 1], C2[:, y], T3[:, y + 1], T1[:, y]; x0=0, incr=-1
            )
        end
        for y in axes(network, 2)
            # corner_fixed_point!(C2[:, y + 0], swapvirtual.(T1[:, y + 0]), VR[:, y + 0], -1)
            corner_fixed_point_alt!(
                reverse(C2[:, y + 0]),
                swapvirtual.(reverse(T1[:, y + 0])),
                reverse(VR[:, y + 0]),
                1,
            )
            # # corner_fixed_point!(C3[:, y + 1], T3[:, y + 1], UR[:, y + 0], -1)
            #
            corner_fixed_point_alt!(
                reverse(C3[:, y + 1]), reverse(T3[:, y + 1]), reverse(UR[:, y + 0]), 1
            )

            # edge_fixed_point!(
            #     T2[:, y + 0], invertaxes.(network[:, y + 0]), VR[:, y + 0], UR[:, y - 1], 1
            # )
            edge_fixed_point_alt!(
                reverse(T2[:, y + 1]),
                invertaxes.(reverse(network[:, y + 1])),
                reverse(VR[:, y + 1]),
                reverse(UR[:, y + 0]),
                1,
            )
        end

        #=
               |
            *--U--*
            |    1|
        --- M --- T
            |    2|
            *--V--*
               |
        =#
    end

    # cj_f = x -> axpby!(1//2, cj(permutedom(x, (2, 1))), 1//2, x)

    #
    # cj_f.(C1)
    # cj_f.(C2)
    # cj_f.(C3)
    # cj_f.(C4)

    # circf = identity

    normalize!.((C1))
    normalize!.((C2))
    normalize!.((C3))
    normalize!.((C4))
    normalize!.((T1))
    normalize!.((T2))
    normalize!.((T3))
    normalize!.((T4))

    return ctmrg
end

function fpcmprojectors!(proj, ctmrg, network)
    C1, C2, C3, C4 = unpack(ctmrg.corners)
    T1, _, T3, _ = unpack(ctmrg.edges)
    UL, VL, UR, VR = unpack(proj)
    for y in axes(network, 2)
        recleftbiorth!(UL[:, y], VL[:, y], C1[:, y], C4[:, y + 1], T1[:, y], T3[:, y + 1])
        recleftbiorth!(
            reverse(VR[:, y]),
            reverse(UR[:, y]),
            reverse(C3[:, y + 1]),
            reverse(C2[:, y]),
            reverse(T3[:, y + 1]),
            reverse(T1[:, y]),
        )
    end
end

function corner_fixed_point!(corners, edges, projectors, dir)
    edges = circshift(edges, -dir)

    x0 = RecursiveVec(corners...)

    val, vec, info = eigsolve(x0, 1, :LM; ishermitian=false, eager=true) do x
        corner_fixed_point_map(x, edges, projectors, dir)
    end

    # @debug "Corner fixed point:" val #vec info

    copy!.(corners, vec[1] * 1 / val[1])
    normalize!.(corners)

    return corners
end
function corner_fixed_point_alt!(corners, edges, projectors, dir)
    edges = circshift(edges, -dir)

    x0 = corners[1]

    val, vec, info = eigsolve(x0, 1, :LM; ishermitian=false, eager=true) do out
        for x in 1:length(corners)
            out = projectcorner(out, edges[x], projectors[x])
        end
        return out
    end

    copy!(corners[1], vec[1])# / val[1])

    for _ in 1:1
        for x in 1:(length(corners) - 1)
            old = copy(corners[x + 1])
            projectcorner!(corners[x + 1], corners[x], edges[x], projectors[x])
            normalize!(corners[x + 1])
            # println(norm(corners[x + 1] - old))
        end
    end

    # @debug "Corner fixed point:" val #vec info

    # copy!.(corners, vec[1] * 1 / val[1])

    return corners
end

function left_up_fixed_point!(C1s, T1s, T4s, VUs, VLs, Ms)
    T1s = circshift(T1s, (-1, 0))
    T4s = circshift(T4s, (0, -1))

    VUs = circshift(VUs, (0, -1))
    VLs = circshift(VLs, (0, -1))

    Ms = circshift(Ms, (-1, -1))

    @assert nx == ny

    x0 = C1s[1, 1]

    for y in 1:ny
        val, vec, info = eigsolve(x0, 1, :LM; ishermitian=false, eager=true) do out
            for x in 1:nx
                out = projectcorner2d(out, T1s[x, y], T4s[x, y], Ms[x, y], UHs[x, y], UVs[x, y]) # x + 1, y + 1
            end
            return out
        end

        copy!(C1s[1, 1], vec[1])# / val[1])

        for x in 1:(nx - 1)
            projectcorner2d!(
                C1s[x + 1, y + 1],
                C1s[x, y],
                T1s[x, y],
                T4s[x, y],
                Ms[x, y],
                UHs[x, y],
                UVs[x, y],
            )
        end
    end

    # @debug "Corner fixed point:" val #vec info

    # copy!.(corners, vec[1] * 1 / val[1])

    return C1s
end

function corner_fixed_point_alt_2d!(
    C1s, T1s, T4s, UHs, UVs, Ms; increment::CartesianIndex{2}
)
    xi = increment[1]
    yi = increment[2]

    nx, ny = size(C1s)

    # number of repeats
    cycle_length = lcm(nx, ny)

    num_cycles = Integer(length(C1s)//cycle_length)

    T1s = circshift(T1s, (-xi, 0))
    T4s = circshift(T4s, (0, -yi))

    UHs = circshift(UHs, (-xi, 0))
    UVs = circshift(UVs, (0, -yi))

    Ms = circshift(Ms, (-xi, -yi))

    for y in 1:num_cycles
        coord0 = CartesianIndex(1, y)

        x0 = C1s[1, y]
        val, vec, info = eigsolve(x0, 1, :LM; ishermitian=false, eager=true) do out
            for _ in 1:cycle_length
                @inbounds out = projectcorner2d(
                    out, T1s[coord0], T4s[coord0], Ms[coord0], UHs[coord0], UVs[coord0]
                ) # x + 1, y + 1
            end
            coord0 += increment
            return out
        end

        coord0 = CartesianIndex(1, y)

        copy!(C1s[coord0], vec[1])# / val[1])

        for _ in 1:(cycle_length - 1)
            @inbounds projectcorner2d!(
                C1s[coord0 + increment],
                C1s[coord0],
                T1s[coord0],
                T4s[coord0],
                Ms[coord0],
                UHs[coord0],
                UVs[coord0],
            )
            coord0 += increment
        end
    end

    # @debug "Corner fixed point:" val #vec info

    # copy!.(corners, vec[1] * 1 / val[1])

    return C1s
end

function edge_fixed_point!(edges, network, uproj, vproj, dir)
    bulk = circshift(network, -dir)

    x0 = RecursiveVec(edges...)

    val, vec, info = eigsolve(x0, 1, :LM; ishermitian=false, eager=true) do x
        edge_fixed_point_map(x, bulk, uproj, vproj, dir)
    end

    # @debug "Edge fixed point:" val #vec info

    copy!.(edges, vec[1] * 1 / val[1])

    normalize!.(edges)

    return edges
end

function edge_fixed_point_alt!(edges, bulk, uproj, vproj, dir)
    bulk = circshift(bulk, -dir)

    x0 = edges[1]

    val, vec, info = eigsolve(x0, 1, :LM; ishermitian=false, eager=true) do out
        for x in 1:length(edges)
            out = projectedge(out, bulk[x], uproj[x], vproj[x])
        end
        return out
    end

    # @debug "Edge fixed point:" val #vec info

    # copy!.(edges, vec[1] * 1 / val[1])
    #
    # normalize!.(edges)

    copy!(edges[1], vec[1])# / val[1])

    for _ in 1:1
        for x in 1:(length(edges) - 1)
            # old = copy(edges[x + 1])
            projectedge!(edges[x + 1], edges[x], bulk[x], uproj[x], vproj[x])
            # normalize!(edges[x + 1])
            # println(norm(edges[x + 1] - normalize(old)))
        end
    end

    return edges
end

function corner_fixed_point_map(corners::RecursiveVec, edges, projectors, dir)
    out = projectcorner.(corners, edges, projectors)
    return RecursiveVec(circshift(out, dir)...)
end
function edge_fixed_point_map(edges::RecursiveVec, bulk, uproj, vproj, dir)
    out = projectedge.(edges, bulk, uproj, vproj) # x -> x + 1
    return RecursiveVec(circshift(out, dir)...)
end

function leftbiorth!(UL, VL, CU, CD, AU, AD)
    @tensoropt C0[x; y] := CU[x 2] * CD[y 2]

    χ = domain(CU)[1]

    val, vec, info = eigsolve(C0, 1, :LM) do x0
        @tensoropt x1[ur; dr] := x0[ul; dl] * AU[p; ur ul] * AD[p; dl dr]
    end

    # truncating here is a convenient way to switch bond from adjoint to not-adjoint. No truncation actually occurs.
    U, S, V = tsvd(vec[1]; trunc=truncspace(χ))

    s = sqrt(S)
    sinv = pinv(s)

    #TODO later.

    @tensoropt UL[i o; n] = s[j; i] * U[k; j] * AU[o; l k] * (U')[m; l] * sinv[n; m]
    @tensoropt VL[n o; i] = sinv[j; i] * (V')[k; j] * AD[o; l k] * V[m; l] * s[n; m]

    id_maybe = _transpose(UL) * VL

    permute!(CU, U * s, (), (1, 2))
    permute!(CD, s * V, (), (2, 1))
    # println(norm(id_maybe / tr(id_maybe) - one(id_maybe) / tr(one(id_maybe))))

    return UL, VL
end

recleftbiorth(args...; kwargs...) = recleftbiorth!(deepcopy.(args)...; kwargs...)

function recleftbiorth_solve(x0, AU, AD)
    rv = x0.vecs
    out = broadcast(rv, AU, AD) do x0, AU, AD
        @tensoropt x1[ur; dr] := x0[ul; dl] * AU[p; ur ul] * AD[p; dl dr]
        return x1
    end
    return RecursiveVec(circshift(out, 1)...)
end

function gauge_upper_edge!(UV, s, U, A, V, t)
    @tensoropt UV[i o; n] = s[j; i] * U[k; j] * A[o; l k] * V[m; l] * t[n; m]
    return UV
end
function gauge_lower_edge!(UV, s, U, A, V, t)
    @tensoropt UV[n o; i] = s[j; i] * U[k; j] * A[o; l k] * V[m; l] * t[n; m]
    return UV
end

function biorth_tsvd(T, χs)
    closure = (x, y) -> tsvd(x; trunc=truncspace(y))

    out = closure.(T, χs)

    U = getindex.(out, 1)
    S = getindex.(out, 2)
    V = getindex.(out, 3)

    return U, S, V
end

function biorth_fixed_point!(C0, AU, AD, coord, increment)
    val, vec, info = eigsolve(C0[coord], 1, :LM; eager=true, maxiter=1) do x0
        temp = similar(x0)
        @inbounds for _ in 1:length(AU)
            # Cant mutate in eigsolve currently
            x0 = biorth_fixed_point_map!(temp, x0, AU[coord], AD[coord])
            coord += increment
        end
        return x0
    end
    copy!(C0[coord], vec[1])
    return C0
end

function biorth_fixed_point_map!(C1, C, AU, AD)
    @tensoropt C1[ur; dr] = C[ul; dl] * AU[p; ur ul] * AD[p; dl dr]
    return C1
end

function biorth_get_gauge_transform(C0, AU, AD; x0, incr, χ)
    biorth_fixed_point!(C0, AU, AD, x0, incr)

    U = similar(C0)
    S = similar(C0)
    V = similar(C0)

    U[x0], S[x0], V[x0] = tsvd(C0[x0]; trunc=truncspace(χ))

    for x in 2:length(C0)
        c = x0 + (x - 1) * incr
        biorth_fixed_point_map!(C0[c], C0[c - incr], AU[c - incr], AD[c - incr])
        U[c], S[c], V[c] = tsvd(C0[c]; trunc=truncspace(χ))
    end

    return U, S, V
end

function biorth_verify(CU, CD, AU, AD, UL, VL; x0, incr, kwargs...)
    r = 1:length(CU)
    ϵu = broadcast(CU) do _
        return 0.0
    end
    ϵd = broadcast(CU) do _
        return 0.0
    end
    for x in r
        c = x0 + (x - 1) * incr

        @tensoropt t1[o; k j] := CU[c][i j] * AU[c][o; k i]
        @tensoropt t2[o; k j] := UL[c][j o; 3] * CU[c + incr][k 3]

        @tensoropt s1[o; i k] := CD[c][i j] * AD[c][o; j k]
        @tensoropt s2[o; i k] := VL[c][i o; 3] * CD[c + incr][3 k]

        ϵu[c] = norm(normalize(t1) - normalize(t2))
        ϵd[c] = norm(normalize(s1) - normalize(s2))

        ϵu[c] < sqrt(eps()) || @warn "CU($c) * TU($c) ≈ PU($c) * CU($(mod(c + incr,r))):" ϵu[c]
        ϵd[c] < sqrt(eps()) || @warn "CD($c) * TD($c) ≈ PD($c) * CD($(mod(c + incr,r))):" ϵd[c]

    end

    return ϵu, ϵd
end


function recleftbiorth!(UL, VL, CU, CD, AU, AD; x0=1, incr=1, χ=domain(CU[1], 1))
    nx = length(AD)

    # S_old = normalize(tsvd(CU[x + 1], (1,), (2,))[2])

    C0 = broadcast(CU, CD) do CU, CD
        @tensoropt C0[a; b] := CU[a c] * CD[b c]
    end


    # val, vec, info = eigsolve(RecursiveVec(C0...), 1, :LM; eager=true, maxiter=1) do x0
    #     recleftbiorth_solve(x0, AU, AD)
    # end

    U, S, V = biorth_get_gauge_transform(C0, AU, AD; x0=x0, incr=incr, χ=χ)

    #
    s = sqrt.(S)
    sinv = pinv.(s)

    biorthness = broadcast(UL) do x
        return 0.0
    end

    for x in 1:nx
        c = x0 + (x - 1) * incr
        gauge_upper_edge!(UL[c], s[c], U[c], AU[c], (U[c + incr])', sinv[c + incr])
        gauge_lower_edge!(VL[c], sinv[c + incr], (V[c + incr])', AD[c], V[c], s[c])
        permute!(CU[c], U[c] * s[c], (), (1, 2))
        permute!(CD[c], s[c] * V[c], (), (1, 2))

        id_maybe = _transpose(UL[c]) * VL[c]
        biorthness[c] = norm(id_maybe / tr(id_maybe) - one(id_maybe) / tr(one(id_maybe)))

        # println(id_maybe)

        # id_maybe = _transpose(UL[x]) * VL[x]
        #
    end

    numiter = 0

    C0 = transpose.(C0)

    while maximum(biorthness) > sqrt(eps()) && numiter <= 10
        ULp = permute.(UL, Ref((2,)), Ref((3, 1)))
        VLp = permute.(VL, Ref((2,)), Ref((1, 3)))

        U, S, V = biorth_get_gauge_transform(C0, ULp, VLp; x0=x0, incr=incr, χ=χ)

        s = sqrt.(S)
        sinv = pinv.(s)

        for x in 1:nx
            c = x0 + (x - 1) * incr
            # gauge_upper_edge!(UL[x], s[x], U[x], ULp[x], (U[x + 1])', sinv[x + 1])
            # gauge_lower_edge!(VL[x], sinv[x + 1], (V[x + 1])', VLp[x], V[x], s[x])
            gauge_upper_edge!(UL[c], s[c], U[c], ULp[c], (U[c + incr])', sinv[c + incr])
            gauge_lower_edge!(VL[c], sinv[c + incr], (V[c + incr])', VLp[c], V[c], s[c])

            permute!(CU[c], permute(CU[c], (1,), (2,)) * U[c] * s[c], (), (1, 2))
            permute!(CD[c], s[c] * V[c] * permute(CD[c], (1,), (2,)), (), (1, 2))

            id_maybe = _transpose(UL[c]) * VL[c]

            biorthness[c] = norm(
                id_maybe / tr(id_maybe) - one(id_maybe) / tr(one(id_maybe))
            )

            # id_maybe = _transpose(UL[x]) * VL[x]
            #
        end
        # println(s)
        numiter += 1
    end

    maximum(biorthness) < sqrt(eps()) || @warn "Biorth tol $biorthness after $numiter"

    biorth_verify(CU,CD,AU,AD,UL,VL; x0=x0, incr=incr)

    return UL, VL
end

function conjugate(t::AbstractTensorMap)
    return TensorMap(conj(t.data), codomain(t), domain(t))
end

const cj = conjugate

# swapvirtual(t::AbsTen{1,2}) = permute(t, (1,), (3, 2))
# swapvirtual(t::AbsTen{2,2}) = permute(t, (1, 2), (4, 3))

swapvirtual(t::AbstractTensorMap) = permutedom(t, (2, 1))

# function flipaxis(t::TensorPair)
#     return TensorPair(permutedom(t.top, (3, 2, 1, 4)), permutedom(t.bot, (3, 2, 1, 4)))
# end

function updatecorners!(ctmrg::CTMRGTensors, ctmrg_p::CTMRGTensors)
    updatecorners!(ctmrg.corners, ctmrg_p.corners)
    return ctmrg
end

function contract(ctmrg::CTMRGTensors, network, i1::UnitRange, i2::UnitRange)
    cs = ctmrg.corners[i1, i2]
    es = map(x -> tuple(x...), ctmrg.edges[i1, i2])
    return _contractall(cs..., es..., network)
end

function testctmrg(data_func)
    βc = log(1 + sqrt(2)) / 2

    s = ℂ^2

    network =
        x -> UnitCell(fill(TensorMap((data_func(x)[1]), one(s), s * s * s' * s'), 2, 2))
    network_magn =
        x -> UnitCell(fill(TensorMap(data_func(x)[2], one(s), s * s * s' * s'), 2, 2))

    rv = []
    rv_exact = []

    alg = CTMRG(;
        bonddim=10,
        verbose=true,
        maxiter=150,
        tol=1e-11,
        svd_alg=TensorKit.SVD(),
        randinit=true,
    )
    # alg = VUMPS(; bonddim=20, verbose=true, maxiter=1)

    X = TensorMap(randn, ComplexF64, s, s)
    Y = TensorMap(randn, ComplexF64, s, s)

    randgauge! =
        (p, X, Y) -> @tensoropt p[a b c d] =
            copy(p)[aa bb cc dd] * X[aa; a] * pinv(X)[c; cc] * Y[bb; b] * pinv(Y)[d; dd]

    for x in 1.111
        # b1 = network(x * βc)
        b1 = randgauge!.(network(x * βc), Ref(X), Ref(Y))
        b2 = randgauge!.(network_magn(x * βc), Ref(X), Ref(Y))
        if x < 1.0
            M = 0.0
        else
            M = abs((1 - sinh(2 * x * βc)^(-4))^(1 / 8))
        end

        cb = (st, args...) -> println(contract(st.tensors, b2) ./ contract(st.tensors, b1))
        state = initialize(b1, alg)#; callback=cb)

        did_converge = false

        calculate!(state)

        Z = contract(state.tensors, b1)
        magn = contract(state.tensors, b2) ./ Z

        push!(rv, abs.(magn)[1, 1])

        push!(rv_exact, M)
    end

    return abs.(rv - rv_exact)
end
function toric(p)
    distr = Dict("X" => p / 3, "Y" => p / 3, "Z" => p / 3, "I" => 1 - p)

    function xprob(dst)
        arr = zeros(2, 2, 2, 2)
        arr[1, 1, 1, 1] = arr[1, 2, 1, 2] = arr[2, 1, 2, 1] = arr[2, 2, 2, 2] = dst["X"]
        arr[1, 2, 1, 1] = arr[1, 1, 1, 2] = arr[2, 2, 2, 1] = arr[2, 1, 2, 2] = dst["I"]
        arr[2, 1, 1, 1] = arr[1, 1, 2, 1] = arr[2, 2, 1, 2] = arr[1, 2, 2, 2] = dst["Y"]
        arr[2, 1, 1, 2] = arr[1, 2, 2, 1] = arr[2, 2, 1, 1] = arr[1, 1, 2, 2] = dst["Z"]
        return arr
    end

    data1 = xprob(distr)
    data2 = permutedims(data1, (2, 1, 4, 3))

    kron = delta(Float64, 2, 2, 2, 2)

    s = ℂ^2

    toten = x -> TensorMap(x, one(s), s * s * s' * s')

    kront = toten(kron)
    ten1 = toten(data1)
    ten2 = toten(data2)

    # bulk = UnitCell([ten1 kront ten1; kront ten2 kront; ten1 kront ten1])
    bulk = UnitCell([ten1 kront; kront ten2])

    alg = CTMRG(;
        bonddim=50,
        verbose=true,
        maxiter=500,
        tol=1e-10,
        svd_alg=TensorKit.SVD(),
        randinit=false,
    )

    state = initialize(bulk, alg)

    calculate!(state)
    out = contract(state.tensors, bulk)

    return out
end

function dimer(D; el=Float64)
    rv = []

    probs = []
    # TS = 0.5:0.01:0.8

    TS = 0.8:0.8

    for T in TS
        β = 1 / T

        a = zeros(el, 4, 4, 4, 4)
        b = zeros(el, 4, 4, 4, 4)

        for ind in CartesianIndices(a)
            if ind[1] ==
                mod(ind[2] + 1, 1:4) ==
                mod(ind[3] + 2, 1:4) ==
                mod(ind[4] + 3, 1:4)
                a[ind] = 1
            end
            if ind[1] ==
                mod(ind[2] - 1, 1:4) ==
                mod(ind[3] - 2, 1:4) ==
                mod(ind[4] - 3, 1:4)
                b[ind] = 1
            end
        end

        q = sqrt([1 0 0 0; 0 exp(β / 2) 1 1; 0 1 1 1; 0 1 1 exp(β / 2)])

        qh = diagm([1, -1, 1, -1]) * q
        qv = diagm([-1, 1, -1, 1]) * q

        @tensoropt aa[ii, jj, kk, ll] :=
            a[i, j, k, l] * q[i, ii] * q[j, jj] * q[k, kk] * q[l, ll]
        @tensoropt bb[ii, jj, kk, ll] :=
            b[i, j, k, l] * q[i, ii] * q[j, jj] * q[k, kk] * q[l, ll]

        δa = zeros(4, 4)
        δa[1, 1] = δa[3, 3] = 1
        δa[2, 2] = δa[4, 4] = -1

        @tensoropt aad[i, j, k, l] := a[ii, j, k, l] * δa[i, ii]
        @tensoropt bbd[i, j, k, l] := b[ii, j, k, l] * δa[i, ii]

        s = ℂ^4

        A = TensorMap(aa, one(s), s * s * s' * s')
        B = TensorMap(bb, one(s), s * s * s' * s')

        AD = TensorMap(aad, one(s), s * s * s' * s')
        BD = TensorMap(bbd, one(s), s * s * s' * s')

        alg = CTMRG(; bonddim=D, maxiter=149, tol=1e-11, randinit=true)
        # alg = VUMPS(; bonddim=D, maxiter=1000)

        bulk = UnitCell([A B; B A])
        dbulk = UnitCell([AD BD; BD AD])

        st = initialize(bulk, alg)

        sto = calculate(st)
        d = (contract(sto.tensors, dbulk) ./ contract(sto.tensors, bulk)) .* 4

        ps = Dict{Int,Matrix{Float64}}([])

        csum = zeros(2, 2)

        for l in 1:4
            δp = zeros(4, 4)
            δp[l, l] = 1

            @tensoropt aap[i, j, k, l] := a[ii, j, k, l] * δp[i, ii]
            @tensoropt bbp[i, j, k, l] := b[ii, j, k, l] * δp[i, ii]

            AP = TensorMap(aap, one(s), s * s * s' * s')
            BP = TensorMap(bbp, one(s), s * s * s' * s')

            pbulk = UnitCell([AP BP; BP AP])

            p = (contract(sto.tensors, pbulk) ./ contract(sto.tensors, bulk))

            ps[l] = abs.(p)

            csum = csum + ps[l]
        end

        for l in 1:4
            ps[l] = ps[l] ./ csum
        end
        push!(probs, ps)
        push!(rv, abs(d[1, 1]))
    end
    return rv, probs
end
