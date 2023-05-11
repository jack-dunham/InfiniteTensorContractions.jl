#TODO: Implement CTMRG; CTMRG is stupid

# Clu[x,y], Tu[x+1,y], Tu[x+2,y], Cru[x+3,y], Tl[x,y+1], a[x+1,y+1],b[x+2,y+1], Tr[x+3,y+1]
#=
Conventions:

(0,3)

(0,2)

(0,1)

(0,0) - (1,0) - (2,0) - (3,0)

C1 T1 T1 C2
T4 M  M  T2
T4 M  M  T2
C3 T3 T3 C3
=#

abstract type AbstractCornerBoundary <: AbstractBoundary end

corners(ctm::AbstractCornerBoundary) = ctm.corners
edges(ctm::AbstractCornerBoundary) = ctm.edges

struct Corners{L,C1Type,C2Type,C4Type}
    C1::OnLattice{L,C1Type,Matrix{C1Type}}
    C2::OnLattice{L,C2Type,Matrix{C2Type}}
    C3::OnLattice{L,C1Type,Matrix{C1Type}}
    C4::OnLattice{L,C4Type,Matrix{C4Type}}
end
struct Edges{L,T1Type,T3Type}
    T1::OnLattice{L,T1Type,Matrix{T1Type}}
    T2::OnLattice{L,T1Type,Matrix{T1Type}}
    T3::OnLattice{L,T3Type,Matrix{T3Type}}
    T4::OnLattice{L,T3Type,Matrix{T3Type}}
end
unpack(ctm::Corners) = ctm.C1, ctm.C2, ctm.C3, ctm.C4
unpack(ctm::Edges) = ctm.T1, ctm.T2, ctm.T3, ctm.T4

struct CTMRG <: AbstractCornerBoundary
    corners::Corners
    edges::Edges
end

function inittensors(f, M, alg::BoundaryAlgorithm{<:AbstractCornerBoundary})
    el = eltype.(M)

    χ = dimtospace(M, alg.bonddim)
    x = Ref(χ)

    C1 = @. TensorMap.(f, el, x, x)
    C2 = @. TensorMap.(f, el, x * x, one(x))
    C3 = @. TensorMap.(f, el, x, x)
    # C3 = C1
    C4 = @. TensorMap.(f, el, one(x), x * x)

    corners = Corners(C1, C2, C3, C4)

    D1 = @. getindex(domain(M), 2)
    D2 = @. getindex(domain(M), 1)
    D3 = @. getindex(codomain(M), 2)
    D4 = @. getindex(codomain(M), 1)

    T1 = @. $circshift(TensorMap(rand, el, x * D1, x), (0, 1))
    # T2 = @. $circshift(TensorMap(rand, el, x * D2 ,x ), (1,0))
    T2 = T1
    T3 = @. $circshift(TensorMap(rand, el, x, D3 * x), (0, -1))
    # T4 = @. $circshift(TensorMap(rand, el, x, D4 * x ), (-1,0))
    T4 = T3

    edges = Edges(T1, T2, T3, T4)
    # edges = initedges(M, χ)

    # return CTMRG(corners, edges)

    return initctm(M, χ)
end

function initctm(M::AbstractMatrix, xi_space)
    out = initctm.(M, Ref(xi_space))

    T1 = getindex.(out, 1)
    T2 = getindex.(out, 2)
    T3 = getindex.(out, 3)
    T4 = getindex.(out, 4)

    C1 = getindex.(out, 5)
    C2 = getindex.(out, 6)
    C3 = getindex.(out, 7)
    C4 = getindex.(out, 8)

    return CTMRG(Corners(C1, C2, C3, C4), Edges(T1, T2, T3, T4))
end

function initctm(M::AbstractTensorMap, xi_space)
    # assume for now that xi_space > phys_space

    west_space = codomain(M)[1]
    south_space = codomain(M)[2]
    east_space = domain(M)[1]
    north_space = domain(M)[2]

    zero_space = one(west_space)

    d_w = isometry(west_space, zero_space)'
    d_s = isometry(south_space, zero_space)'
    d_e = isometry(east_space, zero_space)
    d_n = isometry(north_space, zero_space)

    v_w = isometry(xi_space, west_space)
    v_s = isometry(xi_space, south_space)
    v_e = isometry(xi_space, east_space)'
    v_n = isometry(xi_space, north_space)'

    @tensoropt T1[w s; e] := M[ww s; ee x] * d_n[x] * v_w[w; ww] * v_e[ee; e]
    @tensoropt T2[s w; n] := M[w ss; x nn] * d_e[x] * v_s[s; ss] * v_n[nn; n]
    @tensoropt T3[w; n e] := M[ww x; ee n] * d_s[x] * v_w[w; ww] * v_e[ee; e]
    @tensoropt T4[s; e n] := M[x ss; e nn] * d_w[x] * v_s[s; ss] * v_n[nn; n]

    @tensoropt C1[s; e] := M[y ss; ee x] * d_n[x] * d_w[y] * v_e[ee; e] * v_s[s; ss]
    @tensoropt C2[s w] := M[ww ss; x y] * d_e[x] * v_s[s; ss] * d_n[y] * v_w[w; ww]
    @tensoropt C3[w; n] := M[ww x; y nn] * d_s[x] * v_w[w; ww] * d_e[y] * v_n[nn; n]
    @tensoropt C4[e n] := M[x y; ee nn] * d_w[x] * d_s[y] * v_n[nn; n] * v_e[ee; e]

    return T1, T2, T3, T4, C1, C2, C3, permute(C4, (), (1, 2))
end

#C1[0,2],C2[2,2],C3[2,0],C4[0,0],T1[1,2],T2[2,1],T3[1,0],T4[0,1],M[1,1])
function contractall(c1, c2, c3, c4, t1, t2, t3, t4, m)
    @tensoropt out =
        c1[x3; x1] *
        t1[x1 D1; x2] *
        c2[x2 x5] *
        t4[x6; r1 x3] *
        m[r1 D2; r2 D1] *
        t2[x7 r2; x5] *
        c4[x8 x6] *
        t3[x8; D2 x9] *
        c3[x9; x7]
    return out
end

function contractall2(arguments, x, y)
    return contractall(
        c1[x - 1, y + 1],
        c2[x + 1, y + 1],
        c3[x + 1, y - 1],
        c4[x - 1, y - 1],
        t1[x + 0, y + 1],
        t2[x + 1, y + 0],
        t3[x + 0, y - 1],
        t4[x - 1, y + 0],
        m[x, y],
    )
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
    err = @. norm(S_old - S_new)^2
    S_old .= S_new
    return err
end

function ctmerror(s_old::AbsTen{1,1}, c_new::AbstractTensorMap)
    _, s_new, _ = tsvd(c_new, (2,), (1,))
    normalize!(s_new)
    return s_new
end

function ctmrg_bigtest(xi)
    out = []
    for beta in [0.0:0.01:2.0...]
        val, converged = ctmrg_test(beta; xi=xi)
        count = 0
        while !converged
            val, converged = ctmrg_test(beta; xi=xi)
            count = count + 1
            if count > 5
                break
            end
        end
        push!(out, val)
    end
    return out
end

function ctmrg_test(B; xi=10)
    βc = log(1 + sqrt(2)) / 2
    β = B * βc

    L = Lattice{2,2,Infinite}([ℂ^11 ℂ^12; ℂ^21 ℂ^22])

    aM, aM2 = classicalisingmpo_alt(β)
    tM = TensorMap(Float64.(aM), ℂ^2 * ℂ^2, ℂ^2 * ℂ^2) #mpo
    tM2 = TensorMap(Float64.(aM2), ℂ^2 * ℂ^2, ℂ^2 * ℂ^2) #mpo

    M = OnLattice(L, [tM tM; tM tM])
    M2 = OnLattice(L, [tM2 tM2; tM2 tM2])
    alg = PEPOKit.BoundaryAlgorithm(PEPOKit.CTMRG, xi, 0, 500, 1e-12)

    ctmrg = inittensors(rand, M, alg)
    corners = ctmrg.corners
    edges = ctmrg.edges

    C2_p, C4_p, M_p = ctmrg_permute(corners.C2, corners.C4, M)

    S1 = permute.(corners.C1, Ref((2,)), Ref((1,)))
    S2 = transpose.(corners.C1)
    S3 = permute.(corners.C3, Ref((2,)), Ref((1,)))
    S4 = deepcopy(corners.C1)

    # Error is nonsense from this, we just do this to init S arrays as the singular
    # values of the ctms's>
    ctmerror!(S1, S2, S3, S4, corners)
    mgn = 0.0

    err = Inf

    numiter = 0

    tol = 1e-4

    maybe = false
    while err > tol
        mgn = ctmrgstep!(
            C2_p,
            C4_p,
            M_p,
            unpack(corners)...,
            unpack(edges)...,
            M,
            M2;
            xi=xi,
            runfpcm=maybe,
        )

        err = ctmerror!(S1, S2, S3, S4, corners)

        # println(err)

        if numiter > alg.maxiter
            break
        end

        if mod(numiter, 1:5) == 1
            maybe = false
        else
            maybe = false
        end

        numiter = numiter + 1
    end
    converged = true
    if err > tol
        converged = false
    end
    return mgn, converged
end
function ctmrgstep!(C2_p, C4_p, M_p, C1, C2, C3, C4, T1, T2, T3, T4, M, M2; kwargs...)
    # Left-right move updating all arrays
    leftrightmove!(C1, C2, C3, C4, T1, T2, T3, T4, M; kwargs...)

    # Permute arrays, writing result into placeholder
    ctmrg_permute!(C2_p, C4_p, M_p, C2, C4, M)

    # Up-down sweep updating placeholders and the rest of arrays.
    updownmove!(C1, C2_p, C3, C4_p, T1, T2, T3, T4, M_p; kwargs...)

    # Permute back, writing data into original arrays.
    ctmrg_permute!(C2, C4, M, C2_p, C4_p, M_p)

    Z = contractall(
        C1[0, 2],
        C2[2, 2],
        C3[2, 0],
        C4[0, 0],
        T1[1, 2],
        T2[2, 1],
        T3[1, 0],
        T4[0, 1],
        M[1, 1],
    )
    mgn = contractall(
        C1[0, 2],
        C2[2, 2],
        C3[2, 0],
        C4[0, 0],
        T1[1, 2],
        T2[2, 1],
        T3[1, 0],
        T4[0, 1],
        M2[1, 1],
    )

    println(mgn / Z)

    return mgn / Z
end

function ctmrg_permute(C2, C4, M)
    C2_p = permute.(C2, Ref((2, 1)), Ref(()))
    C4_p = permute.(C4, Ref(()), Ref((2, 1)))
    M_p = permute.(M, Ref((2, 1)), Ref((4, 3)))
    return C2_p, C4_p, M_p
end
function ctmrg_permute!(C2_p, C4_p, M_p, C2, C4, M)
    permute!.(C2_p, C2, Ref((2, 1)), Ref(()))
    permute!.(C4_p, C4, Ref(()), Ref((2, 1)))
    permute!.(M_p, M, Ref((2, 1)), Ref((4, 3)))
    return C2_p, C4_p, M_p
end

# ctmrg_flipargs(C1, C2, C3, C4, T1, T2, T3, T4, M) = C3, C2, C1, C4, T2, T1, T4, T3, M
# function ctmrg_flipall(C1, C2, C3, C4, T1, T2, T3, T4, M) 
#     return ctmrg_flipargs(ctmrg_flip(C1,C2,C3,C4,T1,T2,T3,T4,M)...)
# end

function ctmrg_flip(C1, C2, C3, C4, T1, T2, T3, T4, M)
    C1_f = transpose(C1)
    C2_f = transpose(C2)
    C3_f = transpose(C3)
    C4_f = transpose(C4)
    T1_f = transpose(T1)
    T2_f = transpose(T2)
    T3_f = transpose(T3)
    T4_f = transpose(T4)
    M_f = transpose(M)
    return C1_f, C2_f, C3_f, C4_f, T1_f, T2_f, T3_f, T4_f, M_f
end

function updownmove!(C1, C2, C3, C4, T1, T2, T3, T4, M; kwargs...)
    C1_f, C2_f, C3_f, C4_f, T1_f, T2_f, T3_f, T4_f, M_f = ctmrg_flip(
        C1, C2, C3, C4, T1, T2, T3, T4, M
    )
    leftrightmove!(C3_f, C2_f, C1_f, C4_f, T2_f, T1_f, T4_f, T3_f, M_f; kwargs...)
    return nothing
end

function upthendownmove!(C1, C2, C3, C4, T1, T2, T3, T4, M; kwargs...)
    C1_f, C2_f, C3_f, C4_f, T1_f, T2_f, T3_f, T4_f, M_f = ctmrg_flip(
        C1, C2, C3, C4, T1, T2, T3, T4, M
    )
    leftthenrightmove!(C3_f, C2_f, C1_f, C4_f, T2_f, T1_f, T4_f, T3_f, M_f; kwargs...)
    return nothing
end
function leftthenrightmove!(C1, C2, C3, C4, T1, T2, T3, T4, M; xi=10, runfpcm=false)
    S3 = OnLattice(lattice(M), [ℂ^xi ℂ^xi; ℂ^xi ℂ^xi])
    D = OnLattice(lattice(M), [ℂ^2 ℂ^2; ℂ^2 ℂ^2])
    UL = @. TensorMap(undef, ComplexF64, S3 * D, S3)
    VL = @. TensorMap(undef, ComplexF64, S3, S3 * D)
    UR = @. TensorMap(undef, ComplexF64, S3 * D, S3)
    VR = @. TensorMap(undef, ComplexF64, S3, S3 * D)
    for x in axes(M, 1)
        # Get all the projectors before doing anything
        for y in axes(M, 2)
            # println(lattice(C1)[x,y])
            UL[x + 0, y + 1], VL[x + 0, y + 1], UR[x + 3, y + 1], VR[x + 3, y + 1] = leftrightmove(
                C1[x + 0, y + 3],
                C2[x + 3, y + 3],
                C3[x + 3, y + 0],
                C4[x + 0, y + 0],
                T1[x + 1, y + 3],
                T1[x + 2, y + 3],
                T2[x + 3, y + 2],
                T2[x + 3, y + 1],
                T3[x + 1, y + 0],
                T3[x + 2, y + 0],
                T4[x + 0, y + 2],
                T4[x + 0, y + 1],
                M[x + 1, y + 2],
                M[x + 2, y + 2],
                M[x + 1, y + 1],
                M[x + 2, y + 1];
                xi=xi,
            )
        end
        for y in axes(M, 2)
            project_C!(
                C1[x + 1, y + 0], C1[x + 0, y + 0], T1[x + 1, y + 0], VL[x + 0, y + 0]
            )
            project_T!(
                T4[x + 1, y + 1],
                T4[x + 0, y + 1],
                M[x + 1, y + 1],
                UL[x + 0, y + 0],
                VL[x + 0, y + 1],
            )
            project_C!(
                C4[x + 1, y + 1], C4[x + 0, y + 1], T3[x + 1, y + 1], UL[x + 0, y + 2]
            )
        end
    end
    for x in axes(M, 1)
        # Get all the projectors before doing anything
        for y in axes(M, 2)
            # println(lattice(C1)[x,y])
            UL[x + 0, y + 1], VL[x + 0, y + 1], UR[x + 3, y + 1], VR[x + 3, y + 1] = leftrightmove(
                C1[x + 0, y + 3],
                C2[x + 3, y + 3],
                C3[x + 3, y + 0],
                C4[x + 0, y + 0],
                T1[x + 1, y + 3],
                T1[x + 2, y + 3],
                T2[x + 3, y + 2],
                T2[x + 3, y + 1],
                T3[x + 1, y + 0],
                T3[x + 2, y + 0],
                T4[x + 0, y + 2],
                T4[x + 0, y + 1],
                M[x + 1, y + 2],
                M[x + 2, y + 2],
                M[x + 1, y + 1],
                M[x + 2, y + 1];
                xi=xi,
            )
        end
        for y in axes(M, 2)
            project_C!(
                C2[x + 2, y + 0], C2[x + 3, y + 0], T1[x + 2, y + 0], VR[x + 3, y + 0]
            )
            project_T!(
                T2[x + 2, y + 1],
                T2[x + 3, y + 1],
                M[x + 2, y + 1],
                UR[x + 3, y + 0],
                VR[x + 3, y + 1],
            )
            project_C!(
                C3[x + 2, y + 1], C3[x + 3, y + 1], T3[x + 2, y + 1], UR[x + 3, y + 2]
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
    return normalize!.(T4)
end

function leftrightmove!(C1, C2, C3, C4, T1, T2, T3, T4, M; xi=10, runfpcm=false)
    S3 = OnLattice(lattice(M), [ℂ^xi ℂ^xi; ℂ^xi ℂ^xi])
    D = OnLattice(lattice(M), [ℂ^2 ℂ^2; ℂ^2 ℂ^2])
    UL = @. TensorMap(undef, ComplexF64, S3 * D, S3)
    VL = @. TensorMap(undef, ComplexF64, S3, S3 * D)
    UR = @. TensorMap(undef, ComplexF64, S3 * D, S3)
    VR = @. TensorMap(undef, ComplexF64, S3, S3 * D)
    for x in axes(M, 1)
        # Get all the projectors before doing anything
        for y in axes(M, 2)
            # println(lattice(C1)[x,y])
            UL[x + 0, y + 1], VL[x + 0, y + 1], UR[x + 3, y + 1], VR[x + 3, y + 1] = leftrightmove(
                C1[x + 0, y + 3],
                C2[x + 3, y + 3],
                C3[x + 3, y + 0],
                C4[x + 0, y + 0],
                T1[x + 1, y + 3],
                T1[x + 2, y + 3],
                T2[x + 3, y + 2],
                T2[x + 3, y + 1],
                T3[x + 1, y + 0],
                T3[x + 2, y + 0],
                T4[x + 0, y + 2],
                T4[x + 0, y + 1],
                M[x + 1, y + 2],
                M[x + 2, y + 2],
                M[x + 1, y + 1],
                M[x + 2, y + 1];
                xi=xi,
            )
        end
        # println(normalize(normalize(VR[4,1]) * normalize(UR[4,1])))
        for y in axes(M, 2)
            # Left move
            project_C!(
                C1[x + 1, y + 0], C1[x + 0, y + 0], T1[x + 1, y + 0], VL[x + 0, y + 0]
            )
            project_T!(
                T4[x + 1, y + 1],
                T4[x + 0, y + 1],
                M[x + 1, y + 1],
                UL[x + 0, y + 0],
                VL[x + 0, y + 1],
            )
            project_C!(
                C4[x + 1, y + 1], C4[x + 0, y + 1], T3[x + 1, y + 1], UL[x + 0, y + 2]
            )
            # Right move
            project_C!(
                C2[x + 2, y + 0], C2[x + 3, y + 0], T1[x + 2, y + 0], VR[x + 3, y + 0]
            )
            project_T!(
                T2[x + 2, y + 1],
                T2[x + 3, y + 1],
                M[x + 2, y + 1],
                UR[x + 3, y + 0],
                VR[x + 3, y + 1],
            )
            project_C!(
                C3[x + 2, y + 1], C3[x + 3, y + 1], T3[x + 2, y + 1], UR[x + 3, y + 2]
            )
        end
    end

    if runfpcm == true
        for y in axes(M, 2)
            _, C1s, _ = eigsolve(
                z -> project_C_rec(z, T1[:, y], VL[:, y]; dir=:left),
                RecursiveVec(C1[:, y]...),
                1,
                :LM,
            )
            C1[:, y] .= real.(C1s[1])

            _, C4s, _ = eigsolve(
                z -> project_C_rec(z, T3[:, y], UL[:, y + 1]; dir=:left),
                RecursiveVec(C4[:, y]...),
                1,
                :LM,
            )
            C4[:, y] .= real.(C4s[1])

            _, C2s, _ = eigsolve(
                z -> project_C_rec(z, T1[:, y], VR[:, y]; dir=:right),
                RecursiveVec(C2[:, y]...),
                1,
                :LM,
            )
            C2[:, y] .= real.(C2s[1])

            _, C3s, _ = eigsolve(
                z -> project_C_rec(z, T3[:, y], UR[:, y + 1]; dir=:right),
                RecursiveVec(C3[:, y]...),
                1,
                :LM,
            )
            C3[:, y] .= real.(C3s[1])

            _, T4s, _ = eigsolve(
                z -> project_T_rec(z, M[:, y], UL[:, y - 1], VL[:, y]; dir=:left),
                RecursiveVec(T4[:, y]...),
                1,
                :LM,
            )
            T4[:, y] .= real.(T4s[1])

            _, T2s, _ = eigsolve(
                z -> project_T_rec(z, M[:, y], UR[:, y - 1], VR[:, y]; dir=:right),
                RecursiveVec(T2[:, y]...),
                1,
                :LM,
            )
            T2[:, y] .= real.(T2s[1])
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

    return nothing
end

# C[:, y], T[: + 1,y], UV[:, y]
function project_C_rec(C, T, UV; dir=:left)
    if dir === :left
        i = 1
    elseif dir === :right
        i = -1
    else
        throw(ArgumentError(""))
    end

    rv = project_C.(C, circshift(T, -i), UV)

    rv = circshift(rv, i)
    return RecursiveVec(rv...)
end
function project_T_rec(T, M, U, V; dir=:left)
    if dir === :left
        i = 1
    elseif dir === :right
        i = -1
    else
        throw(ArgumentError(""))
    end

    rv = project_T.(T, circshift(M, -i), U, V)

    rv = circshift(rv, i)
    return RecursiveVec(rv...)
end

# Phys indices come second on projectors 
function project_C!(C1_dst::AbsTen{1,1}, C1_src, T1::AbsTen{2,1}, VL) #done (no update)
    @tensoropt C1_dst[x4; x2] = VL[x4; x3 D1] * C1_src[x3; x1] * T1[x1 D1; x2]
    return C1_dst
end
function project_C!(C4_dst::AbsTen{0,2}, C4_src, T3::AbsTen{1,2}, UL) #done
    @tensoropt C4_dst[x4 x2] = UL[x1 D1; x2] * C4_src[x3 x1] * T3[x3; D1 x4]
    return C4_dst
end
function project_T!(T4_dst::AbsTen{1,2}, T4_src, M, UL, VL) #done (no updated)
    @tensoropt T4_dst[x4; D3 x2] =
        UL[x1 D1; x2] * T4_src[x3; D2 x1] * M[D2 D4; D3 D1] * VL[x4; x3 D4]
    return T4_dst
end

# Phys index conj here on VR. Dont bother correcting as it gets contracted.
function project_C!(C2_dst::AbsTen{2,0}, C2_src, T1::AbsTen{2,1}, VR) #done
    @tensoropt C2_dst[x1 x3] = T1[x1 D1; x2] * C2_src[x2 x4] * VR[x3; x4 D1]
    return C2_dst
end
function project_C!(C3_dst::AbsTen{1,1}, C3_src, T3::AbsTen{1,2}, UR)
    @tensoropt C3_dst[x3; x1] = UR[x2 D1; x1] * T3[x3; D1 x4] * C3_src[x4; x2]
    return C3_dst
end
function project_T!(T2_dst::AbsTen{2,1}, T2_src, M, UR, VR)
    @tensoropt T2_dst[x3 D2; x1] =
        UR[x2 D1; x1] * M[D2 D4; D3 D1] * T2_src[x4 D3; x2] * VR[x3; x4 D4]
    return T2_dst
end

project_C(C, T, UV) = project_C!(similar(C), C, T, UV)
project_T(T, M, U, V) = project_T!(similar(T), T, M, U, V)

leftrightmove_test(args...) = leftrightmove!(deepcopy.(args)...)
function leftrightmove(
    C1_03,
    C2_33,
    C3_30,
    C4_00,
    T1_13,
    T1_23,
    T2_32,
    T2_31,
    T3_10,
    T3_20,
    T4_02,
    T4_01,
    M_12,
    M_22,
    M_11,
    M_21,
    ;
    xi=10,
)
    top = topsvd_new(C1_03, C2_33, T1_13, T1_23, T2_32, T4_02, M_12, M_22)
    U, S, V = tsvd!(top; alg=TensorKit.SVD())
    normalize!(S)
    UL_02 = U * sqrt(S)
    UR_32 = permute(sqrt(S) * V, (2, 3), (1,)) # Should have indices in correct place

    bot = botsvd_new(C3_30, C4_00, T2_31, T3_10, T3_20, T4_01, M_11, M_21)
    V, S, U = tsvd!(bot; alg=TensorKit.SVD())
    normalize!(S)
    VL_02 = sqrt(S) * U
    VR_32 = permute(V * sqrt(S), (3,), (1, 2)) # likewise

    # Biorth

    UL_02, VL_02 = biorthogonalize(UL_02, VL_02; xi=xi)
    UR_32, VR_32 = biorthogonalize(UR_32, VR_32; xi=xi)

    # println(isapprox(VL_02 * UL_02 , one(VL_02 * UL_02)/xi ))
    # println(isapprox(VR_32 * UR_32, one(VR_32 * UR_32)/xi ))

    return UL_02, VL_02, UR_32, VR_32
end

function biorthogonalize(U, V; xi=10)
    W, S, Q = tsvd!(V * U; trunc=truncdim(xi), alg=TensorKit.SVD())

    sqS = sqrt(S)
    normalize!(sqS)

    U = U * Q' * pinv(sqS; rtol=5e-8)
    V = pinv(sqS; rtol=5e-8) * W' * V

    normalize!(U)
    normalize!(V)

    isapprox(normalize(V * U), one(V * U) / sqrt(xi)) || @warn "Biorthogonalization failed"

    return U, V
end

function adv_biorth(C, AU, AV)
    vals, vecs, _ = eigsolve(z -> biorthfp(z, AU, AV), C, 1, :LM)
    return U, S, V = tsvd(vecs[1] / vals[1])
end

function biorthfp(C, AU, AV)
    CC = similar(C)
    @tensoropt CC[x4 x2] = C[x3 x1] * AU[x1 D1; x2] * AV[x3 D1; x4]
    return CC
end

function topsvd(C1_00, C2_30, T1_10, T1_20, T2_31, T4_01, M_11, M_21)
    @tensoropt out[x6 D3; x7 D4] :=
        C1_00[x4; x1] *
        T1_10[x1 D1; x2] *
        T1_20[x2 D2; x3] *
        C2_30[x3; x5] *
        T4_01[x6 r1; x4] *
        M_11[r1 D3; r2 D1] *
        M_21[r2 D4; r3 D2] *
        T2_31[x5 r3; x7]
    return out
end
function topsvd_new(C1_03, C2_33, T1_13, T1_23, T2_32, T4_02, M_12, M_22)
    @tensoropt out[x6 D3; x7 D4] :=
        C1_03[x4; x1] *
        T1_13[x1 D1; x2] *
        T1_23[x2 D2; x3] *
        C2_33[x3 x5] *
        T4_02[x6 r1; x4] *
        M_12[r1 D3; r2 D1] *
        M_22[r2 D4; r3 D2] *
        T2_32[x7 r3; x5]
    return out
end

function botsvd(C3_33, C4_03, T2_32, T3_13, T3_23, T4_02, M_12, M_22)
    @tensoropt out[x2 D2; x1 D1] :=
        T4_02[x3; r1 x1] *
        M_12[r1 D3; r2 D1] *
        M_22[r2 D4; r3 D2] *
        T2_32[x2 r3; x4] *
        C4_03[x5; x3] *
        T3_13[x6; D3 x5] *
        T3_23[x7 D4; x6] *
        C3_33[x4; x7]
    return out
end
function botsvd_new(C3_30, C4_00, T2_31, T3_10, T3_20, T4_01, M_11, M_21)
    @tensoropt out[x2 D2; x1 D1] :=
        T4_01[x3; r1 x1] *
        M_11[r1 D3; r2 D1] *
        M_21[r2 D4; r3 D2] *
        T2_31[x4 r3; x2] *
        C4_00[x5 x3] *
        T3_10[x5; D3 x6] *
        T3_20[x6 D4; x7] *
        C3_30[x7; x4]
    return out
end

# For the corner, first index is always in the direction of move, second index
# is the one to be projected, as such, sometimes we may need to correct the 
function contract_corner(C, T, U)
    @tensoropt C[x1 x2] * T[D1; x2 x3] * U[]
end
#=
D = ℂ^3 # Bond dim
d = ℂ^2 # phys/mpo dim
nil = one(D)

T = ComplexF64

C1_00 = TensorMap(rand,T, nil, D * D)
T1_10 = T1_20 = TensorMap(rand,T, d', D * D')
C2_30 = TensorMap(rand, T, nil, D' * D')
T4_01 = TensorMap(rand, T, d',  D * D')
M_11  = M_21  = TensorMap(rand, T, nil, d * d * d' * d')
T3_31 = TensorMap(rand, T, d, D * D')

=#
function test_halfsvd()
    D = ℂ^3 # Bond dim
    d = ℂ^2 # phys/mpo dim
    nil = one(D)

    L = Lattice{2,2,Infinite}([ℂ^11 ℂ^12; ℂ^21 ℂ^22])

    aM, aM2 = classicalisingmpo_alt(0.44)

    T = ComplexF64

    #Corners
    C1 = C3 = fill(TensorMap(rand, T, nil, D * D), L)
    C2 = C4 = fill(TensorMap(rand, T, nil, D' * D'), L)

    #Edges
    T1 = fill(TensorMap(rand, T, d', D * D'), L)
    T4 = fill(TensorMap(rand, T, d', D * D'), L)

    T2 = fill(TensorMap(rand, T, d, D * D'), L)
    T3 = fill(TensorMap(rand, T, d, D * D'), L)

    #MPO
    # M_11 = M_21 = M_12 = M_22 = fill(TensorMap(aM, nil, d * d * d' * d'),L)
    M = fill(TensorMap(aM, nil, d * d * d' * d'), L)

    for i in 1:100
    end
end

function ctmrg_permute(C1, C3, M)
    C1_p = permute.(C1, Ref(()), Ref((2, 1)))
    C3_p = permute.(C3, Ref(()), Ref((2, 1)))
    M_p = permute.(M, Ref(()), Ref((2, 1, 4, 3)))
    return C1_p, C3_p, M_p
end
function ctmrg_permute!(C1_p, C3_p, M_p, C1, C3, M)
    permute!.(C1_p, C1, Ref(()), Ref((2, 1)))
    permute!.(C3_p, C3, Ref(()), Ref((2, 1)))
    permute!.(M_p, M, Ref(()), Ref((2, 1, 4, 3)))
    return C1_p, C3_p, M_p
end

function new_ctrmg(xi)
    D = ℂ^xi # Bond dim
    d = ℂ^2 # phys/mpo dim
    nil = one(D)

    L = Lattice{2,2,Infinite}([ℂ^11 ℂ^12; ℂ^21 ℂ^22])

    aM, aM2 = classicalisingmpo_alt(0.44)

    T = Float64

    #Corners
    C1 = C3 = fill(TensorMap(rand, T, nil, D * D), L)
    C2 = C4 = fill(TensorMap(rand, T, nil, D' * D'), L)

    #Edges
    T1 = fill(TensorMap(rand, T, d', D * D'), L)
    T4 = fill(TensorMap(rand, T, d', D * D'), L)

    T2 = fill(TensorMap(rand, T, d, D * D'), L)
    T3 = fill(TensorMap(rand, T, d, D * D'), L)

    #MPO
    # M_11 = M_21 = M_12 = M_22 = fill(TensorMap(aM, nil, d * d * d' * d'),L)
    M = fill(TensorMap(aM, nil, d * d * d' * d'), L)
    M2 = fill(TensorMap(aM2, nil, d * d * d' * d'), L)

    C1_p, C3_p, M_p = ctmrg_permute(C1, C3, M)

    for i in 1:100
        _, S_old, _ = tsvd(permute(C1[1, 1], (1,), (2,)))
        normalize!(S_old)
        ctmrg_step!(C1_p, C3_p, M_p, C1, C2, C3, C4, T1, T2, T3, T4, M, M2, D)
        _, S, _ = tsvd(permute(C1[1, 1], (1,), (2,)))
        normalize!(S)
        println(norm(S - S_old))
    end
end
function ctmrg_step!(C1_p, C3_p, M_p, C1, C2, C3, C4, T1, T2, T3, T4, M, M2, D)
    # Left-right move updating all arrays
    ctmrgmove!(C1, C2, C3, C4, T1, T2, T3, T4, M, D)

    # Permute arrays, writing result into placeholder
    ctmrg_permute!(C1_p, C3_p, M_p, C1, C3, M)

    # Up-down sweep updating placeholders and the rest of arrays.
    updownctmrgmove!(C1_p, C2, C3_p, C4, T1, T2, T3, T4, M_p, D)

    # Permute back, writing data into original arrays.
    ctmrg_permute!(C1, C3, M, C1_p, C3_p, M_p)

    Z = ctmrg_expval(
        C1[0, 0],
        C2[2, 0],
        C3[2, 2],
        C4[0, 2],
        T1[1, 0],
        T2[2, 1],
        T3[1, 2],
        T4[0, 1],
        M[1, 1],
    )
    mgn = ctmrg_expval(
        C1[0, 0],
        C2[2, 0],
        C3[2, 2],
        C4[0, 2],
        T1[1, 0],
        T2[2, 1],
        T3[1, 2],
        T4[0, 1],
        M2[1, 1],
    )
    println("Mag:  ", mgn / Z)

    return mgn / Z
end
function updownctmrgmove!(C1, C2, C3, C4, T1, T2, T3, T4, M, D)
    println("updown")
    C1_f, C2_f, C3_f, C4_f, T1_f, T2_f, T3_f, T4_f, M_f = ctmrg_flip(
        C1, C2, C3, C4, T1, T2, T3, T4, M
    )
    ctmrgmove!(C1_f, C4_f, C3_f, C2_f, T4_f, T3_f, T2_f, T1_f, M_f, D)
    return nothing
end

function ctmrgmove!(C1, C2, C3, C4, T1, T2, T3, T4, M, D)
    d = ℂ^2 # phys/mpo dim
    L = lattice(M)
    UL = fill(TensorMap(undef, ComplexF64, D' * d', D'), L)
    VL = fill(TensorMap(undef, ComplexF64, D * d, D), L)
    UR = fill(TensorMap(undef, ComplexF64, D * d', D), L)
    VR = fill(TensorMap(undef, ComplexF64, D' * d, D'), L)
    for x in axes(M, 1)
        for y in axes(M, 2)
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
                M[x + 1, y + 1],
                M[x + 2, y + 1],
                M[x + 1, y + 2],
                M[x + 2, y + 2],
                dim(D),
            )
        end
        for y in axes(M, 2)
            project_corner!(
                C1[x + 1, y + 0], C1[x + 0, y + 0], T1[x + 1, y + 0], VL[x + 0, y + 0]
            )
            project_corner!(
                C2[x + 2, y + 0],
                C2[x + 3, y + 0],
                permute(T1[x + 2, y + 0], (1,), (3, 2)),
                VR[x + 3, y + 0],
            )
            project_corner!(
                C3[x + 2, y + 3], C3[x + 3, y + 3], T3[x + 2, y + 3], UR[x + 3, y + 2]
            )
            project_corner!(
                C4[x + 1, y + 3],
                C4[x + 0, y + 3],
                permute(T3[x + 1, y + 3], (1,), (3, 2)),
                UL[x + 0, y + 2],
            )
            project_edge!(
                T4[x + 1, y + 1],
                T4[x + 0, y + 1],
                M[x + 1, y + 1],
                UL[x + 0, y + 0],
                VL[x + 0, y + 1],
            )
            project_edge!(
                T2[x + 2, y + 1],
                T2[x + 3, y + 1],
                permute(M[x + 2, y + 1], (3, 4, 1, 2)),
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

    return nothing
end

# O(χ^2 D^8) or D^10
function halfsvd(C1_00, T1_10, T1_20, C2_30, T4_01, M_11, M_21, T2_31)
    TensorKit.TensorOperations.@optimalcontractiontree (
        h1 => 2 * D,
        h2 => 2 * D,
        h3 => 2 * D,
        h4 => D^2,
        h5 => D^2,
        h6 => D^2,
        v1 => 2 * D,
        v2 => D^2,
        v3 => D^2,
        v4 => 2 * D,
        v5 => 2 * D,
        v6 => D^2,
        v7 => D^2,
        v8 => D^2,
    ) out[v8 v7; v5 v6] :=
        C1_00[h1 v1] *
        T1_10[v2; h2 h1] *
        T1_20[v3; h3 h2] *
        C2_30[h3 v4] *
        T4_01[h4; v5 v1] *
        M_11[h5 v6 h4 v2] *
        M_21[h6 v7 h5 v3] *
        T2_31[h6; v4 v8]
    return out
end

# O(χ^3 D^6) or D^9 or r^2 χ^3 D^6
function halfcontract(C1_00, T1_10, T1_20, C2_30, T4_01, M_11, M_21, T2_31)
    TensorKit.TensorOperations.@optimalcontractiontree (
        k1 => 2,
        k2 => 2,
        b1 => 2,
        b2 => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D8 => D,
        E1 => D,
        E2 => D,
        E3 => D,
        E4 => D,
        E5 => D,
        E6 => D,
        E8 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
        x5 => D,
        x6 => D,
        x7 => D,
    ) top[x7 D6 E6; x6 D2 E2] :=
        M_11[k1 b1; D1 D2 D3 D4] *
        M_21[k2 b2; D5 D6 D1 D8] *
        (M_11)'[E1 E2 E3 E4; k1 b1] *
        (M_21)'[E5 E6 E1 E8; k2 b2] *
        C1_00[x1 x4] *
        T1_10[D4 E4; x2 x1] *
        T1_20[D8 E8; x3 x2] *
        C2_30[x3 x5] *
        T4_01[D3 E3; x6 x4] *
        T2_31[D5 E5; x5 x7]
end

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
    xi,
)
    # Top
    top = halfsvd(C1_00, T1_10, T1_20, C2_30, T4_01, M_11, M_21, T2_31)
    U, S, V = tsvd(top; alg=TensorKit.SVD())
    # println(normalize(S))
    FUL = sqrt(S) * V
    FUR = U * sqrt(S)

    # Bottom (requires permutation)
    MP_12 = permute(M_12, (), (3, 4, 1, 2))
    MP_22 = permute(M_22, (), (3, 4, 1, 2))

    bot = halfsvd(C3_33, T3_23, T3_13, C4_03, T2_32, MP_22, MP_12, T4_02)
    U, S, V = tsvd(bot)
    FDL = U * sqrt(S)
    FDR = sqrt(S) * V

    # Biorthogonalization
    Q, S, W = tsvd!(FUL * FDL; trunc=truncdim(xi), alg=TensorKit.SVD())
    SX = pinv(sqrt(S); rtol=1e-7)
    normalize!(SX)
    UL = _transpose(SX * Q' * FUL)
    VL = FDL * W' * SX

    W, S, Q = tsvd!(FDR * FUR; trunc=truncdim(xi), alg=TensorKit.SVD())
    SX = pinv(sqrt(S); rtol=1e-7)
    normalize!(SX)
    UR = FUR * Q' * SX
    VR = _transpose(SX * W' * FDR)

    normalize!(UL)
    normalize!(VL)
    normalize!(UR)
    normalize!(VR)

    return UL, VL, UR, VR
end

project_corner(c, t, p) = project_corner!(similar(c), c, t, p)
function project_corner!(c_dst, c_src, t::AbsTen{1,2}, uv)
    # P: (x,D)<-(x_out)
    @tensoropt c_dst[h0 v0] = c_src[h1 v1] * t[v2; h0 h1] * uv[v1 v2; v0]
end
function project_corner!(c_dst, c_src, t::AbsTen{2,2}, uv)
    # P: (x,D)<-(x_out)
    @tensoropt c_dst[h0 v0] = c_src[h1 v1] * t[v2 x; h0 h1] * uv[v1 v2 x; v0]
end

project_edge(t, m, u, v) = project_edge!(similar(t), t, m, u, v)
function project_edge!(t_dst::AbsTen{1,2}, t_src::AbsTen{1,2}, m, u, v)
    @tensoropt t_dst[h0; v0_d v0_u] =
        t_src[h1; v3 v1] * m[h0 v4 h1 v2] * u[v1 v2; v0_u] * v[v3 v4; v0_d]
end
function project_edge!(t_dst::AbsTen{2,2}, t_src::AbsTen{2,2}, m, u, v)
    @tensoropt (
        k => 2,
        b => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D7 => D,
        D8 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
    ) t_dst[D1 D5; x4 x2] =
        t_src[D3 D7; x3 x1] *
        m[k b; D1 D2 D3 D4] *
        (m')[D5 D6 D7 D8; k b] *
        u[x1 D4 D8; x2] *
        v[D2 D6; x4]
end

function ctmrg_expval(C1, C2, C3, C4, T1, T2, T3, T4, M)
    TensorKit.TensorOperations.@optimalcontractiontree (
        h4 => D^2,
        h3 => D^2,
        v5 => D^2,
        v2 => D^2,
        h1 => D,
        h2 => D,
        h5 => D,
        h6 => D,
        v1 => D,
        v3 => D,
        v4 => D,
        v6 => D,
    ) expval =
        C1[h1 v1] *
        T1[v2; h2 h1] *
        C2[h2 v3] *
        T4[h3; v4 v1] *
        M[h4 v5 h3 v2] *
        T2[h4; v3 v6] *
        C4[h5 v4] *
        T3[v5; h5 h6] *
        C3[h6 v6]
    return expval
end

function ctmrg_expval2(C1, C2, C3, C4, T1, T2, T3, T4, M)
    TensorKit.TensorOperations.@optimalcontractiontree (
        b => 2,
        k => 2,
        h1 => D,
        h2 => D,
        h3 => 16 * D,
        h4 => 16 * D,
        h5 => D,
        h6 => D,
        v1 => D,
        v2 => D,
        u1 => D,
        u2 => 16 * D,
        u3 => D,
        u4 => 16 * D,
        v1 => D,
        v2 => D,
        v3 => D,
        v4 => D,
        v5 => D,
        v6 => D,
    ) expval =
        C1[h1 v1] *
        T1[v2 u1; h2 h1] *
        C2[h2 v3] *
        T4[h3 u4; v4 v1] *
        M[b k; h4 v5 h3 v2] *
        (M')[u2 u3 u4 u1; b k] *
        T2[h4 u2; v3 v6] *
        C4[h5 v4] *
        T3[v5 u3; h5 h6] *
        C3[h6 v6]
    return expval
end

# function contractphysical_merge(P::AbstractTensorMap{S,2,4}) where {S}
#     TensorKit.TensorOperations.disable_cache()
#
#     east = fuse(domain(P)[1] ⊗ domain(P)[1]')
#     south = fuse(domain(P)[2] ⊗ domain(P)[2]')
#
#     west = fuse(domain(P)[3] ⊗ domain(P)[3]')'
#     north = fuse(domain(P)[4] ⊗ domain(P)[4]')'
#
#     temp_space = *(map(x -> x * x',domain(P))...)
#
#     out = similar(P, one(S), temp_space)
#
#     @time @tensoropt out[e1 e2 s1 s2 w1 w2 n1 n2] = P[k b; e1 s1 w1 n1] * (P')[e2 s2 w2 n2;k b]
#
#     d = domain(P)[1]
#     D = d
#
#     T = ComplexF64
#     nil = one(d)
#
#     C1 = C3 = TensorMap(rand, T, nil, D * D)
#     C2 = C4 = TensorMap(rand, T, nil, D' * D')
#
#     #Edges
#     T1 = TensorMap(rand, T, north, D * D')
#     T4 = TensorMap(rand, T, west, D * D')
#
#     T2 = TensorMap(rand, T, east, D * D')
#     T3 = TensorMap(rand, T, south, D * D')
#     #####
#     D1 = D3 = TensorMap(rand, T, nil, D * D)
#     D2 = D4 = TensorMap(rand, T, nil, D' * D')
#
#     #Edges
#     S1 = TensorMap(rand, T, d' * d, D * D')
#     S4 = TensorMap(rand, T, d' * d, D * D')
#
#     S2 = TensorMap(rand, T, d * d', D * D')
#     S3 = TensorMap(rand, T, d * d', D * D')
#
#     rv = TensorMap(out.data, one(S), east * south * west * north )
#
#     @time ctmrg_expval(C1,C2,C3,C4,T1,T2,T3,T4,rv)
#     @time ctmrg_expval2(D1,D2,D3,D4,S1,S2,S3,S4,P)
#
#     return nothing
# end
