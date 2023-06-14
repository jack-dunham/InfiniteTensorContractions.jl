"""
    VUMPS <: AbstractBoundaryAlgorithm

Stores the parameters for the variational uniform matrix product state (VUMPS)  boundary algorithm. 

# Fields
- `bonddim::Int`: the bond dimension of the boundary
- `maxiter::Int = 100`: maximum number of iterations
- `tol::Float64 = 1e-12`: convergence tolerance
- `verbose::Bool = true`: when true, will print algorithm convergence progress
"""
@kwdef struct VUMPS <: AbstractBoundaryAlgorithm
    bonddim::Int
    maxiter::Int = 100
    tol::Float64 = 1e-12
    verbose::Bool = true
    function VUMPS(bonddim::Int, maxiter::Float64, tol::Float64, verbose::Bool)
        if maxiter == Inf
            maxiter = typemax(Int)
            return new(bonddim, maxiter, tol, verbose)
        else
            throw(
                ArgumentError(
                    "Unsupported `Float64` value of `maxiter`. Please supply an `Int`, or `Inf` for `maxiter = typemax(Int)`",
                ),
            )
        end
    end
    function VUMPS(bonddim::Int, maxiter::Int, tol::Float64, verbose::Bool)
        return new(bonddim, maxiter, tol, verbose)
    end
end

struct VUMPSTensors{
    AType<:AbstractUnitCell,CType<:AbstractUnitCell,FType<:AbstractUnitCell
} <: AbstractBoundaryTensors
    mps::MPS{AType,CType}
    fixedpoints::FixedPoints{FType}
end

contraction_boundary_type(::VUMPSTensors) = VUMPS

function Base.similar(vumps::VUMPSTensors)
    return VUMPSTensors(similar(vumps.mps), similar(vumps.fixedpoints))
end

function inittensors(f, bulk, alg::VUMPS)
    # D = @. getindex(domain(bulk), 4)

    _, _, _, north_bonds = bondspace(bulk)

    χ = dimtospace(spacetype(bulk), alg.bonddim)

    chi = similar(north_bonds, typeof(χ))

    chi .= Ref(χ)

    boundary_mps = MPS(f, numbertype(bulk), north_bonds, chi)

    fixed_points = fixedpoints(boundary_mps, bulk)

    return VUMPSTensors(boundary_mps, fixed_points)
end

function start(state::BoundaryState{VUMPS})
    vumps = state.tensors
    sing_val = x -> tsvd(x, (1,), (2,))[2]
    return (sing_val.(getbond(vumps.mps)),)
end

function step!(
    vumps::VUMPSTensors,    #mutating
    bulk::ContractableTensors,
    ::VUMPS,
    singular_values,        #mutating
)
    vumpsstep!(vumps, bulk)

    error_per_site = boundaryerror!(singular_values, getbond(vumps.mps))

    @debug "Error per site:" ϵᵢ = error_per_site

    return max(error_per_site...)
end

function vumpsstep!(vumps::VUMPSTensors, bulk)
    mps = vumps.mps
    fps = vumps.fixedpoints

    vumpsupdate!(mps, fps, bulk) # Vectorised

    fixedpoints!(fps, mps, bulk)

    return vumps
end

# Bulk of work
function vumpsupdate!(A::MPS, FP::FixedPoints, M)
    FL = FP.left
    FR = FP.right

    nx, ny = size(M)
    rx = 1:nx
    ry = 1:ny

    AC = getcentral(A)
    C = getbond(A)

    for x in rx

        # take mps[y], get mps[y].AC, send in mps[y].AC[x]
        # First solve for the new AC tensors.
        μ1s, ACs, _ = eigsolve(
            z -> applyhac(z, (FL[x, :]), (FR[x, :]), (M[x, :])),
            RecursiveVec((AC[x, :])...),
            1,
            :LM;
            ishermitian=false,
        )

        for y in ry
            AC[x, y] = ACs[1][y]
        end
        # now set mps[y-1, mod].AC[x] to output of above

        # A[mod(y - 1, ry)].AC[x] = ACs[1]
        μ0s, Cs, _ = eigsolve(
            z -> applyhc(z, (FL[x + 1, :]), (FR[x, :])),
            RecursiveVec((C[x, :])...),
            1,
            :LM;
            ishermitian=false,
        )

        @debug "Effective Hamiltonian eigenvalues:" μ1 = μ1s[1] μ0 = μ0s[1] μ1 / μ0 =
            (μ1s[1] / μ0s[1])

        for y in ry
            C[x, y] = Cs[1][y]
        end
        # A[mod(y - 1, ry)].C[x] = Cs[1]
        # λ = real(μ1s[1] / μ0s[1])
    end

    #A now as updated AC, C, need to update AL, AR
    _, errL, errR = updateboth!(A)

    @debug "MPS update errors: " ϵL = findmax(errL)[1] ϵR = findmax(errR)[1]

    return A
end

# EFFECTIVE HAMILTONIANS

function applyhac(
    z::RecursiveVec, FL::AbstractVector, FR::AbstractVector, M::AbstractVector
)
    rv = deepcopy(circshift([z.vecs...], -1))
    applyhac!.(rv, z, FL, FR, M)
    rv = circshift(rv, 1)
    return RecursiveVec(rv...)::typeof(z)
end

function applyhc(z::RecursiveVec, FL::AbstractVector, FR::AbstractVector)
    rv = deepcopy(circshift([z.vecs...], -1))
    applyhc!.(rv, z, FL, FR)
    rv = circshift(rv, 1)
    return RecursiveVec(rv...)::typeof(z)
end

# MPS UPDATE
# TODO: MOVE TO MPS FILE?

function updateleft!(al::AbstractTensorMap, ac::AbstractTensorMap, c::AbstractTensorMap)
    # qac, rac = leftorth(ac)
    qac, rac = _leftorth(ac)
    qc, rc = _leftorth(c)

    normalize!(rac)
    normalize!(rc)

    mulbond!(al, qac, permute((qc'), (), (2, 1)))
    errL = norm(rac - rc)

    return errL
end

function updateleft!(A::MPS)
    AL, C, _, AC = unpack(A)
    errL = updateleft!.(AL, AC, C)
    return errL
end

function updateright!(ar::AbstractTensorMap, ac::AbstractTensorMap, c::AbstractTensorMap)
    lac, qac = _rightorth(ac)
    lc, qc = _rightorth(c)

    normalize!(lac)
    normalize!(lc)

    mulbond!(ar, permute((qc'), (), (2, 1)), qac)
    errR = norm(lac - lc)

    return errR
end

# function updateright!(ar::AbstractTensorMap, ac::AbstractTensorMap, c::AbstractTensorMap)
#     ac_p = permutedom(ac, (2, 1))
#     c_p = permutedom(c, (2, 1))
#
#     ar_p = deepcopy(ac_p)
#
#     errR = updateleft!(ar_p, ac_p, c_p)
#
#     permute!(ar, ar_p, (1,), (3, 2))
#
#     return errR
# end

function updateright!(A::MPS)
    _, C, AR, AC = unpack(A)
    errR = updateright!.(AR, AC, circshift(C, (1, 0)))
    return errR
end

# function updateright!(A::MPS)
#     errR = updateright!.(A)
#     return A, hcat(errR...)
# end
# function vumps_rightupdate(AC,C)
#     U,S,V = tsvd(_CA(C', AC), (2,),(1,3))
#     AR = _CA(U, permute(V, (2,1),(3,)))
#     ϵR = norm(AC - _CA(C,AR))
#     return AR, ϵR
# end
function updateboth!(A::MPS)
    errL = updateleft!(A)
    errR = updateright!(A)
    return A, errL, errR
end

#= 
function tracecontract(vumps::VUMPSTensors, bulk)
    AC = getcentral(vumps.mps)
    FL = vumps.fixedpoints.left
    FR = vumps.fixedpoints.right
    return tracecontract.(FL, FR, AC, circshift(AC, (0, -1)), bulk)
end
function onelocalcontract(vumps::VUMPSTensors, bulk)
    AC = getcentral(vumps.mps)
    FL = vumps.fixedpoints.left
    FR = vumps.fixedpoints.right

    cod = codomain.(bulk)

    return onelocalcontract.(FL, FR, AC, circshift(AC, (0, -1)), bulk)
end

function metric(pepo::AbstractPEPO, vumps::VUMPSTensors, bond::Bond)
    fs = get_truncmetric_tensors(vumps.fixedpoints, bond)

    as = get_truncmetric_tensors(vumps.mps, bond)

    ms = pepo[bond]

    return truncmetriccontract(fs..., as..., ms...)
end

function truncmetric!(dst, vumps::VUMPSTensors, bulk::ContractableTensors)
    mps = vumps.mps

    AC = getcentral(mps)
    AR = getright(mps)

    FL = vumps.fixedpoints.left
    FR = vumps.fixedpoints.right

    # for x in axes(bulk, 1)
    #     for y in axes(bulk, 2)
    #         truncmetriccontract!(dst[x,y],FL[x,y],AC[x,y],AC[x,y+1],AR[x+1,y],AR[x+1,y+1],bulk[x,y],bulk[x+1,y],FR[x+1,y])
    #     end
    # end
    #
    # return dst

    return truncmetriccontract!.(
        dst,
        FL,
        circshift(FR, (-1, 0)),
        AC,
        circshift(AC, (0, -1)),
        circshift(AR, (-1, 0)),
        circshift(AR, (-1, -1)),
        bulk,
        circshift(bulk, (-1, 0)),
    )
end

### TESTING TODO: DELETE

function vumps_test()
    βc = log(1 + sqrt(2)) / 2
    β = 1.0 * βc

    println("β = \t", β)

    aM, aM2 = classicalisingmpo_alt(β)
    tM = TensorMap(ComplexF64.(aM), ℂ^2 * ℂ^2, ℂ^2 * ℂ^2) #mpo
    tM2 = TensorMap(aM2, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2) #mpo

    L = Lattice{3,3,Infinite}([ℂ^2 ℂ^2; ℂ^2 ℂ^2])
    M = OnLattice(transpose(L), tM)
    # M = OnLattice(L, tM)
    # M = OnLattice(Lattice{1,1, Infinite}( fill(ℂ^2,1,1)), tM)

    alg = BoundaryAlgorithm(; alg=VUMPSTensors, bondspace=ℂ^10, verbosity=1, maxiter=200)
    vumps = inittensors(rand, M, alg)

    return state = BoundaryState(vumps, M, alg, vumps, ConvergenceInfo())

    # alg = initialise_state(VUMPS, M, BoundaryParameters(2, ℂ^4, 100, 1e-10))
    #
    # st = alg.data
    #
    # @assert isgauged(st.A[1])
    #
    # return calculate!(alg)

    # _, FL, FR = vumps!(A, M)

    # al = lefttensor(A)
    # ar = righttensor(A)
    # c = bondtensor(A)
    # ac = centraltensor(A)

    # z2 = expval(tM, tomatrixt(ac), tomatrixt(FL), tomatrixt(FR))
    # magn = expval(tM2, tomatrixt(ac), tomatrixt(FL), tomatrixt(FR))

    # return magn ./ z2
end

# function testflsolve(FL, P, A)
#     @tensoropt FLo[k4 r4; r3 k3] :=
#         FL[k2 r2; r1 k1] *
#         A[k1 D1; k3 D3] *
#         P[r1 D2 d1; r3 D1 d2] *
#         (P')[r4 D3 d2; r2 D4 d1] *
#         (A')[k4 D4; k2 D2]
#     return Flo
# end

=#
