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

function inittensors(f, network, alg::VUMPS)
    # D = @. getindex(domain(network), 4)

    _, _, _, north_bonds = bondspace(network)

    χ = dimtospace(spacetype(network), alg.bonddim)

    chi = similar(north_bonds, typeof(χ))

    chi .= Ref(χ)

    boundary_mps = MPS(f, numbertype(network), north_bonds, chi)

    fixed_points = initfixedpoints(f, boundary_mps, network)

    return VUMPSTensors(boundary_mps, fixed_points)
end

function start(state::BoundaryState{VUMPS})
    vumps = state.tensors
    sing_val = x -> tsvd(x, (1,), (2,))[2]
    return (sing_val.(getbond(vumps.mps)),)
end

function step!(
    vumps::VUMPSTensors,    #mutating
    network::AbstractNetwork,
    ::VUMPS,
    singular_values,        #mutating
)
    vumpsstep!(vumps, network)

    error_per_site = boundaryerror!(singular_values, getbond(vumps.mps))

    @debug "Error per site:" ϵᵢ = error_per_site

    return max(error_per_site...)
end

function vumpsstep!(vumps::VUMPSTensors, network)
    mps = vumps.mps
    fps = vumps.fixedpoints

    vumpsupdate!(mps, fps, network) # Vectorised

    fixedpoints!(fps, mps, network)

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

function updateright!(A::MPS)
    _, C, AR, AC = unpack(A)
    errR = updateright!.(AR, AC, circshift(C, (1, 0)))
    return errR
end

function updateboth!(A::MPS)
    errL = updateleft!(A)
    errR = updateright!(A)
    return A, errL, errR
end

### TESTING TODO: DELETE
#=
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
