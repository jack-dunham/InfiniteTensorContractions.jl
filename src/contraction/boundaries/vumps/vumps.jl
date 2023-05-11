@doc raw"""
    VUMPS <: AbstractBoundary

A struct holding the boundary tensors obtained via the VUMPS algorithm.

# Fields
- `A::MultilineInfMPS`: The VUMPS boundary matrix product state. 
- `FP::FixedPoints`: The left and right fixed points of the transfer matrix generated by
    some external MPO-like object.
"""
struct VUMPS <: AbstractBoundary
    A::MPS
    FP::FixedPoints
end

#=
struct BoundaryState{B} <: AbstractBoundaryState{B}
    initial::B
    mpo::InfiniteMPO
    alg::BoundaryAlgorithm{B}
    tensors::B
    info::ConvergenceInfo
end
@with_kw struct BoundaryAlgorithm{B<:AbstractBoundary,S<:IndexSpace} <:
                AbstractBoundaryAlgorithm
    alg::Type{B}
    bondspace::S
    verbosity::Int = 0
    maxiter::Int = 100
    tol::Float64 = 1e-12
end
=#
const VUMPSState = BoundaryState{VUMPS}

function inittensors(f, M, alg::BoundaryAlgorithm{VUMPS})

    D = @. getindex(domain(M),2)
    χ = dimtospace(M, alg.bonddim)

    boundary_mps = MPS(f, numbertype(M), lattice(M), D, χ)
    
    fixed_points = fixedpoints(boundary_mps, M)
    
    return VUMPS(boundary_mps, fixed_points)
end

# function inittensors(f, T::Union{HilbertSchmidt,Trace}, alg::BoundaryAlgorithm{VUMPS})
#     M = physicaltrace(T)
#     return inittensors(f, M, alg)
# end

calculate!(v::VUMPSState) = vumps!(v)

# algorithm

vumps!(v::VUMPSState) = vumps!(v, Val(v.info.finished))

# Mutates the vumps tensors
function vumps!(v::VUMPSState, ::Val{false})
    ## tensors
    st = v.tensors
    mpo = v.mpo
    ## alg
    alg = v.alg
    ## info (mutable data)
    info = v.info

    vumpsloop!(st.A, st.FP, info,mpo,alg)

    info.finished = true

    return v
end

# Function barrier
function vumpsloop!(mps, fixedpoints, info, mpo, alg)
    # Immutable parameters
    verbosity = alg.verbosity
    maxiter = alg.maxiter
    tol = alg.tol
    # Mutating parameters
    error = info.error
    iterations = info.iterations

    while error > tol && iterations < maxiter
        _, _, error = vumpsstep!(mps, fixedpoints, mpo)
        verbosity > 0 && @info "\t Step $(iterations): error ≈ $(error)"

        iterations += 1
        info.iterations = iterations
        info.error = error
    end

    error > tol ? info.converged = false : info.converged = true

    return mps, fixedpoints, info
end

function vumps!(v::VUMPSState, ::Val{true})
    @info "This algorithm has already finished according to the parameters set"
    return v
end

function vumpsstep!(A::MPS, FP::FixedPoints, M)
    A, errL, errR = vumpsupdate!(A, FP, M) # Vectorised

    fixedpoints!(FP, A, M)

    AC = getcentral(A)

    # Error calculation
    vl = leftnull.(circshift(AC, (0, -1)))
    ac = applyhac.(AC, FP.left, FP.right, M)
    err = @. norm(adjoint(vl) * ac)

    # nx,ny = size(M)
    # err = zeros(Float64, nx, ny)

    # rx = 1:nx; ry=1:ny
    # for x in rx, y in ry
    #     yp = mod(y+1, ry) 

    #     vl = leftnull(AC[x,yp])

    #     ac = applyhac(AC[x,y],FL[x,y],FR[x,y],M[x,y])

    #     err[x,y] = norm(vl[x,y]'* ac[x,y])
    # end

    erri = maximum(err)

    return A, FP, erri
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
        Threads.@sync begin
            Threads.@spawn begin
                # First solve for the new AC tensors.
                _, ACs, _ = eigsolve(
                    z -> applyhac(z, ($FL[x, :]), ($FR[x, :]), ($M[x, :])),
                    RecursiveVec(($AC[x, :])...),
                    1,
                    :LM,
                )

                # @info "μ1 $(μ1s)"
                for y in ry
                    A[y].AC[x] = ACs[1][y]
                end
            end
            # now set mps[y-1, mod].AC[x] to output of above

            # A[mod(y - 1, ry)].AC[x] = ACs[1]
            Threads.@spawn begin
                _, Cs, _ = eigsolve(
                    z -> applyhc(z, ($FL[x + 1, :]), ($FR[x, :])),
                    RecursiveVec(($C[x, :])...),
                    1,
                    :LM,
                )

                # @info "μ0 $(μ0s[1]))"

                for y in ry
                    A[y].C[x] = Cs[1][y]
                end
            end
        end
        # A[mod(y - 1, ry)].C[x] = Cs[1]
        # λ = real(μ1s[1] / μ0s[1])
    end

    #A now as updated AC, C, need to update AL, AR
    _, errL, errR = updateboth!(A)

    errL, _ = findmax(errL)
    errR, _ = findmax(errR)
    # @info "errL = $errL \t errR = $errR"
    return A, errL, errR
end

# EFFECTIVE HAMILTONIANS

function applyhac(
    ac::AbstractTensorMap{S,2,1},
    fl::AbstractTensorMap{S,1,2},
    fr::AbstractTensorMap{S,2,1},
    m::AbstractTensorMap{S,2,2},
) where {S<:IndexSpace}
    @tensoropt begin
        hac[-2 -1; -3] := ac[a 4; b] * fl[-2; a 1] * m[1 -1; 3 4] * fr[b 3; -3]
    end
    return hac
end

function applyhc(
    c::AbstractTensorMap{S,1,1}, fl::AbstractTensorMap{S,1,2}, fr::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
    @tensoropt hc[a; b] := c[x; y] * fl[a; x m] * fr[y m; b]
    return hc
end

function applyhac(
    z::RecursiveVec, FL::AbstractVector, FR::AbstractVector, M::AbstractVector
)
    Ny = length(FL)
    rv = Vector{eltype(z.vecs)}(undef, Ny)
    for y in 1:Ny
        rv[y] = applyhac(z[y], FL[y], FR[y], M[y])
    end
    rv = circshift(rv, 1)
    return RecursiveVec(rv...)
end

function applyhc(z::RecursiveVec, FL::AbstractVector, FR::AbstractVector)
    Ny = length(FL)
    rv = Vector{eltype(z.vecs)}(undef, Ny)
    for y in 1:Ny
        rv[y] = applyhc(z[y], FL[y], FR[y])
    end
    rv = circshift(rv, 1)
    return RecursiveVec(rv...)
end

# MPS UPDATE
# TODO: MOVE TO MPS FILE?

function updateleft!(al::AbstractTensorMap, ac::AbstractTensorMap, c::AbstractTensorMap)
    qac, rac = leftorth(ac)
    qc, rc = leftorth(c)

    mul!(al, qac, qc') # QAC * QC' -> AL
    errL = norm(rac - rc)

    return errL
end

function updateleft!(A::MPS)
    AL, C, _, AC = unpack(A)
    errL = updateleft!.(AL, AC, C)
    return errL
end
# function updateleft!(A::MPS)
#     errL = updateleft!.(A)
#     return A, hcat(errL...)
# end

function updateright!(ar::AbstractTensorMap, ac::AbstractTensorMap, c::AbstractTensorMap)
    lac, qac = rightorth(ac, (1,), (2, 3))
    lc, qc = rightorth(c)

    permute!(ar, qc' * qac, (1, 2), (3,))
    errR = norm(lac - lc)

    return errR
end

function updateright!(A::MPS)
    _, C, AR, AC = unpack(A)
    errR = updateright!.(AR, AC, circshift(C, (1,0)))
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



### TESTING TODO: DELETE

function vumps_test()
    βc = log(1 + sqrt(2)) / 2
    β = 1.0 * βc

    println("β = \t", β)

    aM, aM2 = classicalisingmpo_alt(β)
    tM = TensorMap(ComplexF64.(aM), ℂ^2 * ℂ^2, ℂ^2 * ℂ^2) #mpo
    tM2 = TensorMap(aM2, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2) #mpo

    
    L = Lattice{3,3, Infinite}([ ℂ^2 ℂ^2; ℂ^2 ℂ^2])
    M = OnLattice(transpose(L), tM)
    # M = OnLattice(L, tM)
    # M = OnLattice(Lattice{1,1, Infinite}( fill(ℂ^2,1,1)), tM)

    alg = BoundaryAlgorithm(alg = VUMPS, bondspace = ℂ^10, verbosity=1,maxiter=200)
    vumps = inittensors(rand, M, alg)

    state = BoundaryState(vumps, M, alg, vumps, ConvergenceInfo())

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

function statmechmpo_alt(β, h, D)
    δ1 = zeros(D, D, D, D)
    for i in 1:D
        δ1[i, i, i, i] = 1 # kronecker delta
    end

    δ2 = zeros(D, D, D, D, D)
    for i in 1:D
        δ2[i, i, i, i, i] = 1 # kronecker delta
    end

    X = zeros(D, D)
    for j in 1:D, i in 1:D
        X[i, j] = exp(-β * h(i, j))
    end
    Xsq = sqrt(X)

    Z = [1.0, -1.0]

    @tensor M1[a, b, c, d] :=
        δ1[a', b', c', d'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b']

    @tensor M2[a, b, c, d] :=
        δ2[a', b', c', d', e'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b'] * Z[e']

    return M1, M2
end

function classicalisingmpo_alt(β; J=1.0, h=0.0)
    return statmechmpo_alt(
        β, (s1, s2) -> -J * (-1)^(s1 != s2) - h / 2 * (s1 == 1 + s2 == 1), 2
    )
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
