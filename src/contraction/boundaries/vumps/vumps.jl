@doc raw"""
    VUMPS <: AbstractBoundary

A struct holding the boundary tensors obtained via the VUMPS algorithm.

# Fields
- `A::MultilineInfMPS`: The VUMPS boundary matrix product state. 
- `FL::LeftEnv:` The left environment of the transfer matrix.
- `RF::RightEnv:` The right environment of the transfer matrix.
"""
struct VUMPS <: AbstractBoundary
    A::MultilineInfiniteMPS
    FP::FixedPoints
end

const VUMPSState = BoundaryState{VUMPS}

function inittensors(f, M::InfiniteMPO, alg::BoundaryAlgorithm{VUMPS})
    nx, ny = size(M)

    χsp = alg.bondspace
    Dsp = @. getindex(domain(M), 2)

    c = Ref(χsp) .* Dsp

    data = Tuple(
        InfiniteMPS{nx}(TensorMap.(f, numbertype(M), c[:, y], Ref(χsp))) for y in 1:ny
    )
    A = MultilineInfiniteMPS(data...)
    FL, FR = environments(A, M)

    return VUMPS(A, FL, FR)
end

function inittensors(f, T::Union{HilbertSchmidt,Trace}, alg::BoundaryAlgorithm{VUMPS})
    M = physicaltrace(T)
    return inittensors(f, M, alg)
end

calculate!(v::VUMPSState) = vumps!(v)

include("transfermatrix.jl")
include("fixedpoints.jl")

# algorithm

vumps!(v::VUMPSState) = vumps!(v, Val(v.info.finished))

#mutates the vumps tensors
function vumps!(v::VUMPSState, ::Val{false})
    st = v.data
    mpo = v.mpo
    ## alg
    param = v.param
    #
    verbosity = param.verbosity
    maxiter = param.maxiter
    tol = param.tol
    ## info
    info = v.info
    #
    converged = info.converged
    error = info.error
    iterations = info.iterations

    while error > tol && iterations < maxiter
        _, _, _, error = vumpsstep!(st, mpo)
        verbosity > 0 && @info "\t Step $(iterations): error ≈ $(error)"

        iterations += 1
    end
    error > tol ? converged = false : converged = true
    info.finished = true
    return v
end

function vumps!(v::VUMPSState, ::Val{true})
    @info "This algorithm has already finished according to the parameters set"
    return v
end

vumpsstep!(T::VUMPS, M::InfiniteMPO) = vumpsstep!(T.A, T.FL, T.FR, M) # Function barrier
function vumpsstep!(A::MultilineInfiniteMPS, FP::FixedPoints, M::InfiniteMPO)
    A, errL, errR = vumpsupdate!(A, FL, FR, M) # array form

    fixedpoints!(FP, A, M)

    AC = centraltensor(A)

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

function vumpsupdate!(A::MultilineInfiniteMPS, FP::FixedPoints, M::InfiniteMPO)
    FL = FP.left
    FR = FP.right

    nx, ny = size(M)
    rx = 1:nx
    ry = 1:ny

    AC = centraltensor(A)
    C = bondtensor(A)

    for x in rx

        # take mps[y], get mps[y].AC, send in mps[y].AC[x]
        Threads.@sync begin
            Threads.@spawn begin
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

function updateleft!(A::InfiniteMPS)
    AL, C, _, AC = unpack(A)
    errL = updateleft!.(AL, AC, C)
    return errL
end
function updateleft!(A::MultilineInfiniteMPS)
    errL = updateleft!.(A)
    return A, hcat(errL...)
end

function updateright!(ar::AbstractTensorMap, ac::AbstractTensorMap, c::AbstractTensorMap)
    lac, qac = rightorth(ac, (1,), (2, 3))
    lc, qc = rightorth(c)

    permute!(ar, qc' * qac, (1, 2), (3,))
    errR = norm(lac - lc)

    return errR
end

function updateright!(A::InfiniteMPS)
    _, C, AR, AC = unpack(A)
    errR = updateright!.(AR, AC, circshift(C, 1))
    return errR
end

function updateright!(A::MultilineInfiniteMPS)
    errR = updateright!.(A)
    return A, hcat(errR...)
end
# function vumps_rightupdate(AC,C)
#     U,S,V = tsvd(_CA(C', AC), (2,),(1,3))
#     AR = _CA(U, permute(V, (2,1),(3,)))
#     ϵR = norm(AC - _CA(C,AR))
#     return AR, ϵR
# end
function updateboth!(A::MultilineInfiniteMPS)
    _, errL = updateleft!(A)
    _, errR = updateright!(A)
    return A, errL, errR
end

function testflsolve(FL, P, A)
    @tensoropt FLo[k4 r4; r3 k3] :=
        FL[k2 r2; r1 k1] *
        A[k1 D1; k3 D3] *
        P[r1 D2 d1; r3 D1 d2] *
        (P')[r4 D3 d2; r2 D4 d1] *
        (A')[k4 D4; k2 D2]
    return Flo
end
