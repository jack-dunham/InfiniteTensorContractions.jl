# Solves AL * C = C * A (= AC)
function leftgauge!(
    AL::AbstractUnitCell{G,<:AbstractTensorMap{S,N,2}},
    L::AbstractUnitCell{G,<:AbstractTensorMap{S,0,2}},
    A::AbstractUnitCell{G,<:AbstractTensorMap{S,N,2}};
    tol=1e-12,
    maxiter=100,
    verbose=false,
) where {G,S,N}
    numsites = length(A)
    r = 1:numsites

    ϵ = 2 * tol
    numiter = 1

    normalize!(L[end])

    λ = zeros(numsites)
    ϵ = zeros(numsites)

    ϵ[end] = 2 * tol

    T = multi_transfer(A, A) # dr dl, ur ul -> ul dl, ur dr

    Tp = permute(T, (4, 2), (3, 1))

    λs, Ls, info = KrylovKit.eigsolve(
        y -> y * Tp, L[end], 1, :LM; ishermitian=false, tol=tol, eager=true, maxiter=1
    )

    Lp = permute(Ls[1], (1,), (2,))

    @debug "MPS fixed point info:" λ = λs[1] info = info hermiticity = norm(Lp - Lp')

    Lp = 1 / 2 * (Lp + Lp')

    # Eigenvector can be α * v, so choose α s.t. v positive
    Lp = sqrt(Lp * Lp)

    D, V = eigh(Lp)

    L[end] = permute(V * sqrt(D) * V', (), (1, 2))

    for x in r
        Lold = L[x - 1]

        AL[x], L[x] = leftdecomp!(deepcopy(AL[x]), deepcopy(L[x - 1]), A[x])
        λ[x] = norm(L[x])
        rmul!(L[x], 1 / λ[x])

        ϵ[x] = norm(Lold - L[x])
    end

    verbose && @info "\t error: $ϵ"

    while numiter < maxiter && max(ϵ...) > tol
        for x in r
            Lold = L[x]
            AL[x], L[x] = leftdecomp!(deepcopy(AL[x]), deepcopy(L[x - 1]), A[x])

            λ[x] = norm(L[x])
            rmul!(L[x], 1 / λ[x])

            ϵ[x] = norm(Lold - L[x])
        end

        verbose && @info "\t error: $ϵ"
        numiter += 1
    end
    return AL, L, λ
end

function rightgauge!(
    AR::AbstractUnitCell{U,<:AbstractTensorMap{S,N,2}},
    R::AbstractUnitCell{U,<:AbstractTensorMap{S,0,2}},
    A::AbstractUnitCell{U,<:AbstractTensorMap{S,N,2}};
    kwargs...,
) where {U,S,N}
    # Permute left and right virtual bonds bonds
    AL = permutedom.(AR, Ref((2, 1)))
    L = permutedom.(R, Ref((2, 1)))
    A = permutedom.(A, Ref((2, 1)))

    # Run the gauging algorithm
    AL, L, λ = leftgauge!(AL, L, A; kwargs...)

    # Permute left and right bonds
    permutedom!.(AR, AL, Ref((2, 1)))
    permutedom!.(R, L, Ref((2, 1)))
    return AR, R, λ
end
# AL[x] <- L[x - 1] * A[x]
# TODO: remove inplace stuff, replace with _leftorth from utils.jl
function leftdecomp!(AL, L, A::AbsTen{N,2}) where {N}
    # Overwrite AL
    mulbond!(AL, L, A)
    AL_p = permute(AL, tuple(1:N..., N + 2)::NTuple{N + 1,Int}, (N + 1,))
    temp_AL, temp_L = leftorth!(AL_p)
    permute!(AL, temp_AL, Tuple(1:N)::NTuple{N,Int}, (N + 2, N + 1))
    permute!(L, temp_L, (), (2, 1))
    return AL, L
end

# Find the mixedguage of uniform MPS A, writing result into AL,C,AR
function mixedgauge!(AL, C, AR, A; verbose=false, kwargs...)
    for y in axes(A, 2)
        verbose && @info "Gauging right..."
        @views begin
            rightgauge!(AR[:, y], C[:, y], A[:, y]; verbose=verbose, kwargs...)
            verbose && @info "Gauging left..."
            leftgauge!(AL[:, y], C[:, y], AR[:, y]; verbose=verbose, kwargs...)
        end
    end
    # currently not working 
    # diagonalise!(AL, C, AR)
    return AL, C, AR
end

# function mixedgauge(A::AbstractOnLattice{<:AbstractLattice,<:AbsTen{2,1}})
#     C = TensorMap.(rand, ComplexF64, eastbond.(A), westbond.(A))
#     return mixedgauge(A, C)
# end
# function mixedgauge(
#     A::AbstractOnLattice{L,<:AbstractTensorMap{S,2,1}},
#     C0::AbstractOnLattice{L,<:AbstractTensorMap{S,1,1}},
# ) where {L,S}
#     C = deepcopy(C0)
#     AL, _, _ = leftgauge!(similar.(A), C, A)
#     AR, C, _ = rightgauge!(similar.(A), C, AL)
#     diagonalise!(AL, C, AR)
#     return AL, C, AR
# end

# Diagonalise the bond matrices using the singular value decomposition.
function diagonalise!(
    AL::AbstractUnitCell{G,<:AbstractTensorMap{S}},
    C::AbstractUnitCell{G,<:AbstractTensorMap{S}},
    AR::AbstractUnitCell{G,<:AbstractTensorMap{S}},
) where {G,S}
    U = similar(C)
    V = similar(C)

    for y in axes(C, 2)
        for x in axes(C, 1)
            U[x, y], C[x, y], V[x, y] = _tsvd(C[x, y])
        end
    end
    gauge!(AL, U)
    gauge!(AR, V)
    return AL, C, AR
end
function diagonalise!(mps::MPS)
    AL, C, AR, AC = unpack(mps)
    diagonalise!(AL, C, AR)
    centraltensor!(AC, AL, C)
    return mps
end

function _tsvd(t)
    u, s, v = tsvd(t, (2,), (1,))
    up = permute(u, (), (2, 1))
    sp = permute(s, (), (2, 1))
    vp = permute(s, (), (2, 1))
    return up, sp, vp
end

# Vectorised form of a gauge transformation. That is, A[x] <- U'[x-1]*A[x]*U[x]
function gauge!(
    A::AbstractUnitCell{G,<:AbstractTensorMap{S}},
    U::AbstractUnitCell{G,<:AbstractTensorMap{S,0,2}},
) where {G,S}
    _A = centraltensor(A, U)
    centraltensor!(A, adjoint.(U), _A)
    return A
end

function mps_transfer(
    a::AbstractTensorMap{S,2,2}, al::AbstractTensorMap{S,2,2}
) where {S<:IndexSpace}
    @tensoropt T[dr dl; ur ul] := a[p1 p2; ur ul] * (al')[dr dl; p1 p2]
    return T
end
function mps_transfer(
    a::AbstractTensorMap{S,1,2}, al::AbstractTensorMap{S,1,2}
) where {S<:IndexSpace}
    @tensoropt T[dr dl; ur ul] := a[p1; ur ul] * (al')[dr dl; p1]
    return T
end

function compose_transfer(T1, T2)
    @tensoropt T[dr dl; ur ul] := T1[dr d; ur u] * T2[d dl; u ul]
    return T
end

function multi_transfer(A::AbstractArray, AL::AbstractArray)
    T = mps_transfer(A[1], A[1])
    for i in axes(A, 1)
        if i == 1
            continue
        end
        Ti = mps_transfer(A[i], AL[i])
        T = compose_transfer(T, Ti)
    end
    # T = T / norm(T)
    return T
end

function lgsolve(x, T)
    y = similar(x)
    @tensoropt y[ur dr] = x[ul dl] * T[dr dl; ur ul]
    return y
end
