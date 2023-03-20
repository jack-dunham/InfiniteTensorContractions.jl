# Solves AL * C = C * A (= AC)
function leftgauge!(
    AL::AbstractOnLattice{U,<:AbstractTensorMap{S,2,1}},
    L::AbstractOnLattice{U,<:AbstractTensorMap{S,1,1}},
    A::AbstractOnLattice{U,<:AbstractTensorMap{S,2,1}},
    tol=1e-12,
    maxiter=100,
) where {U<:UnitCell,S}
    numsites = length(A)
    r = 1:numsites

    ϵ = 2 * tol
    numiter = 1

    normalize!(L[end])

    λ = zeros(numsites)

    ϵ = zeros(numsites)
    ϵ[end] = 2 * tol

    T = multi_transfer(A, A)

    λs, Ls, info = KrylovKit.eigsolve(
        y -> lgsolve(y, T),
        L[end],
        1,
        :LM;
        ishermitian=false,
        tol=tol,
        eager=true,
        maxiter=1,
    )

    # R[end] <- C = sqrt(C^2)
    L[end] = sqrt(Ls[1])

    for x in r
        Lold = L[x - 1]

        AL[x], L[x] = leftorth!(mulbond!(AL[x], L[x - 1], A[x]))

        λ[x] = norm(L[x])
        rmul!(L[x], 1 / λ[x])

        ϵ[x] = norm(Lold - L[x])
    end

    println(ϵ)

    while numiter < maxiter && max(ϵ...) > tol
        for x in r
            Lold = L[x]
            AL[x], L[x] = leftorth!(mulbond!(AL[x], L[x - 1], A[x]))

            λ[x] = norm(L[x])
            rmul!(L[x], 1 / λ[x])

            ϵ[x] = norm(Lold - L[x])
        end

        println(ϵ)
        numiter += 1
    end
    return AL, L, λ
end
function rightgauge!(
    AR::AbstractOnLattice{U,<:AbstractTensorMap{S,2,1}},
    R::AbstractOnLattice{U,<:AbstractTensorMap{S,1,1}},
    A::AbstractOnLattice{U,<:AbstractTensorMap{S,2,1}},
    kwargs...,
) where {U<:UnitCell,S}
    # Permute left and right bonds
    AL = permute.(AR, Ref((3, 2)), Ref((1,)))
    L = permute.(R, Ref((2,)), Ref((1,)))
    A = permute.(A, Ref((3, 2)), Ref((1,)))

    # Run the gauging algorithm
    AL, L, λ = leftgauge!(AL, L, A; kwargs...)

    # Permute left and right bonds
    permute!.(AR, AL, Ref((3, 2)), Ref((1,)))
    permute!.(R, L, Ref((2,)), Ref((1,)))
    return AR, R, λ
end

function mixedgauge(A::AbstractOnLattice{<:AbstractLattice,<:AbsTen{2,1}})
    C = TensorMap.(rand, numbertype(A), eastbond(A), westbond(A))
    return mixedgauge(A, C)
end
function mixedgauge(
    A::AbstractOnLattice{L,<:AbstractTensor{S,2,1}},
    C0::AbstractOnLattice{L,<:AbstractTensor{S,1,1}},
) where {L,S}
    C = deepcopy(C0)
    AL, _, _ = leftgauge!(similar.(A), C, A)
    AR, C, _ = rightgauge!(similar.(A), C, AL)
    diagonalise!(AL, C, AR)
    return AL, C, AR
end

# Diagonalise the bond matrices using the singular value decomposition.
function diagonalise!(
    AL::AbstractOnLattice{L,<:AbstractTensor{S,2,1}},
    C::AbstractOnLattice{L,<:AbstractTensor{S,1,1}},
    AR::AbstractOnLattice{L,<:AbstractTensor{S,2,1}},
) where {L,S}
    U = similar(C)
    V = similar(C)
    for y in axes(C, 2)
        for x in axes(C, 1)
            U[x, y], C[x, y], V[x, y] = tsvd!(C[x, y])
            gauge!(AL, U)
            gauge!(AR, V)
        end
    end
    return AL, C, AR
end
function diagonalise!(mps::MPS)
    AL, C, AR, AC = unpack(mps)
    diagonalise!(AL, C, AR)
    centraltensor!(AC, AL, C)
    return mps
end

# Vectorised form of a gauge transformation. That is, A[x] <- U'[x-1]*A[x]*U[x]
function gauge!(
    A::AbstractOnLattice{L,<:AbstractTensor{S,2,1}},
    U::AbstractOnLattice{L,<:AbstractTensor{S,1,1}},
) where {L,S}
    _A = centraltensor(A, U)
    centraltensor!(A, U, _A)
    return A
end

function mps_transfer(
    a::AbstractTensorMap{S,2,1}, al::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
    @tensoropt T[lu rd; ru ld] := a[lu a; ru] * (al')[rd; ld a]
    return T
end

function compose_transfer!(T1, T2)
    Tn = deepcopy(T1)
    @tensoropt T1[lu rd; ru ld] = Tn[lu md; mu ld] * T2[mu rd; ru md]
    return T1
end

function multi_transfer(A::AbstractArray, AL::AbstractArray)
    T = mps_transfer(A[1], A[1])
    for i in 2:length(A)
        Ti = mps_transfer(A[i], AL[i])
        compose_transfer!(T, Ti)
    end
    # T = T / norm(T)
    return T
end

function lgsolve(x, T)
    @tensoropt y[rd; ru] := x[ld; lu] * T[lu rd; ru ld]
    return y
end
# TEMP
#
function mulbond(
    C::AbstractTensorMap{S,1,1}, A::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
    return mulbond!(similar(A), C, A)
end
function mulbond!(
    CA::AbstractTensorMap{S,2,1}, C::AbstractTensorMap{S,1,1}, A::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
    return @tensoropt CA[1 2; 3] = C[1; a] * A[a 2; 3]
end

function mulbond(
    A::AbstractTensorMap{S,2,1}, C::AbstractTensorMap{S,1,1}
) where {S<:IndexSpace}
    return mulbond!(similar(A), A, C)
end
function mulbond!(
    AC::AbstractTensorMap{S,2,1}, A::AbstractTensorMap{S,2,1}, C::AbstractTensorMap{S,1,1}
) where {S<:IndexSpace}
    return @tensoropt AC[1 2; 3] = A[1 2; a] * C[a; 3]
end
