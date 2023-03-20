abstract type AbstractMPS{L<:AbstractLattice} end
const AbstractInfiniteMPS{L} = AbstractMPS{L<:AbstractUnitCell}

struct MPS{
    L,AType<:AbstractOnLattice{L,<:AbsTen{2,1}},CType<:AbstractOnLattice{L,<:AbsTen{1,1}}
} <: AbstractMPS{L}
    AL::AType
    C::CType
    AR::AType
    AC::AType
    function MPS(AL, C, AR, AC)
        mps = new(AL, C, AR, AC)
        return validate(mps)
    end
end

const InfiniteMPS{L<:AbstractUnitCell} = MPS{L}
const iMPS = InfiniteMPS

getleft(mps::AbstractMPS) = mps.AL
getbond(mps::AbstractMPS) = mps.C
getright(mps::AbstractMPS) = mps.AR
getcentral(mps::AbstractMPS) = mps.AC

unpack(mps::AbstractMPS) = (getleft(mps), getbond(mps), getright(mps), getcentral(mps))

function isgauged(mps::AbstractMPS)
    for mpsview in mps
        AL, C, AR, AC = unpack(mpsview)
        for x in axes(mpsview, 1)
            if !(mulbond(AL[x], C[x]) ≈ mulbond(C[x - 1], AR[x]) ≈ AC[x])
                return false
            end
        end
    end
    return true
end

function validate(mps::AbstractMPS)
    if !isgauged(mps)
        @warn "MPS is not in mixed canonical form."
    end
    return mps
end

function MPS(
    f, T, lattice::AbstractLattice{Nx,Ny}, D::IndexSpace, χ::IndexSpace
) where {Nx,Ny}
    return MPS(f, T, lattice, fill(D, Nx, Ny), fill(χ, Nx, Ny))
end
function MPS(f, T, lattice, D::AbstractMatrix{S}, χ::AbstractMatrix{S}) where {S}
    data_mat = OnLattice(lattice, @. TensorMap(f, T, χ * D, χ))
    return MPS(data_mat)
end

function MPS(A::AbstractOnLattice)
    χ_left = getindex(domain(A), 1)
    χ_right = getindex.(codomain.(circshift(A, 1)), 1)
    AL = similar(A)
    C = similar.(A, χ_left, χ_right)
    AR = similar(A)
    for y in axes(lattice, 2)
        AL[:, y], C[:, y], AR[:, y] = mixedgauge(@view data_mat[:, y])
    end
end

function MPS(AL, C, AR)
    AC = centraltensor(AL, C)
    return MPS(AL, C, AR, AC)
end

function Base.getindex(mps::AbstractMPS, y::Int)
    lattice = lattice(mps)
    AL, C, AR, AC = unpack(mps)
    sublattice = LatticeView(lattice, :, y)
    @views begin
        subAL = OnLattice(sublattice, AL[:, y:y])
        subC = OnLattice(sublattice, C[:, y:y])
        subAR = OnLattice(sublattice, AR[:, y:y])
        subAC = OnLattice(sublattice, AC[:, y:y])
    end
    return MPS(subAL, subC, subAR, subAC)
end

_similar_ac(A::AbstractTensorMap{S,2,1}, C::AbstractTensorMap{S,1,1}) where {S} = similar(A)
function _similar_ac(C::AbstractTensorMap{S,1,1}, A::AbstractTensorMap{S,2,1}) where {S}
    return _similar_ac(A, C)
end

centraltensor(A1, A2) = centraltensor!(_similar_ac(A1, A2), A1, A2)
centraltensor!(AC, AL, C) = mpsbond!.(AC, AL, C)
centraltensor!(AC, C, AR) = mpsbond!.(AC, circshift(C, (1, 0)), AR)

function mulbond(A1::AbstractTensorMap{S}, A2::AbstractTensorMap{S})
    return mulbond!(_similar_ac(A1, A2), A1, A2)
end
function mulbond!(
    CA::AbstractTensorMap{S,2,1}, C::AbstractTensorMap{S,1,1}, A::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
    return @tensoropt CA[1 2; 3] = C[1; a] * A[a 2; 3]
end
function mulbond!(
    AC::AbstractTensorMap{S,2,1}, A::AbstractTensorMap{S,2,1}, C::AbstractTensorMap{S,1,1}
) where {S<:IndexSpace}
    return @tensoropt AC[1 2; 3] = A[1 2; a] * C[a; 3]
end

# norm(A::AbstractMPS) = tr(c * c')
