abstract type AbstractMPS{L<:AbstractLattice} end
# const AbstractInfiniteMPS{L} = AbstractMPS{L<:AbstractUnitCell}

@doc raw"""
    struct MPS
"""
struct MPS{
    L,AType<:AbstractOnLattice{L,<:AbsTen{2,1}},CType<:AbstractOnLattice{L,<:AbsTen{1,1}}
} <: AbstractMPS{L}
    AL::AType
    C::CType
    AR::AType
    AC::AType
    function MPS(AL::AType, C::CType, AR::AType, AC::AType) where {
        L,AType<:AbstractOnLattice{L}, CType<:AbstractOnLattice{L}}
        mps = new{L,AType,CType}(AL, C, AR, AC)
        validate(mps)
        return mps
    end
    function MPS(AL::AType, C::CType, AR::AType, AC::AType) where {
        L<:SubLattice,AType<:AbstractOnLattice{L}, CType<:AbstractOnLattice{L}}
        mps = new{L,AType,CType}(AL, C, AR, AC)
        return mps
    end
end

const AbsLatY{Ny,Nx} = AbstractLattice{Nx,Ny}
const SingleMPS = MPS{<:AbsLatY{1}}

# const InfiniteMPS{L<:AbstractUnitCell} = MPS{L}
# const iMPS = InfiniteMPS

getleft(mps::AbstractMPS) = mps.AL
getbond(mps::AbstractMPS) = mps.C
getright(mps::AbstractMPS) = mps.AR
getcentral(mps::AbstractMPS) = mps.AC

unpack(mps::AbstractMPS) = (getleft(mps), getbond(mps), getright(mps), getcentral(mps))

function isgauged(mps_single::AbstractMPS{<:AbstractLattice{Nx,1}}) where {Nx} 
    AL, C, AR, AC = unpack(mps_single)
    for x in axes(AL, 1)
        if !(mulbond(AL[x], C[x]) ≈ mulbond(C[x - 1], AR[x]) ≈ AC[x])
            return false
        end
    end
    return true
end
function isgauged(mps::AbstractMPS)
    for mpsview in mps
        if !isgauged(mpsview)
            return false
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

# function MPS(
#     f, T, lattice::AbstractLattice{Nx,Ny}, D::S, χ::S
# ) where {Nx,Ny,S<:IndexSpace}
#     return MPS(f, T, lattice, fill(D, Nx, Ny), fill(χ, Nx, Ny))
# end
function MPS(f, T, lattice, D, χ)
    # D_lat = OnLattice(lattice, D)
    # χ_lat = OnLattice(lattice, χ)
    return MPS(f, T, _fill_all_maybe(lattice, D, χ)...)
end
function MPS(f, T, D::AbstractOnLattice{L,S}, χ::AbstractOnLattice{L,S}) where {L,S<:IndexSpace}
    χ_dom = χ
    χ_cod = circshift(χ,(1,0))
    data_lat = @. TensorMap(f, T, χ_cod * D, χ_dom)
    return MPS(data_lat)
end

function MPS(A::AbstractOnLattice)
    # Left and right refers to the left and right indices of the BOND MATRIX,
    # not the MPS tensors.
    χ_left = getindex.(domain.(A), 1)
    χ_right = getindex.(codomain.(circshift(A, (1,0))), 1)

    AL = similar.(A)
    C = TensorMap.(rand, ComplexF64, χ_left, χ_right)
    AR = similar.(A)
    mixedgauge!(AL, C, AR, A)
    return MPS(AL, C, AR)
end

function MPS(AL, C, AR)
    AC = centraltensor(AL, C)
    return MPS(AL, C, AR, AC)
end

function Base.getindex(mps::AbstractMPS, y::Int)
    AL, C, AR, AC = unpack(mps)
    subAL = lview(AL, :, y)
    subC = lview(C, :, y)
    subAR = lview(AR, :, y)
    subAC = lview(AC, :, y)
    return MPS(subAL, subC, subAR, subAC)
end

function Base.iterate(mps::AbstractMPS{<:AbstractLattice{Nx,Ny}}, state = 1) where {Nx,Ny}
    if state > Ny
        return nothing
    else
        return mps[state], state + 1
    end
end

Base.length(mps::AbstractMPS{<:AbstractLattice{Nx,Ny}}) where {Nx,Ny} = Ny

_similar_ac(A::AbstractTensorMap{S,2,1}, C::AbstractTensorMap{S,1,1}) where {S} = similar(A)
function _similar_ac(C::AbstractTensorMap{S,1,1}, A::AbstractTensorMap{S,2,1}) where {S}
    return _similar_ac(A, C)
end

centraltensor(A1, A2) = centraltensor!(_similar_ac.(A1, A2), A1, A2)
centraltensor!(AC, AL::AbstractOnLattice{L,<:AbsTen{2,1}}, C) where {L} = mulbond!.(AC, AL, C)
centraltensor!(AC, C::AbstractOnLattice{L,<:AbsTen{1,1}}, AR) where {L} = mulbond!.(AC, circshift(C, (1, 0)), AR)

function mulbond(A1::AbstractTensorMap{S}, A2::AbstractTensorMap{S}) where {S}
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

function Base.transpose(M::MPS)
    return transpose.(unpack(M))
end

westbond(mps::MPS) = westbond.(mps.AC)


# norm(A::AbstractMPS) = tr(c * c')
