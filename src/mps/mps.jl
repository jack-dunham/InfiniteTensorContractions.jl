abstract type AbstractMPS{L<:AbstractLattice} end
# const AbstractInfiniteMPS{L} = AbstractMPS{L<:AbstractUnitCell}

@doc raw"""
    struct MPS
"""
struct MPS{
    L,AType<:AbstractOnLattice{L,<:TenAbs{2}},CType<:AbstractOnLattice{L,<:AbsTen{0,2}}
} <: AbstractMPS{L}
    AL::AType
    C::CType
    AR::AType
    AC::AType
    function MPS(
        AL::AType, C::CType, AR::AType, AC::AType; check=true
    ) where {L,AType<:AbstractOnLattice{L},CType<:AbstractOnLattice{L}}
        mps = new{L,AType,CType}(AL, C, AR, AC)
        if check
            validate(mps)
        end
        return mps
    end
    function MPS(
        AL::AType, C::CType, AR::AType, AC::AType
    ) where {L<:SubLattice,AType<:AbstractOnLattice{L},CType<:AbstractOnLattice{L}}
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


Base.similar(mps::MPS) = MPS(similar.(unpack(mps))...; check=false)::MPS

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

## Contructors
function MPS(f, T, lattice, D, χ; kwargs...)
    return MPS(f, T, _fill_all_maybe(lattice, D, χ)...; kwargs...)
end
function MPS(
    f, T, D::AbstractOnLattice{L}, right_bonds::AbstractOnLattice{L}; kwargs...
) where {L}
    left_bonds = circshift(right_bonds, (1, 0))
    data_lat = @. TensorMap(f, T, D, right_bonds * adjoint(left_bonds))
    return MPS(data_lat; kwargs...)
end

function MPS(A::AbstractOnLattice; kwargs...)
    bonds = @. getindex(domain(A), 1) # Canonical bond (right-hand) bond of mps tensor

    T = numbertype(A)

    AL = similar.(A)
    C = @. TensorMap(rand, T, one(bonds), bonds * adjoint(bonds)) # R * L
    AR = similar.(A)

    mixedgauge!(AL, C, AR, A; kwargs...)
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

function Base.iterate(mps::AbstractMPS{<:AbstractLattice{Nx,Ny}}, state=1) where {Nx,Ny}
    if state > Ny
        return nothing
    else
        return mps[state], state + 1
    end
end

Base.length(mps::AbstractMPS{<:AbstractLattice{Nx,Ny}}) where {Nx,Ny} = Ny

function _similar_ac(ac1::AbstractTensorMap{S,N}, ac2::AbstractTensorMap{S,M}) where {S,N,M}
    return _similar_ac(Val(N), ac1, ac2)
end
_similar_ac(::Val{0}, ac1, ac2) = similar(ac2)
_similar_ac(::Val{N}, ac1, ac2) where {N} = similar(ac1)

centraltensor(A1, A2) = centraltensor!(_similar_ac.(A1, A2), A1, A2)
function centraltensor!(AC, AL::AbstractOnLattice{L,<:AbsTen{N,2}}, C) where {L,N}
    return mulbond!.(AC, AL, C)
end
function centraltensor!(AC, C::AbstractOnLattice{L,<:AbsTen{0,2}}, AR) where {L}
    return mulbond!.(AC, circshift(C, (1, 0)), AR)
end

function mulbond(A1::AbstractTensorMap{S}, A2::AbstractTensorMap{S}) where {S}
    return mulbond!(_similar_ac(A1, A2), A1, A2)
end
# Right
function mulbond!(
    CA::AbstractTensorMap{S,1,2}, C::AbstractTensorMap{S,0,2}, A::AbstractTensorMap{S,1,2}
) where {S<:IndexSpace}
    return @tensoropt CA[p1; xr xl] = C[x_in xl] * A[p1; xr x_in]
end
function mulbond!(
    CA::AbstractTensorMap{S,2,2}, C::AbstractTensorMap{S,0,2}, A::AbstractTensorMap{S,2,2}
) where {S<:IndexSpace}
    return @tensoropt CA[p1 p2; xr xl] = C[x_in xl] * A[p1 p2; xr x_in]
end
# Left
function mulbond!(
    AC::AbstractTensorMap{S,1,2}, A::AbstractTensorMap{S,1,2}, C::AbstractTensorMap{S,0,2}
) where {S<:IndexSpace}
    return @tensoropt AC[p1; xr xl] = A[p1; x_in xl] * C[xr x_in]
end
function mulbond!(
    AC::AbstractTensorMap{S,2,2}, A::AbstractTensorMap{S,2,2}, C::AbstractTensorMap{S,0,2}
) where {S<:IndexSpace}
    return @tensoropt AC[p1 p2; xr xl] = A[p1 p2; x_in xl] * C[xr x_in]
end

function Base.transpose(M::MPS)
    return transpose.(unpack(M))
end

westbond(mps::MPS) = westbond.(mps.AC)

# norm(A::AbstractMPS) = tr(c * c')

function get_truncmetric_tensors(mps::MPS, bond::Bond)
    bond_1 = bond
    # bond_2 = nextrow(bond)

    AC = getcentral(mps)
    AR = getright(mps)

    ac_u = AC[left(bond_1)]
    ac_d = AC[left(bond_1) + CartesianIndex(0, 1)]

    ar_u = AR[right(bond_1)]
    ar_d = AR[right(bond_1) + CartesianIndex(0, 1)]

    return ac_u, ac_d, ar_u, ar_d
end
