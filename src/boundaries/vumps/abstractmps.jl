# DONE
abstract type AbstractMPS end

@doc raw"""
    struct MPS
"""
struct MPS{AType<:AbUnCe{<:TenAbs{2}},CType<:AbUnCe{<:AbsTen{0,2}}
} <: AbstractMPS
    AL::AType
    C::CType
    AR::AType
    AC::AType
    function MPS(
        AL::AType, C::CType, AR::AType, AC::AType
    ) where {AType<:AbstractUnitCell,CType<:AbstractUnitCell}
        check_size_allequal(AL, C, AR, AC)
        mps = new{AType,CType}(AL, C, AR, AC)
        validate(mps)
        return mps
    end
    function MPS(
        AL::AType, C::CType, AR::AType, AC::AType
    ) where {AType<:SubUnitCell,CType<:SubUnitCell}
        mps = new{AType,CType}(AL, C, AR, AC)
        return mps
    end
end

# FIELD GETTERS

getleft(mps::MPS) = mps.AL
getbond(mps::MPS) = mps.C
getright(mps::MPS) = mps.AR
getcentral(mps::MPS) = mps.AC

unpack(mps::AbstractMPS) = (getleft(mps), getbond(mps), getright(mps), getcentral(mps))

# BASE

Base.size(mps::AbstractMPS) = size(getcentral(mps))
Base.length(mps::AbstractMPS) = size(mps)[2]
Base.similar(mps::M) where {M<:MPS} = MPS(similar.(unpack(mps))...)::M

function Base.getindex(mps::AbstractMPS, y::Int)
    AL, C, AR, AC = unpack(mps)
    @views begin
        subAL = AL[:, y]
        subC = C[:, y]
        subAR = AR[:, y]
        subAC = AC[:, y]
    end
    return MPS(subAL, subC, subAR, subAC)
end

function Base.iterate(mps::AbstractMPS, state=1)
    if state > length(mps)
        return nothing
    else
        return mps[state], state + 1
    end
end

function Base.transpose(M::MPS)
    return transpose.(unpack(M))
end

# GAUGING

function isgauged(mps::AbstractMPS)
    for single_mps in mps
        unp = unpack(single_mps)
        if all(isassigned, unpack(single_mps))
            AL, C, AR, AC = unp
            for x in axes(AL, 1)
                if !(mulbond(AL[x], C[x]) ≈ mulbond(C[x-1], AR[x]) ≈ AC[x])
                    return false
                end
            end
        else
            @debug "Undefined MPS. Skipping gauge check..."
            return true
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

# CONSTRUCTORS

# function MPS(f, T, lattice, D, χ; kwargs...)
#     return MPS(f, T, _fill_all_maybe(lattice, D, χ)...; kwargs...)
# end

function MPS(
    f, T, D::AbstractUnitCell, right_bonds::AbstractUnitCell
)
    left_bonds = circshift(right_bonds, (1, 0))
    data_lat = @. TensorMap(f, T, D, right_bonds * adjoint(left_bonds))
    return MPS(data_lat)
end

function MPS(A::AbstractUnitCell)
    bonds = @. getindex(domain(A), 1) # Canonical bond (right-hand) bond of mps tensor

    T = numbertype(A)

    AL = similar.(A)
    C = @. TensorMap(rand, T, one(bonds), bonds * adjoint(bonds)) # R * L
    AR = similar.(A)

    mixedgauge!(AL, C, AR, A)
    return MPS(AL, C, AR)
end

MPS(AL, C, AR) = MPS(AL, C, AR, centraltensor(AL, C))

# TENSOR OPERATIONS

function _similar_ac(ac1::AbstractTensorMap{S,N}, ac2::AbstractTensorMap{S,M}) where {S,N,M}
    return _similar_ac(Val(N), ac1, ac2)
end
_similar_ac(::Val{0}, ac1, ac2) = similar(ac2)
_similar_ac(::Val{N}, ac1, ac2) where {N} = similar(ac1)

centraltensor(A1, A2) = centraltensor!(_similar_ac.(A1, A2), A1, A2)
function centraltensor!(AC, AL::AbUnCe{<:TenAbs{2}}, C)
    return mulbond!.(AC, AL, C)
end
function centraltensor!(AC, C::AbUnCe{<:AbsTen{0,2}}, AR)
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

# MAYBE DEPREC

westbond(mps::MPS) = westbond.(mps.AC)

# DEPREC
#=

function get_truncmetric_tensors(mps::MPS, bond::Bond)
    bond_1 = bond
    # bond_2 = nextrow(bond)

    AC = getcentral(mps)
    AR = getright(mps)

    ac_u = AC[left(bond_1)]
    ac_d = AC[left(bond_1)+CartesianIndex(0, 1)]

    ar_u = AR[right(bond_1)]
    ar_d = AR[right(bond_1)+CartesianIndex(0, 1)]

    return ac_u, ac_d, ar_u, ar_d
end
=#
