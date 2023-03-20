struct TransferMatrix{S,A<:AbstractTensorMap{S,2,1},M<:AbstractTensorMap{S,2,2}}
    above::A
    middle::M
    below::A
end

# do this to get other FLs
function Base.:*(fl::AbstractTensorMap{S,1,2}, T::TransferMatrix{S}) where {S}
    at = T.above
    ab = T.below
    m = T.middle

    flu = similar(fl)

    @tensoropt flu[b2; a2 3] := fl[b1; a1 1] * at[a1 4; a2] * m[1 2; 3 4] * (ab')[b2; b1 2]

    return flu
end
function Base.:*(T::TransferMatrix{S}, fr::AbstractTensorMap{S,2,1}) where {S}
    at = T.above
    ab = T.below
    m = T.middle

    fru = similar(fr)

    @tensoropt fru[a1 1; b1] := at[a1 4; a2] * m[1 2; 3 4] * (ab')[b2; b1 2] * fr[a2 3; b2]

    return fru
end
