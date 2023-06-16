# mpo-like
function applyhac!(
    hac::AbstractTensorMap{S,1,2},
    ac::AbstractTensorMap{S,1,2},
    fl::AbstractTensorMap{S,1,2},
    fr::AbstractTensorMap{S,1,2},
    m::AbstractTensorMap{S,0,4},
) where {S}
    @tensoropt hac[D2; x4 x3] =
        ac[D4; x2 x1] * fl[D3; x1 x3] * m[D1 D2 D3 D4] * fr[D1; x2 x4]
end
# peps-like
function applyhac!(
    hac::AbstractTensorMap{S,2,2},
    ac::AbstractTensorMap{S,2,2},
    fl::AbstractTensorMap{S,2,2},
    fr::AbstractTensorMap{S,2,2},
    ma::AbstractTensorMap{S,N,4},
) where {S,N}
    return applyhac!(hac, ac, fl, fr, ma, ma)
end

function applyhac!(
    hac::AbstractTensorMap{S,2,2},
    ac::AbstractTensorMap{S,2,2},
    fl::AbstractTensorMap{S,2,2},
    fr::AbstractTensorMap{S,2,2},
    ma::AbstractTensorMap{S,1,4},
    mb::AbstractTensorMap{S,1,4},
) where {S}
    @tensoropt (
        k => 2,
        D1 => D,
        E1 => D,
        D2 => D,
        E2 => D,
        D3 => D,
        E3 => D,
        D4 => D,
        E4 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
    ) hac[D2 E2; x4 x3] =
        ac[D4 E4; x2 x1] *
        fl[D3 E3; x1 x3] *
        ma[k; D1 D2 D3 D4] *
        (mb')[E1 E2 E3 E4; k] *
        fr[D1 E1; x2 x4]
end
# pepo-like
function applyhac!(
    hac::AbstractTensorMap{S,2,2},
    ac::AbstractTensorMap{S,2,2},
    fl::AbstractTensorMap{S,2,2},
    fr::AbstractTensorMap{S,2,2},
    ma::AbstractTensorMap{S,2,4},
    mb::AbstractTensorMap{S,2,4},
) where {S}
    @tensoropt (
        k => 2,
        b => 2,
        D1 => D,
        E1 => D,
        D2 => D,
        E2 => D,
        D3 => D,
        E3 => D,
        D4 => D,
        E4 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
    ) hac[D2 E2; x4 x3] =
        ac[D4 E4; x2 x1] *
        fl[D3 E3; x1 x3] *
        ma[k b; D1 D2 D3 D4] *
        (mb')[E1 E2 E3 E4; k b] *
        fr[D1 E1; x2 x4]
end

# mpo-like
function applyhc!(
    hc::AbstractTensorMap{S,0,2},
    c::AbstractTensorMap{S,0,2},
    fl::AbstractTensorMap{S,1,2},
    fr::AbstractTensorMap{S,1,2},
) where {S}
    @tensoropt hc[x4 x3] = c[x2 x1] * fl[D; x1 x3] * fr[D; x2 x4]
end
# pepx-like
function applyhc!(
    hc::AbstractTensorMap{S,0,2},
    c::AbstractTensorMap{S,0,2},
    fl::AbstractTensorMap{S,2,2},
    fr::AbstractTensorMap{S,2,2},
) where {S}
    @tensoropt hc[x4 x3] = c[x2 x1] * fl[D E; x1 x3] * fr[D E; x2 x4]
end
