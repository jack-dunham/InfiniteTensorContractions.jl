# mpo-like
function applyhac!(
    hac::AbsTen{1,2}, ac::AbsTen{1,2}, fl::AbsTen{1,2}, fr::AbsTen{1,2}, m::AbsTen{0,4}
)
    @tensoropt hac[D2; x4 x3] =
        ac[D4; x2 x1] * fl[D3; x1 x3] * m[D1 D2 D3 D4] * fr[D1; x2 x4]
end
# pepo-like
function applyhac!(
    hac::AbsTen{2,2}, ac::AbsTen{2,2}, fl::AbsTen{2,2}, fr::AbsTen{2,2}, m::AbsTen{2,4}
)
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
        m[k b; D1 D2 D3 D4] *
        (m')[E1 E2 E3 E4; k b] *
        fr[D1 E1; x2 x4]
end

# mpo-like
function applyhc!(hc::AbsTen{0,2}, c::AbsTen{0,2}, fl::AbsTen{1,2}, fr::AbsTen{1,2})
    @tensoropt hc[x4 x3] = c[x2 x1] * fl[D; x1 x3] * fr[D; x2 x4]
end
# pepo-like
function applyhc!(hc::AbsTen{0,2}, c::AbsTen{0,2}, fl::AbsTen{2,2}, fr::AbsTen{2,2})
    @tensoropt hc[x4 x3] = c[x2 x1] * fl[D E; x1 x3] * fr[D E; x2 x4]
end
