function applyhac!(
    hac::AbsTen{1,2}, ac::AbsTen{1,2}, fl::AbsTen{1,2}, fr::AbsTen{1,2}, m::AbsTen{0,4}
)
    @tensoropt hac[D2; x4 x3] =
        ac[D4; x2 x1] * fl[D3; x1 x3] * m[D1 D2 D3 D4] * fr[D1; x2 x4]
end
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

function applyhc!(hc::AbsTen{0,2}, c::AbsTen{0,2}, fl::AbsTen{1,2}, fr::AbsTen{1,2})
    @tensoropt hc[x4 x3] = c[x2 x1] * fl[D; x1 x3] * fr[D; x2 x4]
end
function applyhc!(hc::AbsTen{0,2}, c::AbsTen{0,2}, fl::AbsTen{2,2}, fr::AbsTen{2,2})
    @tensoropt hc[x4 x3] = c[x2 x1] * fl[D E; x1 x3] * fr[D E; x2 x4]
end

function tracecontract(vumps::VUMPS, bulk)
    AC = getcentral(vumps.mps)
    FL = vumps.fixedpoints.left
    FR = vumps.fixedpoints.right
    return tracecontract.(FL, FR, AC, circshift(AC, (0, -1)), bulk)
end

function tracecontract(FL_00, FR_00, AT_00, AB_01, M_00)
    @tensoropt rv =
        FL_00[D3; x1 x3] *
        AT_00[D4; x2 x1] *
        FR_00[D1; x2 x4] *
        M_00[D1 D2 D3 D4] *
        (AB_01')[x4 x3; D2]
    return rv
end

function onelocalcontract!(dst, FL_00, FR_00, AT_00, AB_01, M_00)
    @tensoropt dst[k b] =
        FL_00[D3; x1 x3] *
        AT_00[D4; x2 x1] *
        FR_00[D1; x2 x4] *
        M_00[k b; D1 D2 D3 D4] *
        (AB_01')[x4 x3; D2]
    return rv
end

function truncmetric(boundary::AbstractBoundary, bulk::ContractableTensors)
    sp = @. domain(bulk, 1) * domain(bulk, 1)
    dst = similar.(bulk, sp, sp)
    return truncmetric!(dst, boundary, bulk)
end
function truncmetric!(dst, vumps::VUMPS, bulk::ContractableTensors)
    mps = vumps.mps

    AC = getcentral(mps)
    AR = getright(mps)

    FL = vumps.fixedpoints.left
    FR = vumps.fixedpoints.right

    # for x in axes(bulk, 1)
    #     for y in axes(bulk, 2)
    #         truncmetriccontract!(dst[x,y],FL[x,y],AC[x,y],AC[x,y+1],AR[x+1,y],AR[x+1,y+1],bulk[x,y],bulk[x+1,y],FR[x+1,y])
    #     end
    # end
    #
    # return dst
    
    return truncmetriccontract!.(dst,
        FL,
        AC,
        circshift(AC, (0, -1)),
        circshift(AR, (-1, 0)),
        circshift(AR, (-1, -1)),
        bulk,
        circshift(bulk, (-1, 0)),
        circshift(FR, (-1, 0)),
    )
end

function truncmetriccontract!(dst, FL_00, AC_00, AC_01, AR_10, AR_11, M_00, M_10, FR_10)
    @tensoropt (
        k1 => 2,
        k2 => 2,
        b1 => 2,
        b2 => 2,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
        x5 => D,
        x6 => D,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D7 => D,
        D8 => D,
        E1 => D,
        E2 => D,
        E3 => D,
        E4 => D,
        E5 => D,
        E6 => D,
        E7 => D,
        E8 => D,
    ) dst[D7 E1; D1 E7] =
        AC_00[D4 E4; x2 x1] *
        AR_10[D8 E8; x3 x2] *
        (AC_01')[x5 x4; D2 E2] *
        (AR_11')[x6 x5; D6 E6] *
        M_00[k1 b1; D1 D2 D3 D4] *
        (M_00')[E1 E2 E3 E4; k1 b1] *
        M_10[k2 b2; D5 D6 D7 D8] *
        (M_10')[E5 E6 E7 E8; k2 b2] *
        FL_00[D3 E3; x1 x4] *
        FR_10[D5 E5; x3 x6]

    return dst
end
