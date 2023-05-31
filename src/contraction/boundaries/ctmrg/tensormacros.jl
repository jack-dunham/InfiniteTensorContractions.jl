projectcorner(c, t, p) = projectcorner!(similar(c), c, t, p)
function projectcorner!(c_dst, c_src, t, uv)
    if c_dst === c_src
        c_copy = deepcopy(c_src)
        return _projectcorner!(c_dst, c_copy, t, uv)
    else
        return _projectcorner!(c_dst, c_src, t, uv)
    end
end
function _projectcorner!(c_dst, c_src, t::AbsTen{1,2}, uv)
    # P: (x,D)<-(x_out)
    @tensoropt c_dst[h0 v0] = c_src[h1 v1] * t[v2; h0 h1] * uv[v1 v2; v0]
    return c_dst
end
function _projectcorner!(c_dst, c_src, t::AbsTen{2,2}, uv)
    # P: (x,D)<-(x_out)
    @tensoropt c_dst[h0 v0] = c_src[h1 v1] * t[v2 x; h0 h1] * uv[v1 v2 x; v0]
    return c_dst
end

projectedge(t, m, u, v) = projectedge!(similar(t), t, m, u, v)
function projectedge!(t_dst, t_src, m, u, v)
    if t_dst === t_src
        t_copy = deepcopy(t_src)
        return _projectedge!(t_dst, t_copy, m, u, v)
    else
        return _projectedge!(t_dst, t_src, m, u, v)
    end
end
function _projectedge!(t_dst::AbsTen{1,2}, t_src::AbsTen{1,2}, m, u, v)
    @tensoropt t_dst[h0; v0_d v0_u] =
        t_src[h1; v3 v1] * m[h0 v4 h1 v2] * u[v1 v2; v0_u] * v[v3 v4; v0_d]
    return t_dst
end
function _projectedge!(t_dst::AbsTen{2,2}, t_src::AbsTen{2,2}, m, u, v)
    @tensoropt (
        k => 2,
        b => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D7 => D,
        D8 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
    ) t_dst[D1 D5; x4 x2] =
        t_src[D3 D7; x3 x1] *
        m[k b; D1 D2 D3 D4] *
        (m')[D5 D6 D7 D8; k b] *
        u[x1 D4 D8; x2] *
        v[x3 D2 D6; x4]

    return t_dst
end

# O(χ^2 D^8) or D^10
function halfcontract(C1_00, T1_10::AbsTen{1,2}, T1_20, C2_30, T4_01, M_11, M_21, T2_31)
    @tensoropt (
        h1 => D,
        h2 => D,
        h3 => D,
        h4 => D^2,
        h5 => D^2,
        h6 => D^2,
        v1 => D,
        v2 => D^2,
        v3 => D^2,
        v4 => D,
        v5 => D,
        v6 => D^2,
        v7 => D^2,
        v8 => D^2,
    ) out[v8 v7; v5 v6] :=
        C1_00[h1 v1] *
        T1_10[v2; h2 h1] *
        T1_20[v3; h3 h2] *
        C2_30[h3 v4] *
        T4_01[h4; v5 v1] *
        M_11[h5 v6 h4 v2] *
        M_21[h6 v7 h5 v3] *
        T2_31[h6; v4 v8]

    return out
end

# O(χ^3 D^6) or D^9 or r^2 χ^3 D^6
function halfcontract(C1_00, T1_10::AbsTen{2,2}, T1_20, C2_30, T4_01, M_11, M_21, T2_31)
    @tensoropt (
        k1 => 2,
        k2 => 2,
        b1 => 2,
        b2 => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D8 => D,
        E1 => D,
        E2 => D,
        E3 => D,
        E4 => D,
        E5 => D,
        E6 => D,
        E8 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
        x5 => D,
        x6 => D,
        x7 => D,
    ) top[x7 D6 E6; x6 D2 E2] :=
        M_11[k1 b1; D1 D2 D3 D4] *
        M_21[k2 b2; D5 D6 D1 D8] *
        (M_11)'[E1 E2 E3 E4; k1 b1] *
        (M_21)'[E5 E6 E1 E8; k2 b2] *
        C1_00[x1 x4] *
        T1_10[D4 E4; x2 x1] *
        T1_20[D8 E8; x3 x2] *
        C2_30[x3 x5] *
        T4_01[D3 E3; x6 x4] *
        T2_31[D5 E5; x5 x7]

    return top
end

function tracecontract(ctmrg::CornerMethodTensors, bulk::ContractableTensors)
    cor = corners(ctmrg)
    edg = edges(ctmrg)

    rv = similar(bulk, numbertype(bulk))

    for y in axes(bulk, 2)
        for x in axes(bulk, 1)
            cs = cor[x, y]
            es = getindex.(edg[x, y], 1)
            rv[x, y] = tracecontract(_reorderargs(bulk[x, y], cs..., es...)...)
        end
    end
    return rv
end

# Util function for convenience
function _reorderargs(M, C1, C2, C3, C4, T1, T2, T3, T4)
    return C1, T1, C2, T4, M, T2, C4, T3, C3
end

# Contract a tensor representing a trace. That is M_11::AbsTen{0,4}
function tracecontract(C1_00, T1_10, C2_20, T4_01, M_11, T2_21, C4_02, T3_12, C3_22)
    @tensoropt rv =
        C1_00[x1 x3] *
        T1_10[D4; x2 x1] *
        C2_20[x2 x4] *
        T4_01[D3; x5 x3] *
        M_11[D1 D2 D3 D4] *
        T2_21[D1; x4 x6] *
        C4_02[x7 x5] *
        T3_12[D2; x7 x8] *
        C3_22[x8 x6]
    return rv
end

function onelocalcontract(C1_00, T1_10, C2_20, T4_01, M_11, T2_21, C4_02, T3_12, C3_22)
    # Need to specify output structure otherwise @tensoropt will incorrectly guess 
    # index placement.
    physpace = codomain(M_11)
    onelocal = similar(M_11, physpace, one(physpace))
    return onelocalcontract!(
        onelocal, C1_00, T1_10, C2_20, T4_01, M_11, T2_21, C4_02, T3_12, C3_22
    )
end
function onelocalcontract!(
    dst, C1_00, T1_10, C2_20, T4_01, M_11, T2_21, C4_02, T3_12, C3_22
)
    @tensoropt dst[k b] =
        C1_00[x1 x3] *
        T1_10[D4; x2 x1] *
        C2_20[x2 x4] *
        T4_01[D3; x5 x3] *
        M_11[k b; D1 D2 D3 D4] *
        T2_21[D1; x4 x6] *
        C4_02[x7 x5] *
        T3_12[D2; x7 x8] *
        C3_22[x8 x6]
    return rv
end

function metric(pepo::AbstractPEPO, ctmrg::CornerMethodTensors, bond::Bond)
    return ctmrg_trunctensors(pepo, ctmrg, bond)
end

function ctmrg_trunctensors(pepo, ctmrg, bond)
    l = left(bond)
    r = right(bond)

    corners = ctmrg.corners
    edges = ctmrg.edges

    cs = corners[l[1]:r[1], l[2]]
    t1s, t2s, t3s, t4s = edges[l[1]:r[1], l[2]]

    ms = pepo[bond]

    return truncmetriccontract(cs..., t1s..., t2s..., t3s..., t4s..., ms...)
end

function truncmetriccontract(C1, C2, C3, C4, T1_1, T1_2, T2, T3_1, T3_2, T4, M1, M2)
    sp = domain(M1)[1] * domain(M1)[1]
    dst = similar(M1, sp, sp)
    return truncmetriccontract!(dst, C1, T1_1, T1_2, C2, T4, M1, M2, T2, C4, T3_1, T3_2, C3)
end

function truncmetriccontract!(
    dst, C1_00, T1_10, T1_20, C2_30, T4_01, M_11, M_21, T2_31, C4_02, T3_12, T3_22, C3_32
)
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
        x7 => D,
        x8 => D,
        x9 => D,
        x10 => D,
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
        C1_00[x1 x4] *
        T1_10[D4 E4; x2 x1] *
        T1_20[D8 E8; x3 x2] *
        C2_30[x3 x5] *
        T4_01[D3 E3; x6 x4] *
        M_11[k1 b1; D1 D2 D3 D4] *
        (M_11')[E1 E2 E3 E4; k1 b1] *
        M_21[k2 b2; D5 D6 D7 D8] *
        (M_21')[E5 E6 E7 E8; k2 b2] *
        T2_31[D5 E5; x5 x7] *
        C4_02[x8 x6] *
        T3_12[D2 E2; x8 x9] *
        T3_22[D6 E6; x9 x10] *
        C3_32[x10 x7]
    return dst
end
