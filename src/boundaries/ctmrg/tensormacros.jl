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
