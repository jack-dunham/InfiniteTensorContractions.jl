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

## SINGLE LAYER

function _projectedge!(
    t_dst::T, t_src::T, m::M, u::UV, v::UV
) where {
    S,T<:AbstractTensorMap{S,1,2},M<:AbstractTensorMap{S,0,4},UV<:AbstractTensorMap{S,2,1}
}
    @tensoropt t_dst[h0; v0_d v0_u] =
        t_src[h1; v3 v1] * m[h0 v4 h1 v2] * u[v1 v2; v0_u] * v[v3 v4; v0_d]
    return t_dst
end

## DOUBLE LAYER

function _projectedge!(
    t_dst::T, t_src::T, ma::M, u::UV, v::UV
) where {
    S,N,T<:AbstractTensorMap{S,2,2},M<:AbstractTensorMap{S,N,4},UV<:AbstractTensorMap{S,3,1}
}
    return _projectedge!(t_dst, t_src, ma, ma, u, v)
end

function _projectedge!(
    t_dst::T, t_src::T, ma::M, mb::M, u::UV, v::UV
) where {
    S,T<:AbstractTensorMap{S,2,2},M<:AbstractTensorMap{S,1,4},UV<:AbstractTensorMap{S,3,1}
}
    @tensoropt (
        k => 2,
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
        ma[k; D1 D2 D3 D4] *
        (mb')[D5 D6 D7 D8; k] *
        u[x1 D4 D8; x2] *
        v[x3 D2 D6; x4]

    return t_dst
end

function _projectedge!(
    t_dst::T, t_src::T, ma::M, mb::M, u::UV, v::UV
) where {
    S,T<:AbstractTensorMap{S,2,2},M<:AbstractTensorMap{S,2,4},UV<:AbstractTensorMap{S,3,1}
}
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
        ma[k b; D1 D2 D3 D4] *
        (mb')[D5 D6 D7 D8; k b] *
        u[x1 D4 D8; x2] *
        v[x3 D2 D6; x4]

    return t_dst
end

## SINGLE LAYER METHODS

# O(χ^2 D^8) or D^10
function halfcontract(
    C1_00::C, T1_10::T, T1_20::T, C2_30::C, T4_01::T, M_11::M, M_21::M, T2_31::T
) where {
    S,C<:AbstractTensorMap{S,1,1},T<:AbstractTensorMap{S,1,2},M<:AbstractTensorMap{S,0,4}
}
    @tensoropt out[v8 v7; v5 v6] :=
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

## DOUBLE LAYER METHODS

function halfcontract(
    C1_00::C, T1_10::T, T1_20::T, C2_30::C, T4_01::T, M_11a::M, M_21a::M, T2_31::T
) where {
    S,N,C<:AbstractTensorMap{S,1,1},T<:AbstractTensorMap{S,2,2},M<:AbstractTensorMap{S,N,4}
}
    return halfcontract(
        C1_00, T1_10, T1_20, C2_30, T4_01, M_11a, M_11a, M_21a, M_21a, T2_31
    )
end

function halfcontract(
    C1_00::C,
    T1_10::T,
    T1_20::T,
    C2_30::C,
    T4_01::T,
    M_11a::M,
    M_11b::M,
    M_21a::M,
    M_21b::M,
    T2_31::T,
) where {
    S,C<:AbstractTensorMap{S,1,1},T<:AbstractTensorMap{S,2,2},M<:AbstractTensorMap{S,1,4}
}
    @tensoropt (
        k1 => 2,
        k2 => 2,
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
        M_11a[k1; D1 D2 D3 D4] *
        M_21a[k2; D5 D6 D1 D8] *
        (M_11b)'[E1 E2 E3 E4; k1] *
        (M_21b)'[E5 E6 E1 E8; k2] *
        C1_00[x1 x4] *
        T1_10[D4 E4; x2 x1] *
        T1_20[D8 E8; x3 x2] *
        C2_30[x3 x5] *
        T4_01[D3 E3; x6 x4] *
        T2_31[D5 E5; x5 x7]

    return top
end

function halfcontract(
    C1_00::C,
    T1_10::T,
    T1_20::T,
    C2_30::C,
    T4_01::T,
    M_11a::M,
    M_11b::M,
    M_21a::M,
    M_21b::M,
    T2_31::T,
) where {
    S,C<:AbstractTensorMap{S,1,1},T<:AbstractTensorMap{S,2,2},M<:AbstractTensorMap{S,2,4}
}
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
        M_11a[k1 b1; D1 D2 D3 D4] *
        M_21a[k2 b2; D5 D6 D1 D8] *
        (M_11b)'[E1 E2 E3 E4; k1 b1] *
        (M_21b)'[E5 E6 E1 E8; k2 b2] *
        C1_00[x1 x4] *
        T1_10[D4 E4; x2 x1] *
        T1_20[D8 E8; x3 x2] *
        C2_30[x3 x5] *
        T4_01[D3 E3; x6 x4] *
        T2_31[D5 E5; x5 x7]

    return top
end
