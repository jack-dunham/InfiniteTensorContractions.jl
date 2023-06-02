# DONE (NEEDS RESTRUCTURING)
abstract type AbstractFixedPoints end

# Fixed points of a transfer matrix
struct FixedPoints{A<:AbstractUnitCell{<:TenAbs{2}}} <: AbstractFixedPoints
    left::A
    right::A
    function FixedPoints(left::A, right::A) where {A}
        check_size_allequal(left, right)
        return new{A}(left, right)
    end
end

FixedPoints(f, mps::MPS, bulk) = initfixedpoints(f, mps, bulk)

Base.similar(fps::FixedPoints) = FixedPoints(similar(fps.left), similar(fps.right))

# DEPREC
#=
function get_truncmetric_tensors(fp::FixedPoints, bond::Bond)
    l = left(bond)
    r = right(bond)
    return fp.left[l], fp.right[r]
end
=#

function initfixedpoints(f, mps::MPS, bulk)
    mps_tensor = getcentral(mps)
    bulk_tensor = convert.(TensorMap, bulk)
    left = _initfixedpoints.(f, mps_tensor, bulk_tensor, :left)
    right = _initfixedpoints.(f, mps_tensor, bulk_tensor, :right)
    return FixedPoints(left, right)
end

function _initfixedpoints(
    f, mps::AbstractTensorMap{S}, bulk::AbstractTensorMap{S}, leftright::Symbol
) where {S}
    T = promote_type(eltype(mps), eltype(bulk))

    cod = fixedpoint_codomain(bulk, leftright)
    dom = fixedpoint_domain(mps, leftright)

    return TensorMap(f, T, cod, dom)
end

function fixedpoint_codomain(bulk, leftright::Symbol)
    if leftright === :left
        return _fixedpoint_codomain(bulk, 3)
    elseif leftright === :right
        return _fixedpoint_codomain(bulk, 1)
    else
        throw(ArgumentError(""))
    end
end
_fixedpoint_codomain(bulk::AbsTen{0,4}, lr::Int) = domain(bulk)[lr]
_fixedpoint_codomain(bulk::AbsTen{2,4}, lr::Int) = domain(bulk)[lr] * domain(bulk)[lr]'

function fixedpoint_domain(mps_tensor, leftright::Symbol)
    if leftright === :left
        return _fixedpoint_domain(mps_tensor, 2)
    elseif leftright === :right
        return _fixedpoint_domain(mps_tensor, 1)
    else
        throw(ArgumentError(""))
    end
end
_fixedpoint_domain(mps_tensor, lr::Int) = domain(mps_tensor)[lr]' * domain(mps_tensor)[lr]

function renorm(cb, ca, fl, fr)
    c_out = hcapply!(similar(ca), ca, fl, fr)
    N = dot(cb, c_out)
    return N
end

function hcapply!(
    hc::AbstractTensorMap{S,0,2},
    c::AbstractTensorMap{S,0,2},
    fl::AbstractTensorMap{S,1,2},
    fr::AbstractTensorMap{S,1,2},
) where {S}
    @tensoropt hc[dr dl] = c[ur ul] * fl[m; ul dl] * fr[m; ur dr]
    return hc
end
function hcapply!(
    hc::AbstractTensorMap{S,0,2},
    c::AbstractTensorMap{S,0,2},
    fl::AbstractTensorMap{S,2,2},
    fr::AbstractTensorMap{S,2,2},
) where {S}
    @tensoropt hc[dr dl] = c[ur ul] * fl[m1 m2; ul dl] * fr[m1 m2; ur dr]
    return hc
end

function fixedpoints(mps::MPS, bulk)
    initial_fpoints = FixedPoints(rand, mps, bulk)
    return fixedpoints!(initial_fpoints, mps, bulk)
end

#this is now the correct env func

function fixedpoints!(fpoints::FixedPoints, mps::MPS, bulk)
    AL, C, AR, _ = unpack(mps)
    tm_left = TransferMatrix.(AL, bulk, circshift(AL, (0, -1)))
    tm_right = TransferMatrix.(AR, bulk, circshift(AR, (0, -1)))

    return fixedpoints!(fpoints, tm_left, tm_right, C)
end

function fixedpoints!(
    fpoints::FixedPoints,
    tm_left::AbstractTransferMatrices,
    tm_right::AbstractTransferMatrices,
    C::AbstractUnitCell,
)
    FL = fpoints.left
    FR = fpoints.right

    Nx, Ny = size(C)

    for y in 1:Ny
        # _, Ls, _ = eigsolve(z -> fpsolve(z, AL, M, 1, y), FL[1, y], 1, :LM)
        # _, Rs, _ = eigsolve(z -> fpsolve(z, AR, M, Nx, y), FR[Nx, y], 1, :LM)
        scal_left, Ls, linfo = eigsolve(
            z -> leftsolve(z, tm_left[:, y]), FL[1, y], 1, :LM; ishermitian=false
        )
        scal_right, Rs, rinfo = eigsolve(
            z -> rightsolve(z, tm_right[:, y]), FR[Nx, y], 1, :LM; ishermitian=false
        )

        FL[1, y] = Ls[1]
        FR[end, y] = Rs[1]

        # @info "left: $(scal_left[1]) right: $(scal_right[1])"

        for x in 2:Nx
            # FL[x, y] =
            #     FL[x - 1, y] * TransferMatrix(AL[x - 1, y], M[x - 1, y], AL[x - 1, y + 1])#works
            multransfer!(FL[x, y], FL[x-1, y], tm_left[x-1, y])
        end

        NN = renorm(C[Nx, y+1], C[Nx, y], Ls[1], Rs[1]) # Should be positive?

        NN = sqrt(NN)

        # the eigenvalue problem eqn gives us FL[1,y] and FR[end,y], so normalise them
        rmul!(FL[1, y], 1 / NN) #correct NN
        rmul!(FR[end, y], 1 / NN) #correct NN

        for x in (Nx-1):-1:1
            # FR[x, y] =
            #     TransferMatrix(AR[x + 1, y], M[x + 1, y], AR[x + 1, y + 1]) * FR[x + 1, y]#works
            multransfer!(FR[x, y], tm_right[x+1, y], FR[x+1, y])

            NN = renorm(C[x, y+1], C[x, y], FL[x+1, y], FR[x, y])

            rmul!(FR[x, y], 1 / sqrt(NN))
            rmul!(FL[x+1, y], 1 / sqrt(NN))
        end

        # First x seems to be normalized, but not second x
        for x in 1:Nx
            NN = renorm(C[x, y+1], C[x, y], FL[x+1, y], FR[x, y])
            # @warn "NN: $((x,y)),  $NN"
            # rmul!(FR[x, y], 1 / (NN))
            # rmul!(FL[x + 1, y], 1 / (NN))
        end
    end
    return fpoints
end
# FL[x] * T[x] = FL[x + 1]
function leftsolve(f0, Ts)
    f_new = f0
    for T in Ts
        f_new = f_new * T
    end
    return f_new
end
function rightsolve(f0, Ts)
    f_new = f0
    for T in reverse(Ts)
        f_new = T * f_new
    end
    return f_new
end

function simple_environments!(FL, FR, A, M)
    Nx, Ny = size(M)

    # λ = Matrix{numbertype(M)}(undef, Nx, Ny)

    AL, C, AR, _ = unpack(A)

    for x in 1:Nx, y in 1:Ny
        _, Ls, _ = eigsolve(z -> fpsolve(z, AL, M, x, y), FL[x, y], 1, :LM)
        _, Rs, _ = eigsolve(z -> fpsolve(z, AR, M, x, y), FR[x, y], 1, :LM)

        FL[x, y] = Ls[x]
        FR[x, y] = Rs[x]
    end
    return FL, FR
end

function fpsolve(fl::AbsTen{1,2}, A::AbstractUnitCell, M, x::Int, y::Int)
    Nx = size(A)[1]
    fl_n = flsolve(fl, A, M, x, y, Val(Nx))
    return fl_n
end
function fpsolve(fr::AbsTen{2,1}, A::AbstractUnitCell, M, x::Int, y::Int)
    Nx = size(A)[1]
    fr_n = frsolve(fr, A, M, x, y, Val(Nx))
    return fr_n
end

# Right-environment
function flsolve(
    fl::AbstractTensorMap{S,1,2},
    a::AbstractMatrix{<:AbstractTensorMap{S,2,1}},
    m::AbstractMatrix{<:AbstractTensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{1},
) where {S}
    flu = similar(fl)
    @tensoropt begin
        flu[8; 6 7] =
            fl[3; 1 2] * (a[x, y])[1 4; 6] * (m[x, y])[2 5; 7 4] * (a[x, y+1]')[8; 3 5]
    end
    return flu
end
function flsolve(
    fl::AbstractTensorMap{S,1,2},
    a::AbstractMatrix{<:AbstractTensorMap{S,2,1}},
    m::AbstractMatrix{<:AbstractTensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{2},
) where {S}
    flu = similar(fl)
    @tensoropt begin
        flu[13; 11 12] =
            fl[3; 1 2] *
            (a[x, y])[1 4; 6] *
            (m[x, y])[2 5; 7 4] *
            (a[x, y+1]')[8; 3 5] *
            (a[x+1, y])[6 9; 11] *
            (m[x+1, y])[7 10; 12 9] *
            (a[x+1, y+1]')[13; 8 10]
    end
    return flu
end

function flsolve(
    fl::AbstractTensorMap{S,1,2},
    a::AbstractMatrix{<:AbstractTensorMap{S,2,1}},
    m::AbstractMatrix{<:AbstractTensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{3},
) where {S}
    flu = similar(fl)
    @tensoropt begin
        flu[18; 16 17] =
            fl[3; 1 2] *
            (a[x, y])[1 4; 6] *
            (m[x, y])[2 5; 7 4] *
            (a[x, y+1]')[8; 3 5] *
            (a[x+1, y])[6 9; 11] *
            (m[x+1, y])[7 10; 12 9] *
            (a[x+1, y+1]')[13; 8 10] *
            (a[x+2, y])[11 14; 16] *
            (m[x+2, y])[12 15; 17 14] *
            (a[x+2, y+1]')[18; 13 15]
    end
    return flu
end

# Right-environment
function frsolve(
    fr::AbstractTensorMap{S,2,1},
    a::AbstractMatrix{<:AbstractTensorMap{S,2,1}},
    m::AbstractMatrix{<:AbstractTensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{1},
) where {S}
    fru = similar(fr)
    @tensoropt begin
        fru[1 2; 3] =
            (a[x, y])[1 4; 6] * (m[x, y])[2 5; 7 4] * (a[x, y+1]')[8; 3 5] * fr[6 7; 8]
    end
    return fru
end
function frsolve(
    fr::AbstractTensorMap{S,2,1},
    a::AbstractMatrix{<:AbstractTensorMap{S,2,1}},
    m::AbstractMatrix{<:AbstractTensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{2},
) where {S}
    fru = similar(fr)
    @tensoropt begin
        fru[1 2; 3] =
            (a[x-1, y])[1 4; 6] *
            (m[x-1, y])[2 5; 7 4] *
            (a[x-1, y+1]')[8; 3 5] *
            (a[x, y])[6 9; 11] *
            (m[x, y])[7 10; 12 9] *
            (a[x, y+1]')[13; 8 10] *
            fr[11 12; 13]
    end
    return fru
end
function frsolve(
    fr::AbstractTensorMap{S,2,1},
    a::AbstractMatrix{<:AbstractTensorMap{S,2,1}},
    m::AbstractMatrix{<:AbstractTensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{3},
) where {S}
    fru = similar(fr)
    @tensoropt begin
        fru[1 2; 3] =
            (a[x-2, y])[1 4; 6] *
            (m[x-2, y])[2 5; 7 4] *
            (a[x-2, y+1]')[8; 3 5] *
            (a[x-1, y])[6 9; 11] *
            (m[x-1, y])[7 10; 12 9] *
            (a[x-1, y+1]')[13; 8 10] *
            (a[x, y])[11 14; 16] *
            (m[x, y])[12 15; 17 14] *
            (a[x, y+1]')[18; 13 15] *
            fr[16 17; 18]
    end
    return fru
end

function fptest(FL, FR, A, M)
    nx, ny = size(M)
    AL, C, AR, _ = unpack(A)
    for x in 1:nx, y in 1:ny
        println(
            "Left: ",
            isapprox(
                normalize(FL[x, y]), normalize(fpsolve(FL[x, y], AL, M, x, y)); atol=1e-3
            ),
        )
        println(
            "Right: ",
            isapprox(normalize(FR[x, y]), normalize(fpsolve(FR[x, y], AR, M, x, y))),
        )
    end
end
