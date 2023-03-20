abstract type AbstractFixedPoints end

struct FixedPoints{Nx,Ny,L<:AbsTen{1,2},R<:AbsTen{2,1}} <: AbstractFixedPoints
    left::OnUnitCell{Nx,Ny,L}
    right::OnUnitCell{Nx,Ny,R}
end

function initleftfp(f, A::MultilineInfiniteMPS, M::InfiniteMPO)
    nx, ny = size(M)
    Ds = westbond.(M)
    χs = hcat(westbond.(A)...)
    data = [
        TensorMap(
            f, promote_type(numbertype(A), numbertype(M)), χs[x, y + 1], χs[x, y] * Ds[x, y]
        ) for x in 1:nx, y in 1:ny
    ]
    return OnLattice(latticestyle(M), data)
end
function initrightfp(f, A::MultilineInfiniteMPS, M::InfiniteMPO)
    nx, ny = size(M)
    Ds = eastbond.(M)
    χs = hcat(eastbond.(A)...)
    data = [
        TensorMap(
            f, promote_type(numbertype(A), numbertype(M)), χs[x, y] * Ds[x, y], χs[x, y + 1]
        ) for x in 1:nx, y in 1:ny
    ]
    return OnLattice(latticestyle(M), data)
end

function FixedPoints(f, A::MultilineInfiniteMPS, M::InfiniteMPO)
    return FixedPoints(initleftfp(f, A, M), initrightfp(f, A, M))
end

function renorm(cb, ca, fl, fr)
    return dot(cb, hcapply(ca, fl, fr))
end

function hcapply(
    c::AbstractTensorMap{S,1,1}, fl::AbstractTensorMap{S,1,2}, fr::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
    @tensoropt hc[a; b] := c[x; y] * fl[a; x m] * fr[y m; b]
    return hc
end

function fixedpoints(A::MultilineInfiniteMPS, M::InfiniteMPO)
    FPS = FixedPoints(rand, A, M)
    return fixedpoints!(FPS, A, M)
end

#this is now the correct env func
function fixedpoints!(FP::FixedPoints, A::MultilineInfiniteMPS, M::InfiniteMPO)
    FL = FP.left
    FR = FP.right

    Nx, Ny = size(M)

    # λ = Matrix{numbertype(M)}(undef, Nx, Ny)

    AL, C, AR, _ = unpack(A)

    for y in 1:Ny
        _, Ls, _ = eigsolve(z -> fpsolve(z, AL, M, 1, y), FL[1, y], 1, :LM)
        _, Rs, _ = eigsolve(z -> fpsolve(z, AR, M, Nx, y), FR[Nx, y], 1, :LM)

        FL[1, y] = Ls[1]

        for x in 2:Nx
            FL[x, y] =
                FL[x - 1, y] * TransferMatrix(AL[x - 1, y], M[x - 1, y], AL[x - 1, y + 1])#works
        end

        NN = renorm(C[Nx, y + 1], C[Nx, y], Ls[1], Rs[1])

        # the eigenvalue problem eqn gives us FL[1,y] and FR[end,y], so normalise them
        FL[1, y] = rmul!(FL[1, y], 1 / sqrt(NN)) #correct NN 
        FR[Nx, y] = rmul!(Rs[1], 1 / sqrt(NN)) #correct NN

        for x in (Nx - 1):-1:1
            FR[x, y] =
                TransferMatrix(AR[x + 1, y], M[x + 1, y], AR[x + 1, y + 1]) * FR[x + 1, y]#works

            NN = renorm(C[x, y + 1], C[x, y], FL[x + 1, y], FR[x, y])

            rmul!(FR[x, y], 1 / sqrt(NN))
            rmul!(FL[x + 1, y], 1 / sqrt(NN))
        end
    end
    return FP # mutated
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

function fpsolve(
    fl::AbsTen{1,2}, A::AbstractOnUnitCell{MPS,Nx,Ny}, M::InfiniteMPO{Nx,Ny}, x::Int, y::Int
) where {Nx,Ny}
    fl_n = flsolve(fl, A, M, x, y, Val(Nx))
    return fl_n
end
function fpsolve(
    fr::AbsTen{2,1}, A::AbstractOnUnitCell{MPS,Nx,Ny}, M::InfiniteMPO{Nx,Ny}, x::Int, y::Int
) where {Nx,Ny}
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
            fl[3; 1 2] * (a[x, y])[1 4; 6] * (m[x, y])[2 5; 7 4] * (a[x, y + 1]')[8; 3 5]
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
            (a[x, y + 1]')[8; 3 5] *
            (a[x + 1, y])[6 9; 11] *
            (m[x + 1, y])[7 10; 12 9] *
            (a[x + 1, y + 1]')[13; 8 10]
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
            (a[x, y + 1]')[8; 3 5] *
            (a[x + 1, y])[6 9; 11] *
            (m[x + 1, y])[7 10; 12 9] *
            (a[x + 1, y + 1]')[13; 8 10] *
            (a[x + 2, y])[11 14; 16] *
            (m[x + 2, y])[12 15; 17 14] *
            (a[x + 2, y + 1]')[18; 13 15]
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
            (a[x, y])[1 4; 6] * (m[x, y])[2 5; 7 4] * (a[x, y + 1]')[8; 3 5] * fr[6 7; 8]
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
            (a[x - 1, y])[1 4; 6] *
            (m[x - 1, y])[2 5; 7 4] *
            (a[x - 1, y + 1]')[8; 3 5] *
            (a[x, y])[6 9; 11] *
            (m[x, y])[7 10; 12 9] *
            (a[x, y + 1]')[13; 8 10] *
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
            (a[x - 2, y])[1 4; 6] *
            (m[x - 2, y])[2 5; 7 4] *
            (a[x - 2, y + 1]')[8; 3 5] *
            (a[x - 1, y])[6 9; 11] *
            (m[x - 1, y])[7 10; 12 9] *
            (a[x - 1, y + 1]')[13; 8 10] *
            (a[x, y])[11 14; 16] *
            (m[x, y])[12 15; 17 14] *
            (a[x, y + 1]')[18; 13 15] *
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
