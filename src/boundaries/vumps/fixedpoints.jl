# DONE (NEEDS RESTRUCTURING)
abstract type AbstractFixedPoints end

# Fixed points of a transfer matrix
struct FixedPoints{A<:AbUnCe{<:TenAbs{2}}} <: AbstractFixedPoints
    left::A
    right::A
    function FixedPoints(left::A, right::A) where {A}
        check_size_allequal(left, right)
        return new{A}(left, right)
    end
end

FixedPoints(f, mps::MPS, network::AbstractNetwork) = initfixedpoints(f, mps, network)

Base.similar(fps::FixedPoints) = FixedPoints(similar(fps.left), similar(fps.right))

function initfixedpoints(f, mps::MPS, network::AbstractNetwork)
    mps_tensor = getcentral(mps)
    # network_tensor = convert.(TensorMap, network)
    left = _initfixedpoints.(f, mps_tensor, network, :left)
    right = _initfixedpoints.(f, mps_tensor, network, :right)
    return FixedPoints(left, right)
end

function _initfixedpoints(
    f, mps::AbstractTensorMap{S}, network, leftright::Symbol
) where {S}
    T = promote_type(eltype(mps), numbertype(network))

    cod = fixedpoint_codomain(network, leftright)
    dom = fixedpoint_domain(mps, leftright)

    return TensorMap(f, T, cod, dom)
end

function fixedpoint_codomain(network, leftright::Symbol)
    if leftright === :left
        return _fixedpoint_codomain(network, 3)
    elseif leftright === :right
        return _fixedpoint_codomain(network, 1)
    else
        throw(ArgumentError(""))
    end
end
_fixedpoint_codomain(network_tensor::AbsTen{0,4}, lr::Int) = domain(network_tensor)[lr]
function _fixedpoint_codomain(network_tensor::TensorPair, lr::Int)
    return domain(network_tensor.top)[lr] * domain(network_tensor.bot)[lr]'
end

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

function fixedpoints(mps::MPS, network)
    initial_fpoints = FixedPoints(rand, mps, network)
    return fixedpoints!(initial_fpoints, mps, network)
end

#this is now the correct env func

function fixedpoints!(fpoints::FixedPoints, mps::MPS, network)
    AL, C, AR, _ = unpack(mps)

    TransferMatrix(AL[1, 1], network[1, 1], AL[1, 1])

    tm_left = TransferMatrix.(AL, network, circshift(AL, (0, -1)))
    tm_right = TransferMatrix.(AR, network, circshift(AR, (0, -1)))

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
        left, Ls, linfo = eigsolve(
            z -> leftsolve(z, tm_left[:, y]), FL[1, y], 1, :LM; ishermitian=false, eager=true, maxiter=1
        )
        right, Rs, rinfo = eigsolve(
            z -> rightsolve(z, tm_right[:, y]), FR[Nx, y], 1, :LM; ishermitian=false, eager=true, maxiter=1
        )

        FL[1, y] = Ls[1]
        FR[end, y] = Rs[1]

        @debug "Fixed point leading eigenvalues:" left = left[1] right = right[1]
        @debug "Fixed point convergence info:" left = linfo right = rinfo

        for x in 2:Nx
            multransfer!(FL[x, y], FL[x - 1, y], tm_left[x - 1, y])
        end

        NN = renorm(C[Nx, y + 1], C[Nx, y], Ls[1], Rs[1]) # Should be positive?

        @debug "" norm = NN
        
        if isa(NN, AbstractFloat) && NN < 0.0
            NN = sqrt(sqrt(NN^2))
            # the eigenvalue problem eqn gives us FL[1,y] and FR[end,y], so normalise them
            rmul!(FL[1, y], 1 / NN) #correct NN
            rmul!(FR[end, y], 1 / -NN) #correct NN
        else
            NN = sqrt(NN)
            # the eigenvalue problem eqn gives us FL[1,y] and FR[end,y], so normalise them
            rmul!(FL[1, y], 1 / NN) #correct NN
            rmul!(FR[end, y], 1 / NN) #correct NN
        end

        # # the eigenvalue problem eqn gives us FL[1,y] and FR[end,y], so normalise them
        # rmul!(FL[1, y], 1 / NN) #correct NN
        # rmul!(FR[end, y], 1 / NN) #correct NN

        for x in (Nx - 1):-1:1
            multransfer!(FR[x, y], tm_right[x + 1, y], FR[x + 1, y])

            NN = renorm(C[x, y + 1], C[x, y], FL[x + 1, y], FR[x, y])

            rmul!(FR[x, y], 1 / sqrt(NN))
            rmul!(FL[x + 1, y], 1 / sqrt(NN))
        end

        # First x seems to be normalized, but not second x
        for x in 1:Nx
            NN = renorm(C[x, y + 1], C[x, y], FL[x + 1, y], FR[x, y])
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

# This needs updated
# function fptest(FL, FR, A, M)
#     nx, ny = size(M)
#     AL, C, AR, _ = unpack(A)
#     for x in 1:nx, y in 1:ny
#         println(
#             "Left: ",
#             isapprox(
#                 normalize(FL[x, y]), normalize(fpsolve(FL[x, y], AL, M, x, y)); atol=1e-3
#             ),
#         )
#         println(
#             "Right: ",
#             isapprox(normalize(FR[x, y]), normalize(fpsolve(FR[x, y], AR, M, x, y))),
#         )
#     end
# end
