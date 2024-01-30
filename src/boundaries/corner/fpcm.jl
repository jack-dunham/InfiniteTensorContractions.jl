
"""
    FPCM{SVD<:OrthogonalFactorizationAlgorithm}

# Fields
- `bonddim::Int`: the bond dimension of the boundary
- `maxiter::Int = 100`: maximum number of iterations
- `tol::Float64 = 1e-12`: convergence tolerance
- `verbose::Bool = true`: when true, will print algorithm convergence progress
- `randinit::Bool` = false: if true, use random tensors as a starting point.
- `docorners::Symbol = :never`: Determines if and how a fixed point equation for the corners 
should by calculated. The default value `:never` results in no fixed point equation being
used (corners determined from biorthogonalisation only), `:fast` uses equation, and `slow`
uses equation .
"""
@kwdef struct FPCM <: AbstractCornerMethod
    bonddim::Int
    maxiter::Int = 100
    tol::Float64 = 1e-12
    verbose::Bool = true
    randinit::Bool = false
    docorners::Symbol = :never # fast, slow 
end

function fpcmprojectors!(tensors::CornerMethodTensors)
    fpcmprojectors!(tensors.projectors, tensors.corners, tensors.edges)
    return tensors
end

function fpcmprojectors!(projectors::Projectors, corners::Corners, edges::Edges)
    C1, C2, C3, C4 = corners
    T1, _, T3, _ = edges
    UL, VL, UR, VR = projectors
    for y in axes(C1, 2)
        leftbiorth!(UL[:, y], VL[:, y], C1[:, y], C4[:, y + 1], T1[:, y], T3[:, y + 1])
        # Now do biorth from right, but upside down to ensure indices match up correctly.
        rightbiorth!(VR[:, y], UR[:, y], C3[:, y + 1], C2[:, y], T3[:, y + 1], T1[:, y])
    end
    return projectors
end

function step!(runtime::CornerMethodRuntime, algorithm::FPCM)
    fpcm_horizontal_move!(runtime)

    fpcm_vertical_move!(runtime)

    # if algorithm.docorners != :never
    cornermove!(runtime; docorners=algorithm.docorners)
    # end

    normalize!(runtime.primary.corners)
    normalize!(runtime.primary.edges)

    error = ctmerror!(runtime)

    return error
end

## EDGE MOVE

function fpcm_horizontal_move!(runtime)
    _fpcm_horizontal_move!(runtime.primary, runtime.permuted)
    return runtime
end
function fpcm_vertical_move!(runtime)
    _fpcm_horizontal_move!(runtime.permuted, runtime.primary)
    return runtime
end
# This function mutates both arguments
function _fpcm_horizontal_move!(tensors, tensors_permuted)
    fpcmprojectors!(tensors)

    edgemove!(tensors)

    updatecorners!(tensors_permuted, tensors)

    return nothing
end

function edgemove!(tensors::CornerMethodTensors)
    _, T2, _, T4 = tensors.edges
    UL, VL, UR, VR = tensors.projectors
    network = tensors.network

    forward_edge_move!(T4, network, UL, VL)
    backward_edge_move!(T2, network, UR, VR)

    return tensors
end

forward_edge_move!(args...) = coordinate_edge_move!(args...; x0=1, incr=1)
backward_edge_move!(args...) = coordinate_edge_move!(args...; x0=0, incr=-1)

function coordinate_edge_move!(edges, network, upper_projector, lower_projector; kwargs...)
    rv = map(axes(network, 2)) do y
        coordinate_edge_move!(edges, network, upper_projector, lower_projector, y; kwargs...)
    end
    return rv
end

function coordinate_edge_move!(
    edges, network, upper_projector, lower_projector, y; x0=1, incr=1
)
    E = edges[:, y + 1]
    N = network[:, y + 1]
    U = upper_projector[:, y + 0]
    L = lower_projector[:, y + 1]

    # If moving in the backwards direction, everything is upside down.
    if incr < 0
        L, U = U, L
        N = invertaxes.(N)
    end

    return _coordinate_edge_move!(E, N, U, L; x0=x0, incr=incr)
end

function _coordinate_edge_move!(T4, bulk, UL, VL; x0=firstindex(T4), incr=1)
    val, vec, info = eigsolve(T4[x0], 1, :LM; ishermitian=false, eager=true) do out
        c = x0
        for _ in eachindex(T4)
            out = projectedge(out, bulk[c + incr], UL[c], VL[c])
            c += incr
        end
        return out
    end

    copy!(T4[x0], vec[1])

    c = x0
    for _ in 1:(length(T4) - 1)
        projectedge!(T4[c + incr], T4[c], bulk[c + incr], UL[c], VL[c])

        c += incr
    end

    return T4
end

## 

## CORNERS
# Everything here almost certainly doesn't work
# TODO: Refactor all of this and check it actually works.

function cornermove!(runtime; kwargs...)
    return cornermove!(runtime.primary, runtime.permuted.projectors; kwargs...)
end
function cornermove!(primary_tensors, permuted_projectors; docorners=:slow)
    C1, C2, C3, C4 = primary_tensors.corners
    T1, T2, T3, T4 = primary_tensors.edges
    UL, VL, UR, VR = primary_tensors.projectors

    # Need to un-transpose the lattice...
    UU, VU, UD, VD = map(permutedims, permuted_projectors)

    Ms = primary_tensors.network

    corner_fixed_point_reverse!(C1, T1, T4, VU, VL, Ms; corner=:C1)
    corner_fixed_point_reverse!(C2, T1, T2, UU, VR, Ms; corner=:C2)
    corner_fixed_point_reverse!(C3, T3, T2, UD, UR, Ms; corner=:C3)
    corner_fixed_point_reverse!(C4, T3, T4, VD, UL, Ms; corner=:C4)

    return primary_tensors
end

function corner_fixed_point_reverse!(C, TH, TV, PH, PV, M; corner::Symbol)
    flipx = x -> permutedom(x, (3, 2, 1, 4))
    flipy = x -> permutedom(x, (1, 4, 3, 2))

    if corner === :C1
        increment = (1, 1)
    elseif corner === :C2
        increment = (-1, 1)
        TH = swapvirtual.(TH)
        TV = swapvirtual.(TV)
        PH = circshift(PH, (1, 0))
        M = flipx.(M)
    elseif corner === :C3
        increment = (-1, -1)
        PH = circshift(PH, (1, 0))
        PV = circshift(PV, (0, 1))
        M = invertaxes(M)
    elseif corner === :C4
        increment = (1, -1)
        TH = swapvirtual.(TH)
        TV = swapvirtual.(TV)
        PV = circshift(PV, (0, 1))
        M = flipy.(M)
    end

    return corner_fixed_point_alt_2d!(
        C, TH, TV, PH, PV, M; increment=CartesianIndex(increment)
    )
end
function corner_fixed_point_alt!(corners, edges, projectors, dir)
    edges = circshift(edges, -dir)

    x0 = corners[1]

    val, vec, info = eigsolve(x0, 1, :LM; ishermitian=false, eager=true) do out
        for x in 1:length(corners)
            out = projectcorner(out, edges[x], projectors[x])
        end
        return out
    end

    copy!(corners[1], vec[1])# / val[1])

    for x in 1:(length(corners) - 1)
        old = copy(corners[x + 1])
        projectcorner!(corners[x + 1], corners[x], edges[x], projectors[x])
        normalize!(corners[x + 1])
        # println(norm(corners[x + 1] - old))
    end

    # @debug "Corner fixed point:" val #vec info

    # copy!.(corners, vec[1] * 1 / val[1])

    return corners
end
function corner_fixed_point_alt_2d!(
    C1s, T1s, T4s, UHs, UVs, Ms; increment::CartesianIndex{2}
)
    xi = increment[1]
    yi = increment[2]

    nx, ny = size(C1s)

    # number of repeats
    cycle_length = lcm(nx, ny)

    num_cycles = Integer(length(C1s)//cycle_length)

    T1s = circshift(T1s, (-xi, 0))
    T4s = circshift(T4s, (0, -yi))

    UHs = circshift(UHs, (-xi, 0))
    UVs = circshift(UVs, (0, -yi))

    Ms = circshift(Ms, (-xi, -yi))

    for y in 1:num_cycles
        coord0 = CartesianIndex(1, y)

        x0 = C1s[1, y]
        val, vec, info = eigsolve(x0, 1, :LM; ishermitian=false, eager=true) do out
            for _ in 1:cycle_length
                @inbounds out = projectcorner2d(
                    out, T1s[coord0], T4s[coord0], Ms[coord0], UHs[coord0], UVs[coord0]
                ) # x + 1, y + 1
            end
            coord0 += increment
            return out
        end

        coord0 = CartesianIndex(1, y)

        copy!(C1s[coord0], vec[1])# / val[1])

        for _ in 1:(cycle_length - 1)
            @inbounds projectcorner2d!(
                C1s[coord0 + increment],
                C1s[coord0],
                T1s[coord0],
                T4s[coord0],
                Ms[coord0],
                UHs[coord0],
                UVs[coord0],
            )
            coord0 += increment
        end
    end

    # @debug "Corner fixed point:" val #vec info

    # copy!.(corners, vec[1] * 1 / val[1])

    return C1s
end
##
