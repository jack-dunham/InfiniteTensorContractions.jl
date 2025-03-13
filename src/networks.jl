const AbstractNetwork{G,ElType<:AbstractTensorMap,A} = AbstractUnitCell{G,ElType,A}
const AbstractSingleLayerNetwork{G,ElType<:TensorMap,A} = AbstractUnitCell{G,ElType,A}
const AbstractDoubleLayerNetwork{G,ElType<:CompositeTensor{2},A} = AbstractUnitCell{
    G,ElType,A
}

TensorKit.spacetype(uc::AbstractUnitCell) = spacetype(typeof(uc))
TensorKit.spacetype(::Type{<:AbstractNetwork{G,T}}) where {G,T} = spacetype(T)
## Implement functions for contractable tensors etc
## ContractableTensors need a notion of an east, south, west, north bonds

virtualspace(network::AbstractUnitCell, dir) = virtualspace.(network, dir)

swapaxes(network::AbstractNetwork) = swapaxes.(network)
invertaxes(network::AbstractNetwork) = invertaxes.(network)

## INTERFACE

"""
    virtualspace(t, [dir::Integer])

Return an `NTuple{4,<:VectorSpace}` containing the east, south, west, and north vector spaces associated with the respective
bonds, in that order. Used to initialise appropriate algorithm tensors. For custom data types, this function must be 
specified for use in contraction algorithms.
"""
virtualspace(t::AbsTen{0,4}, dir) = domain(t, dir)
virtualspace(t) = map(i -> virtualspace(t, i), (1, 2, 3, 4))

"""
    swapaxes(t)

Swap the "x" and the "y" bonds of an object `t`. For custom data types, this function must be specified for use in 
contraction algorithms.
"""
function swapaxes end

"""
    invertaxes(t)

Invert the "x" and the "y" bonds of an object `t`, that is, south ↔ north and east ↔ west. For custom data types, 
this function must be specified for use in contraction algorithms.
"""
function invertaxes end

swapaxes(t::TenAbs{4}) = permutedom(t, (2, 1, 4, 3))
invertaxes(t::TenAbs{4}) = permutedom(t, (3, 4, 1, 2))

ensure_contractable(x) = x

"""
    adjoining_bondspace(network::AbstractUnitCell) -> E, S, W, N

Returns a tuple of the east, south, west, and north bond spaces as unit cells shifted one 
position in the opposite cardinal direction resepctively. For example, `N[x,y]` is the same 
as `north(network)[x,y + 1]`.
"""
function adjoining_bondspace(network)
    # east, south, west, north ← bondspace(network)
    return map(circshift, virtualspace(network), ((1, 0), (0, 1), (-1, 0), (0, -1)))
end
