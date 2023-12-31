module InfiniteTensorContractions
const ITC = InfiniteTensorContractions

using CircularArrays
using LinearAlgebra
using TensorKit
using KrylovKit

import Base: @kwdef

export ITC

export AbstractUnitCellGeometry
export AbstractUnitCell

export AbstractContractionAlgorithm, AbstractContractionState, AbstractContractionTensors
export AbstractBoundaryAlgorithm, AbstractBoundaryState, AbstractBoundaryTensors

export Square
export UnitCell
export TensorPair, bondspace, swapaxes, invertaxes

export VUMPS, CTMRG
export BoundaryState
export VUMPSTensors, MPS, FixedPoints
export CTMRGTensors, Corners, Edges, corners, edges

export initialize, calculate!, calculate, contract

# No deps
include("convergenceinfo.jl")
include("utils.jl")

include("abstractunitcell.jl")
include("abstractcontraction.jl")
include("networks.jl")

include("boundaries/abstractboundary.jl")

# VUMPS
include("boundaries/vumps/abstractmps.jl")
include("boundaries/vumps/mpsgauge.jl")

include("boundaries/vumps/tensormacros.jl")
include("boundaries/vumps/transfermatrix.jl")
include("boundaries/vumps/fixedpoints.jl")
include("boundaries/vumps/vumps.jl")
#

# CTMRG
include("boundaries/ctmrg/ctmrg.jl")
include("boundaries/ctmrg/init.jl")
include("boundaries/ctmrg/tensormacros.jl")
#

function __init__()
    ENV["ITC_ALWAYS_FORCE"] = false
    return nothing
end

end # module
