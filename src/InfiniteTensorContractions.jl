module InfiniteTensorContractions
const ITC = InfiniteTensorContractions

using CircularArrays
using LinearAlgebra
using TensorKit
using KrylovKit

export ITC

export AbstractUnitCellGeometry
export AbstractUnitCell

export AbstractProblemState
export AbstractBoundaryAlgorithm 

export Square
export UnitCell
export TensorPair, bondspace, swapaxes, invertaxes

export VUMPS, CTMRG, TRG
export VUMPSRuntime, MPS, FixedPoints
export CornerMethodTensors, Corners, Edges

export newproblem, initialize, run!, run, contract

# No deps
include("convergenceinfo.jl")
include("utils.jl")
include("callback.jl")

include("abstractunitcell.jl")
include("abstractproblem.jl")
include("networks.jl")

include("contract.jl")

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
include("boundaries/corner/tensormacros.jl")
include("boundaries/corner/cornermethod.jl")
include("boundaries/corner/ctmrg.jl")
include("boundaries/corner/init.jl")

# FPCM
include("boundaries/corner/fpcm.jl")
include("boundaries/corner/biorth.jl")

include("boundaries/corner/hybrid.jl")

# Graining
include("graining/abstractgraining.jl")
include("graining/trg.jl")

function __init__()
    ENV["ITC_ALWAYS_FORCE"] = false
    return nothing
end

end # module
