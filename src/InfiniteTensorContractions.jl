module InfiniteTensorContractions

using CircularArrays
using LinearAlgebra
using TensorKit
using KrylovKit

import Base: @kwdef

export Square
export UnitCell, ContractableTensors

export VUMPS, CTMRG
export initialize, calculate!, calculate, contract

# No deps
include("convergenceinfo.jl")
include("utils.jl")

include("abstractunitcell.jl")
include("abstractcontraction.jl")
include("contractabletensors.jl")

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
end

end # module
