module InfiniteTensorContractions

using CircularArrays
using LinearAlgebra
using TensorKit
using KrylovKit

# No deps
include("convergenceinfo.jl")
include("utils.jl")

include("abstractunitcell.jl")
include("abstractcontraction.jl")

include("boundaries/abstractboundary.jl")
# VUMPS
## types
include("boundaries/vumps/abstractmps.jl")
# include("boundaries/vumps/abstractfixedpoints.jl")
# include("boundaries/vumps/abstracttransfermatrix.jl")
## methods
include("boundaries/vumps/fixedpoints.jl")
include("boundaries/vumps/mpsgauge.jl")


end # module
