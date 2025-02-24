using Test
using TestExtras
using InfiniteTensorContractions
using CircularArrays
using TensorKit
using LinearAlgebra

@testset "All tests" verbose = true begin
    if isempty(ARGS)
        include("unitcell.jl")
        include("misc.jl")
        include("vumps.jl")
        include("ctmrg.jl")
        include("classicalmodels.jl")
    else
        for file in ARGS
            include(file)
        end
    end
end
