abstract type AbstractContractionAlgorithm end
abstract type AbstractContractionState{<:AbstractContractionAlgorithm} end

# Return AlgorithmState 
function initialize(f, bulk, alg::AbstractContractionAlgorithm; kwargs...) end
const initialise = initialize
