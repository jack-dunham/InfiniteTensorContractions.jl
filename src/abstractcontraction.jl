abstract type AbstractContractionAlgorithm end
abstract type AbstractContractionState{<:AbstractContractionAlgorithm} end
abstract type AbstractContractionTensors end

# Return AlgorithmState 
function initialize(f, bulk, alg::AbstractContractionAlgorithm; kwargs...) end
const initialise = initialize

contract(alg_state::AbstractContractionState; kwargs...) = contract!(deepcopy(alg_state); kwargs...)

function contract! end
