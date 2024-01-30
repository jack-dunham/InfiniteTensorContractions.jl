abstract type AbstractBoundaryAlgorithm <: AbstractAlgorithm end
abstract type AbstractBoundaryRuntime <: AbstractRuntime end

function boundaryerror!(S_old::AbstractMatrix, C_new::AbstractMatrix)
    S_new = boundaryerror.(C_new)
    err = @. norm(S_old - S_new)
    S_old .= S_new
    return err
end

function boundaryerror(c_new::AbstractTensorMap)
    _, s_new, _ = tsvd(c_new, (1,), (2,))
    normalize!(s_new)
    return s_new
end
