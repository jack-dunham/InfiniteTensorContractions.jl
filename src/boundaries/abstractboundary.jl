abstract type AbstractBoundaryAlgorithm <: AbstractAlgorithm end
abstract type AbstractBoundaryRuntime <: AbstractRuntime end

function boundaryerror!(S_old::AbstractMatrix, C_new::AbstractMatrix)
    S_new = boundaryerror.(C_new)
    if all(space.(S_new) == space.(S_old))
        err = @. norm(S_old - S_new)
    else
        err = broadcast(_ -> Inf, S_new)
    end
    S_old .= S_new
    return err
end

function boundaryerror(c_new::AbstractTensorMap)
    _, s_new, _ = tsvd(c_new, (1,), (2,))
    normalize!(s_new)
    return s_new
end
