mutable struct Callback{T}
    func::Function
    output::T
end

(cb::Callback)(prob) = cb.func(prob)
