function contract(tensors, network)
    inds = collect(CartesianIndices(network))

    RT = promote_type(scalartype(tensors), scalartype(network))

    rv = similar(network, RT)

    rv .= contract.(Ref(tensors), Ref(network), inds)
    return rv
end

contract(tensors, network, i1::Int, i2::Int) = contract(tensors, network, i1:i1, i2:i2)
contract(tensors, network, inds) = contract(tensors, network, inds[1], inds[2])

function get_bond_symbol(i, j, dir)
    if dir == :h
        str = "$(i)$(j)_$(i + 1)$(j)"
    elseif dir == :v
        str = "$(i)$(j)_$(i)$(j + 1)"
    end
    return Symbol(str)
end

@generated function _contractall(
    C1,
    C2,
    C3,
    C4,
    T1::NTuple{Nx,E},
    T2::NTuple{Ny,E},
    T3::NTuple{Nx,E},
    T4::NTuple{Ny,E},
    MS::AbstractArray{<:AbsTen{0}},
) where {Nx,Ny,E}
    gh = (i, j) -> get_bond_symbol(i, j, :h)
    gv = (i, j) -> get_bond_symbol(i, j, :v)

    e_T1s = [
        Expr(:ref, Expr(:ref, :T1, i), gv(i + 1, 1), (gh(i + 1, 1), gh(i, 1))...) for
        i in 1:Nx
    ]

    e_T2s = [
        Expr(
            :ref,
            Expr(:ref, :T2, j),
            gh(Nx + 1, j + 1),
            (gv(Nx + 2, j), gv(Nx + 2, j + 1))...,
        ) for j in 1:Ny
    ]
    e_T3s = [
        Expr(
            :ref,
            Expr(:ref, :T3, i),
            gv(i + 1, Ny + 1),
            (gh(i, Ny + 2), gh(i + 1, Ny + 2))...,
        ) for i in 1:Nx
    ]
    e_T4s = [
        Expr(:ref, Expr(:ref, :T4, j), gh(1, j + 1), (gv(1, j + 1), gv(1, j))...) for
        j in 1:Ny
    ]

    e_MS = [
        Expr(
            :ref,
            Expr(:ref, :MS, i, j),
            gh(i + 1, j + 1),
            gv(i + 1, j + 1),
            gh(i, j + 1),
            gv(i + 1, j),
        ) for i in 1:Nx, j in 1:Ny
    ]

    e_C1 = Expr(:ref, :C1, gh(1, 1), gv(1, 1))
    e_C2 = Expr(:ref, :C2, gh(Nx + 1, 1), gv(Nx + 2, 1))
    e_C3 = Expr(:ref, :C3, gh(Nx + 1, Ny + 2), gv(Nx + 2, Ny + 1))
    e_C4 = Expr(:ref, :C4, gh(1, Ny + 2), gv(1, Ny + 1))

    e_einsum = Expr(
        :call, :*, e_C1, e_C2, e_C3, e_C4, e_T1s..., e_T2s..., e_T3s..., e_T4s..., e_MS...
    )

    quote
        @tensoropt rv = $e_einsum
    end
end

@generated function _contractall(
    FL,
    FR,
    ACU::T1,
    ARU::NTuple{N,T1},
    ACD::T2,
    ARD::NTuple{N,T2},
    MS::AbstractArray{<:AbsTen{0}},
) where {N,T1,T2}
    symb = (s, i) -> Symbol("$(s)_$i")

    e_ARU = [
        Expr(:ref, Expr(:ref, :ARU, i), symb(:n, i + 1), symb(:u, i + 1), symb(:u, i)) for
        i in 1:N
    ]
    e_ARD = [
        Expr(:ref, Expr(:ref, :ARD, i), symb(:d, i + 1), symb(:d, i), symb(:s, i + 1)) for
        i in 1:N
    ]

    e_MS = [
        Expr(
            :ref, Expr(:ref, :MS, i), symb(:h, i), symb(:s, i), symb(:h, i - 1), symb(:n, i)
        ) for i in 1:(N + 1)
    ]

    e_ACU = Expr(:ref, :ACU, symb(:n, 1), symb(:u, 1), symb(:u, 0))
    e_ACD = Expr(:ref, :ACD, symb(:d, 1), symb(:d, 0), symb(:s, 1))
    e_FL = Expr(:ref, :FL, symb(:h, 0), symb(:u, 0), symb(:d, 0))
    e_FR = Expr(:ref, :FR, symb(:h, N + 1), symb(:u, N + 1), symb(:d, N + 1))

    e_einsum = Expr(:call, :*, e_FL, e_FR, e_ACU, e_ACD, e_MS..., e_ARU..., e_ARD...)

    quote
        # @tensoropt rv = scalar($e_einsum)
        @tensoropt rv = $e_einsum
    end
end
