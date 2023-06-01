struct TransferMatrix{A<:TenAbs{2},M<:Union{AbsTen{0,4},AbsTen{1,4},AbsTen{2,4}}}
    above::A
    middle::M
    below::A
end

function Base.eltype(t::TransferMatrix)
    return promote_type(eltype(t.above), eltype(t.middle), eltype(t.below))
end

const AbstractTransferMatrices{T} = AbstractUnitCell{T<:TransferMatrix}

# Regular is for getting correct space of resulting F, _ function is for getting 
# correct space compatible F
# TODO: make these functions clearer re function.
function rightspace(transfer::TransferMatrix{A,<:AbsTen{0,4}}) where {A}
    dom = domain(transfer.above)[1] * domain(transfer.below)[1]'
    cod = domain(transfer.middle)[1]'
    return cod, dom
end
function leftspace(transfer::TransferMatrix{A,<:AbsTen{0,4}}) where {A}
    dom = domain(transfer.above)[2] * domain(transfer.below)[2]'
    cod = domain(transfer.middle)[3]'
    return cod, dom
end

function rightspace(transfer::TransferMatrix{A,<:AbsTen{2,4}}) where {A}
    dom = domain(transfer.above)[1] * domain(transfer.below)[1]'
    cod = domain(transfer.middle)[1]' * domain(transfer.middle)[1]
    return cod, dom
end
function leftspace(transfer::TransferMatrix{A,<:AbsTen{2,4}}) where {A}
    dom = domain(transfer.above)[2] * domain(transfer.below)[2]'
    cod = domain(transfer.middle)[3]' * domain(transfer.middle)[3]
    return cod, dom
end

function Base.:*(f, transfer::TransferMatrix)
    cod, dom = rightspace(transfer)
    fdst = similar(f, cod, dom)
    return multransfer!(fdst, f, transfer)
end
function Base.:*(transfer::TransferMatrix, f)
    cod, dom = leftspace(transfer)
    fdst = similar(f, cod, dom)
    return multransfer!(fdst, transfer, f)
end

function multransfer!(fdst, t1, t2)
    left, right, at, ab = _multransfer_tensors(t1, t2)
    return multransfer!(fdst, left, right, at, ab)
end

function _multransfer_tensors(f::AbsTen, transfer::TransferMatrix)
    at = transfer.above
    bulk = transfer.middle
    ab = transfer.below

    return f, bulk, at, ab
end
function _multransfer_tensors(transfer::TransferMatrix, f::AbsTen)
    at = transfer.above
    bulk = transfer.middle
    ab = transfer.below

    return bulk, f, at, ab
end

function multransfer!(
    fdst::AbsTen{1,2},
    fsrc::AbsTen{1,2},
    bulk::AbsTen{0,4},
    at::AbsTen{1,2},
    ab::AbsTen{1,2},
)
    @tensoropt fdst[D1; x2 x4] =
        fsrc[D3; x1 x3] * at[D4; x2 x1] * bulk[D1 D2 D3 D4] * (ab')[x4 x3; D2]
    return fdst
end
function multransfer!(
    fdst::AbsTen{1,2},
    bulk::AbsTen{0,4},
    fsrc::AbsTen{1,2},
    at::AbsTen{1,2},
    ab::AbsTen{1,2},
)
    @tensoropt fdst[D3; x1 x3] =
        fsrc[D1; x2 x4] * at[D4; x2 x1] * bulk[D1 D2 D3 D4] * (ab')[x4 x3; D2]
    return fdst
end
function multransfer!(
    fdst::AbsTen{2,2},
    fsrc::AbsTen{2,2},
    bulk::AbsTen{2,4},
    at::AbsTen{2,2},
    ab::AbsTen{2,2},
)
    @tensoropt (
        k => 2,
        b => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        E1 => D,
        E2 => D,
        E3 => D,
        E4 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
    ) fdst[D1 E1; x2 x4] =
        fsrc[D3 E3; x1 x3] *
        at[D4 E4; x2 x1] *
        bulk[k b; D1 D2 D3 D4] *
        (bulk')[E1 E2 E3 E4; k b] *
        (ab')[x4 x3; D2 E2]
    return fdst
end
function multransfer!(
    fdst::AbsTen{2,2},
    bulk::AbsTen{2,4},
    fsrc::AbsTen{2,2},
    at::AbsTen{2,2},
    ab::AbsTen{2,2},
)
    @tensoropt (
        k => 2,
        b => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        E1 => D,
        E2 => D,
        E3 => D,
        E4 => D,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
    ) fdst[D3 E3; x1 x3] =
        fsrc[D1 E1; x2 x4] *
        at[D4 E4; x2 x1] *
        bulk[k b; D1 D2 D3 D4] *
        (bulk')[E1 E2 E3 E4; k b] *
        (ab')[x4 x3; D2 E2]
    return fdst
end
