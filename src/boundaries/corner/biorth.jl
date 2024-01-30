function leftbiorth!(UL, args...; kwargs...)
    return biorthogonalise!(UL, args...; x0=firstindex(UL), incr=1, kwargs...)
end
function rightbiorth!(UL, args...; kwargs...)
    return biorthogonalise!(UL, args...; x0=lastindex(UL), incr=-1, kwargs...)
end

biorthogonalise(args...; kwargs...) = biorthogonalise!(deepcopy.(args)...; kwargs...)
function biorthogonalise!(
    UL::CircularVector,
    VL::CircularVector,
    CU::CircularVector,
    CD::CircularVector,
    AU::CircularVector,
    AD::CircularVector;
    x0=firstindex(UL),
    incr=1,
    maxiter=10,
)
    χ = domain(CU[1], 1)
    iter = eachindex(AD)

    C0 = broadcast(CU, CD) do CU, CD
        @tensoropt C0[a; b] := CU[a c] * CD[b c]
    end

    U, Σ, V = biorth_get_gauge_transform(C0, AU, AD; x0=x0, incr=incr, χ=χ)

    s = sqrt.(Σ)
    sinv = pinv.(s)

    biorthness = broadcast(_ -> 0.0, UL)

    for x in iter
        c = x0 + (x - 1) * incr
        __gauge_upper_edge!(UL[c], s[c], U[c], AU[c], (U[c + incr])', sinv[c + incr])
        __gauge_lower_edge!(VL[c], sinv[c + incr], (V[c + incr])', AD[c], V[c], s[c])

        permute!(CU[c], U[c] * s[c], ((), (1, 2)))
        permute!(CD[c], s[c] * V[c], ((), (1, 2)))

        id_maybe = _transpose(UL[c]) * VL[c]
        biorthness[c] = norm(id_maybe / tr(id_maybe) - one(id_maybe) / tr(one(id_maybe)))
    end

    numiter = 0

    C0 = transpose.(C0)

    while maximum(biorthness) > sqrt(eps()) && numiter < maxiter
        ULp = map(i -> permute(i, ((2,), (3, 1))), UL)
        VLp = map(i -> permute(i, ((2,), (1, 3))), VL)

        U, Σ, V = biorth_get_gauge_transform(C0, ULp, VLp; x0=x0, incr=incr, χ=χ)

        s = sqrt.(Σ)
        sinv = pinv.(s)

        for x in iter
            c = x0 + (x - 1) * incr
            __gauge_upper_edge!(UL[c], s[c], U[c], ULp[c], (U[c + incr])', sinv[c + incr])
            __gauge_lower_edge!(VL[c], sinv[c + incr], (V[c + incr])', VLp[c], V[c], s[c])

            permute!(CU[c], permute(CU[c], ((1,), (2,))) * U[c] * s[c], ((), (1, 2)))
            permute!(CD[c], s[c] * V[c] * permute(CD[c], ((1,), (2,))), ((), (1, 2)))

            id_maybe = _transpose(UL[c]) * VL[c]

            biorthness[c] = norm(
                id_maybe / tr(id_maybe) - one(id_maybe) / tr(one(id_maybe))
            )
        end
        numiter += 1
    end

    @debug "Biorthgonalisation:" tol = biorthness iterations = numiter

    ϵu, ϵd = biorth_verify_translational_invariance(
        CU, CD, AU, AD, UL, VL; x0=x0, incr=incr
    )

    @debug ϵu ϵd

    return UL, VL
end

function biorth_verify_translational_invariance(CU, CD, AU, AD, UL, VL; x0, incr, kwargs...)
    @debug "Verifying translational invariance of biorthogonalisation..."
    r = 1:length(CU)

    ϵu = broadcast(_ -> 0.0, CU)
    ϵd = broadcast(_ -> 0.0, CU)

    for x in eachindex(CU)
        c = x0 + (x - 1) * incr

        @tensoropt t1[o; k j] := CU[c][i j] * AU[c][o; k i]
        @tensoropt t2[o; k j] := UL[c][j o; 3] * CU[c + incr][k 3]

        @tensoropt s1[o; i k] := CD[c][i j] * AD[c][o; j k]
        @tensoropt s2[o; i k] := VL[c][i o; 3] * CD[c + incr][3 k]

        ϵu[c] = norm(normalize(t1) - normalize(t2))
        ϵd[c] = norm(normalize(s1) - normalize(s2))

        ϵu[c] < sqrt(eps()) ||
            @warn "CU($c) * TU($c) ≈ PU($c) * CU($(mod(c + incr,r))):" ϵu[c]
        ϵd[c] < sqrt(eps()) ||
            @warn "CD($c) * TD($c) ≈ PD($c) * CD($(mod(c + incr,r))):" ϵd[c]
    end

    @debug "...done"

    return ϵu, ϵd
end

function biorth_get_gauge_transform(C0, AU, AD; x0, incr, χ)
    # Solve the fixed point equation at x0
    biorth_fixed_point!(C0, AU, AD; x0=x0, incr=incr)

    U = similar(C0)
    S = similar(C0)
    V = similar(C0)

    for x in eachindex(C0)
        c = x0 + (x - 1) * incr

        # Already solved for C at x0, The remaining matrices can be obtained by application
        # of the fixed point map.
        if c != x0
            __biorth_fixed_point_map!(C0[c], C0[c - incr], AU[c - incr], AD[c - incr])
        end

        # We use `trunc=truncspace` here as is a convenient way to enforce χ on bond rather 
        # than χ'. No actual truncation takes place.
        U[c], S[c], V[c] = tsvd(C0[c]; trunc=truncspace(χ))
    end

    return U, S, V
end

# UV = s * U * A * U(+1) * s^{-1}(+1)
function __gauge_upper_edge!(UV, s, U, A, V, t)
    @tensoropt UV[i o; n] = s[j; i] * U[k; j] * A[o; l k] * V[m; l] * t[n; m]
    return UV
end
function __gauge_lower_edge!(UV, s, U, A, V, t)
    @tensoropt UV[n o; i] = s[j; i] * U[k; j] * A[o; l k] * V[m; l] * t[n; m]
    return UV
end

function biorth_fixed_point!(C0, AU, AD; x0=1, incr=1)
    val, vec, info = eigsolve(C0[x0], 1, :LM; eager=true, maxiter=1) do vec0
        c = x0

        # Need to use a temp as can't mutate in eigsolve currently
        temp = similar(vec0)

        # Recurrence relation is defined:
        # C(x + 1) = T(x)⋅C(x)
        # So fixed point equation to solve becomes (N = length(C0)):
        # C(x) = T(x-1)⋯T(x+1)⋅T(x)⋅C(x) mod(x, N)
        @inbounds for _ in eachindex(C0)
            __biorth_fixed_point_map!(temp, vec0, AU[c], AD[c])

            vec0 = temp
            c += incr
        end
        return vec0
    end
    copy!(C0[x0], vec[1])
    return C0
end

function __biorth_fixed_point_map!(C1, C, AU, AD)
    @tensoropt C1[ur; dr] = C[ul; dl] * AU[p; ur ul] * AD[p; dl dr]
    return C1
end
