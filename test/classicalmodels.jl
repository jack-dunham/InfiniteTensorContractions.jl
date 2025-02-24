function statmechmpo(β, h, D)
    δ1 = zeros(D, D, D, D)
    for i in 1:D
        δ1[i, i, i, i] = 1 # kronecker delta
    end

    δ2 = zeros(D, D, D, D, D)
    for i in 1:D
        δ2[i, i, i, i, i] = 1 # kronecker delta
    end

    X = zeros(D, D)
    for j in 1:D, i in 1:D
        X[i, j] = exp(-β * h(i, j))
    end
    Xsq = sqrt(X)

    Z = [1.0, -1.0]

    @tensor M1[a, b, c, d] :=
        δ1[a', b', c', d'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b']

    @tensor M2[a, b, c, d] :=
        δ2[a', b', c', d', e'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b'] * Z[e']

    return M1, M2
end

function classicalisingmpo(β; J=1.0, h=0.0)
    return statmechmpo(β, (s1, s2) -> -J * (-1)^(s1 != s2) - h / 2 * (s1 == 1 + s2 == 1), 2)
end

function exactZ(β)
    function quad_midpoint(f, a, b, N)
        h = (b - a) / N
        int = 0.0
        for k in 1:N
            xk_mid = (b - a) * (2k - 1) / (2N) + a
            int = int + h * f(xk_mid)
        end
        return int
    end

    function exact_integrand(x, β)
        return log(
            cosh(2 * β)^2 +
            sinh(2 * β) * sqrt(sinh(2 * β)^2 + sinh(2 * β)^(-2) - 2 * cos(2 * x)),
        )
    end
    f = x -> exact_integrand(x, β)
    quad_out = quad_midpoint(f, 0, pi, 100)
    return 1 / (2 * pi) * quad_out + log(2) / 2
end

@testset "Classical models" verbose = true begin
    @testset "Classical Ising model" verbose = true begin
        βc = log(1 + sqrt(2)) / 2

        x = 1.1

        M = abs((1 - sinh(2 * x * βc)^(-4)))^(1 / 8)

        D = ℂ^2

        zt, mt = map(
            data -> TensorMap(complex.(data), one(D), D * D * D' * D'),
            classicalisingmpo(x * βc),
        )
        zbulk = UnitCell([zt;;])
        mtbulk = UnitCell([mt;;])

        χs = (2, 5, 20)

        algs = (
            (VUMPS(; verbose=false, bonddim=i) for i in χs)...,
            (CTMRG(; verbose=false, svdalg=TensorKit.SVD(), bonddim=i) for i in χs)...,
        )

        @testset "Using $(typeof(alg)) with χ = $(alg.bonddim)" for (alg, tol) in zip(
            algs, (1e-3, 1e-6, 1e-9, 1e-3, 1e-6, 1e-9)
        )
            rt = @constinferred(initialize(zbulk, alg))
            st = @constinferred(newcontraction(zbulk, rt; alg=alg))

            runcontraction!(st)

            z_val = @constinferred(contract(st.runtime, zbulk))
            m_val = contract(st.runtime, mtbulk) ./ z_val

            @test abs(m_val[1, 1]) ≈ M atol = tol
        end

        χs = (2, 5, 10)
        trgalgs = (
            TRG(; verbose=false, maxiter=100, trunc=TensorKit.truncdim(i)) for i in χs
        )

        z_exact = exactZ(βc * x)

        @testset "Using $(typeof(alg)) with χ = $(alg.trunc.dim)" for (alg, tol) in zip(
            trgalgs, (1e-2, 1e-3, 1e-4)
        )
            rt = @constinferred(initialize(zbulk, alg))
            st = @constinferred(newcontraction(zbulk, rt; alg=alg))

            runcontraction!(st)

            z_val = st.runtime.cumsum

            @test abs(z_val[1, 1]) ≈ z_exact atol = tol
        end
    end
    #=
        @testset "Interacting dimers" verbose = true begin
            β = 1 / 0.85

            a = zeros(4, 4, 4, 4)
            b = zeros(4, 4, 4, 4)

            for I in CartesianIndices(a)
                if I[1] == mod(I[2] + 1, 1:4) == mod(I[3] + 2, 1:4) == mod(I[4] + 3, 1:4)
                    a[I] = 1
                end
                if I[1] == mod(I[2] - 1, 1:4) == mod(I[3] - 2, 1:4) == mod(I[4] - 3, 1:4)
                    b[I] = 1
                end
            end

            q = sqrt([1 0 0 0; 0 exp(β / 2) 1 1; 0 1 1 1; 0 1 1 exp(β / 2)])

            qh = diagm([1, -1, 1, -1]) * q
            qv = diagm([-1, 1, -1, 1]) * q

            @tensoropt aa[ii, jj, kk, ll] :=
                a[i, j, k, l] * q[i, ii] * q[j, jj] * q[k, kk] * q[l, ll]
            @tensoropt bb[ii, jj, kk, ll] :=
                b[i, j, k, l] * q[i, ii] * q[j, jj] * q[k, kk] * q[l, ll]

            δa = zeros(4, 4)
            δa[1, 1] = δa[3, 3] = 1
            δa[2, 2] = δa[4, 4] = -1

            @tensoropt aad[i, j, k, l] := a[ii, j, k, l] * δa[i, ii]
            @tensoropt bbd[i, j, k, l] := b[ii, j, k, l] * δa[i, ii]

            s = ℂ^4

            A = TensorMap(aa, one(s), s * s * s' * s')
            B = TensorMap(bb, one(s), s * s * s' * s')

            AD = TensorMap(aad, one(s), s * s * s' * s')
            BD = TensorMap(bbd, one(s), s * s * s' * s')

            alg = VUMPS(; bonddim=5, maxiter=1000)

            bulk = UnitCell(([A B; B A]))
            dbulk = UnitCell(([AD BD; BD AD]))

            st = initialize(bulk, alg)
            dst = initialize(dbulk, alg)

            sto = calculate(st)
            contract(sto.tensors, dbulk) ./ contract(sto.tensors, bulk)
        end
        =#
end

#=
@testset "Double layer" verbose = true begin
    @testset "Classical Ising model" verbose = true begin
        βc = log(1 + sqrt(2)) / 2

        x = 1.5

        M = abs((1 - sinh(2 * x * βc)^(-4)))^(1 / 8)

        D = ℂ^2

        zt, mt = map(
            data -> TensorMap(data, one(D), D * D * D' * D'), classicalisingmpo(x * βc)
        )
        zbulk = UnitCell([zt;;])
        mtbulk = UnitCell([mt;;])

        χs = [2, 5, 20]

        algs = (
            (VUMPS(; bonddim=i) for i in χs)...,
            (CTMRG(; svd_alg=TensorKit.SVD(), bonddim=i) for i in χs)...,
        )

        @testset "Using $(typeof(alg)) with χ = $(alg.bonddim)" for alg in algs
            st = @constinferred(initialize(zbulk, alg))
            calculate!(st)
            z_val = @constinferred(contract(st.tensors, zbulk))
            m_val = contract(st.tensors, mtbulk) ./ z_val

            @test isapprox(abs(m_val[1, 1]), M; atol=1e-5)
        end
    end
end
=#
