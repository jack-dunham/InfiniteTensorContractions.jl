@testset verbose = true begin
    e = ℂ^2
    s = ℂ^3
    w = ℂ^4
    n = ℂ^5

    p = ℂ^6

    t = TensorMap(rand, p * p', e * s * w' * n')

    @test swapaxes(t) == permute(t, ((1, 2), (4, 3, 6, 5)))
    @test invertaxes(t) == permute(t, ((1, 2), (5, 6, 3, 4)))

    ct = CompositeTensor(t, t')

    @test ct[1] == ct[2]'
    @test @constinferred(first(ct)) === parent(@constinferred(last(ct)))

    @test @constinferred(copy(ct)) == ct
    @test !(copy(ct) === ct)

    @test virtualspace(ct, 1) == e * e'
    @test virtualspace(ct, 2) == s * s'
    @test virtualspace(ct, 3) == w' * w
    @test virtualspace(ct, 4) == n' * n

    @testset "SquareSymmetric" verbose = true begin
        data = [
            1 2 3
            2 3 1
            3 1 2
        ]

        ucsym = UnitCell{SquareSymmetric}([1, 2, 3])

        @test UnitCell{Square}(data) == ucsym

        @test ucsym == UnitCell{SquareSymmetric}([1 2 3])
        @test ucsym == UnitCell{SquareSymmetric}([1; 2; 3;;])

        for i in 1:3
            @test ucsym[i, :] == data[i, :]
            @test ucsym[:, i] == data[:, i]
        end

        @test identity.(ucsym) == ucsym
        @test exp2.(ucsym) == UnitCell{SquareSymmetric}([2, 4, 8])
        @test ucsym .+ ucsym == 2 * ucsym
        @test ucsym + ucsym == 2 * ucsym

        let rv1 = ucsym + data, rv2 = ucsym .+ data
            for rv in (rv1, rv2)
                @test rv == 2 * data
                @test isa(rv, UnitCell{Square})
            end
        end

        @test ucsym == ucsym'
    end
end
