@testset "Other tests" verbose = true begin
    d = ℂ^1

    s2 = ℂ^2
    s3 = ℂ^3
    s4 = ℂ^4
    s5 = ℂ^5

    t = TensorMap(rand, ComplexF64, s2 * s3, s4 * s5)

    dom = s2 * s3 * s4' * s5'

    p1 = TensorMap(rand, ComplexF64, one(d), dom)
    p2 = TensorMap(rand, ComplexF64, d, dom)
    p3 = TensorMap(rand, ComplexF64, d * d', dom)

    tp = CompositeTensor(p3, p3')

    test_t = (p1, p2, p3)

    @testset "Utils" verbose = true begin
        @test space(ITC._transpose(t)) == ((s4' * s5') ← (s2' * s3'))
        @test @constinferred(ITC._transpose(t)).data == permutedims(t.data)
        @test @constinferred(ITC.permutedom(t, (2, 1))) == permute(t, (1, 2), (4, 3))
        @test @constinferred(ITC.permutecod(t, (2, 1))) == permute(t, (2, 1), (3, 4))

        perm = [1:10...]

        @test isa(@constinferred(ITC.tcircshift(Tuple(perm), 0)), NTuple{10,Int})

        for i in (-1, 0, 1)
            @test ITC.tcircshift(Tuple(perm), i) == Tuple(circshift(perm, i))
        end

        for tn in test_t
            @test @constinferred(ITC.rotate(tn, 0)) == tn
            @test domain(ITC.rotate(tn, 1)) == s5' * s2 * s3 * s4'
        end
        @test @constinferred(ITC.rotate(tp, 0)) == tp
        @test domain(ITC.rotate(tp, 1).top) == s5' * s2 * s3 * s4'
        @test domain(ITC.rotate(tp, 1).bot) == s5' * s2 * s3 * s4'
    end

    @testset "Networks" verbose = true begin
        s1 = ℂ^1
        s2 = ℂ^2
        s3 = ℂ^3
        s4 = ℂ^4

        d = ℂ^6

        cod = d * d'
        dom = s1 * s2 * s3' * s4'

        t = TensorMap(rand, ComplexF64, one(d), dom)
        p = TensorMap(rand, ComplexF64, cod, dom)

        tp = CompositeTensor(p, p')

        up = UnitCell([p p; p p])
        ut = UnitCell([t t; t t])
        utp = UnitCell([tp tp; tp tp])

        @test ITC.tensortype(tp) == typeof(p)
        @test eltype(tp) == eltype(t)

        @test bondspace(t) == tuple(dom...)
        @test isa(bondspace(tp), NTuple{4})
        @test bondspace(tp) == tuple(s1 * s1', s2 * s2', s3' * s3, s4' * s4)

        @test eltype(ut) == typeof(t)
        @test eltype(up) == typeof(p)
        @test eltype(utp) == typeof(tp)

        @test eltype(ITC.ensure_contractable(ut)) == typeof(t)
        @test eltype(ITC.ensure_contractable(up)) == typeof(tp)
        @test eltype(ITC.ensure_contractable(utp)) == typeof(tp)
    end
end
