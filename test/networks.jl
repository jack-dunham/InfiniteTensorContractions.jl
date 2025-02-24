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
end
