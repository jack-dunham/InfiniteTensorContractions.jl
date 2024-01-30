@testset "Corner Methods" verbose = true begin
    s = ℂ^3
    suc = UnitCell([s;;])

    tf = s -> TensorMap(rand, ComplexF64, one(s), s * s)

    C1 = map(s -> tf(s), suc)
    C2 = map(s -> tf(s'), suc)
    C3 = map(s -> tf(s), suc)
    C4 = map(s -> tf(s'), suc)

    corners = Corners((C1, C2, C3, C4))

    @test @constinferred(Corners(C1, C2, C3, C4)) == corners

    CT = typeof(C1)

    @test length(corners) == 4

    @test iterate(corners, 5) === nothing
    @test convert(Tuple, corners) == (C1, C2, C3, C4)
    @test convert(Tuple, (C1, C2, C3, C4)) == (C1, C2, C3, C4)

    @test isa(identity.(corners), NTuple{4,CT})

    @test isa(@constinferred(convert(Tuple, corners)), NTuple{4,CT})

    @test isa(@constinferred(map(identity, corners)), NTuple{4,CT})

    @test scalartype(corners) == ComplexF64

    @test ITC.chispace(corners) == s
end

@testset "CTMRG" verbose = true begin
    d = ℂ^2
    s1 = ℂ^3
    s2 = ℂ^4
    chi = ℂ^5

    ta = TensorMap(rand, ComplexF64, d * d', s1 * s1 * s2' * s2')
    tb = TensorMap(rand, ComplexF64, d * d', s2 * s2 * s1' * s1')

    net = UnitCell([ta tb; tb ta])
    alg = CTMRG(; verbose=false, bonddim=5)

    local st

    @testset "Initialization" verbose = true begin
        uc = ITC.ensure_contractable(net)

        ten = @constinferred(ITC.inittensors(uc, alg; randinit=false))

        st = @constinferred(initialize(alg, uc; randinit=false))

        prob = @constinferred(newproblem(alg, uc, st))

        @test prob.runtime.primary == ten
        @test eltype(prob.network) <: TensorPair

        for sp in convert.(ProductSpace, domain(ta))
            for ch in convert.(ProductSpace, (ℂ^2, sp, ℂ^6))
                eiso = @constinferred(ITC.get_embedding_isometry(sp, ch))

                @test codomain(eiso) == sp
                @test domain(eiso) == ch
            end
            riso = @constinferred(ITC.get_removal_isometry(sp))

            @test codomain(riso) == sp
            @test domain(riso) == one(sp)
        end

        c = i -> ITC.init_single_corner(TensorPair(ta, ta), chi, i)
        e = i -> ITC.init_single_edge(TensorPair(ta, ta), chi, i)

        @test domain(c(1)) == chi * chi
        @test domain(c(2)) == chi' * chi'
        @test domain(c(3)) == chi * chi
        @test domain(c(4)) == chi' * chi'

        b = bondspace(TensorPair(ta, ta))

        for i in 1:4
            @info "E($i):"
            # cod should be 'swapped' bondspace of south bond (aka north bond of tensor below)
            @test codomain(e(i)) == ITC.swap(b[mod(i + 1, 1:4)])
            @test domain(e(i)) == chi * chi'
        end

        c_raw = @constinferred(ITC.initcorners(uc, chi))
        e_raw = @constinferred(ITC.initedges(uc, chi))

        # tens = @constinferred(ITC.inittensors(TensorPair.(net), alg, randinit=true))
        c_ran = ten.corners
        e_ran = ten.edges

        for i in 1:4
            @info "C/E($i):"
            @test domain.(c_raw[i]) == domain.(c_ran[i])
            @test domain.(e_raw[i]) == domain.(e_ran[i])
            @test codomain.(c_raw[i]) == codomain.(c_ran[i])
            @test codomain.(e_raw[i]) == codomain.(e_ran[i])
        end
    end

    corn = st.primary.corners

    @testset "Error calculation" verbose = true begin
        Ss = @constinferred(ITC.initerror(st.tensors))
        Ss_old = deepcopy(Ss)

        @test isa(@constinferred(ITC.ctmerror!(Ss..., corn)), AbstractFloat)
        # Singular values should have updated:
        @test Ss !== Ss_old
        # Should get zero error with corners unchanged:
        @test @constinferred(ITC.boundaryerror!(Ss[1], corn.C1)) ≈ zero.(eltype.(corn.C1))

        local t = TensorMap([2.0 0.0; 0.0 1.0], one(d), d * d)

        @test @constinferred(ITC.boundaryerror(t)).data ≈ normalize([2.0 0.0; 0.0 1.0])
    end
end
