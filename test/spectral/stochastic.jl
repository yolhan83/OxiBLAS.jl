# Written for the supervised campaign; no execution implied.
@testset "Stochastic estimates: analytic references and reproducibility" begin
    for T in TYPES
        R=real(T); n=9; values=T.(1:n)
        T<:Complex && (values .+= T(im).*T.(reverse(1:n)))
        D=Matrix(LA.Diagonal(values)); A=CSRMatrix(D)
        config=StochasticConfig(num_probes=12,seed=9)
        for input in (D,A,CSCMatrix(A),COOMatrix(A))
            h=trace_hutchinson(input;config)
            @test h.estimate≈sum(values) rtol=tol(T)
            @test h.std_error<=tol(T)
            @test h.matvecs==12 && h.n_probes_used==12
            diagonal=stochastic_diagonal(input;config)
            @test diagonal.diagonal≈values rtol=tol(T)
            @test maximum(diagonal.std_error)<=tol(T)
            f=stochastic_frobenius(input;config)
            @test f.squared_estimate≈sum(abs2,values) rtol=4tol(T)
            @test f.estimate≈LA.norm(D) rtol=4tol(T)
        end
        # A rank-one diagonal has an exact captured range for any unit-phase probe.
        low=zeros(T,n,n); low[1,1]=T(7)
        for estimator in (trace_hutch_plus_plus,trace_xtrace)
            result=estimator(CSRMatrix(low);config)
            @test result.estimate≈T(7) rtol=10tol(T) atol=10tol(T)
            @test result.matvecs>=result.n_probes_used
        end
        @test isnan(trace_xtrace(A;config).std_error)
        @test trace_xtrace(A;config).error_kind===:unavailable_correlated_samples
        @test isinf(trace_hutchinson(A;config=StochasticConfig(num_probes=1)).std_error)
        # Rectangular operator: expectation is over the input dimension.
        rectangular=zeros(T,11,7)
        for j in 1:7
            rectangular[j,j]=T(j)
        end
        @test stochastic_frobenius(CSRMatrix(rectangular);config).squared_estimate≈T(140) rtol=4tol(T)
        B=randn(T,n,n); saved=copy(B)
        for distribution in (:rademacher,:gaussian,:spherical)
            cfg=StochasticConfig(num_probes=20,seed=77,probe_type=distribution)
            first_result=trace_hutchinson(B;config=cfg); two=trace_hutchinson(B;config=cfg)
            @test isequal(first_result,two)
            @test trace_hutchinson(CSRMatrix(B);config=cfg).estimate≈first_result.estimate rtol=4tol(T) atol=4tol(T)
            results=Vector{typeof(first_result)}(undef,2)
            @sync for j in 1:2
                Threads.@spawn results[j]=trace_hutchinson(B;config=cfg)
            end
            @test isequal(results[1],first_result) && isequal(results[2],first_result)
            GC.gc(); @test isequal(trace_hutchinson(B;config=cfg),first_result)
        end
        @test B==saved
        # Negative row stride views are read-only operands.
        viewB=view(B,n:-1:1,n:-1:1)
        @test trace_hutchinson(viewB;config).estimate≈trace_hutchinson(Matrix(viewB);config).estimate rtol=tol(T) atol=tol(T)
        # Repeated eigenvalue: one Lanczos step suffices, no quadrature bias.
        scalar=T(3).*Matrix{T}(LA.I,n,n)
        logresult=stochastic_logdet(CSRMatrix(scalar);config,lanczos_steps=1)
        @test logresult.estimate≈n*log(R(3)) rtol=10tol(T)
        @test logresult.matvecs==config.num_probes
        pd=Matrix(LA.Diagonal(T.(1:n)))
        @test stochastic_logdet(pd;config,lanczos_steps=n).estimate≈sum(log,R.(1:n)) rtol=30tol(T)
        nonhermitian=copy(pd); nonhermitian[1,2]=one(T)
        @test_throws ArgumentError stochastic_logdet(nonhermitian;config)
        @test_throws ArgumentError stochastic_logdet(CSRMatrix(nonhermitian);config)
        @test_throws ArgumentError stochastic_logdet(-scalar;config,lanczos_steps=1)
        for estimator in (trace_hutchinson,trace_hutch_plus_plus,trace_xtrace,stochastic_logdet)
            @test iszero(estimator(zeros(T,0,0);config).estimate)
            @test_throws DimensionMismatch estimator(zeros(T,2,3);config)
        end
        @test isempty(stochastic_diagonal(zeros(T,0,0);config).diagonal)
        @test iszero(stochastic_frobenius(zeros(T,3,0);config).estimate)
        @test_throws ArgumentError trace_hutchinson(fill(T(NaN),1,1);config)
        @test_throws ArgumentError stochastic_diagonal(fill(T(Inf),1,1);config)
    end
    # Check the distribution normalization against a known scalar identity trace.
    for kind in (:gaussian,:spherical)
        cfg=StochasticConfig(num_probes=2048,seed=824,probe_type=kind)
        result=trace_hutchinson(Matrix{Float64}(LA.I,5,5);config=cfg)
        @test abs(result.estimate-5)<0.5
        kind===:spherical && @test result.estimate≈5 atol=1e-12
    end
    @test_throws ArgumentError StochasticConfig(num_probes=0)
    @test_throws ArgumentError StochasticConfig(seed=-1)
    @test_throws ArgumentError StochasticConfig(probe_type=:unknown)
    @test_throws ArgumentError StochasticConfig(confidence=NaN)
    @test_throws ArgumentError StochasticConfig(confidence=1)
    @test_throws ArgumentError trace_hutch_plus_plus(ones(1,1);config=StochasticConfig(num_probes=2))
    @test_throws ArgumentError stochastic_logdet(ones(1,1);lanczos_steps=0)
    @test_throws ArgumentError stochastic_logdet(ones(1,1);lanczos_steps=2)
end
