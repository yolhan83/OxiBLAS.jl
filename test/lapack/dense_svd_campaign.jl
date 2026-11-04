# Dense SVD campaign: Golub-Kahan and parallel Jacobi against the reference.
# The other dense_svd_*.jl files and svd.jl are separate entries of
# groups.jl. The campaign keeps its own seed; its former local `tol` was
# unused and is removed (it would have replaced the suite's).
Random.seed!(0x5ad)

function check_dense_svd(A,algorithm;threaded=false,full=false)
    T=eltype(A);R=real(T);m,n=size(A);k=min(m,n);original=copy(A)
    f=algorithm===:golub_kahan ? svd_bidiagonal : svd_factor
    F=f(A;algorithm,threaded,full)
    @test A==original
    @test issorted(F.S;rev=true) && all(isfinite,F.S) && all(>=(0),F.S)
    scale=isempty(A) ? one(R) : maximum(abs,A)
    scale=iszero(scale) ? one(R) : scale
    bound=128eps(R)*max(m,n,1)
    @test LA.norm(A/scale-F.U[:,1:k]*LA.Diagonal(F.S/scale)*F.Vt[1:k,:])<=bound*max(one(R),LA.norm(A/scale))
    @test LA.norm(F.U'*F.U-Matrix{T}(LA.I,size(F.U,2),size(F.U,2)))<=bound*sqrt(max(1,size(F.U,2)))
    @test LA.norm(F.Vt*F.Vt'-Matrix{T}(LA.I,size(F.Vt,1),size(F.Vt,1)))<=bound*sqrt(max(1,size(F.Vt,1)))
    values=f(A;algorithm,threaded,vectors=false)
    @test values.U===nothing && values.Vt===nothing
    @test values.S≈F.S rtol=bound atol=bound*scale
    @test F.S/scale≈LA.svdvals(A/scale) rtol=bound atol=bound
end

@testset "Dense SVD campaign: Golub-Kahan and parallel Jacobi" begin
    for T in TYPES, algorithm in (:golub_kahan,:parallel_jacobi), threaded in (false,true)
        for (m,n) in ((0,0),(3,0),(0,3),(1,1),(1,7),(7,1),(9,9),(19,7),(7,19),(151,96),(96,151))
            check_dense_svd(randn(T,m,n),algorithm;threaded)
        end
        for A in (zeros(T,9,7),ones(T,9,7),randn(T,9,2)*randn(T,2,7))
            check_dense_svd(A,algorithm;threaded)
        end
        for exponent in (real(T)===Float32 ? (-120,110) : (-1000,1000))
            check_dense_svd(randn(T,9,7)*ldexp(one(real(T)),exponent),algorithm;threaded)
        end
        M=randn(T,19,17)
        check_dense_svd(view(M,19:-2:1,17:-2:1),algorithm;threaded)
        check_dense_svd(randn(T,13,7),algorithm;threaded,full=true)
        check_dense_svd(randn(T,7,13),algorithm;threaded,full=true)
    end
    # Cross the actual task thresholds, rather than merely setting a flag.
    for algorithm in (:golub_kahan,:parallel_jacobi)
        check_dense_svd(randn(Float64,960,320),algorithm;threaded=true)
    end
    @test_throws OxiBLAS.ConvergenceError svd_bidiagonal(randn(7,5);maxiter=0)
    @test_throws ArgumentError svd_bidiagonal([1.0 Inf;0 1])
    @test_throws ArgumentError svd_bidiagonal(ones(3,2);rtol=0)
end

@testset "Dense Jacobi graded pair guard and generic fallback" begin
    for T in TYPES
        R=real(T);tiny=ldexp(one(R),R===Float32 ? -80 : -600)
        # Correlated tiny columns exercise the normalized fallback, not only a
        # diagonal with an exactly zero cross product.
        A=zeros(T,5,3);A[1,1]=1;A[2,2]=tiny;A[2,3]=tiny/2;A[3,3]=tiny
        for algorithm in (:jacobi,:parallel_jacobi)
            F=svd_factor(A;algorithm)
            @test all(>(zero(R)),F.S)
            @test F.S[2:3]/tiny≈LA.svdvals(A[2:3,2:3]/tiny) rtol=128eps(R)
            @test LA.norm(F.U'*F.U-Matrix{T}(LA.I,3,3))<256eps(R)
        end
    end
    setprecision(BigFloat,128) do
        A=BigFloat[1 2;3 5;7 11]
        for f in (svd_factor,svd_bidiagonal)
            F=f(A;threaded=false)
            @test LA.norm(A-F.U*LA.Diagonal(F.S)*F.Vt)<big"1e-30"
        end
    end
end

