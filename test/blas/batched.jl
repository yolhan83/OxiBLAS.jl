@testset "Batched GEMM and overlap semantics" begin
    for T in TYPES
        A=[randn(T,9,7) for _ in 1:4]; B=[randn(T,7,11) for _ in 1:4]; C=[zeros(T,9,11) for _ in 1:4]
        gemm_batched!(C,A,B)
        @test all(isapprox(C[i],A[i]*B[i];rtol=tol(T),atol=tol(T)) for i in 1:4)
        aa=randn(T,9,7,4); bb=randn(T,7,11,4); cc=fill(T(NaN),9,11,4)
        gemm_strided_batched!(cc,aa,bb)
        @test all(isapprox(cc[:,:,i],aa[:,:,i]*bb[:,:,i];rtol=tol(T),atol=tol(T)) for i in 1:4)
        # Input batch 2 shares output batch 1: snapshot before any writes.
        X=randn(T,5,5); Y=randn(T,5,5); Z=zeros(T,5,5); W=randn(T,5,5)
        expected1=Y*W; expected2=X*W
        gemm_batched!([X,Z],[Y,X],[W,W])
        @test X≈expected1 rtol=tol(T)
        @test Z≈expected2 rtol=tol(T)
        for ta in (:N,:T,:C), tb in (:N,:T,:C)
            aa=randn(T,ta===:N ? (9,7,2) : (7,9,2))
            bb=randn(T,tb===:N ? (7,11,2) : (11,7,2))
            cc=zeros(T,9,11,2)
            gemm_strided_batched!(cc,aa,bb;transA=ta,transB=tb)
            @test all(isapprox(cc[:,:,i],op(aa[:,:,i],ta)*op(bb[:,:,i],tb);rtol=tol(T),atol=tol(T)) for i in 1:2)
            cs=[zeros(T,9,11) for _ in 1:2]
            gemm_batched!(cs,[aa[:,:,i] for i in 1:2],[bb[:,:,i] for i in 1:2];transA=ta,transB=tb)
            @test all(isapprox(cs[i],cc[:,:,i];rtol=tol(T),atol=tol(T)) for i in 1:2)
        end
    end
    @test gemm_batched!(Matrix{Float64}[],Matrix{Float64}[],Matrix{Float64}[])==[]
    @test_throws DimensionMismatch gemm_batched!([zeros(3,3)],[],[])
    X=zeros(3,3)
    @test_throws ArgumentError gemm_batched!([X,X],[ones(3,3),ones(3,3)],[ones(3,3),ones(3,3)])
end
