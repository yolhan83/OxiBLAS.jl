@testset "GEMM algorithms, types, dimensions and operations" begin
    shapes=((0,0,0),(0,3,7),(4,0,5),(3,4,0),(1,1,1),(2,3,5),
            (7,9,11),(8,6,4),(17,19,23),(33,13,35),(65,67,49))
    for T in TYPES, method in (:reference,:blocked,:packed,:register,:simd,:auto),
        ta in (:N,:T,:C), tb in (:N,:T,:C), (m,n,k) in shapes
        A=randn(T,ta===:N ? (m,k) : (k,m))
        B=randn(T,tb===:N ? (k,n) : (n,k))
        C=randn(T,m,n); old=copy(C)
        α=T<:Real ? T(0.7) : T(0.7+0.3im)
        β=T<:Real ? T(-0.4) : T(-0.4+0.2im)
        expected=α*(op(A,ta)*op(B,tb))+β*old
        @test gemm!(C,A,B,α,β;transA=ta,transB=tb,algorithm=method,threaded=false)===C
        @test isapprox(C,expected;rtol=tol(T),atol=tol(T)*max(1,k))
    end
end

@testset "Views, wrappers, packed edges and guard regions" begin
    for T in TYPES, method in (:reference,:blocked,:packed,:register,:simd)
        A0=randn(T,75,81); B0=randn(T,85,71); C0=fill(T(123),80,77)
        A=view(A0,73:-2:3,4:2:80) # 36×39
        B=view(B0,2:2:78,69:-2:3) # 39×34
        C=view(C0,5:2:75,4:2:70)
        expected=T(0.8)*(A*B)
        config=GemmConfig(T;mc=17,kc=7,nc=19,parallel_threshold=0)
        gemm!(C,A,B,T(0.8),zero(T);algorithm=method,config)
        @test C≈expected rtol=tol(T) atol=tol(T)*39
        @test all(C0[i,j]==123 for j in 1:77 for i in 1:80 if !(i in 5:2:75 && j in 4:2:70))
        # Unit row stride with padded leading dimension; unaligned first row.
        A=randn(T,17,15); B=randn(T,15,19); storage=fill(T(88),23,22)
        C=view(storage,2:18,3:21)
        gemm!(C,A,B;algorithm=method)
        @test C≈A*B rtol=tol(T) atol=tol(T)*15
        @test storage[1,:]==fill(T(88),22)
        # Output wrappers route through the layout fallback.
        backing=zeros(T,19,17); C=adjoint(backing)
        gemm!(C,A,B;algorithm=method)
        @test C≈A*B rtol=tol(T) atol=tol(T)*15
    end
end

# UnreadableMatrix: test/helpers/probes.jl

@testset "No-read cases and overlap snapshots" begin
    for T in TYPES, method in (:reference,:blocked,:packed,:register,:simd)
        for β in (zero(T),one(T),T(0.5))
            C=fill(T(2),9,7)
            gemm!(C,UnreadableMatrix{T}((9,3)),UnreadableMatrix{T}((3,7)),zero(T),β;algorithm=method)
            @test C==fill(2β,9,7)
        end
        C=fill(T(NaN),17,19); A=randn(T,17,13); B=randn(T,13,19)
        gemm!(C,A,B,one(T),zero(T);algorithm=method)
        @test C≈A*B rtol=tol(T) atol=tol(T)*13
        C=fill(T(4),5,7)
        gemm!(C,zeros(T,5,0),zeros(T,0,7),T(NaN),T(0.5);algorithm=method)
        @test C==fill(T(2),5,7)
        # C is A, C is B, and all three share storage.
        A=randn(T,17,17); B=randn(T,17,17); expected=A*B+T(0.3)*A
        gemm!(A,A,B,one(T),T(0.3);algorithm=method)
        @test A≈expected rtol=tol(T) atol=tol(T)*17
        expected=A*B
        gemm!(B,A,B;algorithm=method)
        @test B≈expected rtol=tol(T) atol=tol(T)*17
        X=randn(T,20,20); C=view(X,2:18,2:18); A=view(X,1:17,1:17); B=view(X,3:19,3:19)
        expected=A*B+copy(C)
        gemm!(C,A,B,one(T),one(T);algorithm=method)
        @test C≈expected rtol=tol(T) atol=tol(T)*17
    end
end

@testset "Workspace, threading and validation" begin
    for T in TYPES, dims in ((97,101,83),(263,7,51),(9,263,71))
        m,n,k=dims; A=randn(T,m,k); B=randn(T,k,n); C=zeros(T,m,n)
        config=GemmConfig(T;mc=32,kc=17,nc=48,parallel_threshold=0)
        w=GemmWorkspace(T,m,n,k;config)
        @test gemm!(C,A,B;config,workspace=w)≈A*B rtol=tol(T) atol=tol(T)*k
        # Shared workspace must serialize two callers even under task migration.
        D=similar(C)
        @sync begin
            Threads.@spawn gemm!(C,A,B;config,workspace=w)
            Threads.@spawn gemm!(D,A,B;config,workspace=w)
        end
        @test C≈D rtol=tol(T) atol=tol(T)*k
        GC.gc()
        @test gemm!(C,A,B;config,workspace=w,threaded=false)≈A*B rtol=tol(T) atol=tol(T)*k
    end
    C=zeros(64,64); A=randn(64,64)
    @test_throws DimensionMismatch gemm!(C,A,zeros(63,64))
    @test_throws ArgumentError gemm!(C,A,A;transA=:bad)
    @test_throws ArgumentError gemm!(C,A,A;algorithm=:bad)
    @test_throws ArgumentError gemm!(C,A,A;kernel=:bad)
    @test_throws ArgumentError GemmConfig(;mc=0)
    @test_throws ArgumentError GemmWorkspace(Float64,3,3,3;workers=0)
    @test_throws ArgumentError gemm!(C,A,A;workspace=GemmWorkspace(Float64,3,3,3))
    @test_throws ArgumentError gemm!(C,A,A;workspace=GemmWorkspace(Float32,64,64,64))
    @test_throws ArgumentError gemm!(view(C,[1,1],1:2),ones(2,3),ones(3,2))
    if Sys.ARCH===:aarch64
        @test_throws ArgumentError gemm!(C,A,A;kernel=:avx2)
        @test C==zeros(64,64)
    end
    @test gemm!(C,A,A;kernel=:scalar,algorithm=:simd)≈A*A
end
