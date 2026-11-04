using Test, Random, OxiBLAS
import LinearAlgebra as LA
@testset "Subnormal active bidiagonal block beside a unit singular value" begin
    for T in (Float32,Float64,ComplexF32,ComplexF64)
        R=real(T);tiny=16nextfloat(zero(R))
        B=Matrix{T}(LA.Bidiagonal(R[1,2,3,4],R[0.5,0.5,0.5],:U))
        A=zeros(T,5,5);A[1,1]=one(T);A[2:5,2:5]=B*tiny
        F=svd_bidiagonal(A)
        @test maximum(abs.(F.S[2:5]/tiny-LA.svdvals(B)))<=R(0.125)
        @test LA.norm(adjoint(F.U)*F.U-Matrix{T}(LA.I,5,5))<=128eps(R)
        @test LA.norm(F.Vt*adjoint(F.Vt)-Matrix{T}(LA.I,5,5))<=128eps(R)
    end
end
@testset "SVD Givens at subnormal scales" begin
    for R in (Float32,Float64), i in 1:5
        tiny=nextfloat(zero(R));a=R(i)*tiny;b=R(i+1)*tiny
        c,s,r=OxiBLAS._real_givens(a,b)
        @test abs(c*c+s*s-one(R))<=4eps(R)
        @test abs(-s*R(i)+c*R(i+1))<=8eps(R)*R(i+1)
        @test r>=max(a,b)
    end
end
@testset "SVD Householder at subnormal scales" begin
    for T in (Float32,Float64,ComplexF32,ComplexF64),i in 1:5
        R=real(T);tiny=nextfloat(zero(R))
        x=T[R(i)*tiny,R(i+1)*tiny,R(i+2)*tiny]
        T<:Complex && (x[2]=complex(real(x[2]),tiny))
        v=copy(x);beta,tau=OxiBLAS._svd_dense_householder!(v)
        H=Matrix{T}(LA.I,3,3)-tau*v*adjoint(v)
        @test LA.norm(adjoint(H)*H-Matrix{T}(LA.I,3,3))<=32eps(R)
        # beta itself is rounded to a subnormal scalar; allow its unavoidable
        # final quantization while checking the direction in scaled units.
        target=T[beta/tiny,0,0]
        @test LA.norm(H*(x/tiny)-target)<=R(2)
    end
end
@testset "Dense rectangular SVD: native QR preprocessing" begin
    rng=MersenneTwister(0x9a)
    for T in (Float32,Float64,ComplexF32,ComplexF64), algorithm in (:jacobi,:parallel_jacobi,:golub_kahan)
        R=real(T);f=algorithm===:golub_kahan ? svd_bidiagonal : svd_factor
        for (m,n) in ((192,64),(64,192),(384,128),(128,384))
            A=randn(rng,T,m,n);old=copy(A);k=min(m,n);F=f(A;algorithm,threaded=true)
            bound=128eps(R)*max(m,n)
            @test A==old
            @test LA.norm(A-F.U*LA.Diagonal(F.S)*F.Vt)<=bound*LA.norm(A)
            @test LA.norm(F.U'*F.U-Matrix{T}(LA.I,k,k))<=bound*sqrt(k)
            @test LA.norm(F.Vt*F.Vt'-Matrix{T}(LA.I,k,k))<=bound*sqrt(k)
            @test F.S≈LA.svdvals(A) rtol=64eps(R)*sqrt(R(max(m,n)))
            values=f(A;algorithm,vectors=false)
            @test values.S≈F.S rtol=64eps(R)*sqrt(R(max(m,n)))
        end
        for A in (ones(T,144,64),zeros(T,144,64),randn(rng,T,144,3)*randn(rng,T,3,64))
            F=f(A;algorithm)
            @test LA.norm(A-F.U*LA.Diagonal(F.S)*F.Vt)<=128eps(R)*144*max(one(R),LA.norm(A))
            @test LA.norm(F.U'*F.U-Matrix{T}(LA.I,64,64))<=128eps(R)*144*8
        end
        for (m,n) in ((96,32),(32,96))
            A=randn(rng,T,m,n);F=f(A;algorithm,full=true)
            @test size(F.U)==(m,m) && size(F.Vt)==(n,n)
            @test LA.norm(A-F.U[:,1:32]*LA.Diagonal(F.S)*F.Vt[1:32,:])<=128eps(R)*96*LA.norm(A)
        end
        M=randn(rng,T,384,128);A=view(M,384:-2:1,128:-2:1);old=copy(A)
        F=f(A;algorithm)
        @test A==old
        @test LA.norm(A-F.U*LA.Diagonal(F.S)*F.Vt)<=128eps(R)*192*LA.norm(A)
    end
end
@testset "Column-scaled QR preserves graded Jacobi inputs" begin
    for T in (Float32,Float64,ComplexF32,ComplexF64)
        R=real(T);powers=round.(Int,range(0,R===Float32 ? -100 : -900;length=128))
        d=R[ldexp(one(R),p) for p in powers];A=zeros(T,384,128)
        for i in 1:128
            A[i,i]=d[i]
        end
        for i in 1:2:127
            OxiBLAS._svd_rotate_columns!(A,i,i+1,R(0.8),T(0.6))
        end
        for algorithm in (:jacobi,:parallel_jacobi)
            F=svd_factor(A;algorithm)
            @test all(F.S.>0)
            @test maximum(abs.(F.S./d.-1))<=1024eps(R)
        end
    end
end
