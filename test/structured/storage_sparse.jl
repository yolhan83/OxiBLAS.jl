@testset "Packed and banded BLAS storage" begin
    for T in TYPES, n in (0,1,9), uplo in (:U,:L), kind in (:symmetric,:hermitian,:triangular)
        A=randn(T,n,n); A[LA.diagind(A)].+=T(10)
        P=pack_matrix(A;uplo,kind)
        expected=kind===:symmetric ? LA.Symmetric(A,uplo) : kind===:hermitian ? LA.Hermitian(A,uplo) : uplo===:U ? LA.UpperTriangular(A) : LA.LowerTriangular(A)
        @test Matrix(P)==Matrix(expected)
        x=randn(T,n); y=fill(T(NaN),n)
        if kind===:triangular
            for trans in (:N,:T,:C)
                z=copy(x)
                @test tpmv!(z,P;trans)≈op(expected,trans)*x rtol=tol(T) atol=tol(T)
                @test tpsv!(z,P;trans)≈x rtol=tol(T) atol=tol(T)
            end
        else
            f=kind===:symmetric ? spmv! : hpmv!
            @test f(y,P,x)≈expected*x rtol=tol(T) atol=tol(T)
            for rank2 in (true,false)
                pp=pack_matrix(A;uplo,kind); z=randn(T,n)
                if kind===:symmetric
                    rank2 ? spr2!(pp,x,z) : spr!(pp,x)
                    update=rank2 ? x*transpose(z)+z*transpose(x) : x*transpose(x)
                else
                    rank2 ? hpr2!(pp,x,z) : hpr!(pp,x)
                    update=rank2 ? x*z'+z*x' : x*x'
                end
                @test Matrix(pp)≈expected+update rtol=tol(T) atol=tol(T)
            end
        end
        for kd in (0,2)
            B=band_matrix(A;kl=kd,ku=kd,kind,uplo)
            D=Matrix(B)
            if kind===:triangular
                for tr in (:N,:T,:C)
                    z=copy(x)
                    @test tbmv!(z,B;trans=tr)≈op(D,tr)*x rtol=tol(T) atol=tol(T)
                    @test tbsv!(z,B;trans=tr)≈x rtol=tol(T) atol=tol(T)
                end
            else
                f=kind===:symmetric ? sbmv! : hbmv!
                @test f(y,B,x)≈D*x rtol=tol(T) atol=tol(T)
            end
        end
    end
    for T in TYPES, (m,n) in ((0,0),(5,9),(9,5)), kl in (0,2), ku in (0,3)
        A=randn(T,m,n); P=band_matrix(A;kl,ku)
        expected=copy(A)
        for j in 1:n, i in 1:m
            (i-j>kl || j-i>ku) && (expected[i,j]=0)
        end
        @test Matrix(P)==expected
        for trans in (:N,:T,:C)
            x=randn(T,size(op(P,trans),2)); y=fill(T(NaN),size(op(P,trans),1))
            @test gbmv!(y,P,x;trans)≈op(expected,trans)*x rtol=tol(T) atol=tol(T)
        end
    end
    @test_throws DimensionMismatch PackedMatrix(ones(3),3)
    @test_throws ArgumentError PackedMatrix(ones(3),2;kind=:bad)
    @test_throws DimensionMismatch BandedMatrix(ones(1,5),5;kl=3)
end

function test_csr(A::Matrix{T}) where T
    m,n=size(A); ptr=Int[1]; col=Int[]; vals=T[]
    for i in 1:m
        for j in 1:n
            if !iszero(A[i,j])
                push!(col,j); push!(vals,A[i,j])
            end
        end
        push!(ptr,length(vals)+1)
    end
    CSRMatrix(m,n,ptr,col,vals)
end
@testset "CSR and conjugate gradient" begin
    for T in TYPES, (m,n) in ((0,0),(1,1),(9,7),(7,9))
        A=randn(T,m,n)
        A[abs.(A).<0.7].=0
        S=test_csr(A)
        @test Matrix(S)==A
        for trans in (:N,:T,:C)
            AA=op(A,trans); x=randn(T,size(AA,2)); y=fill(T(NaN),size(AA,1))
            @test sparse_mul!(y,S,x;trans)≈AA*x rtol=tol(T) atol=tol(T)
        end
    end
    for T in TYPES, n in (0,1,19,67)
        M=randn(T,n,n); A=M*M'+T(n+1)*Matrix{T}(LA.I,n,n); b=randn(T,n)
        for sparse in (true,false), precond in (true,false)
            aa=sparse ? test_csr(A) : A
            result=cg(aa,b;rtol=real(T)===Float32 ? 2f-5 : 1e-11,diagonal=precond ? LA.diag(A) : nothing)
            @test result.converged
            @test LA.norm(A*result.x-b)<=tol(T)*10*max(1,LA.norm(b))
        end
    end
    @test_throws ArgumentError CSRMatrix(2,2,[1,3,3],[1,1],[1.0,2.0])
    @test_throws ArgumentError CSRMatrix(2,2,[1,2,2],[3],[1.0])
    @test_throws ArgumentError CSRMatrix(2,2,[0,1,2],[1],[1.0])
    @test !cg([-1.0 0;0 -1],ones(2)).converged
    @test !cg([2.0 0;0 3],ones(2);maxiter=0).converged
    @test_throws ArgumentError cg([2.0 0;0 3],ones(2);diagonal=[0.0,1.0])
end
