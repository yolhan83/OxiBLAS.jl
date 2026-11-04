@testset "Iterative sparse SVD: both singular equations" begin
    for T in TYPES
        R=real(T)
        small=reshape(T.(1:6),3,2); T<:Complex && (small .*= T(im))
        dilation=OxiBLAS._SVDHermitianDilation(CSRMatrix(small))
        explicit=[zeros(T,3,3) small; small' zeros(T,2,2)]
        x=randn(T,5); expected=explicit*x; alias=copy(x)
        OxiBLAS._native_mv!(alias,dilation,alias)
        @test alias≈expected rtol=10tol(T) atol=10tol(T)
        X=randn(T,5,3); expected=explicit*X
        OxiBLAS._block_apply!(X,dilation,X)
        @test X≈expected rtol=10tol(T) atol=10tol(T)
        @test_throws ArgumentError dilation[1,1]
        for (m,n) in ((9,6),(6,9),(7,7))
            A=zeros(T,m,n)
            for i in 1:min(m,n)
                A[i,i]=T(i)
            end
            if T<:Complex
                # Diagonal phases test conjugation in A' applications.
                for i in 1:min(m,n)
                    A[i,i]*=T(cis(R(i)/3))
                end
            end
            expected=LA.svdvals(A)[1:3]
            for format in (CSRMatrix,CSCMatrix,COOMatrix)
                sparse=format(CSRMatrix(A)); before=Matrix(sparse)
                result=sparse_svd_lanczos(sparse,3;subspace=m+n,maxrestarts=0)
                @test result.converged
                @test result.S≈expected rtol=40tol(T) atol=40tol(T)
                @test A*result.Vt'≈result.U*LA.Diagonal(result.S) rtol=40tol(T) atol=40tol(T)
                @test A'*result.U≈result.Vt'*LA.Diagonal(result.S) rtol=40tol(T) atol=40tol(T)
                @test result.U'*result.U≈Matrix{T}(LA.I,3,3) rtol=40tol(T) atol=40tol(T)
                @test result.Vt*result.Vt'≈Matrix{T}(LA.I,3,3) rtol=40tol(T) atol=40tol(T)
                left=[LA.norm(A*conj.(result.Vt[j,:])-result.S[j]*result.U[:,j]) for j in 1:3]
                right=[LA.norm(A'*result.U[:,j]-result.S[j]*conj.(result.Vt[j,:])) for j in 1:3]
                @test result.left_residuals≈left atol=100eps(R)*LA.norm(A)
                @test result.right_residuals≈right atol=100eps(R)*LA.norm(A)
                @test Matrix(sparse)==before
            end
            initial=randn(T,n,2)
            result=sparse_svd_lanczos(CSRMatrix(A),3;subspace=m+n,maxrestarts=0,initial)
            @test result.S≈expected rtol=40tol(T) atol=40tol(T)
        end
        A=zeros(T,8,5); A[1,1]=T(7); A[2,2]=T(3)
        deficient=sparse_svd_lanczos(CSRMatrix(A),4;subspace=13,maxrestarts=0)
        @test length(deficient.S)==2 && !deficient.converged
        @test deficient.reason===:insufficient_positive_ritz_values
        @test deficient.S≈R[7,3] rtol=40tol(T) atol=40tol(T)
        zero_result=sparse_svd_lanczos(CSRMatrix(zeros(T,4,3)),2;subspace=7)
        @test isempty(zero_result.S) && !zero_result.converged
        for shape in ((0,0),(3,0),(0,3),(3,4))
            result=sparse_svd_lanczos(CSRMatrix(zeros(T,shape)),0)
            @test result.converged && isempty(result.S)
            @test size(result.U)==(shape[1],0) && size(result.Vt)==(0,shape[2])
        end
        repeated=zeros(T,8,6); repeated[1,1]=repeated[2,2]=T(5)
        result=sparse_svd_lanczos(CSRMatrix(repeated),2;subspace=14,blocksize=2)
        @test result.S≈fill(R(5),2) rtol=40tol(T) atol=40tol(T)
        @test result.U'*result.U≈Matrix{T}(LA.I,2,2) rtol=40tol(T) atol=40tol(T)
        for scale in (R(2)^(-60),R(2)^60)
            A=Matrix(LA.Diagonal(T.(scale.*R[1,2,4,8])))
            result=sparse_svd_lanczos(CSRMatrix(A),2;subspace=8,maxrestarts=0)
            @test result.S./scale≈R[8,4] rtol=80tol(T) atol=80tol(T)
        end
        A=CSRMatrix(Matrix(LA.Diagonal(T.(1:8))))
        GC.gc(); result=sparse_svd_lanczos(A,2;subspace=16)
        @test result.S≈R[8,7] rtol=40tol(T) atol=40tol(T)
        results=Vector{Any}(undef,2)
        @sync for j in 1:2
            Threads.@spawn results[j]=sparse_svd_lanczos(A,2;subspace=16)
        end
        @test results[1].S==results[2].S
        @sync begin
            Threads.@spawn GC.gc()
            Threads.@spawn begin
                result=sparse_svd_lanczos(A,2;subspace=16)
                @test result.S≈R[8,7] rtol=40tol(T) atol=40tol(T)
            end
        end
        @test_throws ArgumentError sparse_svd_lanczos(A,-1)
        @test_throws ArgumentError sparse_svd_lanczos(A,9)
        @test_throws ArgumentError sparse_svd_lanczos(A,2;singular_tol=NaN)
        @test_throws ArgumentError sparse_svd_lanczos(A,2;rtol=-1)
        @test_throws ArgumentError sparse_svd_lanczos(A,0;maxrestarts=-1)
        @test_throws ArgumentError sparse_svd_lanczos(A,0;subspace=-1)
        @test_throws DimensionMismatch sparse_svd_lanczos(A,2;initial=zeros(T,7,2))
    end
end
