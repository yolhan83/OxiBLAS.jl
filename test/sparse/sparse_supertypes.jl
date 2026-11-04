# The OxiBLAS sparse types in the SparseArrays hierarchy (AbstractSparseMatrix
# / AbstractSparseArray, not AbstractSparseMatrixCSC), and Base.* on them
# (routed to OxiBLAS.mul).
import LinearAlgebra as LA
import SparseArrays as SA
using SparseArrays: SparseMatrixCSC, SparseVector, sparse, sparsevec, nnz, nonzeros, rowvals, nzrange

@testset "CSCMatrix <: AbstractSparseMatrix" begin
    @test CSCMatrix{Float64,Int} <: SA.AbstractSparseMatrix{Float64,Int}
    @test !(CSCMatrix{Float64,Int} <: SA.AbstractSparseMatrixCSC)
    for T in (Float32,Float64,ComplexF64,BigFloat)
        S=SparseMatrixCSC{T,Int}(SA.sprand(9,7,0.4)); C=CSCMatrix(copy(S))
        @test rowvals(C)===C.rowind && nonzeros(C)===C.values
        @test nnz(C)==nnz(S) && all(nzrange(C,j)==nzrange(S,j) for j in 1:7)
        @test C==S && SA.issparse(C)
        # Zero-copy in both directions (row indices are always sorted).
        W=CSCMatrix(S); @test nonzeros(W)===nonzeros(S)
        @test nonzeros(SparseMatrixCSC(C))===C.values
        # copy/similar/zero keep the CSCMatrix type.
        @test copy(C) isa CSCMatrix{T,Int} && copy(C)==C && nonzeros(copy(C))!==C.values
        @test similar(C) isa CSCMatrix{T,Int} && similar(C).colptr==C.colptr
        @test zero(C) isa CSCMatrix{T,Int} && nnz(zero(C))==0 && size(zero(C))==size(C)
    end
    # The validating constructor sorts row indices within each column.
    C=CSCMatrix(3,2,[1,3,5],[3,1,2,1],[30.0,10.0,20.0,1.0])
    @test C.rowind==[1,3,1,2] && C.values==[10.0,30.0,1.0,20.0] && Matrix(C)==[10 1;0 20;30 0]
    @test_throws ArgumentError CSCMatrix(2,1,[1,3],[1,1],[1.0,2.0])   # duplicate entry
    # OxiBLAS kernels are selected for CSCMatrix.
    C=CSCMatrix(SA.sprand(20,20,0.2)); x=randn(20); y=zeros(20)
    @test which(sparse_mul!,Tuple{typeof(y),typeof(C),typeof(x)}).module===OxiBLAS
    @test sparse_mul!(y,C,x)≈Matrix(C)*x
end

@testset "Other formats and SparseTensor in the SparseArrays hierarchy" begin
    S=SA.sprand(12,10,0.3); R=CSRMatrix(S)
    for (A,Ti) in ((R,Int),(COOMatrix(S),Int),(DIAMatrix(R),Int),(ELLMatrix(R),Int),(BSRMatrix(R;blocksize=(3,2)),Int),
                   (BSCMatrix(R),Int),(HYBMatrix(R),Int),(SELLMatrix(R),Int))
        @test A isa SA.AbstractSparseMatrix{Float64,Ti} && SA.issparse(A) && SA.indtype(A)===Ti
        @test Array(A)==Matrix(S)
        # OxiBLAS kernels stay selected.
        x=randn(10); y=zeros(12)
        @test which(sparse_mul!,Tuple{typeof(y),typeof(A),typeof(x)}).module===OxiBLAS && sparse_mul!(y,A,x)≈Matrix(S)*x
        # zero is sparse (same format for CSR/COO, SparseMatrixCSC otherwise).
        Z=zero(A); @test SA.issparse(Z) && iszero(Z) && size(Z)==size(A)
        A isa Union{CSRMatrix,COOMatrix} && @test typeof(Z)==typeof(A)
        # Concatenation goes through SparseArrays' sparse concatenation.
        H=hcat(A,A); @test SA.issparse(H) && H==hcat(Matrix(S),Matrix(S))
    end
    # SparseTensor: order 1 is an AbstractSparseVector, order 2 an AbstractSparseMatrix.
    x=SparseTensor((10,),[2,5,9],[1.0,-2.0,3.0]); v=Array(x)
    @test x isa SA.AbstractSparseVector{Float64,Int} && SparseTensor(S) isa SA.AbstractSparseMatrix{Float64,Int}
    @test SparseTensor(zeros(2,2,2)) isa SA.AbstractSparseArray{Float64,Int,3}
    @test SA.nonzeroinds(x)==[2,5,9] && nonzeros(x)==[1.0,-2.0,3.0] && nnz(x)==3
    @test sum(x)==sum(v) && LA.norm(x)≈LA.norm(v) && LA.dot(x,x)≈LA.dot(v,v) && x[3:6]==v[3:6]
    C=CSCMatrix(SA.sprand(10,10,0.3))
    for A in (C,R[1:10,:] |> Matrix |> sparse |> CSRMatrix,SparseTensor(SA.sprand(10,10,0.3)))
        P=A*x; @test P isa SparseTensor{Float64,1} && Array(P)≈Array(A)*v
        @test which(*,Tuple{typeof(A),typeof(x)}).module===OxiBLAS
    end
end

@testset "Base.* on OxiBLAS sparse types" begin
    S=SA.sprand(12,12,0.25)+LA.I; x=randn(12); X=randn(12,3); D=LA.Diagonal(randn(12))
    U=LA.UpperTriangular(randn(12,12)); H=LA.Hermitian(randn(12,12))
    formats=(CSCMatrix(copy(S)),CSRMatrix(S),COOMatrix(S),ELLMatrix(CSRMatrix(S)),DIAMatrix(CSRMatrix(S)),
             BSRMatrix(CSRMatrix(S);blocksize=(3,3)),SELLMatrix(CSRMatrix(S)),HYBMatrix(CSRMatrix(S)),SparseTensor(S))
    M=Matrix(S)
    for A in formats
        cases=((A,x),(A,X),(X',A),(A,A),(transpose(A),x),(A',X),(A,S),(S,A),(D,A),(A,D),(A,U),(U,A),(A,H),(A,sparsevec([1,5],[1.0,2.0],12)),
               (LA.UpperTriangular(A),x),(LA.Symmetric(A),X),(LA.LowerTriangular(A),A),(LA.Hermitian(A),A))
        for (L,R) in cases
            ref=(L===A ? M : L==transpose(A) ? transpose(M) : L==A' ? M' : Array(L))*(R===A ? M : Array(R))
            P=L*R
            @test Array(P)≈ref
            @test which(*,Tuple{typeof(L),typeof(R)}).module===OxiBLAS
        end
        # Shapes and storage of the results.
        @test (A*x) isa Vector && (x'*A) isa LA.Adjoint{<:Any,<:AbstractVector} && (x'*A)≈x'*M
        P=A*A; @test SA.issparse(P) || P isa SparseTensor || P isa CSRMatrix
        @test (A*S) isa SparseMatrixCSC && (S*A) isa SparseMatrixCSC
        P=2A; @test typeof(P)==typeof(A) && Array(P)≈2M
        @test Array(A*3)≈3M
    end
    C=CSCMatrix(copy(S)); R=CSRMatrix(S)
    @test (C*C) isa CSCMatrix && (R*R) isa CSRMatrix && (C*R) isa CSRMatrix
    @test (D*C) isa CSCMatrix && (C*D) isa CSCMatrix && (D*R) isa CSRMatrix
    @test nnz(2C)==nnz(C) && (2.0im*C) isa CSCMatrix{ComplexF64}
    @test COOMatrix(S)*0.0 isa COOMatrix   # canonical COO drops the zeros
end

@testset "Base/LinearAlgebra operations end in OxiBLAS kernels" begin
    S=SA.sprand(30,30,0.15)+LA.I; M=Matrix(S); x=randn(30); X=randn(30,3)
    formats=(CSRMatrix(S),CSCMatrix(copy(S)),COOMatrix(S),ELLMatrix(CSRMatrix(S)),DIAMatrix(CSRMatrix(S)),
             BSRMatrix(CSRMatrix(S);blocksize=(3,3)),HYBMatrix(CSRMatrix(S)),SELLMatrix(CSRMatrix(S)))
    for A in formats
        # LinearAlgebra.mul! forwards to OxiBLAS.mul!.
        @test LA.mul!(zeros(30),A,x)≈M*x && LA.mul!(ones(30,3),A,X,2.0,1.0)≈2M*X.+1
        @test LA.mul!(zeros(30,3),A',X)≈M'*X && LA.mul!(zeros(3,30),X',A)≈X'*M
        @test which(LA.mul!,Tuple{Vector{Float64},typeof(A),Vector{Float64},Bool,Bool}).module===OxiBLAS
        # +, -, negation, division, transposes, permutation: sparse results.
        for (P,R) in ((A+A,2M),(A-S,zeros(30,30)),(S+A,2M),(-A,-M),(A/2,M/2),(copy(A),M),(copy(transpose(A)),transpose(M)),
                      (copy(A'),M'),(permutedims(A),permutedims(M)),(SA.permute(A,collect(30:-1:1),collect(1:30)),M[30:-1:1,:]))
            @test SA.issparse(P) && Array(P)≈R
        end
        @test which(+,Tuple{typeof(A),typeof(A)}).module===OxiBLAS
        @test typeof(copy(A))==typeof(A) && typeof(-A)==typeof(A) && typeof(A/2)==typeof(A)
        # Norms.
        @test LA.norm(A)≈LA.norm(M) && LA.norm(A,1)≈LA.norm(M,1) && LA.norm(A,Inf)≈LA.norm(M,Inf) && LA.norm(A,3)≈LA.norm(M,3)
        @test LA.opnorm(A,1)≈LA.opnorm(M,1) && LA.opnorm(A,Inf)≈LA.opnorm(M,Inf)
        @test_throws ArgumentError LA.opnorm(A,2)
    end
    C=CSCMatrix(copy(S)); R=CSRMatrix(S)
    @test (C+C) isa CSCMatrix && (R+R) isa CSRMatrix && (C+R) isa CSRMatrix && (C+S) isa SparseMatrixCSC
    @test copy(transpose(C)) isa CSCMatrix && permutedims(R) isa CSRMatrix
    Z=CSRMatrix(SparseMatrixCSC{ComplexF64}(S.*(1+2im))); @test conj(Z) isa CSRMatrix && Array(conj(Z))≈conj(Array(Z))
    @test (CSRMatrix(S)+CSRMatrix(SparseMatrixCSC{Float32}(S))) isa CSRMatrix{Float64}
    T=SparseTensor(randn(3,4,2)); @test -T isa SparseTensor && Array(-T)==-Array(T) && LA.norm(T)≈LA.norm(Array(T))
end

@testset "No dispatch ambiguity involving OxiBLAS" begin
    amb=Test.detect_ambiguities(OxiBLAS,SA,LA,Base;recursive=true)
    @test isempty(filter(p->p[1].module===OxiBLAS || p[2].module===OxiBLAS,amb))
end
