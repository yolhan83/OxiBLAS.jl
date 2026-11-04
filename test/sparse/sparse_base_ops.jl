# Base/LinearAlgebra structural and elementwise operations on the OxiBLAS
# sparse types (src/sparse/sparse_base_ops.jl): results against dense references,
# result formats, and the method selected (OxiBLAS, not a generic fallback).
import LinearAlgebra as LA
import SparseArrays as SA
using SparseArrays: SparseMatrixCSC, sparse, nnz, findnz, dropzeros, dropzeros!, droptol!

_fam(A,R)=A isa CSCMatrix ? R isa CSCMatrix : A isa COOMatrix ? R isa COOMatrix : A isa SparseTensor ? R isa SparseTensor : R isa CSRMatrix
_oxi(f,args...)=which(f,Tuple{map(typeof,args)...}).module===OxiBLAS

@testset "Sparse structural and elementwise operations $T" for T in (Float64,ComplexF64)
    S=SparseMatrixCSC{T,Int}(SA.sprand(9,7,0.35)); M=Matrix(S)
    formats=(CSRMatrix(S),CSCMatrix(copy(S)),COOMatrix(S),SparseTensor(S),ELLMatrix(CSRMatrix(S)),DIAMatrix(CSRMatrix(S)))
    for A in formats
        mutable=A isa Union{CSRMatrix,CSCMatrix,COOMatrix,SparseTensor}
        # Indexing.
        @test A[2:5,3:6]==M[2:5,3:6] && SA.issparse(A[2:5,3:6]) && _oxi(getindex,A,2:5,3:6)
        @test A[[3,1,3],:]==M[[3,1,3],:] && A[:,[2,2]]==M[:,[2,2]]
        @test Array(A[:,3])==M[:,3] && A[:,3] isa SparseTensor{T,1} && Array(A[4,:])==M[4,:]
        A isa SparseTensor || @test _fam(A,A[1:3,1:3])
        # Broadcast and map: zero-preserving functions stay sparse.
        for (R,ref) in ((A.*A,M.*M),(abs.(A),abs.(M)),(2 .*A,2 .*M),(A.*2.5,M.*2.5),(map(abs2,A),map(abs2,M)),(real.(A),real.(M)),(A.*S,M.*M))
            @test SA.issparse(R) && Array(R)≈ref
        end
        @test (A.+1)≈M.+1 && (A.+1) isa Matrix          # f(0)≠0: dense by nature
        @test (A.*ones(T,9,7))≈M && (A.+ones(T,9)) ≈ M.+1  # dense operands: dense result
        @test Base.Broadcast.BroadcastStyle(typeof(A)) isa OxiBLAS.OxiSparseStyle
        # Reductions.
        @test sum(A)≈sum(M) && sum(abs,A)≈sum(abs,M) && sum(A;dims=1)≈sum(M;dims=1) && sum(A;dims=2)≈sum(M;dims=2)
        @test maximum(abs,A)≈maximum(abs,M) && minimum(real,A)≈minimum(real,M) && prod(x->x+1,A)≈prod(x->x+1,M)
        @test count(!iszero,A)==count(!iszero,M) && count(!iszero,A;dims=1)==count(!iszero,M;dims=1)
        @test any(x->abs(x)>0.5,A)==any(x->abs(x)>0.5,M) && all(x->abs(x)<2,A)==all(x->abs(x)<2,M) && !iszero(A) && iszero(0 .*A)
        @test _oxi(mapreduce,identity,+,A) && _oxi(count,!iszero,A)
        # Comparison, search, stored entries.
        @test A==M && A==S && S==A && A==copy(A) && !(A==2 .*A)
        @test findall(!iszero,A)==findall(!iszero,M)
        A isa Union{CSRMatrix,CSCMatrix,COOMatrix} && @test findnz(A)==findnz(S)
        # Triangles, diagonal, trace, symmetry.
        @test LA.triu(A)==LA.triu(M) && LA.tril(A,-1)==LA.tril(M,-1) && SA.issparse(LA.triu(A))
        @test Array(LA.diag(A))==LA.diag(M) && Array(LA.diag(A,2))==LA.diag(M,2) && LA.diag(A,-1) isa SparseTensor
        Q=A[1:7,1:7]; @test LA.tr(Q)≈LA.tr(M[1:7,1:7]) && !LA.issymmetric(Q)
        H=Q+copy(Q'); @test LA.ishermitian(H) && LA.issymmetric(H)==LA.issymmetric(Matrix(H))
        # Inner products.
        x=randn(T,9); y=randn(T,7)
        @test LA.dot(A,A)≈LA.dot(M,M) && LA.dot(A,S)≈LA.dot(M,M) && LA.dot(A,M)≈LA.dot(M,M) && LA.dot(x,A,y)≈LA.dot(x,M,y)
        # Kronecker product, concatenation, reversal, reshape.
        B=A[1:3,1:2]; MB=M[1:3,1:2]
        @test LA.kron(B,B)≈LA.kron(MB,MB) && SA.issparse(LA.kron(B,B)) && LA.kron(B,MB)≈LA.kron(MB,MB)
        @test hcat(A,A)==hcat(M,M) && vcat(A,A)==vcat(M,M) && hcat(A,M)==hcat(M,M) && SA.issparse(hcat(A,A))
        @test reverse(A;dims=1)==reverse(M;dims=1) && reverse(A)==reverse(M) && circshift(A,(1,2))==circshift(M,(1,2))
        @test Array(vec(A))==vec(M) && vec(A) isa SparseTensor{T,1} && reshape(A,7,9)==reshape(M,7,9) && Array(reshape(A,3,3,7))==reshape(M,3,3,7)
        # Dense conversion, zero, similar.
        @test Matrix(A)==M && Array(A)==M && copyto!(zeros(T,9,7),A)==M
        @test iszero(zero(A)) && SA.issparse(zero(A)) && SA.issparse(similar(A,T,(4,4))) && similar(A,T,(2,2,2)) isa SparseTensor
        # Mutation: setindex!, dropzeros!, droptol!, triu!, in-place broadcast.
        if mutable
            C=copy(A); i,j=findfirst(iszero,M).I
            C[i,j]=T(7); @test C[i,j]==7 && count(!iszero,C)==count(!iszero,M)+1
            C[i,j]=zero(T); @test C==M
            M2=copy(M); M2[1,1]=0; D=copy(A); D[1,1]=zero(T); dropzeros!(D)
            @test D==M2 && nnz(D)==count(!iszero,M2)
            E=droptol!(copy(A),0.5); @test Array(E)≈M.*(abs.(M).>0.5)
            F=copy(A); LA.triu!(F); @test F==LA.triu(M)
            G=copy(A); G.=abs.(G); @test Array(G)≈abs.(M)
            Z=zeros(T,9,7); Z.=A.*2; @test Z≈2M
        end
        @test Array(dropzeros(A))==M
    end
    # Result formats.
    C=CSCMatrix(copy(S)); R=CSRMatrix(S)
    @test (C.*C) isa CSCMatrix && (R.*R) isa CSRMatrix && C[2:4,:] isa CSCMatrix && LA.triu(C) isa CSCMatrix && hcat(C,C) isa CSCMatrix
    @test LA.kron(C,C) isa CSCMatrix && reverse(C) isa CSCMatrix && dropzeros(C) isa CSCMatrix
    Ts=SparseTensor(S); @test LA.opnorm(Ts,1)≈LA.opnorm(Matrix(S),1) && LA.opnorm(Ts,Inf)≈LA.opnorm(Matrix(S),Inf)
end

@testset "SparseTensor operations of higher order" begin
    for dims in ((4,3,5),(3,2,2,3,2))
        D=zeros(dims); foreach(i->rand()<0.3 && (D[i]=randn()),eachindex(D)); X=SparseTensor(D)
        Y=SparseTensor(D.*2 .- (D.!=0))
        @test Array(X+Y)≈D+Array(Y) && (X+Y) isa SparseTensor && Array(X-Y)≈D-Array(Y) && Array(-X)==-D
        @test Array(X.*Y)≈D.*Array(Y) && Array(abs.(X))==abs.(D) && (abs.(X)) isa SparseTensor
        @test sum(X)≈sum(D) && sum(X;dims=2)≈sum(D;dims=2) && maximum(X)==maximum(D) && count(!iszero,X)==count(!iszero,D)
        @test LA.norm(X)≈LA.norm(D) && LA.dot(X,Y)≈LA.dot(D,Array(Y)) && X==SparseTensor(copy(D)) && X!=Y
        I=(2:3,:,ntuple(_->1,length(dims)-2)...)
        @test Array(X[I...])==D[I...] && X[2,1,ntuple(_->1,length(dims)-2)...]==D[2,1,ntuple(_->1,length(dims)-2)...]
        @test Array(reverse(X;dims=1))==reverse(D;dims=1) && Array(cat(X,X;dims=3))==cat(D,D;dims=3)
        @test Array(zero(X))==zero(D) && zero(X) isa SparseTensor && (X/2) isa SparseTensor && Array(X/2)==D/2
    end
    # Work in nnz: 100 stored entries in 10^9 positions.
    dims=(1000,1000,1000); idx=sort(unique(rand(1:prod(dims),100))); X=SparseTensor(dims,idx,randn(length(idx)))
    @test sum(X)≈sum(X.values) && (X+X).values≈2X.values && LA.norm(X)≈LA.norm(X.values) && abs.(X).values==abs.(X.values)
    @test count(!iszero,X)==length(idx) && LA.dot(X,X)≈sum(abs2,X.values) && X==copy(X) && maximum(abs,X)==maximum(abs,X.values)
    @test nnz(X[1:500,:,:])==count(i->(i-1)%1000+1<=500,idx)
end
