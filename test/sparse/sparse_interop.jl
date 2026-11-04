# SparseArrays interoperability, SparseTensor, and sparse einsum/contract.
import SparseArrays as SA
using SparseArrays: SparseMatrixCSC, SparseVector, sparse, sparsevec, nnz, nonzeros, rowvals

# Random sparse test data of any element type (sprand is native only).
function _rand_sparse(::Type{T},m,n,density=0.3) where T
    I=Int[];J=Int[];V=T[]
    for j in 1:n, i in 1:m
        rand()<density && (push!(I,i);push!(J,j);push!(V,T(randn(real(T)<:BigFloat ? Float64 : real(T)))+(T<:Complex ? T(0,1)*randn() : 0)))
    end
    sparse(I,J,V,m,n)
end
function _rand_dense_sparse(::Type{T},dims,density=0.4) where T
    A=zeros(T,dims)
    for i in eachindex(A)
        rand()<density && (A[i]=T<:Complex ? T(randn()+im*randn()) : T(randn()))
    end
    A
end
_close(a,b;T=eltype(b))=maximum(abs,Array(a)-Array(b);init=zero(real(T)))<=1000*eps(real(T))*max(1,maximum(abs,Array(b);init=zero(real(T))))

@testset "SparseArrays conversions" begin
    for T in (TYPES...,BigFloat), (m,n) in ((0,0),(0,4),(5,0),(1,1),(7,4),(4,9),(13,13))
        S=_rand_sparse(T,m,n)
        C=CSCMatrix(S)
        @test C isa CSCMatrix{T,Int} && size(C)==(m,n) && C==S
        # Zero copy: CSC shares SparseMatrixCSC's vectors; so does the CSR of transpose(S).
        @test C.colptr===SA.getcolptr(S) && C.rowind===rowvals(S) && C.values===nonzeros(S)
        R=CSRMatrix(transpose(S))
        @test R==transpose(S) && R.rowptr===SA.getcolptr(S) && R.values===nonzeros(S)
        @test CSRMatrix(S)==S && COOMatrix(S)==S && CSRMatrix(S') ≈ Matrix(S')
        @test SparseMatrixCSC(C)==S && SparseMatrixCSC(CSRMatrix(S))==S && SparseMatrixCSC(COOMatrix(S))==S
        @test sparse(CSRMatrix(S))==S && convert(SparseMatrixCSC,CSRMatrix(S))==S && convert(CSCMatrix,S)==S
        @test SparseMatrixCSC(transpose(CSRMatrix(S)))==transpose(S) && SparseMatrixCSC(adjoint(CSCMatrix(S)))==adjoint(S)
        @test SparseMatrixCSC(C).colptr===C.colptr   # sorted CSC is shared back
        if T<:OxiBLAS.NativeFloat && m>0 && n>0
            for F in (DIAMatrix,ELLMatrix,BSRMatrix,HYBMatrix,SELLMatrix)
                @test SparseMatrixCSC(F(S))==SA.dropzeros(S)
            end
        end
        # Index type preserved, or converted on request.
        S32=SparseMatrixCSC{T,Int32}(S)
        @test CSCMatrix(S32) isa CSCMatrix{T,Int32} && CSRMatrix{T,Int32}(S) isa CSRMatrix{T,Int32} && SparseMatrixCSC{T,Int32}(CSRMatrix(S)) isa SparseMatrixCSC{T,Int32}
        # Tensor view of the same entries.
        X=SparseTensor(S); @test X==S && nnz(X)==nnz(S) && SparseMatrixCSC(X)==S && CSRMatrix(X)==S
    end
    # Explicit stored zeros survive compressed conversions; COO is canonical.
    S=SparseMatrixCSC(3,3,[1,3,4,5],[1,3,2,3],[1.0,0.0,2.0,0.0])
    @test nnz(CSCMatrix(S))==4 && nnz(CSRMatrix(S))==4 && nnz(SparseMatrixCSC(CSRMatrix(S)))==4 && nnz(SparseTensor(S))==4
    @test nnz(COOMatrix(S))==2
    # Duplicates of user COO data are summed (COOMatrix and SparseTensor).
    @test COOMatrix(2,2,[1,1,2],[1,1,2],[1.0,2.0,3.0])[1,1]==3 && SparseTensor((2,2),[(1,1),(1,1)],[1.0,2.0])[1,1]==3
    # Unsorted CSC rows are sorted for SparseMatrixCSC.
    U=CSCMatrix(3,1,[1,3],[3,1],[5.0,7.0]); V=SparseMatrixCSC(U)
    @test V==Matrix(U) && issorted(rowvals(V))
    for T in TYPES
        v=sparsevec([2,5,9],T[1,2,3],10)
        @test CSCMatrix(v)==reshape(Vector(v),:,1) && CSCMatrix(v).rowind===SA.nonzeroinds(v)
        @test SparseVector(CSCMatrix(v))==v && CSRMatrix(transpose(v))==transpose(v) && SparseVector(CSRMatrix(transpose(v)))==v
        @test SparseTensor(v)==v && SparseVector(SparseTensor(v))==v && sparse(SparseTensor(v))==v
        @test CSRMatrix(v')≈Matrix(v')
    end
    # Hypersparse: no conversion may touch m*n storage.
    H=sparse([1,10^6],[1,10^6],[1.0,2.0],10^6,10^6)
    @test nnz(CSRMatrix(H))==2 && nnz(SparseTensor(H))==2 && SparseMatrixCSC(COOMatrix(H))==H
end

@testset "OxiBLAS sparse routines on SparseArrays" begin
    for T in TYPES
        n=30; S=_rand_sparse(T,n,n,0.15)+SA.spdiagm(0=>fill(T(n),n)); x=randn(T,n); X=randn(T,n,3)
        y=zeros(T,n); sparse_mul!(y,S,x); @test y≈Matrix(S)*x
        Y=zeros(T,n,3); sparse_mul!(Y,S,X); @test Y≈Matrix(S)*X
        P=sparse_matmul(S,S); @test P isa SparseMatrixCSC{T} && P≈S*S
        @test sparse_matmul(S,CSRMatrix(S))≈Matrix(S)*Matrix(S)
        F=sparse_lu(S); @test Matrix(S)*solve(F,x)≈x
        H=S'S; G=sparse_cholesky(H); @test Matrix(H)*solve(G,x)≈x
        @test Matrix(S)*gmres(S,x;rtol=real(T)(1e-6)).x≈x rtol=1e-4   # default rtol is sqrt(eps): 3.5e-4 in Float32
        @test Matrix(H)*cg(H,x;rtol=1e-10).x≈x rtol=1e-4
        @test sparse_transpose(S)≈transpose(Matrix(S)) && sparse_norm(S)≈sqrt(sum(abs2,nonzeros(S)))
        @test sort(reverse_cuthill_mckee(S))==1:n && ilu0(S) isa ILUFactor
    end
    S=_rand_sparse(BigFloat,6,6)+SA.spdiagm(0=>fill(big(6.0),6)); x=BigFloat.(randn(6))
    y=zeros(BigFloat,6); sparse_mul!(y,S,x); @test y≈Matrix(S)*x
    @test CSRMatrix(S) isa CSRMatrix{BigFloat} && sparse_matmul(S,S)≈Matrix(S)^2
end

@testset "SparseTensor" begin
    for T in (TYPES...,BigFloat)
        D=_rand_dense_sparse(T,(3,4,5)); X=SparseTensor(D)
        @test X==D && Array(X)==D && nnz(X)==count(!iszero,D) && size(X)==(3,4,5)
        @test permutedims(X,(3,1,2))==permutedims(D,(3,1,2)) && reshape(X,(12,5))==reshape(D,12,5)
        @test conj(X)==conj.(D) && copy(X)==X && SA.dropzeros(X)==X
        coords,vals=SA.findnz(X); @test all(D[c...]==v for (c,v) in zip(coords,vals))
        Y=SparseTensor((2,3),[1 2;1 3],T[4,5]); @test Y[1,1]==4 && Y[2,3]==5 && nnz(Y)==2
        Y[2,2]=zero(T); @test nnz(Y)==3 && Y[2,2]==0      # explicit zero stored
        @test 2*SparseTensor(D)==2D
    end
    @test_throws ArgumentError SparseTensor((2,2),[(3,1)],[1.0])
end

@testset "Sparse einsum and contract" begin
    specs=["ij,jk->ik","ij,jk->ki","ijk,jl->ilk","ii->","ii->i","ij->ji","ij->","ijk->kji","ij,ij->ij","ij,ij->",
           "ij,kl->ijkl","ijk,ikl->ijl","abc,cde->abde","iij,jk->ik","ij,j->i","i,i->","i,j->ij","ij,jk,kl->il","ijk,jkl,lm->im"]
    for T in (Float64,ComplexF32,BigFloat), spec in specs
        inputs=split(split(spec,"->")[1],','); sizes=Dict('i'=>3,'j'=>4,'k'=>2,'l'=>5,'m'=>3,'a'=>2,'b'=>3,'c'=>4,'d'=>2,'e'=>3)
        dense=[_rand_dense_sparse(T,Tuple(sizes[c] for c in s)) for s in inputs]
        reference=einsum(spec,dense...)
        for mask in 0:2^length(dense)-1
            ops=Any[isodd(mask>>(p-1)) ? (ndims(A)==2 && T<:OxiBLAS.NativeFloat && iseven(p) ? CSRMatrix(A) : ndims(A)==2 ? sparse(A) : ndims(A)==1 ? sparsevec(A) : SparseTensor(A)) : A for (p,A) in enumerate(dense)]
            for optimize in (length(ops)>2 ? (:greedy,:left,:none,:optimal) : (:greedy,))
                R=einsum(spec,ops...;optimize)
                @test _close(R,reference;T)
                allsparse=all(OxiBLAS._is_sparse_operand,ops)
                @test (R isa SparseTensor)==(allsparse && ndims(reference)>0)
            end
            @test einsum(spec,ops...;output=:dense) isa Array && _close(einsum(spec,ops...;output=:dense),reference;T)
            @test einsum(spec,ops...;output=:sparse) isa SparseTensor && _close(einsum(spec,ops...;output=:sparse),reference;T)
        end
    end
    for T in TYPES
        A=_rand_dense_sparse(T,(6,5)); B=_rand_dense_sparse(T,(5,4)); SA_=sparse(A); SB=sparse(B)
        # Transposed/adjoint sparse operands; broadcasting ellipsis with a sparse operand.
        @test _close(einsum("ji,jk->ik",transpose(SA_),B),transpose(transpose(A))*B;T)
        @test _close(einsum("ij,jk->ik",SA_',sparse(A)),A'*A;T)
        X=_rand_dense_sparse(T,(2,1,3,4)); Y=_rand_dense_sparse(T,(1,5,4,6))
        @test _close(einsum("...ij,...jk->...ik",SparseTensor(X),Y),einsum("...ij,...jk->...ik",X,Y);T)
        # Plans, workspaces and reports keep the sparse information.
        P=plan_einsum("ij,jk->ik",SA_,B)
        @test P.sparse_operands==[true,false] && P.operand_nnz==[nnz(SA_),0] && einsum_layout_report(P).sparse_operands==[true,false]
        @test _close(einsum(P,SA_,B),A*B;T) && _close(einsum(P,A,SB),A*B;T)          # same shapes, other kinds
        W=EinsumWorkspace(T); @test _close(einsum(P,SA_,B;workspace=W),A*B;T)
        Q=plan_einsum_sliced("ij,jk->ik",SA_,B;chunks=(i=2,k=3)); @test _close(einsum(Q,SA_,B),A*B;T)
        Q=plan_einsum_sliced("ij,jk->ik",SA_,SB;chunks=(j=2,)); @test _close(einsum(Q,SA_,SB),A*B;T)
        # Mutating forms: dense and sparse destinations, alpha/beta, aliasing.
        C=randn(T,6,4); old=copy(C); einsum!(C,"ij,jk->ik",SA_,SB;α=T(2),β=T(-1)); @test _close(C,2A*B-old;T)
        for dest in (SparseTensor(_rand_dense_sparse(T,(6,4))),sparse(_rand_dense_sparse(T,(6,4))),CSRMatrix(_rand_dense_sparse(T,(6,4))),CSCMatrix(_rand_dense_sparse(T,(6,4))))
            old=Array(dest); einsum!(dest,"ij,jk->ik",SA_,B;α=T(3),β=T(1//2)); @test _close(dest,3A*B+old/2;T)
        end
        v=sparsevec(_rand_dense_sparse(T,(6,))); old=Vector(v); einsum!(v,"ij,j->i",SA_,B[:,1]); @test _close(v,A*B[:,1];T)
        M=sparse(_rand_dense_sparse(T,(5,5))); MM=Matrix(M); einsum!(M,"ij,jk->ik",M,M); @test _close(M,MM*MM;T)   # destination is an input
        # contract / contract! / outer_product / mode_product.
        X3=_rand_dense_sparse(T,(2,3,4)); Y3=_rand_dense_sparse(T,(4,5,2)); S3=SparseTensor(X3)
        ref=contract(X3,Y3,(3,),(1,))
        @test _close(contract(S3,Y3,(3,),(1,)),ref;T) && _close(contract(S3,SparseTensor(Y3),(3,),(1,)),ref;T) && _close(contract(X3,SparseTensor(Y3),(3,),(1,)),ref;T)
        @test contract(S3,SparseTensor(Y3),(3,),(1,)) isa SparseTensor
        @test _close(contract(S3,Y3,(3,1),(1,3);conjugateA=true),contract(conj.(X3),Y3,(3,1),(1,3));T)
        @test _close(contract(S3,SparseTensor(Y3),(3,),(1,);conjugateB=true),contract(X3,conj.(Y3),(3,),(1,));T)
        @test _close(outer_product(sparse(A),B),outer_product(A,B);T)
        U=randn(T,7,3); @test _close(mode_product(S3,U,2),mode_product(X3,U,2);T)
        C=randn(T,2,3,5,2); old=copy(C); contract!(C,S3,Y3,(3,),(1,);α=T(2),β=T(1)); @test _close(C,2ref+old;T)
        D=SparseTensor(zeros(T,2,3,5,2)); contract!(D,S3,SparseTensor(Y3),(3,),(1,)); @test _close(D,ref;T)
    end
    # Hypersparse 10^6 x 10^6 product stays sparse (a dense path would need terabytes).
    H=sparse([1,7,10^6],[3,10^6,2],[1.0,2.0,3.0],10^6,10^6)
    R=einsum("ij,jk->ik",H,H); @test R isa SparseTensor && nnz(R)==1 && R[7,2]==6
    @test contract(H,H,(2,),(1,))[7,2]==6 && sparse_matmul(H,H)[7,2]==6
end
