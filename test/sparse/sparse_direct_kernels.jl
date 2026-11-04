# Low-level sparse direct kernels: triangular solves, LU (exact, threshold
# incomplete, blocked), Cholesky (exact, IC(0), threshold IC), QR and its Q.
# Residuals against the input, factor invariants, options, edge cases and
# input formats.
import LinearAlgebra as LA
import SparseArrays as SA
using SparseArrays: SparseMatrixCSC, sparse

_sd_rand(::Type{T},n,d) where T=SparseMatrixCSC{T,Int}(SA.sprand(n,n,d)) .+ (T<:Complex ? im .* SparseMatrixCSC{T,Int}(SA.sprand(n,n,d)) : 0)
_sd_unsym(::Type{T},n) where T=(S=SparseMatrixCSC{T,Int}(_sd_rand(T,n,0.12)); S+T(n)*LA.I)
_sd_spd(::Type{T},n) where T=(S=_sd_rand(T,n,0.1); SparseMatrixCSC{T,Int}(S'*S+T(n)*LA.I))
_sd_tol(::Type{T}) where T=real(T)===Float32 ? 1f-3 : 1e-9
_sd_res(A,x,b)=LA.norm(Matrix(A)*x-b)/max(LA.norm(b),1)
_sd_formats(S)=(S,CSRMatrix(S),CSCMatrix(copy(S)),COOMatrix(S),ELLMatrix(CSRMatrix(S)))

@testset "sparse_triangular_solve $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    n=25; S=_sd_unsym(T,n); b=randn(T,n); B=randn(T,n,3)
    for (uplo,Tri) in ((:L,SA.tril),(:U,SA.triu)), A in _sd_formats(SparseMatrixCSC{T,Int}(Tri(S)))
        M=Matrix(A)
        for trans in (:N,:T,:C), unit in (false,true)
            ref=unit ? (uplo===:L ? LA.UnitLowerTriangular(M) : LA.UnitUpperTriangular(M)) : (uplo===:L ? LA.LowerTriangular(M) : LA.UpperTriangular(M))
            op=trans===:N ? ref : trans===:T ? transpose(ref) : adjoint(ref)
            x=sparse_triangular_solve(A,b;uplo,trans,unit); X=sparse_triangular_solve(A,B;uplo,trans,unit)
            @test LA.norm(op*x-b)<=_sd_tol(T)*LA.norm(b) && LA.norm(op*X-B)<=_sd_tol(T)*LA.norm(B)
        end
        @test sparse_triangular_solve(A,copy(b);uplo)≈(uplo===:L ? LA.LowerTriangular(M) : LA.UpperTriangular(M))\b rtol=_sd_tol(T)
    end
    # Edge cases: empty, 1x1, zero pivot, mismatched sizes or types.
    @test sparse_triangular_solve(CSRMatrix(SparseMatrixCSC{T,Int}(SA.spzeros(0,0))),T[])==T[]
    @test sparse_triangular_solve(CSRMatrix(sparse(T[2;;])),T[4])≈T[2]
    Z=CSRMatrix(sparse(T[1 0;1 0])); @test_throws OxiBLAS.SingularException sparse_triangular_solve(Z,T[1,1])
    @test sparse_triangular_solve(Z,T[1,1];unit=true)≈T[1,0]          # unit: the diagonal is not read
    @test_throws DimensionMismatch sparse_triangular_solve(CSRMatrix(sparse(T[1;;])),T[1,2])
    real(T)===Float64 && @test_throws DimensionMismatch sparse_triangular_solve(CSRMatrix(sparse(Float32[1;;])),T[1])
end

@testset "sparse_lu $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    n=40; S=_sd_unsym(T,n); b=randn(T,n); B=randn(T,n,4)
    for A in _sd_formats(S), ordering in (:auto,:natural,:amd_ata,collect(n:-1:1))
        F=sparse_lu(A;ordering)
        # L unit lower, U upper, L*U = A[p,q].
        @test LA.istril(Matrix(F.L)) && all(LA.diag(Matrix(F.L)).==1) && LA.istriu(Matrix(F.U)) && sort(F.p)==sort(F.q)==1:n
        @test Matrix(F.L)*Matrix(F.U)≈Matrix(S)[F.p,F.q] rtol=10_sd_tol(T)
        ordering===:natural && @test F.q==1:n
        ordering isa Vector && @test F.q==ordering
        @test _sd_res(S,solve(F,b),b)<=_sd_tol(T) && _sd_res(S,solve(F,B),B)<=_sd_tol(T)
    end
    # Pivoting is required: zero leading diagonal.
    P=sparse(T[0 1 0;1 0 2;0 3 1])
    for ordering in (:auto,:natural)
        Fp=sparse_lu(P;ordering); x=solve(Fp,T[1,2,3])
        @test _sd_res(P,x,T[1,2,3])<=_sd_tol(T)
        ordering===:natural && @test Fp.p!=1:3
    end
    # Threshold pivoting keeps the diagonal when it is large enough.
    D=sparse(T[2 1;3 1]); @test sparse_lu(D;pivot_threshold=0.5,ordering=:natural).p==[1,2] && sparse_lu(D;ordering=:natural).p==[2,1]
    @test_throws ArgumentError sparse_lu(S;ordering=:other)
    @test_throws ArgumentError sparse_lu(S;droptol=0.1,ordering=:amd_ata)
    # Threshold incomplete LU (ILUT): sparser factors, still a good preconditioner.
    F=sparse_lu(S); Fi=ilut(S;droptol=0.05,maxfill=3)
    @test SA.nnz(SparseMatrixCSC(Fi.L))+SA.nnz(SparseMatrixCSC(Fi.U))<=SA.nnz(SparseMatrixCSC(F.L))+SA.nnz(SparseMatrixCSC(F.U))
    @test _sd_res(S,solve(Fi,b),b)<0.5
    # Edge cases and errors.
    @test solve(sparse_lu(sparse(T[3;;])),T[6])≈T[2]
    @test_throws OxiBLAS.SingularException sparse_lu(sparse(T[1 2;2 4]))
    @test_throws OxiBLAS.SingularException sparse_lu(sparse(T[0 0;0 1]))
    @test_throws ArgumentError sparse_lu(S;droptol=-1)
    @test_throws ArgumentError sparse_lu(S;pivot_threshold=2)
    @test_throws DimensionMismatch sparse_lu(sparse(ones(T,2,3)))
    @test_throws DimensionMismatch solve(F,ones(T,n+1))
end

@testset "sparse_lu_blocked $T" for T in (Float64,ComplexF64)
    n=60; S=_sd_unsym(T,n); b=randn(T,n)
    for bs in (1,7,24,100), threaded in (false,true)
        F=sparse_lu_blocked(S;blocksize=bs,threaded)
        @test _sd_res(S,solve(F,b),b)<=_sd_tol(T)
    end
    @test solve(sparse_lu_blocked(S;threaded=false),b)≈solve(sparse_lu_blocked(S;threaded=true),b)
    @test_throws ArgumentError sparse_lu_blocked(S;blocksize=0)
end

@testset "sparse_cholesky $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    n=40; S=_sd_spd(T,n); b=randn(T,n); B=randn(T,n,3)
    for A in _sd_formats(S)
        F=sparse_cholesky(A); L=Matrix(F.L)
        @test LA.istril(L) && all(isreal,LA.diag(L)) && all(>(0),real(LA.diag(L)))
        @test L*L'≈Matrix(S) rtol=10_sd_tol(T)
        @test _sd_res(S,solve(F,b),b)<=_sd_tol(T) && _sd_res(S,solve(F,B),B)<=_sd_tol(T)
    end
    # Only the requested triangle is read.
    Lw=SparseMatrixCSC{T,Int}(SA.tril(S)); Uw=SparseMatrixCSC{T,Int}(SA.triu(S))
    @test _sd_res(S,solve(sparse_cholesky(Lw;uplo=:L),b),b)<=_sd_tol(T)
    @test _sd_res(S,solve(sparse_cholesky(Uw;uplo=:U),b),b)<=_sd_tol(T)
    # IC(0): the factor keeps the pattern of tril(A); threshold IC drops entries.
    F0=ic0(S); P0=SparseMatrixCSC(F0.L)
    @test all(i->!iszero(Lw[i]),findall(!iszero,P0)) && F0.incomplete
    Ft=ict(S;droptol=0.05); @test SA.nnz(SparseMatrixCSC(Ft.L))<=SA.nnz(SparseMatrixCSC(sparse_cholesky(S).L))
    @test _sd_res(S,solve(F0,b),b)<0.5 && _sd_res(S,solve(Ft,b),b)<0.5
    # Errors.
    @test_throws OxiBLAS.PosDefException sparse_cholesky(sparse(T[1 2;2 1]))
    @test_throws OxiBLAS.PosDefException sparse_cholesky(sparse(T[-1;;]))
    @test_throws ArgumentError sparse_cholesky(S;fill=:other)
    @test_throws ArgumentError sparse_cholesky(S;droptol=-1)
    @test solve(sparse_cholesky(sparse(T[4;;])),T[8])≈T[2]
    # The exact factor goes through the supernodal code: no stored zeros,
    # sorted rows, equal to the up-looking factor; a band (contiguous rows in
    # the solve) and a 2D grid (supernodes, postorder undone).
    lap(k)=SA.spdiagm(0=>fill(T(2),k),1=>fill(T(-1),k-1),-1=>fill(T(-1),k-1))
    G=kron(sparse(T(1)*LA.I,15,15),lap(15))+kron(lap(15),sparse(T(1)*LA.I,15,15))
    for A in (S,lap(60)+T(0.5)*LA.I,G)
        F=sparse_cholesky(A); L=SparseMatrixCSC(F.L); n2=size(A,1); b2=randn(T,n2)
        @test !any(iszero,SA.nonzeros(L)) && all(j->issorted(SA.rowvals(L)[SA.nzrange(L,j)]),1:n2)
        Ls=SparseMatrixCSC(multifrontal_cholesky(A;ordering=:natural,supernodal=false).L)
        @test LA.norm(Matrix(L-Ls))<=_sd_tol(T)*LA.norm(Matrix(Ls))
        @test _sd_res(A,solve(F,b2),b2)<=_sd_tol(T) && _sd_res(A,solve(F,[b2 b2]),[b2 b2])<=_sd_tol(T)
    end
    # The failing pivot is reported at its column of A.
    N=Matrix(G); N[7,7]=-1; e=try sparse_cholesky(sparse(N)) catch err err end
    @test e isa OxiBLAS.PosDefException && e.info==7
end

@testset "sparse_qr $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    m,n=50,20; S=SparseMatrixCSC{T,Int}(_sd_rand(T,m,0.15)[:,1:n])+SparseMatrixCSC{T,Int}(sparse(1:n,1:n,ones(T,n),m,n)); b=randn(T,m); M=Matrix(S)
    for A in (S,CSRMatrix(S),CSCMatrix(copy(S)),COOMatrix(S)), pivot in (true,false)
        F=sparse_qr(A;pivot)
        # Least squares: matches the dense solution, normal-equations residual.
        x=solve(F,b); @test LA.norm(M'*(M*x-b))<=10_sd_tol(T)*LA.norm(M)^2*LA.norm(b)
        @test x≈M\b rtol=100_sd_tol(T)
        X=solve(F,[b b]); @test X[:,1]≈x && X[:,2]≈x
        # Q is unitary: Qᴴ(Q B) = B, and Qᴴ A[:,p] = [R; 0].
        Bq=randn(T,m,3); @test sparse_apply_q!(sparse_apply_q!(copy(Bq),F),F;trans=:C)≈Bq rtol=10_sd_tol(T)
        QtA=sparse_apply_q!(Matrix{T}(M[:,F.p]),F;trans=:C)
        @test QtA[1:n,:]≈Matrix(F.R)[1:n,:] rtol=100_sd_tol(T) atol=100_sd_tol(T)*LA.norm(M)
        @test LA.norm(QtA[n+1:end,:])<=100_sd_tol(T)*LA.norm(M)
        pivot || @test F.p==1:n
    end
    # Square systems, rank deficiency, wide matrices.
    Q=_sd_unsym(T,15); bq=randn(T,15); @test _sd_res(Q,solve(sparse_qr(Q),bq),bq)<=10_sd_tol(T)
    R=sparse(T[1 2;2 4;3 6]); @test_throws OxiBLAS.SingularException solve(sparse_qr(R),T[1,2,3])
    @test_throws DimensionMismatch solve(sparse_qr(sparse(ones(T,2,3))),T[1,2])
    @test_throws DimensionMismatch sparse_apply_q!(ones(T,m+1,1),sparse_qr(S))
    # Multifrontal QR (pivot=false): natural and AMD(AᵀA) orders, a larger
    # tall matrix (blocked fronts), a grid least-squares problem, rank
    # deficiency, and the option checks.
    lap(k)=SA.spdiagm(0=>fill(T(2),k),1=>fill(T(-1),k-1),-1=>fill(T(-1),k-1))
    G=kron(sparse(T(1)*LA.I,12,12),lap(12))+kron(lap(12),sparse(T(1)*LA.I,12,12))
    big=SparseMatrixCSC{T,Int}(_sd_rand(T,600,0.01)[:,1:200])+SparseMatrixCSC{T,Int}(sparse(1:200,1:200,fill(T(3),200),600,200))
    for A in (S,big,[G;SparseMatrixCSC{T,Int}(_sd_rand(T,144,0.02))[1:40,:]]), ordering in (:natural,:amd,collect(size(A,2):-1:1))
        F=sparse_qr(A;pivot=false,ordering); M=Matrix(A); n2=size(A,2); b2=randn(T,size(A,1))
        ordering===:natural && @test F.p==1:n2
        ordering isa Vector && @test F.p==ordering
        x=solve(F,b2); @test LA.norm(M'*(M*x-b2))<=10_sd_tol(T)*LA.norm(M)^2*LA.norm(b2)
        QtA=sparse_apply_q!(Matrix{T}(M[:,F.p]),F;trans=:C)
        @test LA.istriu(Matrix(F.R)[1:n2,:])
        @test QtA[1:n2,:]≈Matrix(F.R)[1:n2,:] rtol=100_sd_tol(T) atol=100_sd_tol(T)*LA.norm(M)
        @test LA.norm(QtA[n2+1:end,:])<=100_sd_tol(T)*LA.norm(M)
        Bq=randn(T,size(A,1),2); @test sparse_apply_q!(sparse_apply_q!(copy(Bq),F),F;trans=:C)≈Bq rtol=10_sd_tol(T)
    end
    @test_throws OxiBLAS.SingularException solve(sparse_qr(R;pivot=false),T[1,2,3])
    @test_throws ArgumentError sparse_qr(S;pivot=false,ordering=:other)
    @test_throws ArgumentError sparse_qr(S;ordering=:amd)
end

@testset "AMD ordering and symbolic analysis" begin
    k=20; L1=SA.spdiagm(0=>fill(2.0,k),1=>fill(-1.0,k-1),-1=>fill(-1.0,k-1)); I1=sparse(1.0LA.I,k,k)
    A=kron(I1,L1)+kron(L1,I1); n=size(A,1)
    for M in (A,CSRMatrix(A),CSCMatrix(copy(A)),COOMatrix(A))
        p=amd_ordering(M); @test sort(p)==1:n
        @test sparse_ordering(M,:amd)==p
    end
    p=amd_ordering(A); Sn=symbolic_cholesky(A;ordering=:natural); Sa=symbolic_cholesky(A;ordering=:amd)
    @test Sa.nnz<Sn.nnz/2 && Sa.permutation==p
    # The symbolic structure is the exact pattern of the numeric factor.
    for S in (Sn,Sa)
        q=S.permutation; F=sparse_cholesky(A[q,q]); Lf=SparseMatrixCSC(F.L)
        @test SA.nnz(Lf)==S.nnz && all(j->sort(findall(!iszero,Lf[:,j]))==S.columns[j],1:n)
        @test all(j->S.parent[j]==(length(S.columns[j])>1 ? S.columns[j][2] : 0),1:n)
        @test sort(S.postorder)==1:n && all(j->S.parent[j]==0 || findfirst(==(j),S.postorder)<findfirst(==(S.parent[j]),S.postorder),1:n)
    end
    # A dense row and column, isolated nodes, tiny and empty matrices.
    D=copy(A); D[1,:].=1; D[:,1].=1; @test sort(amd_ordering(D))==1:n
    @test amd_ordering(sparse(zeros(0,0)))==Int[] && amd_ordering(sparse([1.0;;]))==[1]
    @test sort(amd_ordering(SA.spdiagm(0=>ones(7))))==1:7
    # AᵀA kinds (LU, QR) on rectangular and unsymmetric matrices.
    R=SA.sprand(60,40,0.08)+sparse(1:40,1:40,ones(40),60,40)
    @test sort(amd_ordering(R;kind=:ata))==1:40 && sort(amd_ordering(R;kind=:ata_sparse))==1:40
    @test sort(sparse_ordering(R,:amd_qr))==1:40
    @test_throws DimensionMismatch amd_ordering(R)
    @test_throws ArgumentError amd_ordering(A;kind=:other)
    # Multifrontal solvers with their AMD defaults.
    b=randn(n); @test LA.norm(A*solve(multifrontal_cholesky(A),b)-b)<=1e-9*LA.norm(b)
    U=A+0.3*SA.sprand(n,n,0.01); @test LA.norm(U*solve(multifrontal_lu(U),b)-b)<=1e-8*LA.norm(b)
end

@testset "multifrontal_cholesky paths $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    # Chain (simplicial under :auto), 2D grid (supernodal, small fronts) and
    # a 3D grid whose top fronts exceed the in-place kernel (POTRF/TRSM/HERK).
    lap(k)=SA.spdiagm(0=>fill(T(2),k),1=>fill(T(-1),k-1),-1=>fill(T(-1),k-1))
    I(k)=sparse(T(1)*LA.I,k,k)
    G=kron(I(12),lap(12))+kron(lap(12),I(12))
    V=kron(I(9),kron(I(9),lap(9)))+kron(I(9),kron(lap(9),I(9)))+kron(lap(9),kron(I(9),I(9)))
    S=_sd_spd(T,90)
    for A in (lap(200)+T(0.1)*LA.I,G,V,S), sn in (:auto,true,false), uplo in (:L,:U)
        n=size(A,1); b=randn(T,n); H=SparseMatrixCSC{T,Int}(uplo===:L ? SA.tril(A) : SA.triu(A))
        F=multifrontal_cholesky(H;uplo,supernodal=sn)
        @test _sd_res(A,solve(F,b),b)<=_sd_tol(T)
        L=SparseMatrixCSC(F.L); p=F.p
        @test LA.istril(L) && all(j->isreal(L[j,j]) && real(L[j,j])>0,1:n) && sort(p)==1:n
        @test LA.norm(Matrix(L*L')-Matrix(A[p,p]))<=_sd_tol(T)*LA.norm(Matrix(A))*n
        # Supernodes partition the columns in order.
        @test reduce(vcat,collect.(F.nodes))==1:n
    end
    # A width cap splits the supernodes; the factor is unchanged up to rounding.
    F1=multifrontal_cholesky(V;maxwidth=3,supernodal=true); F2=multifrontal_cholesky(V;supernodal=true)
    @test all(<=(3),length.(F1.nodes)) && F1.p==F2.p && LA.norm(Matrix(F1.L)-Matrix(F2.L))<=_sd_tol(T)*LA.norm(Matrix(F2.L))
    # Simplicial (exact pattern) and supernodal (relaxed, weighted postorder:
    # another equivalent permutation) paths solve alike.
    Fs=multifrontal_cholesky(G;supernodal=false); Fm=multifrontal_cholesky(G;supernodal=true)
    b=randn(T,size(G,1))
    @test SA.nnz(SparseMatrixCSC(Fs.L))<=SA.nnz(SparseMatrixCSC(Fm.L)) && LA.norm(solve(Fs,b)-solve(Fm,b))<=_sd_tol(T)*LA.norm(solve(Fm,b))
    # Not positive definite, at the failing column, on both paths.
    N=copy(G); N[1,1]=T(-1)
    @test_throws PosDefException multifrontal_cholesky(N;supernodal=false)
    @test_throws PosDefException multifrontal_cholesky(N;supernodal=true)
    @test_throws ArgumentError multifrontal_cholesky(G;supernodal=:maybe)
    @test_throws ArgumentError multifrontal_cholesky(G;maxwidth=0)
end

@testset "multifrontal_cholesky subtree parallelism" begin
    # Large enough for the concurrent subtrees (Σ f³ ≥ 2·10⁸): the factor is
    # the sequential one, and a failing pivot inside a subtree is reported
    # at its column of A.
    k=22; L1=SA.spdiagm(0=>fill(2.0,k),1=>fill(-1.0,k-1),-1=>fill(-1.0,k-1)); I1=sparse(1.0LA.I,k,k)
    A=kron(I1,kron(I1,L1))+kron(I1,kron(L1,I1))+kron(L1,kron(I1,I1))
    F=multifrontal_cholesky(A); G=multifrontal_cholesky(A;threaded=false)
    @test F.p==G.p && F.L.values==G.L.values
    b=randn(size(A,1)); @test LA.norm(A*solve(F,b)-b)<=1e-12*LA.norm(b)
    N=copy(A); N[500,500]=-10
    e=try multifrontal_cholesky(N) catch err err end
    @test e isa PosDefException && e.info==500
end

@testset "COLAMD ordering" begin
    # Permutations, the sparse_ordering entry, empty and dense columns, and
    # fill: COLAMD keeps the LU/QR factors of an unsymmetric matrix sparser
    # than the natural order (identical permutations to SuiteSparse's colamd
    # were checked during the port).
    for (m,n,d) in ((1,1,1.0),(5,3,0.6),(40,40,0.1),(300,200,0.03),(200,300,0.03))
        A=SA.sprandn(m,n,d); q=colamd_ordering(A)
        @test sort(q)==1:n && sparse_ordering(A,:colamd)==q
        @test colamd_ordering(CSCMatrix(A))==q && colamd_ordering(CSRMatrix(A))==q
    end
    @test colamd_ordering(SA.spzeros(4,0))==Int[] && colamd_ordering(SA.spzeros(3,4))==1:4
    # Empty columns go last, dense columns just before them.
    A=SA.sprandn(400,30,0.05); A[:,7].=1.0; A[:,12].=0; A=SA.dropzeros(A)
    q=colamd_ordering(A); @test q[end]==12 && q[end-1]==7
    B=SA.sprandn(600,600,0.006)+sparse(1.0*LA.I(600)[randperm(600),:])
    fill(q)=(F=sparse_lu(B;ordering=q); SA.nnz(SparseMatrixCSC(F.L))+SA.nnz(SparseMatrixCSC(F.U)))
    @test fill(colamd_ordering(B))<fill(collect(1:600))
    F=sparse_qr(B;pivot=false,ordering=:colamd); x=randn(600)
    @test LA.norm(B*solve(F,B*x)-B*x)<=1e-8*LA.norm(B*x)
end

@testset "weighted postorder" begin
    # Star with children of weights 3,1,2 under root 4: visited by weight.
    @test OxiBLAS._postorder([4,4,4,0],[3,1,2,9])==[2,3,1,4]
    @test OxiBLAS._postorder([4,4,4,0])==[1,2,3,4]
    parent=[3,3,5,5,0]; w=[1,5,2,7,0]; q=OxiBLAS._postorder(parent,w)
    @test sort(q)==1:5 && all(j->parent[j]==0 || findfirst(==(j),q)<findfirst(==(parent[j]),q),1:5)
end
