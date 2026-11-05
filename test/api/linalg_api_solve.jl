# Linear-solve API (src/api/linalg_api_solve.jl): OxiBLAS.ldiv / ldiv!, and `\` /
# LinearAlgebra.ldiv! on the OxiBLAS sparse formats. Dense LinearAlgebra
# solves serve only as references. Can be run standalone or from runtests.jl.
using Test, Random, OxiBLAS, SparseArrays
import LinearAlgebra as LA
const OBS=OxiBLAS

_sv_rt(::Type{T}) where T=real(T)===Float32 ? 2f-3 : 1e-9
_sv_sparse(rng,::Type{T},n) where T=sprand(rng,T,n,n,0.08)+sparse(T(n)*LA.I,n,n)

# The six forms of the API, which must agree.
function _sv_all_forms(A,B,F)
    X1=A\B
    Y2=similar(X1); LA.ldiv!(Y2,A,B)
    X3=OBS.ldiv(A,B)
    Y4=similar(X1); OBS.ldiv!(Y4,A,B)
    X5=OBS.ldiv(F,B)
    Y6=similar(X1); OBS.ldiv!(Y6,F,B)
    (X1,Y2,X3,Y4,X5,Y6)
end

@testset "linear-solve API" begin

@testset "namespace, standard types untouched" begin
    # Piracy and ambiguities are checked package-wide by Aqua (test/qa.jl).
    @test OBS.ldiv! !== LA.ldiv! && OBS.ldiv !== LA.ldiv!
    @test :ldiv ∉ names(OxiBLAS) && :ldiv! ∉ names(OxiBLAS)
    # No method of these functions on standard types comes from OxiBLAS.
    S=sparse([1.0 2;0 3]); b=[1.0,2]
    for (f,args) in ((\,(S,b)),(\,(Matrix(S),b)),(LA.ldiv!,(similar(b),S,b)),(LA.istril,(S,)),(LA.factorize,(S,)))
        @test which(f,typeof.(args)).module !== OxiBLAS
    end
end

@testset "sparse general square $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    rng=Xoshiro(7); n=40; r=_sv_rt(T)
    S=_sv_sparse(rng,T,n); D=Matrix(S)
    b=randn(rng,T,n); B=randn(rng,T,n,3); W=randn(rng,T,n,5)
    rhs=(b,B,view(W,:,2:4),view(W,:,5),CSRMatrix(sparse(B)),sparse(b))
    for A in (CSRMatrix(S),CSCMatrix(S),COOMatrix(S))
        A0=copy(A); F=OBS.factorize(A)
        @test F isa OBS.LU{T,<:OBS.MultifrontalLUFactor}
        for Bx in rhs
            Bc=copy(Bx); ref=D\Array(Bx)
            forms=_sv_all_forms(A,Bx,F)
            for X in forms
                @test X ≈ ref rtol=r
                @test X isa (Bx isa AbstractVector ? Vector{T} : Matrix{T})
            end
            @test all(X->isapprox(X,forms[1];rtol=r),forms)
            @test Bx==Bc                                   # B preserved
        end
        @test A==A0                                        # A preserved
        x1=OBS.ldiv(F,b); x2=OBS.ldiv(F,b); @test x1==x2  # F reusable, unchanged
        # LinearAlgebra.ldiv!(A, B): B overwritten, square only
        X=copy(B); LA.ldiv!(A,X); @test X ≈ D\B rtol=r
        # in place through OxiBLAS.ldiv!(B, F, B)
        X=copy(B); @test OBS.ldiv!(X,F,X)===X; @test X ≈ D\B rtol=r
        # transpose / adjoint
        @test transpose(A)\b ≈ transpose(D)\b rtol=r
        @test A'\B ≈ D'\B rtol=r
        Y=similar(B); LA.ldiv!(Y,A',B); @test Y ≈ D'\B rtol=r
        @test OBS.ldiv(transpose(F),b) ≈ transpose(D)\b rtol=r
    end
    # SparseMatrixCSC goes through OxiBLAS only with the OxiBLAS functions.
    @test OBS.ldiv(S,B) ≈ D\B rtol=r
    Y=similar(B); @test OBS.ldiv!(Y,S,B)===Y; @test Y ≈ D\B rtol=r
end

@testset "sparse structure choice $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    rng=Xoshiro(11); n=30; r=_sv_rt(T)
    S=_sv_sparse(rng,T,n); b=randn(rng,T,n); B=randn(rng,T,n,2)
    Uu=sparse(LA.triu(S)); Ll=sparse(LA.tril(S))
    for (M,upper) in ((Uu,true),(Ll,false)), A in (CSRMatrix(M),CSCMatrix(M),COOMatrix(M))
        @test LA.istriu(A)==upper && LA.istril(A)==!upper
        F=OBS.factorize(A); @test F isa OBS.LU{T,<:OBS.TriangularEngine}
        for X in _sv_all_forms(A,B,F); @test X ≈ Matrix(M)\B rtol=r; end
        @test F'\b ≈ Matrix(M)'\b rtol=r
        @test F.L*F.U ≈ Matrix(M) rtol=r
    end
    # triangular wrappers: only the wrapped triangle is read
    A=CSRMatrix(S); Dd=Matrix(S)
    for Wr in (LA.UpperTriangular,LA.LowerTriangular,LA.UnitUpperTriangular,LA.UnitLowerTriangular)
        F=OBS.factorize(Wr(A))
        for X in _sv_all_forms(Wr(A),B,F); @test X ≈ Wr(Dd)\B rtol=r; end
        X=copy(b); LA.ldiv!(Wr(A),X); @test X ≈ Wr(Dd)\b rtol=r
    end
    # Hermitian: Cholesky when positive definite, Bunch–Kaufman otherwise
    P=S*S'+sparse(T(n)*LA.I,n,n); Pd=Matrix(P)
    for A in (CSRMatrix(P),CSCMatrix(P))
        @test OBS.factorize(A) isa OBS.Cholesky
        @test A\B ≈ Pd\B rtol=r
        @test LA.Hermitian(A,:U)\b ≈ Pd\b rtol=r
    end
    K=S+S'; K=K-sparse(T(3n)*LA.I,n,n)          # Hermitian part plus a shift: indefinite
    K=(K+K')/2; Kd=Matrix(K); A=CSRMatrix(K)
    @test OBS.factorize(A) isa OBS.BunchKaufman
    for X in _sv_all_forms(A,B,OBS.factorize(A)); @test X ≈ Kd\B rtol=r; end
    if T<:Complex
        Z=S+transpose(S); Zs=LA.Symmetric(CSRMatrix(Z))
        @test OBS.factorize(Zs) isa OBS.BunchKaufman
        @test Zs\b ≈ Matrix(Z)\b rtol=r
    end
end

@testset "rectangular, dimensions, singular $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    rng=Xoshiro(13); m,n=60,25; r=_sv_rt(T)
    R=sprand(rng,T,m,n,0.15)+[sparse(T(m)*LA.I,n,n);spzeros(T,m-n,n)]; Rd=Matrix(R)
    c=randn(rng,T,m); C=randn(rng,T,m,3)
    A=CSRMatrix(R); F=OBS.factorize(A); @test F isa OBS.QRSparse
    for Bx in (c,C)
        ref=Rd\Bx                                       # least squares
        for X in _sv_all_forms(A,Bx,F); @test X ≈ ref rtol=10r; end
    end
    @test_throws DimensionMismatch LA.ldiv!(A,copy(c))  # not square
    @test_throws ArgumentError CSRMatrix(sparse(transpose(R)))\randn(rng,T,n)  # wide: no sparse solve
    S=_sv_sparse(rng,T,n); As=CSRMatrix(S); b=randn(rng,T,n)
    @test_throws DimensionMismatch As\randn(rng,T,n+1)
    @test_throws DimensionMismatch OBS.ldiv!(zeros(T,n+1),As,b)
    @test_throws DimensionMismatch OBS.ldiv!(zeros(T,n,2),OBS.factorize(As),b)
    W=randn(rng,T,n,2)
    w=randn(rng,T,n+1)                                    # overlapping views, not ===
    @test_throws ArgumentError OBS.ldiv!(view(w,2:n+1),OBS.factorize(As),view(w,1:n))
    Y=view(W,:,2); OBS.ldiv!(Y,OBS.factorize(As),view(W,:,1))  # disjoint columns of one parent: allowed
    @test Y ≈ Matrix(S)\W[:,1] rtol=r
    @test_throws ArgumentError OBS.qr(CSRMatrix(sparse(transpose(R))))   # wide sparse QR: clear error
    @test_throws ArgumentError OBS.ldiv!(CSRMatrix(sparse(zeros(T,n,1))),As,reshape(b,n,1))  # sparse destination
    # singular matrices
    Z=copy(S); Z[3,:].=0; Z=dropzeros(Z)
    @test_throws LA.SingularException CSRMatrix(Z)\b
    U=sparse(LA.triu(S)); U[4,4]=0; U=dropzeros(U)
    @test_throws LA.SingularException CSCMatrix(U)\b
    @test_throws LA.SingularException OBS.ldiv(OBS.factorize(CSCMatrix(U)),b)
end

@testset "promotion" begin
    rng=Xoshiro(17); n=20; S=_sv_sparse(rng,Float64,n); A=CSRMatrix(S); D=Matrix(S)
    z=randn(rng,ComplexF64,n)
    @test A\z isa Vector{ComplexF64} && A\z ≈ D\z            # real factor, complex B
    @test A\collect(1:n) isa Vector{Float64} && A\collect(1:n) ≈ D\collect(1:n)
    A32=CSRMatrix(SparseMatrixCSC{Float32}(S))
    @test OBS.ldiv(A32,randn(rng,n)) isa Vector{Float64}     # promote_op(\, Float32, Float64)
    Y=zeros(Float64,n); @test_throws InexactError OBS.ldiv!(Y,A,z)
end

@testset "dense and structured $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    rng=Xoshiro(19); n=12; r=_sv_rt(T)
    A=randn(rng,T,n,n)+n*LA.I; b=randn(rng,T,n); B=randn(rng,T,n,3); A0=copy(A)
    for F in (OBS.lu(A),OBS.factorize(A),OBS.qr(A))
        @test OBS.ldiv(F,B) ≈ A\B rtol=r
        Y=similar(B); @test OBS.ldiv!(Y,F,B)===Y; @test Y ≈ A\B rtol=r
        @test OBS.ldiv(F,view(B,:,1)) ≈ A\B[:,1] rtol=r
    end
    @test OBS.ldiv(A,b) ≈ A\b rtol=r; @test A==A0
    H=LA.Hermitian(A*A'+n*LA.I)
    @test OBS.factorize(H) isa OBS.Cholesky; @test OBS.ldiv(H,B) ≈ Matrix(H)\B rtol=r
    Ft=randn(rng,T,3n,n); c=randn(rng,T,3n)        # tall: least squares
    @test OBS.ldiv(Ft,c) ≈ Ft\c rtol=10r
    Fqr=OBS.qr(Ft); yqr=zeros(T,n); ws=zeros(T,3n)
    @test OBS.ldiv!(yqr,Fqr,c;workspace=ws)===yqr; @test yqr ≈ Ft\c rtol=10r
    @test_throws DimensionMismatch OBS.ldiv!(yqr,Fqr,c;workspace=zeros(T,3n-1))
    @test_throws ArgumentError OBS.ldiv!(yqr,Fqr,c;workspace=c)
    Fw=randn(rng,T,n,3n)                             # wide: a solution of A*x = b
    x=OBS.ldiv(Fw,b); @test Fw*x ≈ b rtol=10r; @test length(x)==3n
    for M in (LA.Diagonal(randn(rng,T,n).+3),LA.Tridiagonal(randn(rng,T,n-1),randn(rng,T,n).+4,randn(rng,T,n-1)),
              LA.UpperTriangular(A),LA.LowerTriangular(A))
        @test OBS.ldiv(M,B) ≈ Matrix(M)\B rtol=r
        Y=similar(B); OBS.ldiv!(Y,M,B); @test Y ≈ Matrix(M)\B rtol=r
    end
    @test_throws MethodError OBS.ldiv(reshape(1.0:n*n,n,n),b)  # no factorize method: no fallback
end

@testset "O(nnz) structure tests" begin
    n=200_000; e=ones(n)
    S=spdiagm(-1=>-e[2:end],0=>4e,1=>-e[2:end]); A=CSRMatrix(S)
    @test !LA.istril(A) && !LA.istriu(A) && LA.istril(CSRMatrix(LA.tril(S)))
    LA.istril(A); @test @elapsed(LA.istril(A))<0.1   # the generic method reads n² positions
    b=randn(n); @test A\b ≈ S\b
end

end
