# High-level API on structured and compact matrices
# (src/api/linalg_api_structured.jl). LinearAlgebra is used only as a reference.
# Standalone or included.
using Test, Random, OxiBLAS
import LinearAlgebra as LA
const OB=OxiBLAS

_api_srt(::Type{T}) where T=real(T)===Float32 ? 2f-3 : 1e-9
_api_st_spd(::Type{T},n) where T=(X=randn(T,n,n); LA.Hermitian(X*X'+n*LA.I))   # own helper: this file runs on its own
_api_resid(A,x,b)=LA.norm(A*x-b)/(LA.opnorm(A,1)*LA.norm(x)+LA.norm(b))
# Allocation budget showing that no n×n dense copy is made: well below
# n^2·sizeof(T).
_api_linear_alloc(f,n,::Type{T}) where T=(f(); @allocated(f())<200*n*sizeof(T)+100_000)

@testset "structured high-level API" begin

@testset "Tridiagonal $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    r=_api_srt(T); n=60
    A=LA.Tridiagonal(randn(T,n-1),randn(T,n),randn(T,n-1)); D=Matrix(A); b=randn(T,n); B=randn(T,n,3)
    F=OB.lu(A)
    @test F isa OB.LU{T,<:OB.BandLUFactor}
    @test F\b ≈ D\b rtol=r
    @test F\B ≈ D\B rtol=r
    @test F'\b ≈ D'\b rtol=r
    @test transpose(F)\b ≈ transpose(D)\b rtol=r
    @test Matrix(F.L)*Matrix(F.U) ≈ D[F.p,:] rtol=r
    @test LA.det(F) ≈ LA.det(D) rtol=r
    @test OB.det(A) ≈ LA.det(D) rtol=r
    @test LA.inv(F) ≈ LA.inv(D) rtol=r
    @test b'/F ≈ b'/D rtol=r
    y=similar(b); @test LA.ldiv!(y,F,b) ≈ D\b rtol=r
    P=LA.Tridiagonal(ones(T,n-1),fill(T(1e-6),n),ones(T,n-1))   # forces row swaps
    G=OB.lu(P); @test G.p != 1:n
    @test _api_resid(Matrix(P),G\b,b) < 100eps(real(T))
    @test OB.factorize(A) isa OB.LU{T,<:OB.BandLUFactor}
    @test_throws LA.SingularException OB.lu(LA.Tridiagonal(zeros(T,n-1),zeros(T,n),zeros(T,n-1)))
    N=2000; At=LA.Tridiagonal(randn(T,N-1),randn(T,N).+4,randn(T,N-1)); bt=randn(T,N)
    @test _api_linear_alloc(()->OB.lu(At),N,T)
    Ft=OB.lu(At); @test _api_linear_alloc(()->Ft\bt,N,T)
    @test LA.lu(A) isa LA.LU   # LinearAlgebra keeps its own method
end

@testset "SymTridiagonal $T" for T in (Float32,Float64)
    r=_api_srt(T); n=50; b=randn(T,n)
    S=LA.SymTridiagonal(fill(T(4),n),randn(T,n-1)); D=Matrix(S)
    C=OB.cholesky(S)
    @test C isa OB.Cholesky{T,<:OB.BandCholeskyFactor}
    @test C.L isa OB.BandedMatrix
    @test C\b ≈ D\b rtol=r
    @test C'\b ≈ D\b rtol=r
    @test Matrix(C.L)*Matrix(C.L)' ≈ D rtol=r
    @test LA.logdet(C) ≈ LA.logdet(D) rtol=r
    @test OB.isposdef(S)
    @test OB.factorize(S) isa OB.Cholesky
    @test OB.ldlt(S) isa OB.LDLt
    @test OB.ldlt(S)\b ≈ D\b rtol=r
    K=LA.SymTridiagonal(randn(T,n),randn(T,n-1)); Dk=Matrix(K)
    @test !OB.isposdef(K)
    @test !LA.issuccess(OB.cholesky(K;check=false))
    @test_throws LA.PosDefException OB.cholesky(K)
    Fk=OB.factorize(K); @test Fk isa OB.BunchKaufman
    @test Fk\b ≈ Dk\b rtol=r
    @test LA.det(Fk) ≈ LA.det(Dk) rtol=r
    @test Matrix(Fk.L)*Matrix(Fk.D)*Matrix(Fk.L)' ≈ Dk[Fk.p,Fk.p] rtol=r
    # 2×2 pivot on zero leading diagonal (ldlt without pivoting must fail; bunchkaufman must succeed)
    Z2=LA.SymTridiagonal(T[0,0,2],T[3,1]); Dz2=Matrix(Z2); bz2=T[1,2,3]
    @test_throws LA.SingularException OB.ldlt(Z2)
    Fz2=OB.bunchkaufman(Z2)
    @test Fz2\bz2 ≈ Dz2\bz2 rtol=r
    @test OB.det(Z2) ≈ LA.det(Dz2) rtol=r
    @test OB.det(LA.SymTridiagonal(T[0,0],T[0])) == 0
    @test OB.lu(K)\b ≈ Dk\b rtol=r
    E=OB.eigen(S); λ,V=E
    @test D*V ≈ V*LA.Diagonal(λ) rtol=r
    @test OB.eigvals(S) ≈ LA.eigvals(D) rtol=r
    @test OB.norm(S) ≈ LA.norm(D) rtol=r
    @test OB.opnorm(S,1) ≈ LA.opnorm(D,1) rtol=r
    @test OB.opnorm(S,Inf) ≈ LA.opnorm(D,Inf) rtol=r
    @test _api_linear_alloc(()->OB.eigvals(LA.SymTridiagonal(randn(T,2000),randn(T,1999))),2000,T)
    @test _api_linear_alloc(()->OB.cholesky(LA.SymTridiagonal(fill(T(4),2000),ones(T,1999))),2000,T)
    @test _api_linear_alloc(()->OB.ldlt(LA.SymTridiagonal(fill(T(4),2000),ones(T,1999))),2000,T)
    @test _api_linear_alloc(()->OB.bunchkaufman(LA.SymTridiagonal(randn(T,2000),randn(T,1999))),2000,T)
end

@testset "complex symmetric SymTridiagonal $T" for T in (ComplexF32,ComplexF64)
    r=_api_srt(T); n=30; b=randn(T,n)
    S=LA.SymTridiagonal(randn(T,n),randn(T,n-1)); D=Matrix(S)
    F=OB.bunchkaufman(S)
    @test F\b ≈ D\b rtol=r
    @test transpose(F)\b ≈ transpose(D)\b rtol=r
    @test F'\b ≈ D'\b rtol=r
    @test Matrix(F.L)*Matrix(F.D)*transpose(Matrix(F.L)) ≈ D[F.p,F.p] rtol=r
    @test OB.logabsdet(S)[1] ≈ LA.logabsdet(D)[1] rtol=r
    @test OB.logabsdet(S)[2] ≈ LA.logabsdet(D)[2] rtol=r
    @test OB.factorize(S)\b ≈ D\b rtol=r
    @test _api_linear_alloc(()->OB.factorize(LA.SymTridiagonal(randn(T,2000),randn(T,1999))),2000,T)
end

@testset "Bidiagonal $T $uplo" for T in (Float32,Float64,ComplexF32,ComplexF64), uplo in (:U,:L)
    r=_api_srt(T); n=40; b=randn(T,n)
    A=LA.Bidiagonal(randn(T,n).+2,randn(T,n-1),uplo); D=Matrix(A)
    F=OB.factorize(A)
    @test F isa OB.LU{T,<:OB.TriangularEngine}
    @test F\b ≈ D\b rtol=r
    @test F'\b ≈ D'\b rtol=r
    @test transpose(F)\b ≈ transpose(D)\b rtol=r
    @test Matrix(F.L)*Matrix(F.U) ≈ D rtol=r
    @test F.L isa Union{LA.Diagonal,LA.Bidiagonal}
    L=OB.lu(A)
    @test L isa OB.LU{T,<:OB.BandLUFactor}
    @test L\b ≈ D\b rtol=r
    @test OB.svdvals(A) ≈ LA.svdvals(D) rtol=r
    U,S,V=OB.svd(A); @test U*LA.Diagonal(S)*V' ≈ D rtol=r
    Abig=LA.Bidiagonal(randn(T,160).+2,randn(T,159),uplo); Dbig=Matrix(Abig)
    Ubig,Sbig,Vbig=OB.svd(Abig)
    @test Ubig*LA.Diagonal(Sbig)*Vbig' ≈ Dbig rtol=r
    @test Ubig'*Ubig ≈ LA.I rtol=r
    @test Vbig'*Vbig ≈ LA.I rtol=r
    @test OB.svdvals(Abig) ≈ Sbig rtol=r
    @test OB.rank(A)==n
    @test OB.cond(A) ≈ LA.cond(D) rtol=sqrt(r)
    @test OB.eigvals(A) ≈ A.dv
    @test OB.det(A) ≈ prod(A.dv) rtol=r
    @test OB.inv(A) ≈ LA.inv(D) rtol=r
    @test _api_linear_alloc(()->OB.factorize(LA.Bidiagonal(randn(T,3000),randn(T,2999),uplo)),3000,T)
    @test _api_linear_alloc(()->OB.svdvals(LA.Bidiagonal(randn(T,3000),randn(T,2999),uplo)),3000,T)
    Z=LA.Bidiagonal(T[1,0,1],T[1,1],uplo)
    @test !LA.issuccess(OB.factorize(Z))
    @test_throws LA.SingularException OB.factorize(Z)\ones(T,3)
end

@testset "Diagonal $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    r=_api_srt(T); n=25; b=randn(T,n)
    A=LA.Diagonal(randn(T,n)); D=Matrix(A)
    for F in (OB.lu(A),OB.factorize(A))
        @test F\b ≈ D\b rtol=r
        @test F'\b ≈ D'\b rtol=r
        @test transpose(F)\b ≈ transpose(D)\b rtol=r
        @test LA.det(F) ≈ LA.det(D) rtol=r
    end
    @test OB.factorize(A).U isa LA.Diagonal
    @test OB.inv(A) isa LA.Diagonal
    @test OB.inv(A) ≈ LA.inv(D) rtol=r
    P=LA.Diagonal(rand(real(T),n).+1); C=OB.cholesky(P)
    @test C.L isa LA.Diagonal
    @test C\b ≈ Matrix(P)\b rtol=r
    @test OB.isposdef(P)
    @test !OB.isposdef(-P)
    E=OB.eigen(A); @test D*E.vectors ≈ E.vectors*LA.Diagonal(E.values) rtol=r
    # Explicit order: LinearAlgebra sorts Diagonal eigenvalues from 1.11 on, not in 1.10.
    @test OB.eigvals(A) ≈ sort(LA.diag(D);by=x->(real(x),imag(x))) rtol=r
    @test OB.eigen(A;sortby=nothing).vectors isa LA.Diagonal
    U,S,V=OB.svd(A); @test U*LA.Diagonal(S)*V' ≈ D rtol=r
    @test OB.svdvals(A) ≈ LA.svdvals(D) rtol=r
    @test OB.rank(LA.Diagonal(T[1,0,2]))==2
    @test_throws LA.SingularException OB.inv(LA.Diagonal(T[1,0]))
    @test _api_linear_alloc(()->OB.factorize(LA.Diagonal(randn(T,5000)))\randn(T,5000),5000,T)
end

@testset "dense triangular $T $W" for T in (Float32,Float64,ComplexF32,ComplexF64),
        W in (LA.UpperTriangular,LA.LowerTriangular,LA.UnitUpperTriangular,LA.UnitLowerTriangular)
    r=_api_srt(T); n=30; b=randn(T,n)
    M=randn(T,n,n)+4n*LA.I   # the other triangle holds data that must be ignored
    A=W(M); D=Matrix(A)
    F=OB.factorize(A)
    @test F isa OB.LU{T,<:OB.TriangularEngine}
    @test F\b ≈ D\b rtol=r
    @test F'\b ≈ D'\b rtol=r
    @test transpose(F)\b ≈ transpose(D)\b rtol=r
    @test Matrix(F.L)*Matrix(F.U) ≈ D rtol=r
    # det itself overflows Float32 here (|d|≈4n). atol: for a unit triangle the
    # exact value is 0, and a dense reference (1.10 uses LU) is off by roundoff.
    @test OB.logabsdet(A)[1] ≈ LA.logabsdet(D)[1] rtol=r atol=r
    @test OB.logabsdet(A)[2] ≈ LA.logabsdet(D)[2] rtol=r
    @test OB.inv(A) ≈ LA.inv(D) rtol=r
    @test OB.eigvals(A) ≈ LA.diag(D)
    G=OB.lu(A)
    @test G\b ≈ D\b rtol=r
    W in (LA.UpperTriangular,LA.UnitUpperTriangular) && @test G isa OB.LU{T,<:OB.TriangularEngine}
    if W in (LA.UpperTriangular,LA.UnitUpperTriangular)
        E=OB.eigen(A); @test D*E.vectors ≈ E.vectors*LA.Diagonal(E.values) rtol=sqrt(r)
    end
    @test M==parent(A)   # the input is not modified
end

@testset "BandedMatrix $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    r=_api_srt(T); n=50; b=randn(T,n)
    X=randn(T,n,n); kl,ku=2,3
    G=OB.band_matrix(X;kl,ku); D=Matrix(G)
    F=OB.lu(G)
    @test F\b ≈ D\b rtol=r
    @test F'\b ≈ D'\b rtol=r
    @test transpose(F)\b ≈ transpose(D)\b rtol=r
    @test Matrix(F.L)*Matrix(F.U) ≈ D[F.p,:] rtol=r
    @test LA.lu(G) isa OB.LU
    @test LA.factorize(G) isa OB.LU
    @test LA.det(G) ≈ LA.det(D) rtol=r
    H=X*X'+4n*LA.I; Hd=Matrix(OB.band_matrix(H;kl=2,ku=2)); Hb=OB.band_matrix(Hd;kl=2,ku=2,kind=:hermitian,uplo=:L)
    C=OB.cholesky(Hb)
    @test C isa OB.Cholesky{T,<:OB.BandCholeskyFactor}
    @test C\b ≈ Hd\b rtol=r
    @test LA.cholesky(Hb) isa OB.Cholesky
    @test LA.isposdef(Hb)
    @test OB.ldlt(Hb)\b ≈ Hd\b rtol=r
    @test OB.bunchkaufman(Hb)\b ≈ Hd\b rtol=r
    Ib=OB.band_matrix(Hd-8n*LA.I;kl=2,ku=2,kind=:hermitian,uplo=:L); Id=Matrix(Ib)
    @test OB.factorize(Ib) isa OB.BunchKaufman
    @test OB.factorize(Ib)\b ≈ Id\b rtol=r
    for uplo in (:U,:L)
        Tb=OB.band_matrix(X+4n*LA.I;kl=uplo===:L ? 2 : 0,ku=uplo===:U ? 2 : 0,kind=:triangular,uplo); Td=Matrix(Tb)
        Ft=OB.factorize(Tb); @test Ft isa OB.LU{T,<:OB.TriangularEngine}
        @test Ft\b ≈ Td\b rtol=r
        @test Ft'\b ≈ Td'\b rtol=r
        @test OB.lu(Tb)\b ≈ Td\b rtol=r
    end
    N=3000; Xb=OB.band_matrix(randn(T,N,N);kl=2,ku=2)   # the dense source is built outside the budget
    @test _api_linear_alloc(()->OB.lu(Xb),N,T)
end

@testset "PackedMatrix $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    r=_api_srt(T); n=30; b=randn(T,n)
    X=randn(T,n,n); H=X*X'+n*LA.I
    P=OB.pack_matrix(H;uplo=:L,kind=:hermitian); Hd=Matrix(P)
    C=OB.cholesky(P)
    @test C isa OB.Cholesky{T,<:OB.PackedCholeskyFactor}
    @test C.L isa OB.PackedMatrix
    @test C\b ≈ Hd\b rtol=r
    @test LA.logdet(C) ≈ LA.logdet(Hd) rtol=r
    @test OB.factorize(P) isa OB.Cholesky
    @test LA.cholesky(P) isa OB.Cholesky
    @test OB.isposdef(P)
    Pi=OB.pack_matrix(H-4n*LA.I;uplo=:L,kind=:hermitian)
    @test !OB.isposdef(Pi)
    @test_throws ArgumentError OB.factorize(Pi)   # no packed indefinite engine
    @test_throws ArgumentError OB.lu(P)
    U=OB.pack_matrix(X+4n*LA.I;uplo=:U,kind=:triangular); Ud=Matrix(U)
    for F in (OB.lu(U),OB.factorize(U))
        @test F\b ≈ Ud\b rtol=r
        @test F'\b ≈ Ud'\b rtol=r
    end
    L=OB.pack_matrix(X+4n*LA.I;uplo=:L,kind=:triangular); Ld=Matrix(L)
    Fl=OB.factorize(L); @test Fl\b ≈ Ld\b rtol=r
    @test Matrix(Fl.L)*Matrix(Fl.U) ≈ Ld rtol=r
end

@testset "Hermitian/Symmetric wrappers of structured matrices $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    r=_api_srt(T); n=40; b=randn(T,n)
    X=LA.Tridiagonal(randn(T,n-1),fill(T(6),n),randn(T,n-1))
    for H in (LA.Hermitian(X,:U),LA.Hermitian(X,:L),LA.Hermitian(LA.Bidiagonal(fill(T(6),n),randn(T,n-1),:U)),
              LA.Hermitian(LA.Diagonal(rand(real(T),n).+1)))
        D=Matrix(H)
        C=OB.cholesky(H)
        @test C isa OB.Cholesky{<:Any,<:OB.BandCholeskyFactor}   # real for Hermitian(Diagonal(real))
        @test C\b ≈ D\b rtol=r
        @test OB.ldlt(H)\b ≈ D\b rtol=r
        @test OB.bunchkaufman(H)\b ≈ D\b rtol=r
        @test OB.lu(H)\b ≈ D\b rtol=r
        @test OB.factorize(H) isa OB.Cholesky
        @test OB.logabsdet(H)[1] ≈ LA.logabsdet(D)[1] rtol=r
        @test OB.isposdef(H)
        if T<:Real
            E=OB.eigen(H); @test D*E.vectors ≈ E.vectors*LA.Diagonal(E.values) rtol=sqrt(r)
            @test OB.eigvals(H) ≈ LA.eigvals(D) rtol=sqrt(r)
        end
    end
    Hi=LA.Hermitian(LA.Tridiagonal(randn(T,n-1),randn(T,n),randn(T,n-1)),:L); Di=Matrix(Hi)
    @test !OB.isposdef(Hi)
    @test OB.factorize(Hi)\b ≈ Di\b rtol=r
    if T<:Complex
        Sy=LA.Symmetric(X); Ds=Matrix(Sy)
        @test OB.bunchkaufman(Sy)\b ≈ Ds\b rtol=r
        @test transpose(OB.bunchkaufman(Sy))\b ≈ transpose(Ds)\b rtol=r
    end
    @test _api_linear_alloc(()->OB.cholesky(LA.Hermitian(LA.Tridiagonal(ones(T,2999),fill(T(6),3000),ones(T,2999)))),3000,T)
end

@testset "matrix functions on structured matrices $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    r=sqrt(_api_srt(T)); n=12
    Tr=LA.Tridiagonal(randn(T,n-1)/4,randn(T,n)/4,randn(T,n-1)/4)
    Sp=LA.SymTridiagonal(fill(real(T)(3),n),randn(real(T),n-1)/4)          # positive definite
    Bi=LA.Bidiagonal(rand(real(T),n).+1,randn(real(T),n-1)/4,:U)            # positive diagonal
    Ba=OB.band_matrix(randn(T,n,n)/4;kl=2,ku=1)
    Hp=Matrix(_api_st_spd(T,n))/n; Pk=OB.pack_matrix(Hp;uplo=:L,kind=:hermitian)
    Hs=LA.Hermitian(LA.Tridiagonal(randn(T,n-1)/4,fill(T(3),n),randn(T,n-1)/4),:U)
    for A in (Tr,Ba), (f,g) in ((OB.exp,exp),(OB.sin,sin),(OB.cos,cos))
        @test f(A) ≈ g(Matrix(A)) rtol=r
    end
    for A in (Sp,Bi,Pk,Hs,LA.UpperTriangular(Matrix(Bi)),LA.Hermitian(Hp)), (f,g) in ((OB.exp,exp),(OB.log,log),(OB.sqrt,sqrt))
        @test f(A) ≈ g(Matrix(A)) rtol=r
    end
    @test OB.exp(LA.Tridiagonal([1,2],[3,4,5],[6,7])) ≈ exp(Float64[3 6 0;1 4 7;0 2 5]) rtol=1e-10   # Int input
    @test OB.exp(LA.Diagonal(T[1,2])) isa LA.Diagonal
    # Structure of the result
    U=LA.UpperTriangular(Matrix(Bi)); L=LA.LowerTriangular(Matrix(transpose(Bi)))
    @test OB.exp(U) isa LA.UpperTriangular
    @test OB.sqrt(L) isa LA.LowerTriangular
    @test OB.log(Bi) isa LA.UpperTriangular{real(T)}          # real, positive diagonal
    @test OB.exp(LA.Bidiagonal(randn(T,n),randn(T,n-1),:L)) isa LA.LowerTriangular
    He=LA.Hermitian(Hp)
    for f in (OB.exp,OB.sin,OB.cos,OB.log,OB.sqrt)
        @test f(He) isa (T<:Real ? LA.Symmetric{T} : LA.Hermitian{T})
    end
    Uu=LA.UnitUpperTriangular(randn(T,n,n)/4)
    Eu=OB.exp(Uu)
    @test Eu isa LA.UpperTriangular                             # not Unit: the diagonal is e
    @test all(i->Eu[i,i] ≈ exp(one(T)),1:n)
    @test OB.exp(LA.UnitLowerTriangular(randn(T,n,n)/4)) isa LA.LowerTriangular
    if T<:Complex
        Sh=OB.sqrt(He)
        @test Sh isa LA.Hermitian{T}
        @test any(!iszero∘imag,Sh[i,j] for i in 1:n for j in 1:n if i!=j)   # imaginary parts are kept
        @test Sh*Sh ≈ Matrix(He) rtol=r
        Cs2=LA.Symmetric(randn(T,4,4)/4)
        @test !(OB.exp(Cs2) isa LA.Hermitian)
    end
    Z=LA.Hermitian(T[0 0; 0 4])                                # semidefinite, not definite
    @test !OB.isposdef(Z)
    @test OB.sqrt(Z) isa (T<:Real ? LA.Symmetric{T} : LA.Hermitian{T})
    @test OB.sqrt(Z) ≈ T[0 0; 0 2] atol=r
    @test_throws DomainError OB.log(Z)                         # singular: no finite logarithm
    Hi=LA.Hermitian(Hp-2LA.I)                                   # indefinite
    @test OB.sqrt(Hi) isa Matrix
    @test OB.sqrt(Hi) ≈ sqrt(Matrix(Hi)) rtol=r
    @test OB.exp(Hi) isa (T<:Real ? LA.Symmetric : LA.Hermitian)
    @test OB.exp(Sp) isa LA.Symmetric{real(T)}
    @test OB.sqrt(Sp) isa LA.Symmetric{real(T)}
    @test OB.exp(Pk) isa (T<:Real ? LA.Symmetric : LA.Hermitian)
    @test OB.sin(Tr) isa Matrix{T}                            # real input -> real result
    @test OB.exp(Tr) isa Matrix
    if T<:Complex
        Cs=LA.Symmetric(randn(T,n,n)/4)
        @test OB.exp(Cs) isa LA.Symmetric
        @test OB.exp(Cs) ≈ exp(Matrix(Cs)) rtol=r
    end
end

@testset "boundaries" begin
    A=LA.Tridiagonal(randn(4),randn(5),randn(4))
    @test_throws MethodError OB.qr(A)       # no structured QR engine
    @test_throws MethodError OB.eigen(A)    # no nonsymmetric tridiagonal eigensolver
    @test_throws MethodError OB.lu!(A)      # band storage cannot hold the fill
    @test_throws MethodError OB.svd(A)
    @test_throws MethodError OB.exp(LA.UpperHessenberg(randn(4,4)))   # no method: no Base/LAPACK fallback
    @test parentmodule(which(LA.lu,(typeof(A),)))===LA
    @test parentmodule(which(LA.factorize,(LA.Diagonal{Float64,Vector{Float64}},)))===LA
end

end
