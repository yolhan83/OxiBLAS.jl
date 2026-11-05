# High-level API (src/linalg_api*.jl). LinearAlgebra/SparseArrays serve only
# as references here. Can be run standalone or included from runtests.jl.
using Test, Random, OxiBLAS, SparseArrays
import LinearAlgebra as LA
const OB=OxiBLAS

_api_rt(::Type{T}) where T=real(T)===Float32 ? 2f-3 : 1e-8
_api_spd(::Type{T},n) where T=(X=randn(T,n,n); LA.Hermitian(X*X'+n*LA.I))
_api_indef(::Type{T},n) where T=(X=randn(T,n,n); LA.Hermitian(X+X'))

@testset "high-level API" begin

@testset "namespace" begin
    @test OB.lu !== LA.lu && OB.qr !== LA.qr && OB.cholesky !== LA.cholesky && OB.det !== LA.det
    @test OB.inv !== Base.inv && OB.inv(4.0)==0.25
    for s in (:lu,:qr,:cholesky,:LU,:QR,:QRSparse,:factorize,:det,:SVD,:Eigen)
        @test s ∉ names(OxiBLAS)
    end
    @test OB.SingularException===LA.SingularException && OB.PosDefException===LA.PosDefException
    @test OB.LU <: LA.Factorization && OB.QRPackedQ <: LA.AbstractQ
end

@testset "dense LU $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    n=9; A=randn(T,n,n); b=randn(T,n); B=randn(T,n,3); r=_api_rt(T)
    F=OB.lu(A)
    @test F isa OB.LU && size(F)==(n,n) && LA.issuccess(F)
    @test F.L*F.U ≈ A[F.p,:] rtol=r
    @test F.P*A ≈ F.L*F.U rtol=r
    @test F\b ≈ A\b rtol=r
    @test F\B ≈ A\B rtol=r
    @test F'\b ≈ A'\b rtol=r
    @test transpose(F)\b ≈ transpose(A)\b rtol=r
    @test LA.ldiv!(F,copy(B)) ≈ A\B rtol=r
    y=similar(b); @test LA.ldiv!(y,F,b) ≈ A\b rtol=r
    @test B'/F ≈ B'/A rtol=r
    @test LA.det(F) ≈ LA.det(A) rtol=r
    @test OB.det(A) ≈ LA.det(A) rtol=r
    @test OB.logabsdet(A)[1] ≈ LA.logabsdet(A)[1] rtol=r
    @test LA.inv(F) ≈ LA.inv(A) rtol=r
    @test OB.inv(A) ≈ LA.inv(A) rtol=r
    if T<:Real
        bc=randn(Complex{T},n); @test F\bc ≈ A\bc rtol=r
        @test LA.ldiv!(F,copy(bc)) ≈ A\bc rtol=r
    end
    @test F\Int.(round.(real.(b))) ≈ A\Int.(round.(real.(b))) rtol=r
    S=copy(A); S[:,2].=0
    G=OB.lu(S;check=false); @test !LA.issuccess(G)
    @test_throws LA.SingularException G\b
    @test_throws LA.SingularException OB.lu(S)
    @test LA.det(G)==0
    @test propertynames(F)==(:L,:U,:p,:P)
end

@testset "in-place lu!/cholesky! on views $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    r=_api_rt(T)
    for n in (9,300)   # 300: blocked path (panels of 128, TRSM/GEMM updates on views)
        b=randn(T,n)
        Big=randn(T,n+7,n+5); V=view(Big,4:n+3,3:n+2)      # leading dimension > m
        Big2=randn(T,2n,n); W=view(Big2,1:2:2n,:)            # row stride 2
        for X in (randn(T,n,n),V,W)
            A=copy(X); F=OB.lu!(X)
            @test F.factors===X
            @test F\b ≈ A\b rtol=r
            @test F.L*F.U ≈ A[F.p,:] rtol=r
        end
        outside=copy(Big); OB.lu!(view(Big,4:n+3,3:n+2))
        @test Big[1:3,:]==outside[1:3,:]
        @test Big[:,1:2]==outside[:,1:2]
        H=Matrix(_api_spd(T,n))
        for X in (copy(H),(Bh=zeros(T,n+3,n+2); Bh[2:n+1,3:n+2]=H; view(Bh,2:n+1,3:n+2)),
                  (Bs=zeros(T,2n,n); Bs[1:2:2n,:]=H; view(Bs,1:2:2n,:)))
            C=OB.cholesky!(X)
            @test C.factors===X
            @test C\b ≈ H\b rtol=r
            @test C.L*C.U ≈ H rtol=r
        end
        U=copy(H); for j in 1:n, i in j+1:n; U[i,j]=zero(T); end     # only the upper triangle is meaningful
        C=OB.cholesky!(LA.Hermitian(U,:U))
        @test C.factors===U
        @test C\b ≈ H\b rtol=r
    end
    S=randn(T,6,6); S[:,3].=0; F=OB.lu!(S;check=false)
    @test !LA.issuccess(F)
    @test F.factors===S
    N=Matrix(_api_indef(T,6)); C=OB.cholesky!(N;check=false)
    @test !LA.issuccess(C)
    @test_throws MethodError OB.lu!(rand(1:9,4,4))   # no implicit conversion for a mutating call
end

@testset "dense Cholesky $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    n=8; H=_api_spd(T,n); A=Matrix(H); b=randn(T,n); r=_api_rt(T)
    for X in (H,A,LA.Hermitian(A,:U),LA.Hermitian(A,:L))
        F=OB.cholesky(X)
        @test F.L*F.U ≈ A rtol=r
        @test F\b ≈ A\b rtol=r
        @test F'\b ≈ A\b rtol=r
        @test transpose(F)\b ≈ transpose(A)\b rtol=r
        @test LA.logdet(F) ≈ LA.logdet(A) rtol=r
        @test b'/F ≈ b'/A rtol=r
    end
    @test_throws LA.PosDefException OB.cholesky(randn(T,n,n)+5n*LA.I)   # not Hermitian
    N=Matrix(_api_indef(T,n))
    F=OB.cholesky(N;check=false); @test !LA.issuccess(F)
    @test_throws LA.PosDefException F\b
    @test_throws LA.PosDefException OB.cholesky(N)
    @test OB.isposdef(A) && !OB.isposdef(N)
    @test OB.factorize(A) isa OB.Cholesky && OB.factorize(N) isa OB.BunchKaufman
end

@testset "dense Bunch–Kaufman / LDLt $T" for T in (Float64,ComplexF64)
    n=10; H=_api_indef(T,n); A=Matrix(H); b=randn(T,n); r=_api_rt(T)
    F=OB.bunchkaufman(H)
    @test F\b ≈ A\b rtol=r
    @test F'\b ≈ A'\b rtol=r
    @test transpose(F)\b ≈ transpose(A)\b rtol=r
    @test LA.det(F) ≈ LA.det(A) rtol=r
    @test F.P*A*F.P' ≈ F.L*Matrix(F.D)*F.L' rtol=r
    if T<:Complex
        X=randn(T,n,n); Sy=LA.Symmetric(X+transpose(X)); As=Matrix(Sy)
        G=OB.bunchkaufman(Sy)
        @test G\b ≈ As\b rtol=r
        @test transpose(G)\b ≈ transpose(As)\b rtol=r
        @test G'\b ≈ As'\b rtol=r
    end
    P=Matrix(_api_spd(T,n)); L=OB.ldlt(P)
    @test L isa OB.LDLt
    @test L\b ≈ P\b rtol=r
    @test LA.logdet(L) ≈ LA.logdet(P) rtol=r
    @test_throws ArgumentError OB.bunchkaufman(randn(T,n,n))
end

@testset "dense QR / QRPivoted / LQ $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    r=_api_rt(T)
    for (m,n) in ((12,7),(7,12),(8,8))
        A=randn(T,m,n); b=randn(T,m); B=randn(T,m,2); c=randn(T,n)
        F=OB.qr(A)
        @test Matrix(F.Q)*F.R ≈ A rtol=r
        @test size(Matrix(F.Q))==(m,min(m,n))
        @test F.Q*(F.Q'*b) ≈ b rtol=r
        @test F\b ≈ LA.pinv(A)*b rtol=r     # least squares (tall), minimum norm (wide)
        @test F\B ≈ LA.pinv(A)*B rtol=r
        @test F'\c ≈ LA.pinv(A')*c rtol=r
        y=zeros(T,n); @test LA.ldiv!(y,F,b) ≈ LA.pinv(A)*b rtol=r
        @test abs(LA.det(F.Q)) ≈ 1 rtol=r
        @test b'*F.Q ≈ (F.Q'*b)' rtol=r
        P=OB.qr(A,LA.ColumnNorm())
        @test A[:,P.p] ≈ Matrix(P.Q)*P.R rtol=r
        @test P\b ≈ LA.pinv(A)*b rtol=r
        L=OB.lq(A)
        @test L.L*Matrix(L.Q) ≈ A rtol=r
        m<=n && @test L\b ≈ LA.pinv(A)*b rtol=r
        m>=n && @test L'\c ≈ LA.pinv(A')*c rtol=r
        if T<:Real
            bc=randn(Complex{T},m)
            @test F\bc ≈ LA.pinv(A)*bc rtol=r
            @test F.Q*complex.(randn(T,m)) isa Vector{Complex{T}}
        end
    end
    A=randn(T,10,4)*randn(T,4,7)   # rank 4
    b=randn(T,10)
    @test OB.qr(A,LA.ColumnNorm())\b ≈ LA.pinv(A)*b rtol=sqrt(r)
end

@testset "SVD / eigen / Schur / Hessenberg $T" for T in (Float64,ComplexF64)
    r=_api_rt(T); A=randn(T,9,6); b=randn(T,9)
    F=OB.svd(A); U,S,V=F
    @test U*LA.Diagonal(S)*V' ≈ A rtol=r
    @test F\b ≈ LA.pinv(A)*b rtol=r
    @test OB.svdvals(A) ≈ LA.svdvals(A) rtol=r
    H=_api_spd(T,7); E=OB.eigen(H); λ,X=E
    @test Matrix(H)*X ≈ X*LA.Diagonal(λ) rtol=r
    @test LA.det(E) ≈ LA.det(Matrix(H)) rtol=r
    @test OB.eigvals(H) ≈ LA.eigvals(H) rtol=r
    G=randn(T,7,7); Eg=OB.eigen(G)
    @test G*Eg.vectors ≈ Eg.vectors*LA.Diagonal(Eg.values) rtol=1e-6
    @test sort(OB.eigvals(G);by=x->(real(x),imag(x))) ≈ sort(complex.(LA.eigvals(G));by=x->(real(x),imag(x))) rtol=1e-6
    Sc=OB.schur(G); @test Sc.Z*Sc.T*Sc.Z' ≈ G rtol=r
    B=randn(T,7,7); Q=OB.schur(G,B)
    @test Q.Q*Q.S*Q.Z' ≈ G rtol=1e-6
    @test Q.Q*Q.T*Q.Z' ≈ B rtol=1e-6
    Hs=OB.hessenberg(G); @test Matrix(Hs.Q)*Matrix(Hs.H)*Matrix(Hs.Q)' ≈ G rtol=r
    @test Hs.Q'*(Hs.Q*b[1:7]) ≈ b[1:7] rtol=r
    @test !hasmethod(LA.ldiv!,Tuple{typeof(Sc),Vector{T}})
end

@testset "generalized eigen, Sylvester, Lyapunov $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    r=_api_rt(T); n=8
    H=_api_spd(T,n); K=_api_indef(T,n); Ah=Matrix(K); Bh=Matrix(H)
    G=OB.eigen(K,H)
    @test G isa OB.GeneralizedEigen
    @test Ah*G.vectors ≈ Bh*G.vectors*LA.Diagonal(G.values) rtol=sqrt(r)
    @test G.vectors'*Bh*G.vectors ≈ LA.I rtol=sqrt(r)
    @test sort(OB.eigvals(K,H)) ≈ sort(LA.eigvals(LA.Hermitian(Ah),LA.Hermitian(Bh))) rtol=sqrt(r)
    λ,V=G; @test λ===G.values
    @test_throws LA.PosDefException OB.eigen(H,K)
    A=randn(T,n,n); B=randn(T,n,n)
    P=OB.eigen(A,B)
    @test A*P.vectors ≈ B*P.vectors*LA.Diagonal(P.values) rtol=sqrt(r)
    @test length(OB.eigvals(A,B))==n
    p,q=7,5; A=randn(T,p,p); B=randn(T,q,q); C=randn(T,p,q)
    X=OB.sylvester(A,B,C)
    @test A*X+X*B+C ≈ zeros(T,p,q) atol=sqrt(r)*LA.norm(C)
    @test X ≈ LA.sylvester(A,B,C) rtol=sqrt(r)
    S=randn(T,n,n)-3n*LA.I; Q=Matrix(_api_spd(T,n))
    Y=OB.lyap(S,Q)
    @test S*Y+Y*S'+Q ≈ zeros(T,n,n) atol=sqrt(r)*LA.norm(Q)
    @test Y ≈ LA.lyap(S,Q) rtol=sqrt(r)
end

@testset "ordschur, norms, kron, eigmax/eigmin, eigen options $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    r=_api_rt(T); n=8
    A=randn(T,n,n); F=OB.schur(A)
    sel=[real(λ)>0 for λ in F.values]
    G=OB.ordschur(F,sel)
    @test G.Z*G.T*G.Z' ≈ A rtol=sqrt(r)
    k=count(sel); @test all(real(λ)>0 for λ in G.values[1:k])
    @test LA.ordschur(F,sel) isa OB.Schur
    H=OB.schur(A); same=H.engine; OB.ordschur!(H,sel)
    @test H.engine!==same
    @test H.Z*H.T*H.Z' ≈ A rtol=sqrt(r)
    B=randn(T,n,n); Q=OB.schur(A,B); qs=[abs(λ)<1 for λ in Q.values]
    R=OB.ordschur(Q,qs)
    @test R.Q*R.S*R.Z' ≈ A rtol=sqrt(r)
    @test R.Q*R.T*R.Z' ≈ B rtol=sqrt(r)
    @test all(abs(λ)<1 for λ in R.values[1:count(qs)])
    @test OB.norm(A) ≈ LA.norm(A) rtol=r
    @test OB.norm(A,1) ≈ LA.norm(A,1) rtol=r
    @test OB.norm(A,Inf) ≈ LA.norm(A,Inf) rtol=r
    @test OB.norm(A,3) ≈ LA.norm(A,3) rtol=r
    x=randn(T,n); @test OB.norm(x) ≈ LA.norm(x) rtol=r
    @test OB.norm(x,1) ≈ LA.norm(x,1) rtol=r
    @test OB.norm(1:4) ≈ LA.norm(1:4)
    for p in (1,2,Inf); @test OB.opnorm(A,p) ≈ LA.opnorm(A,p) rtol=sqrt(r); end
    C=randn(T,3,2); @test OB.kron(A,C) ≈ LA.kron(A,C) rtol=r
    Hm=_api_spd(T,n)
    @test OB.eigmax(Hm) ≈ LA.eigmax(Hm) rtol=r
    @test OB.eigmin(Hm) ≈ LA.eigmin(Hm) rtol=r
    Rot=T[0 -1; 1 0]; @test_throws DomainError OB.eigmax(Rot)   # eigenvalues ±i
    Up=T[3 1 2; 0 -1 5; 0 0 2]   # not symmetric, real eigenvalues (stored as complex with zero imaginary part)
    @test OB.eigmax(Up) ≈ 3
    @test OB.eigmin(Up) ≈ -1
    λ=OB.eigvals(A); @test issorted(λ;by=OB._eigsortby)
    E=OB.eigen(A); @test issorted(E.values;by=OB._eigsortby)
    @test A*E.vectors ≈ E.vectors*LA.Diagonal(E.values) rtol=sqrt(r)
    for (pm,sc) in ((false,false),(true,false),(false,true))
        Eb=OB.eigen(A;permute=pm,scale=sc); @test A*Eb.vectors ≈ Eb.vectors*LA.Diagonal(Eb.values) rtol=sqrt(r)
    end
    μ=OB.eigvals(A;sortby=abs); @test issorted(μ;by=abs)   # conjugate pairs tie: compare the sets below
    @test sort(μ;by=OB._eigsortby) ≈ sort(λ;by=OB._eigsortby) rtol=sqrt(r)
    W=T[1 1e6; 1e-6 1]; @test sort(OB.eigvals(W);by=abs) ≈ sort(complex.(LA.eigvals(W));by=abs) rtol=sqrt(r)
end

@testset "matrix functions exp/log/sqrt/sin/cos $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    r=_api_rt(T); n=6
    A=randn(T,n,n)/4; P=Matrix(_api_spd(T,n))
    @test OB.exp(A) ≈ exp(A) rtol=sqrt(r)
    @test OB.sin(A) ≈ sin(A) rtol=sqrt(r)
    @test OB.cos(A) ≈ cos(A) rtol=sqrt(r)
    @test OB.sqrt(P) ≈ sqrt(P) rtol=sqrt(r)
    @test OB.log(P) ≈ log(P) rtol=sqrt(r)
    @test OB.exp(OB.log(P)) ≈ P rtol=sqrt(r)
    @test OB.sqrt(P)*OB.sqrt(P) ≈ P rtol=sqrt(r)
    @test OB.exp(LA.Hermitian(P)/n) ≈ exp(P/n) rtol=sqrt(r)
    D=LA.Diagonal(rand(real(T),n).+1)
    for f in (OB.exp,OB.log,OB.sqrt,OB.sin,OB.cos)
        @test f(D) isa LA.Diagonal
    end
    @test OB.exp(D) ≈ exp(D) rtol=r
    @test OB.sqrt(T(4))==sqrt(T(4))
    @test OB.exp(T(1))==exp(T(1))
    @test OB.log(2,8.0)==3.0
    @test OB.exp !== Base.exp
    @test_throws MethodError OB.exp(sparse(T(1)*LA.I,3,3))           # no fallback to SparseArrays/LinearAlgebra
    @test_throws MethodError OB.sqrt(LA.UpperHessenberg(ones(3,3)))   # no method: no Base/LAPACK fallback
end

@testset "sparse shift, fkeep!, blockdiag $T" for T in (Float64,ComplexF64)
    Dg=OB.spdiagm(-1=>T[1,2,3],0=>T[4,5,6,7],2=>T[8,9])
    @test Dg isa CSCMatrix
    @test Matrix(Dg)==Matrix(SparseArrays.spdiagm(-1=>T[1,2,3],0=>T[4,5,6,7],2=>T[8,9]))
    @test Matrix(OB.spdiagm(0=>T[1,2],0=>T[3,4]))==Matrix(SparseArrays.spdiagm(0=>T[1,2],0=>T[3,4]))
    @test size(OB.spdiagm(5,6,1=>T[1,2]))==(5,6)
    @test Matrix(OB.spdiagm(T[1,0,2]))==Matrix(SparseArrays.spdiagm(T[1,0,2]))
    @test eltype(OB.spdiagm(0=>[1,2]))==Float64
    sv=OB.sparsevec([1,4,4],T[1,2,3],6)
    @test sv isa OB.SparseTensor{T,1}
    @test Vector(sv)==Vector(SparseArrays.sparsevec([1,4,4],T[1,2,3],6))
    @test Vector(OB.sparsevec(T[0,1,0,2]))==T[0,1,0,2]
    r=_api_rt(T); n=40; b=randn(T,n)
    S=sprandn(T,n,n,0.1); P=S*S'; Pd=Matrix(P)
    C=OB.cholesky(P;shift=2.5); @test C\b ≈ (Pd+2.5LA.I)\b rtol=r
    @test LA.cholesky(CSRMatrix(P);shift=2.5)\b ≈ (Pd+2.5LA.I)\b rtol=r
    @test OB.ldlt(P;shift=1.0)\b ≈ (Pd+LA.I)\b rtol=r
    D=OB.cholesky(P;shift=1.0); OB.cholesky!(D,P;shift=4.0); @test D\b ≈ (Pd+4LA.I)\b rtol=r
    for mk in (CSRMatrix,CSCMatrix)
        Y=mk(S); SparseArrays.fkeep!((i,j,x)->i>=j,Y)
        @test Matrix(Y) ≈ LA.tril(Matrix(S))
        @test SparseArrays.nnz(Y)==SparseArrays.nnz(LA.tril(S))
    end
    K=SparseArrays.blockdiag(CSRMatrix(S),CSCMatrix(sparse(T(2)*LA.I,3,3)))
    @test K isa CSCMatrix
    @test Matrix(K) ≈ Matrix(SparseArrays.blockdiag(S,sparse(T(2)*LA.I,3,3)))
end

@testset "matrix functions $T" for T in (Float64,ComplexF64)
    r=_api_rt(T); A=randn(T,8,8); W=randn(T,8,5)*randn(T,5,6)
    @test OB.rank(W)==5 && OB.rank(A)==8
    @test OB.cond(A) ≈ LA.cond(A) rtol=1e-6
    @test OB.cond(A,1) ≈ LA.cond(A,1) rtol=1e-6
    @test OB.pinv(W) ≈ LA.pinv(W) rtol=1e-6
    N=OB.nullspace(W); @test size(N,2)==1 && LA.norm(W*N)<1e-8
    Pd=Matrix(_api_spd(T,8)); @test OB.logdet(Pd) ≈ LA.logdet(Pd) rtol=r
    @test OB.factorize(A) isa OB.LU && OB.factorize(W) isa OB.QRPivoted
    @test_throws MethodError OB.lu(LA.UpperHessenberg(randn(3,3)))   # no engine: no silent fallback
end

@testset "sparse $T" for T in (Float64,ComplexF64)
    r=_api_rt(T); n=60
    S=sprandn(T,n,n,0.08)+sparse(T(n)*LA.I,n,n); b=randn(T,n); D=Matrix(S)
    for X in (S,CSRMatrix(S),CSCMatrix(S))
        F=OB.lu(X)
        @test F isa OB.LU
        @test F\b ≈ D\b rtol=r
        @test F'\b ≈ D'\b rtol=r
        @test transpose(F)\b ≈ transpose(D)\b rtol=r
        @test F.L*F.U ≈ S[F.p,F.q] rtol=r
        @test LA.det(F) ≈ LA.det(D) rtol=r
        @test SparseArrays.nnz(F)>0
    end
    @test LA.lu(CSRMatrix(S)) isa OB.LU && LA.factorize(CSRMatrix(S)) isa OB.LU
    @test CSRMatrix(S)\b ≈ D\b rtol=r            # OxiBLAS `\` (linalg_api_solve.jl) -> factorize -> LU
    F=OB.lu(S); S2=copy(S); S2.nzval .*= 2; OB.lu!(F,S2); @test F\b ≈ (2D)\b rtol=r
    P=S*S'+sparse(T(n)*LA.I,n,n); Pd=Matrix(P)
    for X in (P,LA.Hermitian(P,:U),LA.Hermitian(CSRMatrix(P),:L))
        C=OB.cholesky(X)
        @test C\b ≈ Pd\b rtol=r
        @test C.PtL*C.PtL' ≈ P rtol=r
        @test LA.logdet(C) ≈ LA.logdet(Pd) rtol=r
        @test OB.logdet(X) ≈ LA.logdet(Pd) rtol=r
        @test OB.logabsdet(X)[1] ≈ LA.logabsdet(Pd)[1] rtol=r
        @test OB.logabsdet(X)[2] ≈ one(T) rtol=r
        @test OB.norm(X) ≈ LA.norm(Pd) rtol=r
        @test OB.opnorm(X,1) ≈ LA.opnorm(Pd,1) rtol=r
        @test OB.opnorm(X,Inf) ≈ LA.opnorm(Pd,Inf) rtol=r
    end
    @test LA.cholesky(LA.Hermitian(CSRMatrix(P))) isa OB.Cholesky
    @test LA.logdet(LA.Hermitian(CSRMatrix(P))) ≈ LA.logdet(Pd) rtol=r
    C=OB.cholesky(P); P2=copy(P); P2.nzval .*= 3; OB.cholesky!(C,P2); @test C\b ≈ (3Pd)\b rtol=r
    @test OB.factorize(P) isa OB.Cholesky
    K=S+S'; K[1,1]=-K[1,1]-n; Kd=Matrix(K)
    @test OB.factorize(K) isa OB.BunchKaufman
    @test OB.bunchkaufman(K)\b ≈ Kd\b rtol=r
    @test OB.logabsdet(K)[1] ≈ LA.logabsdet(Kd)[1] rtol=r
    @test OB.logabsdet(K)[2] ≈ LA.logabsdet(Kd)[2] rtol=r
    @test OB.logabsdet(S)[1] ≈ LA.logabsdet(D)[1] rtol=r
    @test OB.logabsdet(S)[2] ≈ LA.logabsdet(D)[2] rtol=r
    @test OB.norm(CSRMatrix(S)) ≈ LA.norm(D) rtol=r
    @test OB.opnorm(CSRMatrix(S),1) ≈ LA.opnorm(D,1) rtol=r
    @test OB.opnorm(CSRMatrix(S),Inf) ≈ LA.opnorm(D,Inf) rtol=r
    Ssing=copy(S); Ssing[:,2].=0; SparseArrays.dropzeros!(Ssing)
    @test OB.det(Ssing) == 0 && OB.logabsdet(Ssing)[1] == -Inf && iszero(OB.logabsdet(Ssing)[2])
    @test_throws DimensionMismatch OB.det(sprandn(T,4,3,0.5))
    @test OB.ldlt(P)\b ≈ Pd\b rtol=r
    @test OB.ldlt(P) isa OB.LDLt
    M=sprandn(T,3n,n,0.1)+[sparse(T(n)*LA.I,n,n);spzeros(T,2n,n)]; c=randn(T,3n); Md=Matrix(M)
    Q=OB.qr(M)
    @test Q isa OB.QRSparse
    @test Q\c ≈ Md\c rtol=r
    @test Matrix(Q.Q)[:,1:n]*Matrix(Q.R) ≈ Md[Q.prow,Q.pcol] rtol=r
    @test Q.prow==1:3n
    @test Q.Q'*(Q.Q*c) ≈ c rtol=r
    @test OB.factorize(M) isa OB.QRSparse && LA.qr(CSRMatrix(M)) isa OB.QRSparse
    @test_throws ArgumentError OB.inv(S)
    @test OB.isposdef(P) && !OB.isposdef(K)
end

end
