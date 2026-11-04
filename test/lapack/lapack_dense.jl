# Dense LU/Cholesky/QR family: backward errors, pivots, orthogonality,
# singular and ill-conditioned inputs, rectangular shapes, block sizes.
# Errors are normwise and scaled by size and eps, as in LAPACK's test suite.
_bwd(E,A)=LA.norm(E)/(max(LA.norm(A),floatmin(real(eltype(A))))*max(1,size(A)...)*eps(real(eltype(A))))
_orth(Q)=LA.norm(Q'*Q-LA.I)/(max(1,size(Q,2))*eps(real(eltype(Q))))
_spd(T,n)=(M=randn(T,n,n); M'*M+n*LA.I)

@testset "Dense LU" begin
    for T in TYPES, (m,n,bs) in ((1,1,32),(5,5,1),(17,17,16),(33,20,16),(20,33,16),(64,64,32),(130,97,32),(97,130,64),(257,257,32),(257,257,64))
        A=randn(T,m,n); F=lu_factor(A;blocksize=bs)
        @test _bwd(F.L*F.U-A[F.p,:],A)<=30
        # Same partial-pivot choices as getrf for real data; complex getrf
        # ranks |re|+|im| (izamax), OxiBLAS the modulus.
        T<:Real && @test F.p==LA.lu(A;check=false).p
        if m==n
            b=randn(T,n); B=randn(T,n,3)
            @test _bwd(A*solve(F,b)-b,A)<=30*LA.cond(A)
            @test _bwd(A*solve(F,B)-B,A)<=30*LA.cond(A)
            @test _bwd(transpose(A)*solve_transposed(F,b;trans=:T)-b,A)<=30*LA.cond(A)
            @test _bwd(A'*solve_transposed(F,b;trans=:C)-b,A)<=30*LA.cond(A)
        end
    end
    for T in TYPES
        # Exactly singular: zero column inside a block, info reported.
        A=randn(T,40,40); A[:,23].=0
        @test_throws OxiBLAS.SingularException lu_factor(A)
        F=lu_factor(A;check=false); @test F.info==23 && _bwd(F.L*F.U-A[F.p,:],A)<=30
        # Ill-conditioned (graded columns) and nearly singular.
        G=randn(T,80,80)*LA.Diagonal(real(T)(10) .^ range(0,real(T)===Float32 ? -5 : -12;length=80))
        F=lu_factor(G); @test _bwd(F.L*F.U-G[F.p,:],G)<=30
        N=randn(T,60,60); N[:,end]=N[:,1]+eps(real(T))*randn(T,60)
        F=lu_factor(N;check=false); @test _bwd(F.L*F.U-N[F.p,:],N)<=30
        # Lookahead path (threaded, m*n*q >= 2^27): same factors and pivots as
        # the sequential order, also with a singular column inside a panel.
        A=randn(T,600,600); F1=lu_factor(A); F2=lu_factor(A;threaded=false)
        @test F1.factors==F2.factors && F1.p==F2.p && _bwd(F1.L*F1.U-A[F1.p,:],A)<=30
        A[:,300].=0; F1=lu_factor(A;check=false); F2=lu_factor(A;check=false,threaded=false)
        @test F1.info==F2.info==300 && F1.factors==F2.factors
        # Views and strided input.
        P=randn(T,70,60); V=view(P,3:52,5:54); F=lu_factor(V)
        @test _bwd(F.L*F.U-V[F.p,:],V)<=30
        # Inverse (getri path: TRTRI then right TRSM) and determinant.
        for n in (1,2,15,16,17,70,129)
            A=randn(T,n,n)+n*LA.I
            @test LA.norm(A*native_inverse(A)-LA.I)<=30*n*eps(real(T))*LA.cond(A)
        end
        @test_throws OxiBLAS.SingularException native_inverse(zeros(T,4,4))
        # Determinant of a well-scaled matrix (det(A) above overflows Float32).
        D=LA.I+randn(T,70,70)/(4*sqrt(real(T)(70)))
        @test abs(determinant(D)-LA.det(D))<=1000*70*eps(real(T))*abs(LA.det(D))
    end
    setprecision(BigFloat,128) do
        A=BigFloat.(randn(30,30))+30LA.I; F=lu_factor(A;blocksize=8)
        @test _bwd(F.L*F.U-A[F.p,:],A)<=30
    end
end

@testset "Dense Cholesky" begin
    for T in TYPES, (n,bs) in ((1,32),(2,1),(7,16),(33,16),(64,32),(97,32),(130,64),(257,32))
        A=_spd(T,n)
        # Only the lower triangle is read: poison the strict upper one.
        X=copy(A); for j in 2:n, i in 1:j-1; X[i,j]=T(NaN); end
        F=cholesky_factor(X;blocksize=bs)
        @test LA.istril(F.L) && all(isreal,LA.diag(F.L)) && all(>(0),real(LA.diag(F.L)))
        @test _bwd(F.L*F.L'-A,A)<=30
        b=randn(T,n); B=randn(T,n,4)
        @test _bwd(A*solve(F,b)-b,A)<=30*LA.cond(A)
        @test _bwd(A*solve(F,B)-B,A)<=30*LA.cond(A)
        Ai=cholesky_inverse(F)   # potri
        @test Ai==Ai' && LA.norm(A*Ai-LA.I)<=30*n*eps(real(T))*LA.cond(A)
    end
    for T in TYPES
        A=_spd(T,50); A[30,30]=-A[30,30]
        @test_throws OxiBLAS.PosDefException cholesky_factor(A)
        # Ill-conditioned SPD.
        D=LA.Diagonal(real(T)(10) .^ range(0,real(T)===Float32 ? -4 : -10;length=60)); Q=LA.qr(randn(T,60,60)).Q
        S=Matrix(LA.Hermitian(Q*D*Q')); F=cholesky_factor(S)
        @test _bwd(F.L*F.L'-S,S)<=100
    end
    setprecision(BigFloat,128) do
        A=_spd(Float64,25); F=cholesky_factor(BigFloat.(A);blocksize=8)
        @test _bwd(F.L*F.L'-BigFloat.(A),BigFloat.(A))<=30
    end
end

@testset "Dense QR family" begin
    for T in TYPES, (m,n,bs) in ((1,1,32),(9,9,1),(40,17,16),(17,40,16),(64,64,32),(130,97,32),(97,130,64),(257,129,32))
        A=randn(T,m,n); F=qr_factor(A;blocksize=bs); Q=F.Q; R=F.R; k=min(m,n)
        @test LA.istriu(R) && _orth(Q)<=30
        @test _bwd(Q*R-A,A)<=30
        @test _orth(qr_q(F))<=30 && qr_q(F)≈Q[:,1:k]
        for side in (:L,:R), tr in (:N,:T,:C)
            B=side===:L ? randn(T,m,5) : randn(T,5,m); Qt=op(Q,tr)
            @test _bwd(apply_q!(copy(B),F;side,trans=tr)-(side===:L ? Qt*B : B*Qt),B)<=30
        end
        if m>=n
            b=randn(T,m); x=least_squares(F,b)
            @test LA.norm(A'*(A*x-b))<=100*m*eps(real(T))*LA.norm(A)^2*max(LA.norm(x),LA.norm(b)/LA.norm(A))*LA.cond(A)
        end
    end
    for T in TYPES
        # Rank-deficient columns: zero and repeated.
        A=randn(T,50,30); A[:,7].=0; A[:,12]=A[:,3]
        F=qr_factor(A); @test _bwd(F.Q*F.R-A,A)<=30 && _orth(F.Q)<=30
        P=qr_pivoted(A); @test _bwd(P.Q*P.R-A[:,P.p],A)<=30
        d=abs.(LA.diag(P.R)); @test all(d[i]>=d[i+1]*(1-100eps(real(T))) for i in 1:length(d)-1)
        for (f,check) in ((lq_factor,F->F.L*F.Q),(rq_factor,F->F.R*F.Q),(ql_factor,F->F.Q*F.L))
            for (m,n) in ((30,50),(50,30),(40,40))
                B=randn(T,m,n); G=f(B); @test _bwd(check(G)-B,B)<=30 && _orth(G.Q)<=30
            end
        end
        V=view(randn(T,90,80),5:84,3:62); F=qr_factor(V); @test _bwd(F.Q*F.R-V,V)<=30
        # Blocked application with enough right-hand sides for the parallel
        # column/row split, both sides and all transposes.
        A=randn(T,300,200); F=qr_factor(A); Q=F.Q
        for side in (:L,:R), tr in (:N,:T,:C)
            B=side===:L ? randn(T,300,160) : randn(T,160,300); Qt=op(Q,tr)
            @test _bwd(apply_q!(copy(B),F;side,trans=tr)-(side===:L ? Qt*B : B*Qt),B)<=30
        end
        @test _orth(qr_q(F))<=30 && qr_q(F;columns=37)≈Q[:,1:37]
    end
    setprecision(BigFloat,128) do
        A=BigFloat.(randn(25,15)); F=qr_factor(A;blocksize=4)
        @test _bwd(F.Q*F.R-A,A)<=30 && _orth(F.Q)<=30
    end
end

@testset "Dense LDL" begin
    for T in TYPES, herm in (true,false), uplo in (:L,:U), (n,bs) in ((1,64),(2,1),(9,3),(33,8),(97,64),(130,32),(257,64))
        M=randn(T,n,n); A=herm ? Matrix(LA.Hermitian(M+M')) : Matrix(LA.Symmetric(M+transpose(M)))
        # Zero diagonal forces 2x2 pivots on an indefinite matrix.
        isodd(n) || (for i in 1:n; A[i,i]=zero(T); end)
        F=ldlt_factor(A;hermitian=herm,uplo,blocksize=bs)
        Lt=herm ? F.L' : transpose(F.L)
        @test _bwd(F.L*F.D*Lt-A[F.p,F.p],A)<=30*max(1,LA.norm(F.L)^2/n)
        @test LA.istril(F.L) && all(==(one(T)),LA.diag(F.L)) && sum(F.blocks)==n
        b=randn(T,n); x=solve(F,b)
        @test _bwd(A*x-b,A)<=30*LA.cond(A)*max(1,LA.norm(F.L)^2/n)
    end
    @test_throws OxiBLAS.SingularException ldlt_factor(zeros(3,3))
end
