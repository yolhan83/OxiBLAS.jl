# Prepared only: supervised numerical execution requires user authorization.
qsrel(A,B)=LA.norm(A-B)/max(LA.norm(A),LA.norm(B),floatmin(real(eltype(A))))
function qscheck(F,A,B;factor=2000)
    T=eltype(A); R=real(T); n=size(A,1); bound=factor*max(1,n)*eps(R)
    @test qsrel(A,F.Q*F.S*F.Z')<=bound
    @test qsrel(B,F.Q*F.T*F.Z')<=bound
    @test LA.norm(F.Q'*F.Q-Matrix{T}(LA.I,n,n))<=bound
    @test LA.norm(F.Z'*F.Z-Matrix{T}(LA.I,n,n))<=bound
    @test all(iszero(F.T[i,j]) for j in 1:n for i in j+1:n)
    @test all(iszero(F.S[i,j]) for j in 1:n for i in j+(T<:Real ? 2 : 1):n)
    T<:Real && @test all(iszero(F.S[i,i-1]) || iszero(F.S[i+1,i]) for i in 2:n-1)
end
function qsblock(::Type{T},p,center) where T
    p==1 ? reshape(T[center],1,1) : T[center -2; 3 center]
end

@testset "QZ real/complex: full pencil invariants" begin
    for T in TYPES, n in (0,1,2,3,7,17)
        A=randn(T,n,n); B=randn(T,n,n)+T(3)*Matrix{T}(LA.I,n,n)
        oldA=copy(A); oldB=copy(B); F=qz_factor(A,B)
        qscheck(F,A,B); @test A==oldA && B==oldB
        @test eltype(F.S)===T
        n>0 && @test spectrum_error(F.values,LA.eigvals(A,B))<=tol(T)*max(1,nrm2(A),maximum(abs,F.values))*100
        if n>0
            E=eigen_pencil(A,B;left=true)
            for j in 1:n
                a,b=E.alpha[j],E.beta[j]; v=E.vectors[:,j]; w=E.left_vectors[:,j]
                scale=max(abs(a),abs(b)); a/=scale; b/=scale
                @test LA.norm(b*A*v-a*B*v)<=tol(T)*max(1,nrm2(A),nrm2(B))*100
                @test LA.norm(conj(b)*A'*w-conj(a)*B'*w)<=tol(T)*max(1,nrm2(A),nrm2(B))*100
            end
        end
    end
    for T in (Float32,Float64)
        # Rust's extraction ignores B12 and incorrectly uses H's discriminant.
        A=T[1 2;-3 4]; B=T[2 5;0 3]; F=qz_factor(A,B); qscheck(F,A,B)
        expected=Complex{T}.([(26+sqrt(T(436)))/12,(26-sqrt(T(436)))/12])
        @test spectrum_error(F.values,expected)<=100tol(T)
        A=T[0 -1;1 0]; B=T[2 1;0 3]; F=qz_factor(A,B); qscheck(F,A,B)
        expected=Complex{T}[(-1+sqrt(complex(T(-23))))/12,(-1-sqrt(complex(T(-23))))/12]
        @test spectrum_error(F.values,expected)<=100tol(T)
        @test !iszero(F.S[2,1]); @test all(iszero,imag.(F.beta))
        @test isapprox(F.values[1],conj(F.values[2]);rtol=100eps(T))
        # Correct reduction applies each left transform to BOTH matrices.
        A=T[1 2 3;4 5 6;7 8 10]; B=Matrix{T}(LA.I,3,3)
        H=copy(A); R=copy(B); Q=copy(B); Z=copy(B)
        OxiBLAS._qz_reduce!(H,R,Q,Z,1,3)
        @test qsrel(A,Q*H*Z')<=100tol(T)
        @test qsrel(B,Q*R*Z')<=100tol(T)
        @test iszero(H[3,1]) && all(iszero(R[i,j]) for j in 1:3 for i in j+1:3)
        # Infinite / indeterminate pairs retain their homogeneous representation.
        A=T[1 2 0;0 3 1;0 0 4]; B=T[0 1 0;0 2 0;0 0 3]
        F=qz_factor(A,B); qscheck(F,A,B)
        @test count(iszero,F.beta)==1
        F=qz_factor(zeros(T,2,2),zeros(T,2,2)); @test all(isnan,F.values)
        @test_throws ArgumentError eigen_pencil(zeros(T,2,2),zeros(T,2,2))
        A=T[0 -1;1 0]; B=Matrix{T}(LA.I,2,2)
        F=qz_factor(A,B;form=:complex); @test eltype(F.S)===Complex{T}
        @test qsrel(A,F.Q*F.S*F.Z')<=100tol(T)
    end
    @test_throws DimensionMismatch qz_factor(ones(3,3),ones(2,2))
    @test_throws ArgumentError qz_factor([Inf 0.;0 1],ones(2,2))
    @test_throws ArgumentError qz_factor(ones(2,2),ones(2,2);form=:invalid)
    @test_throws ConvergenceError qz_factor(randn(7,7),Matrix{Float64}(LA.I,7,7);maxiter=0)
    parent=randn(4,5); old=copy(parent); A=view(parent,:,1:4); B=view(parent,:,2:5)
    F=qz_factor(A,B); qscheck(F,A,B); @test parent==old
    A=randn(5,5); old=copy(A); F=qz_factor(A,A)
    qscheck(F,A,A); @test A==old
end

@testset "Real Schur and generalized swaps: all four block sizes" begin
    for T in (Float32,Float64), p in (1,2), q in (1,2)
        n=p+q; S=zeros(T,n,n); S[1:p,1:p]=qsblock(T,p,1); S[p+1:n,p+1:n]=qsblock(T,q,5)
        S[1:p,p+1:n].=T(1)/8
        I=Matrix{T}(LA.I,n,n); vals=OxiBLAS._schur_values(S,one(T))
        F=SchurFactor(copy(S),copy(I),vals,0); G=schur_reorder(F,z->real(z)>3)
        @test eltype(G.T)===T; @test qsrel(S,G.Z*G.T*G.Z')<=100tol(T)
        @test all(real.(G.values[1:q]).>3)
        @test S==F.T && F.Z==I
        for block in OxiBLAS._schur_blocks(G.T)
            if length(block)==2
                @test G.T[first(block),last(block)]*G.T[last(block),first(block)]<0
            end
        end
        B=copy(I); B[1:p,p+1:n].=T(1)/16
        a,b=OxiBLAS._qz_pairs(S,B); QF=QZFactor(copy(S),B,copy(I),copy(I),a,b,0)
        QG=qz_reorder(QF,z->real(z)>3); qscheck(QG,S,B)
        @test all(real.(QG.values[1:q]).>3)
        if q==2
            H=schur_reorder(F,z->imag(z)>0 && real(z)>3)
            @test all(real.(H.values[1:2]).>3) # selecting one member moves BOTH
        end
    end
    for T in (ComplexF32,ComplexF64)
        S=T[1 .2 .1;0 2 .3;0 0 4]; B=T[2 .1 0;0 1 .1;0 0 1]
        I=Matrix{T}(LA.I,3,3); a,b=OxiBLAS._qz_pairs(S,B)
        G=qz_reorder(QZFactor(S,B,I,I,a,b,0),z->real(z)>3); qscheck(G,S,B)
        @test real(G.values[1])>3
    end
end

@testset "Block TREVC / TRSNA, preserving preceding and following real pairs" begin
    for T in TYPES
        S=zeros(T,6,6)
        if T<:Real
            S[1:2,1:2]=T[1 -2;3 1]; S[3,3]=3; S[4:5,4:5]=T[5 -4;2 5]; S[6,6]=8
        else
            for i in 1:6
                S[i,i]=T(i,i/8)
            end
        end
        for j in 2:6, i in 1:j-1
            iszero(S[i,j]) && (S[i,j]=T(1)/16)
        end
        values=OxiBLAS._schur_values(S,one(real(T)))
        X=trevc_right(S;packed=false); Y=trevc_left(S;packed=false)
        @test qsrel(S*X,X*LA.Diagonal(values))<=100tol(T)
        @test qsrel(S'*Y,Y*LA.Diagonal(conj.(values)))<=100tol(T)
        s=trsna_s(S); expected=[abs(LA.dot(Y[:,j],X[:,j])) for j in 1:6]
        @test LA.norm(s-expected)<=100tol(T)
        @test all(z->0<=z<=1,s)
        gap=trsna_sep(S); @test all(gap.>=0)
        if T<:Real
            V=trevc_right(S); W=trevc_left(S)
            for block in OxiBLAS._schur_blocks(S)
                j=first(block)
                if length(block)==2
                    @test LA.norm(V[:,j]+im*V[:,j+1]-X[:,j])<=100tol(T)
                    @test LA.norm(W[:,j]+im*W[:,j+1]-Y[:,j])<=100tol(T)
                    @test abs(s[j]-s[j+1])<=100tol(T)
                end
            end
        end
        D=T[1 0 0;0 1 0;0 0 3]
        @test trsna_sep(D)[1:2]==[0,0]
        @test LA.norm(trsna_s(D).-1)<=100tol(T)
    end
end

@testset "Coupled block Sylvester and pencil conditions" begin
    for T in TYPES
        A=T[1 .2;0 2]; D=T[2 .1;0 3]; B=T[5 .2;0 7]; E=T[1 .1;0 2]
        if T<:Real
            A=T[1 -2;3 1]; B=T[6 -1;2 6]
        end
        R=randn(T,2,2); L=randn(T,2,2); C=A*R-L*B; F=D*R-L*E
        G=generalized_sylvester(A,D,B,E,C,F)
        @test qsrel(A*G.R-G.L*B,G.scale*C)<=100tol(T)
        @test qsrel(D*G.R-G.L*E,G.scale*F)<=100tol(T)
        # Independent explicit small Kronecker reference belongs ONLY in tests.
        K=[LA.kron(Matrix{T}(LA.I,2,2),A) -LA.kron(transpose(B),Matrix{T}(LA.I,2,2));
           LA.kron(Matrix{T}(LA.I,2,2),D) -LA.kron(transpose(E),Matrix{T}(LA.I,2,2))]
        c=randn(T,2,2); f=randn(T,2,2)
        x,y=OxiBLAS._sylvester_inverse(A,D,B,E,c,f;adj=true)
        @test qsrel(K'*vcat(vec(x),vec(y)),vcat(vec(c),vec(f)))<=100tol(T)
        estimate=OxiBLAS._coupled_sep(A,D,B,E)
        exact=inv(LA.opnorm(inv(K),1)); @test exact/10<=estimate<=10exact
        S=T[1 .2 0;0 2 .3;0 0 5]; BB=T[2 .1 0;0 1 .1;0 0 1]
        Q=qz_factor(S,BB); report=qz_condition(Q;split=1)
        @test all(report.eigenvalue.>=0); @test 0<=report.left<=1 && 0<=report.right<=1
        @test report.separation>0
        SF=schur_factor(S); report=schur_condition(SF;split=1)
        @test all(report.eigenvalue.<=1); @test report.subspace.separation>0
    end
    F=qz_factor([0. -1;1 0],Matrix{Float64}(LA.I,2,2))
    @test_throws ArgumentError qz_condition(F;split=1)
    # No hidden n^4 operator allocation: implementation uses local systems <=8.
    @test_throws DimensionMismatch generalized_sylvester(ones(2,2),ones(2,2),ones(1,1),ones(1,1),ones(3,1),ones(3,1))
    # The unscaled solution cannot fit in Float64: local pivot scaling must
    # precede division, not merely detect Inf after it occurred.
    A=reshape([1e-320],1,1); I=ones(1,1); O=zeros(1,1)
    G=generalized_sylvester(A,I,O,I,I,O)
    @test 0<G.scale<1 && all(isfinite,G.R) && all(isfinite,G.L)
    @test isapprox(A*G.R,G.scale*I;rtol=1e-12,atol=0)
    @test G.R==G.L
end

@testset "Simultaneous small-bulge train and blocked far updates" begin
    for T in TYPES, blocksize in (1,5,13), nshifts in (1,2,6)
        n=23; A=randn(T,n,n); HF=hessenberg_factor(A)
        H=OxiBLAS._hessenberg_matrix(HF;rescale=false); Q=hessenberg_q(HF); original=copy(H)
        shifts=schur_factor(H[n-7:n,n-7:n]).T
        packets=OxiBLAS._shift_packets(shifts,nshifts)
        w=SchurWorkspace(T,n;blocksize,nshifts)
        QQ=Matrix{T}(LA.I,n,n)
        OxiBLAS._small_bulge_train!(H,QQ,1,n,packets,w,blocksize,false)
        @test qsrel(original,QQ*H*QQ')<=100tol(T)
        @test LA.norm(QQ'*QQ-Matrix{T}(LA.I,n,n))<=100tol(T)
        @test all(abs(H[i,j])<=100tol(T) for j in 1:n for i in j+2:n)
        # Small window forces actual train execution, rather than all-window AED.
        F=schur_multishift(A;window=4,nshifts,blocksize,workspace=w)
        @test eltype(F.T)===T
        @test qsrel(A,F.Z*F.T*F.Z')<=100tol(T)
        @test spectrum_error(F.values,LA.eigvals(A))<=tol(T)*nrm2(A)*500
    end
    for T in TYPES
        A=randn(T,37,37); w=SchurWorkspace(T,37;blocksize=5,nshifts=4)
        before=copy(A); F=schur_multishift(A;window=6,blocksize=5,nshifts=4,workspace=w)
        GC.gc() # no pointers survive a call; retained buffers remain rooted
        @test A==before
        jobs=Task[]
        for _ in 1:2
            push!(jobs,Threads.@spawn schur_multishift(copy(A);window=6,blocksize=5,nshifts=4,workspace=w))
        end
        for task in jobs
            G=fetch(task); @test qsrel(A,G.Z*G.T*G.Z')<=200tol(T)
        end
        @test_throws DimensionMismatch schur_multishift(A;workspace=SchurWorkspace(T,2))
        @test_throws ArgumentError schur_multishift(A;workspace=SchurWorkspace(T===Float64 ? Float32 : Float64,37))
        @test_throws ArgumentError schur_multishift(w.localH;window=6,blocksize=5,nshifts=4,workspace=w)
    end
    @test_throws ConvergenceError schur_multishift(randn(13,13);window=3,maxiter=0)
end

@testset "Generic precision, strided inputs and extreme scales" begin
    setprecision(BigFloat,192) do
        A=BigFloat[0 -1 1;1 0 2;0 0 4]; B=BigFloat[2 1 0;0 3 1;0 0 1]
        F=qz_factor(A,B); qscheck(F,A,B;factor=10000)
        G=schur_multishift(A;window=2,nshifts=2,blocksize=2)
        @test qsrel(A,G.Z*G.T*G.Z')<=10000eps(BigFloat)
        E=eigen_pencil(A,B;left=true)
        for j in 1:3
            @test LA.norm(E.beta[j]*A*E.vectors[:,j]-E.alpha[j]*B*E.vectors[:,j])<=10000eps(BigFloat)
        end
    end
    for T in (Float32,Float64)
        A0=randn(T,14,14); B0=randn(T,14,14)+10*Matrix{T}(LA.I,14,14)
        A=view(A0,14:-2:2,14:-2:2); B=view(B0,14:-2:2,14:-2:2)
        F=qz_factor(A,B); qscheck(F,A,B)
        for exponent in (T===Float32 ? (-60,60) : (-600,600))
            scale=ldexp(one(T),exponent); AA=T[1 2;-3 4].*scale; BB=T[2 5;0 3].*scale
            qscheck(qz_factor(AA,BB),AA,BB)
        end
    end
    # Distinct A/B scales must not turn a nonzero beta into an exact infinity.
    A=[1e200 0.;0 2e200]; B=[1e-200 0.;0 2e-200]
    E=eigen_pencil(A,B;left=true)
    @test all(!iszero(b) for b in E.beta)
    for j in 1:2
        @test LA.norm(E.beta[j]*A*E.vectors[:,j]-E.alpha[j]*B*E.vectors[:,j])<=1e-12
    end
end

@testset "QZ/Schur optional scalar providers and half working precision" begin
    setprecision(BigFloat,256) do
        for T in EXT_TYPES
            T in TYPES && continue
            A=T[1 -.5 .25;.5 1 .125;.125 .25 4]; B=T[2 .125 0;0 2 .125;0 0 1]
            F=qz_factor(A,B); G=qz_reorder(F,z->real(z)>2)
            check_extended_array(extref(G.Q)*extref(G.S)*extref(G.Z)',extref(A),T;factor=3000)
            check_extended_array(extref(G.Q)*extref(G.T)*extref(G.Z)',extref(B),T;factor=3000)
            report=qz_condition(G;split=1)
            @test report.separation>=0 && 0<=report.left<=1 && 0<=report.right<=1
            E=eigen_pencil(A,B;left=true)
            V=extref(E.vectors); W=extref(E.left_vectors)
            for j in 1:3
                a,b=extref(E.alpha[j]),extref(E.beta[j])
                check_extended_array(b*extref(A)*V[:,j],a*extref(B)*V[:,j],T;factor=3000)
                check_extended_array(conj(b)*extref(A)'*W[:,j],conj(a)*extref(B)'*W[:,j],T;factor=3000)
            end
            context=SchurWorkspace(T,3;blocksize=1,nshifts=2)
            S=schur_multishift(A;window=2,nshifts=2,blocksize=1,workspace=context)
            check_extended_array(extref(S.Z)*extref(S.T)*extref(S.Z)',extref(A),T;factor=3000)
        end
    end
end
