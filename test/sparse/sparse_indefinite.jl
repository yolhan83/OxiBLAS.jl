# Prepared numerical/alias/concurrency tests; not executed in supervised phase A.
@testset "Sparse pivoted LDL" begin
    for T in TYPES, hermitian in (true,false), n in (0,1,2,7,13)
        R=real(T); seed=randn(T,n,n); seed[abs.(seed).<R(.6)].=zero(T)
        A=hermitian ? seed+seed' : seed+transpose(seed)
        for i in 1:n
            A[i,i]+=T(isodd(i) ? 4 : -4)
        end
        # Force a nonsingular 2x2 block without a usable diagonal pivot.
        if n==2
            v=T<:Complex ? T(2+im) : T(2)
            A=T[0 (hermitian ? conj(v) : v); v 0]
        end
        for ordering in (:natural,:minimum_degree,collect(n:-1:1)), uplo in (:L,:U)
            stored=copy(A)
            for j in 1:n, i in 1:n
                (uplo===:L ? i<j : i>j) && (stored[i,j]=T(NaN))
            end
            C=CSRMatrix(stored); before=copy(C.values)
            F=sparse_ldlt(C;hermitian,ordering,uplo)
            L=Matrix(F.L); D=Matrix(F.D)
            @test F isa SparseLDLFactor
            @test size(F)==(n,n)
            @test sort(F.p)==collect(1:n)
            @test sum(F.blocks;init=0)==n
            @test all(step->step in (1,2),F.blocks)
            @test A[F.p,F.p]≈L*D*(hermitian ? L' : transpose(L)) rtol=40tol(T) atol=40tol(T)
            n==2 && @test F.blocks==[2]
            @test isequal(C.values,before)
            @test isfinite(F.growth_factor) && F.growth_factor>=1
            rhs=randn(T,n,3); original=copy(rhs)
            for trans in (:N,:T,:C)
                X=solve(F,view(rhs,:,3:-1:1);trans)
                @test op(A,trans)*X≈rhs[:,3:-1:1] rtol=80tol(T) atol=80tol(T)
                @test op(A,trans)*solve(F,view(rhs,:,1);trans)≈rhs[:,1] rtol=80tol(T) atol=80tol(T)
            end
            @test rhs==original
            value,phase=logabsdet_factor(F); refvalue,refphase=LA.logabsdet(A)
            @test value≈refvalue rtol=100tol(T) atol=100tol(T)
            @test phase≈refphase rtol=100tol(T) atol=100tol(T)
            @test determinant(F)≈LA.det(A) rtol=150tol(T) atol=150tol(T)
            if hermitian || T<:Real
                eigen=LA.eigvals(LA.Hermitian(A)); counts=ldl_inertia(F)
                @test counts.positive==count(>(0),eigen)
                @test counts.negative==count(<(0),eigen)
                @test counts.zero==0
            else
                @test_throws ArgumentError ldl_inertia(F)
            end
            GC.gc(); tasks=[Threads.@spawn solve(F,rhs;threaded=false) for _ in 1:3]
            GC.gc()
            @test all(X->isapprox(A*X,rhs;rtol=80tol(T),atol=80tol(T)),fetch.(tasks))
        end
        # The no-pivot variant is still an exact factorization when admissible.
        D=CSRMatrix(Matrix(LA.Diagonal(T[2,-3,4])))
        @test Matrix(sparse_ldlt(D;pivot=:none).D)==Matrix(D)
        for scale in (sqrt(floatmin(R)),sqrt(floatmax(R))/R(4))
            block=T[0 scale; scale 0]; F=sparse_ldlt(CSRMatrix(block))
            rhs=T[scale,-scale]
            @test solve(F,rhs)≈T[-1,1] rtol=20tol(T)
            @test ldl_inertia(F)==(;positive=1,negative=1,zero=0)
        end
    end
    # A later symmetric pivot must carry previously computed L multipliers.
    A=[8.0 1 2 0;1 0 0 3;2 0 0 4;0 3 4 2]
    F=sparse_ldlt(CSRMatrix(A)); L=Matrix(F.L)
    @test A[F.p,F.p]≈L*Matrix(F.D)*L'
    A=[10.0 1 2;1 .1 1;2 1 4]
    G=sparse_ldlt(CSRMatrix(A)); L=Matrix(G.L)
    @test G.p[2]==3 && G.blocks==[1,1,1]
    @test A[G.p,G.p]≈L*Matrix(G.D)*L'
    D=sparse_ldlt(CSRMatrix(Matrix(LA.Diagonal([2.0,-3,4]))))
    @test ldl_inertia(D;tol=3)==(;positive=1,negative=0,zero=2)
    for bad in (zeros(2,2),[1.0 1;1 1])
        @test_throws SingularException sparse_ldlt(CSRMatrix(bad))
    end
    @test_throws SingularException sparse_ldlt(CSRMatrix([0.0 2;2 0]);pivot=:none)
    @test_throws ArgumentError sparse_ldlt(CSRMatrix([Inf 0.0;0 1]))
    @test_throws ArgumentError sparse_ldlt(CSRMatrix([1.0;;]);pivot=:rook)
    @test_throws ArgumentError sparse_ldlt(CSRMatrix([1.0;;]);uplo=:bad)
    @test_throws DimensionMismatch sparse_ldlt(CSRMatrix(ones(2,3)))
    @test_throws DimensionMismatch solve(F,ones(3))
    @test_throws ArgumentError solve(F,ones(4);trans=:bad)
    @test_throws ArgumentError ldl_inertia(F;tol=-1)
    @test_throws ArgumentError ldl_inertia(F;tol=Inf)
    # Imaginary Hermitian diagonal is ignored, including a nonfinite component.
    C=CSRMatrix(ComplexF64[complex(2,NaN) 0;1+im complex(3,NaN)])
    H=ComplexF64[2 1-im;1+im 3]; F=sparse_ldlt(C)
    @test H*solve(F,ones(ComplexF64,2))≈ones(2)
end

@testset "Static sparse LU and honest diagnostics" begin
    for T in TYPES
        R=real(T); A=randn(T,11,11); A[abs.(A).<R(.7)].=zero(T); A+=T(20)*LA.I
        for ordering in (:natural,:minimum_degree,collect(11:-1:1)), relative in (true,false)
            C=CSRMatrix(A); before=copy(C.values); F=sparse_lu_static(C;ordering,relative)
            @test F isa SparseStaticLUFactor
            @test size(F)==size(A)
            @test F.p==F.q
            @test A[F.p,F.q]+LA.Diagonal(F.perturbations)≈Matrix(F.L)*Matrix(F.U) rtol=30tol(T) atol=30tol(T)
            @test all(iszero,F.perturbations)
            @test C.values==before
            for trans in (:N,:T,:C)
                b=randn(T,11,3); x=solve(F,view(b,:,3:-1:1);trans)
                @test op(A,trans)*x≈b[:,3:-1:1] rtol=40tol(T) atol=40tol(T)
                @test op(A,trans)*solve(F,b[:,1];trans)≈b[:,1] rtol=40tol(T) atol=40tol(T)
            end
            b=randn(T,11); x=solve(F,b); d=sparse_direct_diagnostics(C,F,b,x)
            @test !d.perturbed && d.finite
            independent=maximum(abs.(b-A*x)./(abs.(b)+abs.(A)*abs.(x)))
            @test only(d.componentwise_backward_error)≈independent atol=20eps(R)
            @test only(d.residual_norms)≈LA.norm(b-A*x) atol=100eps(R)
            B=randn(T,11,3); X=solve(F,B)
            diagnostics=sparse_direct_diagnostics(C,F,view(B,:,3:-1:1),view(X,:,3:-1:1))
            @test diagnostics.finite && length(diagnostics.residual_norms)==3
            @test diagnostics.residual_norms≈[LA.norm(B[:,j]-A*X[:,j]) for j in 3:-1:1] atol=100eps(R)
            @test logabsdet_factor(F)[1]≈LA.logabsdet(A)[1] rtol=30tol(T)
            @test determinant(F)≈LA.det(A) rtol=60tol(T)
            GC.gc(); tasks=[Threads.@spawn solve(F,b) for _ in 1:3]
            @test all(x->isapprox(A*x,b;rtol=40tol(T),atol=40tol(T)),fetch.(tasks))
        end
        phase=T<:Complex ? T(im) : T(-1)
        A=T[phase*R(1e-5) 0;0 2]; C=CSRMatrix(A)
        F=sparse_lu_static(C;epsilon=R(.1))
        @test F.U[1,1]≈phase*R(.1)
        @test count(!iszero,F.perturbations)==1
        @test A+LA.Diagonal(F.perturbations)≈Matrix(F.L)*Matrix(F.U) rtol=10tol(T)
        b=ones(T,2); x=solve(F,b); d=sparse_direct_diagnostics(C,F,b,x)
        @test d.perturbed && d.finite
        @test only(d.residual_norms)>R(.5) # Original system is not certified solved.
        @test only(d.componentwise_backward_error)>R(.5)
        Z=CSRMatrix(zeros(T,2,2)); F=sparse_lu_static(Z;epsilon=R(.125))
        @test F.perturbations==fill(T(.125),2)
        @test solve(F,ones(T,2))==fill(T(8),2)
        @test sparse_direct_diagnostics(Z,F,ones(T,2)).perturbed
        @test sparse_direct_diagnostics(C,F,zeros(T,2),zeros(T,2)).componentwise_backward_error==R[0]
        E=sparse_lu_static(CSRMatrix(zeros(T,0,0)))
        @test isempty(solve(E,T[])) && determinant(E)==one(T)
        @test size(solve(E,zeros(T,0,2)))==(0,2)
    end
    C=CSRMatrix([1.0 0;0 2]); F=sparse_lu_static(C)
    @test_throws ArgumentError sparse_lu_static(C;epsilon=0)
    @test_throws ArgumentError sparse_lu_static(C;epsilon=NaN)
    @test_throws ArgumentError sparse_lu_static(CSRMatrix([Inf;;]))
    @test_throws ArgumentError sparse_lu_static(C;ordering=[1,1])
    @test_throws DimensionMismatch sparse_lu_static(CSRMatrix(ones(2,3)))
    @test_throws DimensionMismatch solve(F,ones(3))
    @test_throws ArgumentError solve(F,ones(2);trans=:bad)
    @test_throws DimensionMismatch sparse_direct_diagnostics(C,F,ones(2),ones(3))
    @test_throws DimensionMismatch sparse_direct_diagnostics(C,F,ones(2),ones(Float32,2))
    @test !sparse_direct_diagnostics(C,F,ones(2),fill(Inf,2)).finite
end
