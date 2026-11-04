# Prepared only; independent dense references belong in test/, never src/.
@testset "Row-element multifrontal LU" begin
    for T in TYPES
        R=real(T)
        cases=Matrix{T}[zeros(T,0,0),T[3;;],T[0 1;2 3],
            T[0 0 2;3 0 4;0 5 6],
            T[5 1 0 0 2;0 6 1 0 0;0 0 7 1 0;0 0 0 8 1;1 0 0 0 9]]
        # A forest of independent fronts with nonsymmetric, off-diagonal pivots.
        branches=zeros(T,8,8)
        for first in (1,3,5,7)
            branches[first:first+1,first:first+1]=T[0 2;3 4]
        end
        push!(cases,branches)
        for n in (7,13)
            dense=randn(T,n,n); dense[abs.(dense).<R(.8)].=zero(T); dense+=T(12)*LA.I
            push!(cases,dense[randperm(n),randperm(n)])
        end
        for A in cases, maxwidth in (1,2,32), ordering in (:natural,:column_minimum_degree), threaded in (false,true)
            n=size(A,1); C=CSRMatrix(A); before=copy(C.values)
            F=multifrontal_lu(C;ordering,maxwidth,threaded)
            @test F isa MultifrontalLUFactor
            @test size(F)==size(A)
            @test sort(F.p)==sort(F.q)==collect(1:n)
            @test A[F.p,F.q]≈Matrix(F.L)*Matrix(F.U) rtol=60tol(T) atol=60tol(T)
            @test C.values==before
            @test sum(length,F.nodes;init=0)==n
            @test all(node->length(node)<=maxwidth,F.nodes)
            @test length(F.front_shapes)==length(F.nodes)==length(F.parents)
            @test all(g->F.parents[g]==0 || F.parents[g]>g,eachindex(F.parents))
            @test F.growth_factor>=1 && isfinite(F.growth_factor)
            b=randn(T,n,3); original=copy(b)
            for trans in (:N,:T,:C)
                @test op(A,trans)*solve(F,view(b,:,3:-1:1);trans)≈b[:,3:-1:1] rtol=80tol(T) atol=80tol(T)
                @test op(A,trans)*solve(F,view(b,n:-1:1,1);trans)≈b[n:-1:1,1] rtol=80tol(T) atol=80tol(T)
            end
            @test b==original
            @test determinant(F)≈LA.det(A) rtol=100tol(T) atol=100tol(T)
            logvalue,phase=logabsdet_factor(F); refvalue,refphase=LA.logabsdet(A)
            @test logvalue≈refvalue rtol=60tol(T) atol=60tol(T)
            @test phase≈refphase rtol=60tol(T) atol=60tol(T)
            @test sparse_direct_diagnostics(C,F,b).finite
        end
        for threshold in (0,R(.1),1)
            A=T[0 2;3 4]; F=multifrontal_lu(CSRMatrix(A);pivot_threshold=threshold,threaded=false)
            @test F.p[1]==2
            @test A*solve(F,ones(T,2))≈ones(T,2) rtol=20tol(T)
        end
        A=T[4 2 0 1;0 0 3 1;0 1 0 2;1 0 0 4]; C=CSRMatrix(A)
        F=multifrontal_lu(C;ordering=[3,1,4,2],maxwidth=2)
        GC.gc(); b=randn(T,4,4); tasks=[Threads.@spawn solve(F,b) for _ in 1:4]
        @test all(x->isapprox(A*x,b;rtol=60tol(T),atol=60tol(T)),fetch.(tasks))
        # Every permutation matrix must pivot correctly, including scalar trees
        # with no contribution and grouped fronts with independent roots.
        for p in ([1,2,3,4],[4,3,2,1],[2,3,4,1],[3,1,4,2]), width in (1,4)
            A=Matrix{T}(LA.I,4,4)[p,:]; F=multifrontal_lu(CSRMatrix(A);maxwidth=width)
            @test Matrix(F.L)*Matrix(F.U)==A[F.p,F.q]
        end
        for scale in (sqrt(floatmin(R)),sqrt(floatmax(R))/R(8))
            A=T[2 1;1 3]*scale; F=multifrontal_lu(CSRMatrix(A))
            @test solve(F,T[3,4]*scale)≈ones(T,2) rtol=20tol(T)
        end
    end
    for A in (zeros(2,2),[1.0 0;2 0],[1.0 2;2 4])
        @test_throws SingularException multifrontal_lu(CSRMatrix(A);threaded=false)
    end
    C=CSRMatrix([1.0 0;0 2]); F=multifrontal_lu(C)
    @test_throws DimensionMismatch multifrontal_lu(CSRMatrix(ones(2,3)))
    @test_throws ArgumentError multifrontal_lu(C;maxwidth=0)
    @test_throws ArgumentError multifrontal_lu(C;pivot_threshold=-1)
    @test_throws ArgumentError multifrontal_lu(C;pivot_threshold=NaN)
    @test_throws ArgumentError multifrontal_lu(C;ordering=[1,1])
    @test_throws ArgumentError multifrontal_lu(CSRMatrix([Inf;;]))
    @test_throws DimensionMismatch solve(F,ones(3))
    @test_throws ArgumentError solve(F,ones(2);trans=:bad)
end

import SparseArrays as SA
using SparseArrays: sparse
@testset "Supernodal LU, symmetric strategy" begin
    for T in TYPES
        R=real(T)
        # Grid convection-diffusion (unsymmetric values, symmetric pattern),
        # an indefinite Helmholtz-like matrix, and random patterns with zero
        # diagonal entries (delayed pivots).
        k=12; D(c)=SA.spdiagm(-1=>fill(T(-1-c),k-1),0=>fill(T(2),k),1=>fill(T(-1+c),k-1)); I=sparse(T(1)*LA.I,k,k)
        G=kron(I,D(0.4))+kron(D(-0.3),I); H=G-T(3)*sparse(T(1)*LA.I,k*k,k*k)
        cases=Any[G,H]
        for n in (5,30,120)
            S=SA.sprandn(T,n,n,0.08); S=S+copy(transpose(S))*T(0.6)
            M=S+SA.spdiagm(0=>T.(rand(n).<0.7).*randn(T,n))+T(1e-3)*LA.I
            push!(cases,M)
        end
        for A in cases, threshold in (R(1),R(0.1)), maxwidth in (nothing,1,4)
            n=size(A,1); F=multifrontal_lu(A;ordering=:amd,pivot_threshold=threshold,maxwidth)
            @test sort(F.p)==sort(F.q)==collect(1:n)
            @test LA.norm(Matrix(A)[F.p,F.q]-Matrix(F.L)*Matrix(F.U))<=60tol(T)*max(1,LA.norm(Matrix(A)))
            Lm=Matrix(F.L); @test LA.istril(Lm) && all(isone,LA.diag(Lm)) && LA.istriu(Matrix(F.U))
            # The threshold bounds the multipliers.
            @test maximum(abs,Lm;init=zero(R))<=inv(threshold)*(1+sqrt(eps(R)))
            @test sum(length,F.nodes;init=0)==n && (maxwidth===nothing || all(node->length(node)<=maxwidth,F.nodes))
            @test all(g->F.parents[g]==0 || F.parents[g]>g,eachindex(F.parents))
            @test F.growth_factor>=1 && isfinite(F.growth_factor)
            b=randn(T,n,2)
            for trans in (:N,:T,:C)
                @test op(Matrix(A),trans)*solve(F,b;trans)≈b rtol=1e3tol(T) atol=1e3tol(T)
                @test op(Matrix(A),trans)*solve(F,b[:,1];trans)≈b[:,1] rtol=1e3tol(T) atol=1e3tol(T)
            end
        end
        # :auto picks this strategy for a symmetric pattern with a full
        # diagonal (no width cap, so the grid gets wide supernodes) and the
        # row-element one otherwise.
        @test maximum(length,multifrontal_lu(G).nodes)>1
        @test OxiBLAS._lu_symmetric_strategy(CSCMatrix(G)) && !OxiBLAS._lu_symmetric_strategy(CSCMatrix(sparse(T[0 1;1 0])))
        # Structurally and numerically singular matrices.
        @test_throws SingularException multifrontal_lu(sparse(T[1 2;2 4]);ordering=:amd)
        @test_throws SingularException multifrontal_lu(sparse(T[1 0 0;0 0 0;0 0 2]);ordering=:amd)
        @test_throws ArgumentError multifrontal_lu(sparse(T[Inf 0;0 1]))
    end
end

@testset "Column-front LU, unsymmetric patterns" begin
    for T in TYPES
        R=real(T)
        for n in (40,150), d in (0.02,0.06)
            # Random pattern plus a permutation: no usable diagonal, a core
            # larger than n/4 after the singletons (column-front engine).
            P=Matrix{T}(LA.I,n,n)[randperm(n),:]
            A=SA.sprandn(T,n,n,d)+sparse(T(3)*P)
            OxiBLAS._lu_symmetric_strategy(CSCMatrix(A)) && continue
            F=multifrontal_lu(A); M=Matrix(A)
            @test sort(F.p)==sort(F.q)==collect(1:n)
            @test LA.norm(M[F.p,F.q]-Matrix(F.L)*Matrix(F.U))<=100tol(T)*max(1,LA.norm(M))
            Lm=Matrix(F.L); @test LA.istril(Lm) && all(isone,LA.diag(Lm)) && LA.istriu(Matrix(F.U))
            # Threshold pivoting bounds the multipliers by 1/τ (τ = 0.1).
            @test maximum(abs,Lm;init=zero(R))<=10*(1+sqrt(eps(R)))
            b=randn(T,n)
            for trans in (:N,:T,:C)
                @test op(M,trans)*solve(F,b;trans)≈b rtol=1e3tol(T) atol=1e3tol(T)
            end
            # sparse_lu gives the same solution through its :auto path and
            # through Gilbert–Peierls in an AMD(AᵀA) order.
            @test solve(sparse_lu(A),b)≈solve(sparse_lu(A;ordering=:amd_ata),b) rtol=1e3tol(T) atol=1e3tol(T)
        end
    end
end
