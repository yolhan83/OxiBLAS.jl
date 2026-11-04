# Written reference tests; supervised phase A does not execute these cases.
isvd_reconstruction(snapshot)=snapshot.U*LA.Diagonal(snapshot.S)*snapshot.Vt
function check_incremental_svd(F,expected,T;factor=100)
    snapshot=svd_snapshot(F); k=length(snapshot.S)
    @test svd_rank(F)==k
    @test snapshot.shape==size(expected)==size(F)
    @test size(snapshot.U)==(size(expected,1),k) && size(snapshot.Vt)==(k,size(expected,2))
    @test isvd_reconstruction(snapshot)≈expected rtol=factor*tol(T) atol=factor*tol(T)
    @test snapshot.U'*snapshot.U≈Matrix{T}(LA.I,k,k) rtol=factor*tol(T) atol=factor*tol(T)
    @test snapshot.Vt*snapshot.Vt'≈Matrix{T}(LA.I,k,k) rtol=factor*tol(T) atol=factor*tol(T)
    @test issorted(snapshot.S;rev=true) && all(isfinite,snapshot.S) && all(>=(0),snapshot.S)
    snapshot
end

@testset "Incremental SVD: row/column augmentation and zero residual" begin
    for T in TYPES, reorthogonalize in (false,true)
        A=randn(T,5,3); expected=copy(A)
        F=incremental_svd(A;maxrank=10,reorthogonalize)
        @test F isa IncrementalSVDFactor{T,real(T)} && eltype(F)===T
        check_incremental_svd(F,expected,T)
        @test size(F,1)==5 && size(F,2)==3 && size(F,3)==1
        for kind in (:rows,:columns,:rows)
            if kind===:rows
                B=randn(T,2,size(expected,2)); before=copy(B)
                svd_add_rows!(F,B); expected=vcat(expected,B)
            else
                B=randn(T,size(expected,1),4); before=copy(B)
                svd_add_columns!(F,B); expected=hcat(expected,B)
            end
            @test B==before
            snapshot=check_incremental_svd(F,expected,T)
            @test snapshot.S≈LA.svdvals(expected) rtol=100tol(T) atol=100tol(T)
            @test snapshot.last_projection_residual<=100tol(T)*max(1,LA.norm(B))
        end
        @test svd_snapshot(F).updates==3
        snapshot=svd_snapshot(F); saved=isvd_reconstruction(snapshot)
        reorthogonalize_svd!(F)
        check_incremental_svd(F,saved,T)
        @test svd_snapshot(F).updates==snapshot.updates
        svd_add_rows!(F,zeros(T,0,size(F,2)))
        svd_add_columns!(F,zeros(T,size(F,1),0))
        @test svd_snapshot(F).updates==snapshot.updates
        # Within-span updates must change Sigma and rotate factors as needed.
        phase=T<:Complex ? T(im) : one(T)
        initial=T[3 0;0 0]; rows=reshape(T[4phase,0],1,2)
        F=incremental_svd(initial;rank=1,maxrank=2,reorthogonalize)
        svd_add_rows!(F,rows)
        check_incremental_svd(F,vcat(initial,rows),T)
        @test svd_snapshot(F).S≈real(T)[5] rtol=20tol(T) atol=20tol(T)
        F=incremental_svd(initial;rank=1,maxrank=2,reorthogonalize)
        columns=reshape(T[4phase,0],2,1); svd_add_columns!(F,columns)
        check_incremental_svd(F,hcat(initial,columns),T)
        @test svd_snapshot(F).S≈real(T)[5] rtol=20tol(T) atol=20tol(T)
        # Dependent residual columns/rows add only their numerical rank.
        initial=zeros(T,4,3); initial[1,1]=T(2)
        F=incremental_svd(initial;rank=1,maxrank=3,reorthogonalize)
        rows=T[0 1 0;0 2 0]; svd_add_rows!(F,rows)
        check_incremental_svd(F,vcat(initial,rows),T)
        @test length(svd_snapshot(F).S)==2
    end
end

@testset "Incremental truncation, alias snapshots, empty states and errors" begin
    for T in TYPES
        R=real(T)
        F=incremental_svd(randn(T,6,4);rank=2,maxrank=2)
        for kind in (:rows,:columns,:rows,:columns)
            old=svd_snapshot(F); approximation=isvd_reconstruction(old)
            if kind===:rows
                block=randn(T,2,size(F,2)); augmented=vcat(approximation,block)
                svd_add_rows!(F,block)
            else
                block=randn(T,size(F,1),2); augmented=hcat(approximation,block)
                svd_add_columns!(F,block)
            end
            independent=LA.svd(augmented;full=false); rank=2
            expected=independent.U[:,1:rank]*LA.Diagonal(independent.S[1:rank])*independent.Vt[1:rank,:]
            check_incremental_svd(F,expected,T;factor=200)
            @test svd_snapshot(F).S≈independent.S[1:rank] rtol=100tol(T) atol=100tol(T)
            @test svd_snapshot(F).last_truncation_norm≈LA.norm(independent.S[rank+1:end]) rtol=200tol(T) atol=200tol(T)
        end
        # Snapshot/copy storage is independent; updates can safely read factor views.
        F=incremental_svd(randn(T,4,4);maxrank=8); before=svd_snapshot(F)
        owned=svd_snapshot(F); fill!(owned.U,zero(T))
        @test svd_snapshot(F).U==before.U
        clone=copy(F); old=isvd_reconstruction(before); rows=copy(F.Vt)
        svd_add_rows!(F,view(F.Vt,:,:))
        check_incremental_svd(F,vcat(old,rows),T)
        check_incremental_svd(clone,old,T)
        column_source=copy(clone.U); svd_add_columns!(clone,view(clone.U,:,:))
        check_incremental_svd(clone,hcat(old,column_source),T)
        # Reconditioning must preserve U*Sigma*V' while repairing both bases.
        repair=incremental_svd(randn(T,5,3)); represented=isvd_reconstruction(svd_snapshot(repair))
        repair.U[:,1].*=T(2); repair.Vt[1,:]./=T(2)
        reorthogonalize_svd!(repair); check_incremental_svd(repair,represented,T)
        # A valid zero approximation can later acquire new rank and dimensions.
        F=incremental_svd(zeros(T,0,0);rank=0,maxrank=8)
        svd_add_rows!(F,zeros(T,3,0)); columns=randn(T,3,2)
        svd_add_columns!(F,columns); check_incremental_svd(F,columns,T)
        F=incremental_svd(zeros(T,0,3);rank=0,maxrank=3)
        rows=randn(T,2,3); svd_add_rows!(F,rows); check_incremental_svd(F,rows,T)
        F=incremental_svd(zeros(T,3,4);rank=0,maxrank=0)
        svd_add_columns!(F,ones(T,3,2))
        @test isempty(svd_snapshot(F).S) && size(F)==(3,6)
        # Errors leave all factors, dimensions and update count unchanged.
        F=incremental_svd(Matrix{T}(LA.I,3,3)); before=svd_snapshot(F)
        @test_throws DimensionMismatch svd_add_rows!(F,zeros(T,1,2))
        @test_throws DimensionMismatch svd_add_columns!(F,zeros(T,2,1))
        @test_throws ArgumentError svd_add_rows!(F,fill(T(NaN),1,3))
        @test_throws ArgumentError svd_add_columns!(F,fill(T(Inf),3,1))
        other=T===Float64 ? Float32 : Float64
        @test_throws ArgumentError svd_add_rows!(F,zeros(other,1,3))
        @test isequal(svd_snapshot(F),before) && !F.active
        budget=incremental_svd(Matrix{T}(LA.I,2,2);maxsweeps=0)
        before=svd_snapshot(budget)
        @test_throws ConvergenceError svd_add_rows!(budget,ones(T,1,2))
        @test isequal(svd_snapshot(budget),before) && !budget.active
        replacement=randn(T,4,2)
        initialize_svd!(F,replacement)
        check_incremental_svd(F,replacement,T)
        @test svd_snapshot(F).updates==0
        before=svd_snapshot(F)
        @test_throws ArgumentError initialize_svd!(F,fill(T(NaN),4,2))
        @test_throws ArgumentError initialize_svd!(F,replacement;rank=3)
        @test isequal(svd_snapshot(F),before) && !F.active
        @test_throws ArgumentError size(F,0)
        # Negative-stride input and concurrent writers/readers with one shared state.
        A=randn(T,5,3); F=incremental_svd(view(A,5:-1:1,:);maxrank=3)
        B=randn(T,2,3); C=randn(T,3,3); snapshots=Vector{Any}(undef,1)
        @sync begin
            Threads.@spawn svd_add_rows!(F,B)
            Threads.@spawn svd_add_rows!(F,C)
            Threads.@spawn snapshots[1]=svd_snapshot(F)
            Threads.@spawn GC.gc()
        end
        S=svd_snapshot(F); result=isvd_reconstruction(S); initial=A[5:-1:1,:]
        @test min(LA.norm(result-vcat(initial,B,C)),LA.norm(result-vcat(initial,C,B)))<=200tol(T)*max(1,LA.norm(result))
        @test S.updates==2
        seen=snapshots[1]
        @test size(seen.U)==(seen.shape[1],length(seen.S))
        @test size(seen.Vt)==(length(seen.S),seen.shape[2])
        for scale in (R(2)^(-60),R(2)^60)
            A=scale.*T[3 0;0 0]; F=incremental_svd(A;rank=1,maxrank=2)
            svd_add_rows!(F,scale.*T[4 0])
            @test svd_snapshot(F).S./scale≈R[5] rtol=40tol(T) atol=40tol(T)
        end
    end
    @test_throws ArgumentError incremental_svd(ones(2,2);rank=3)
    @test_throws ArgumentError incremental_svd(ones(2,2);rank=2,maxrank=1)
    @test_throws ArgumentError incremental_svd(ones(2,2);ranktol=NaN)
    @test_throws ArgumentError incremental_svd(ones(2,2);svd_rtol=0)
    @test_throws ArgumentError incremental_svd(ones(2,2);method=:unknown)
    @test_throws ArgumentError incremental_svd(ones(2,2);method=:lanczos)
end

@testset "Sparse initialization of incremental states" begin
    for T in TYPES
        A=Matrix(LA.Diagonal(T[1,2,4,8,16,32]))
        for method in (:randomized,:lanczos)
            F=incremental_svd(CSRMatrix(A);rank=2,maxrank=4,method,oversampling=4)
            S=svd_snapshot(F)
            @test S.S≈real(T)[32,16] rtol=100tol(T) atol=100tol(T)
            @test isnan(S.last_truncation_norm) && S.initial_method===method
            represented=isvd_reconstruction(S); B=randn(T,2,6)
            svd_add_rows!(F,B)
            check_incremental_svd(F,vcat(represented,B),T;factor=300)
        end
        dense_randomized=incremental_svd(A;rank=2,method=:randomized,oversampling=4)
        @test svd_snapshot(dense_randomized).S≈real(T)[32,16] rtol=100tol(T) atol=100tol(T)
        default=incremental_svd(CSRMatrix(Matrix(LA.Diagonal(T.(1:20)))))
        @test length(svd_snapshot(default).S)<=16
        @test_throws ArgumentError incremental_svd(CSRMatrix(A);method=:jacobi)
    end
end
