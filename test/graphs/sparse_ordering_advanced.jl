function reference_column_degree(A;dense_threshold=nothing,tie=:column_nnz)
    m,n=size(A); rows=dense_threshold===nothing ? collect(1:m) : [r for r in 1:m if count(!iszero,A[r,:])<=floor(Int,n*dense_threshold)]
    adjacency=[i!=j && any(!iszero(A[r,i]) && !iszero(A[r,j]) for r in rows) for i in 1:n,j in 1:n]
    remaining=collect(1:n); order=Int[]
    while !isempty(remaining)
        scores=[(count(j->adjacency[i,j],remaining),tie===:column_nnz ? count(!iszero,A[:,i]) : 0,i) for i in remaining]
        pivot=remaining[argmin(scores)]; neighbors=[i for i in remaining if adjacency[pivot,i]]
        for i in neighbors,j in neighbors
            i!=j && (adjacency[i,j]=true)
        end
        adjacency[pivot,:].=false; adjacency[:,pivot].=false
        push!(order,pivot); deleteat!(remaining,findfirst(==(pivot),remaining))
    end
    order
end
@testset "Column elements, multiple degree and ordering integration" begin
    for T in TYPES
        for (m,n) in ((0,0),(0,5),(5,0),(7,9),(9,7)), sample in 1:3
            A=T.(rand(0:1,m,n)); C=CSRMatrix(A)
            for threshold in (nothing,0.0,0.5,1.0), absorb in (false,true)
                tie=threshold===nothing ? :column_nnz : :index
                p=column_minimum_degree(C;dense_threshold=threshold,absorb,tie)
                @test p==reference_column_degree(A;dense_threshold=threshold,tie)
                @test sort(p)==collect(1:n)
            end
            @test sparse_ordering(C,:column_minimum_degree)==column_minimum_degree(C)
            @test Matrix(C)==A
        end
        n=8; star=zeros(T,n,n)
        for j in 2:n
            star[1,j]=star[j,1]=T(-1)
        end
        for i in 1:n
            star[i,i]=T(2+count(!iszero,star[i,:]))
        end
        C=CSRMatrix(star); original=copy(C.values)
        algorithms=(:natural,:rcm,:minimum_degree,:multiple_minimum_degree,:nested_dissection,:column_minimum_degree,:multilevel_nested_dissection)
        for algorithm in algorithms
            p=sparse_ordering(C,algorithm); @test sort(p)==collect(1:n)
            metrics=ordering_metrics(C,p); factor=LA.cholesky(LA.Hermitian(star[p,p]))
            @test metrics.factor_nnz==count(!iszero,Matrix(factor.L))
            @test metrics.fill==metrics.factor_nnz-metrics.pattern_nnz>=0
            @test symbolic_cholesky(C;ordering=p).permutation==p
            @test symbolic_cholesky(C;ordering=algorithm).permutation==p
        end
        @test ordering_metrics(C,multiple_minimum_degree(C)).fill==0
        @test ordering_metrics(C,collect(1:n)).fill>0
        comparisons=compare_orderings(C;algorithms)
        selected=select_sparse_ordering(C;algorithms)
        @test selected.best.factor_nnz==minimum(c.factor_nnz for c in comparisons)
        @test C.values==original
        F=multifrontal_cholesky(C;ordering=:multilevel_nested_dissection,threaded=false)
        rhs=ones(T,n); @test star*solve(F,rhs)≈rhs rtol=30tol(T) atol=30tol(T)
        GC.gc(); @test column_minimum_degree(C)==column_minimum_degree(C)
        @test ordering_metrics(CSRMatrix(zeros(T,0,0)),Int[]).factor_nnz==0
        @test isempty(multiple_minimum_degree(CSRMatrix(zeros(T,0,0))))
        @test_throws ArgumentError column_minimum_degree(C;dense_threshold=NaN)
        @test_throws ArgumentError column_minimum_degree(C;dense_threshold=-1)
        @test_throws ArgumentError sparse_ordering(C,:unknown)
        @test_throws ArgumentError symbolic_cholesky(C;ordering=ones(Int,n))
        @test_throws DimensionMismatch multiple_minimum_degree(CSRMatrix(zeros(T,2,3)))
        @test_throws ArgumentError select_sparse_ordering(C;algorithms=())
    end
end
