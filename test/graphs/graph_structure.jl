function dense_graph_distances(A,start)
    n=size(A,1); distance=fill(-1,n); distance[start]=0
    # Independent dense relaxation, no reuse of package BFS.
    for _ in 1:n
        changed=false
        for i in 1:n, j in 1:n
            (i==j || (iszero(A[i,j]) && iszero(A[j,i]))) && continue
            if distance[i]>=0 && (distance[j]<0 || distance[j]>distance[i]+1)
                distance[j]=distance[i]+1; changed=true
            end
        end
        changed || break
    end
    distance
end
function inspect_separator_tree(tree,A;maxdepth=50)
    n=size(A,1); @test sort(tree.permutation)==collect(1:n)
    positions=invperm(tree.permutation)
    n==0 && @test tree.root==0 && isempty(tree.nodes)
    for (index,node) in enumerate(tree.nodes)
        hasproperty(node,:depth) && @test node.depth<=maxdepth
        if node.leaf
            @test node.children==(0,0)
        else
            @test sort(vcat(node.left,node.right,node.separator))==sort(node.vertices)
            @test all(iszero(A[i,j]) && iszero(A[j,i]) for i in node.left for j in node.right)
            @test all(child->0<child<index,node.children)
            if !isempty(node.separator)
                @test maximum(positions[v] for v in vcat(node.left,node.right))<minimum(positions[v] for v in node.separator)
            end
        end
    end
end
@testset "Structural graph coverage and dissection configuration" begin
    for T in TYPES, n in (0,1,2,11,25)
        A=zeros(T,n,n)
        for i in 1:n-1
            i%5!=0 && (A[i,i+1]=one(T))
            i+3<=n && i%5==1 && (A[i,i+3]=T(2))
        end
        C=CSRMatrix(A); saved=copy(C.values)
        result=connected_components(C)
        @test sort(vcat(Int[],result.components...))==collect(1:n)
        @test result.count==length(result.components)
        @test result.count==maximum(result.labels;init=0)
        degrees=[count(j->i!=j && (!iszero(A[i,j]) || !iszero(A[j,i])),1:n) for i in 1:n]
        @test degree_sequence(C)==degrees
        for start in 1:n
            expected=dense_graph_distances(A,start); levels=level_sets(C,start)
            distance=fill(-1,n)
            for (level,vertices) in enumerate(levels), v in vertices
                distance[v]=level-1
            end
            @test distance==expected
            @test all((expected[i]<0)==(result.labels[i]!=result.labels[start]) for i in 1:n)
        end
        band=graph_bandwidth_profile(C)
        lo=maximum((i-j for i in 1:n for j in 1:n if !iszero(A[i,j]));init=0)
        hi=maximum((j-i for i in 1:n for j in 1:n if !iszero(A[i,j]));init=0)
        profile=sum((maximum(vcat(i,findall(!iszero,A[i,:])))-minimum(vcat(i,findall(!iszero,A[i,:]))) for i in 1:n);init=0)
        @test band.lower==lo && band.upper==hi && band.bandwidth==max(lo,hi)
        @test band.profile==profile && band.average_bandwidth==(n==0 ? 0.0 : profile/n)
        @test graph_bandwidth_profile(CSCMatrix(C))==band
        metrics=ordering_metrics(C,collect(1:n))
        @test metrics.graph_bandwidth==band.bandwidth && metrics.graph_profile==band.profile
        for maxdepth in (0,1,4), leafsize in (1,7), balance_tolerance in (0.0,.2,.8)
            tree=nested_dissection(C;maxdepth,leafsize,balance_tolerance,return_tree=true)
            inspect_separator_tree(tree,A;maxdepth)
            @test nested_dissection(C;maxdepth,leafsize,balance_tolerance)==tree.permutation
            @test sparse_ordering(C,:nested_dissection;maxdepth,leafsize,balance_tolerance)==tree.permutation
        end
        for refinement in (:greedy,:fm), maxdepth in (0,2)
            tree=multilevel_nested_dissection(C;refinement,maxdepth,leafsize=1,coarse_size=2,return_tree=true)
            inspect_separator_tree(tree,A;maxdepth)
            maxdepth==0 && n>0 && @test length(tree.nodes)==1
        end
        symbolic=symbolic_cholesky(C)
        @test sort(symbolic.postorder)==collect(1:n)
        position=invperm(symbolic.postorder)
        @test all(symbolic.parent[i]==0 || position[i]<position[symbolic.parent[i]] for i in 1:n)
        @test symbolic.nnz==sum(length,symbolic.columns;init=0)
        @test all(symbolic.columns[i][1]==i for i in 1:n)
        for i in 1:n
            @test symbolic.parent[i]==(length(symbolic.columns[i])>1 ? symbolic.columns[i][2] : 0)
        end
        @test Matrix(sparse_permute(C,collect(n:-1:1)))==A[n:-1:1,n:-1:1]
        @test C.values==saved
    end
    # Selected triangle must ignore arbitrary structure in the other half.
    L=[2.0 0 0 0;1 2 0 0;0 1 2 0;0 0 1 2]
    poisoned=copy(L); poisoned[1,4]=NaN
    @test symbolic_cholesky(CSRMatrix(poisoned);uplo=:L).columns==symbolic_cholesky(CSRMatrix(L)).columns
    @test symbolic_cholesky(CSRMatrix(transpose(poisoned));uplo=:U).columns==symbolic_cholesky(CSRMatrix(L)).columns
    @test_throws ArgumentError symbolic_cholesky(CSRMatrix(L);uplo=:bad)
    # Exact local minimum-degree leaves, rather than sorting static degrees.
    n=9; star=zeros(n,n)
    for i in 2:n
        star[1,i]=1
    end
    C=CSRMatrix(star)
    @test nested_dissection(C;maxdepth=0)==minimum_degree_ordering(C)
    @test_throws ArgumentError nested_dissection(C;maxdepth=-1)
    @test_throws ArgumentError nested_dissection(C;balance_tolerance=NaN)
    @test_throws ArgumentError nested_dissection(C;balance_tolerance=1)
    @test_throws ArgumentError multilevel_nested_dissection(C;maxdepth=-1)
    @test_throws DimensionMismatch graph_bandwidth_profile(CSRMatrix(zeros(2,3)))
    @test_throws BoundsError level_sets(C,0)
    @test_throws BoundsError level_sets(C,n+1)
    GC.gc(); tasks=[Threads.@spawn nested_dissection(C;leafsize=1) for _ in 1:3]
    @test all(p->p==nested_dissection(C;leafsize=1),fetch.(tasks))
    # Iterative postorder remains safe for a long elimination chain.
    n=4096; C=CSRMatrix(COOMatrix(n,n,collect(1:n-1),collect(2:n),ones(n-1)))
    symbolic=symbolic_cholesky(C)
    @test symbolic.postorder==collect(1:n)
    @test symbolic.parent==vcat(collect(2:n),0)
end
