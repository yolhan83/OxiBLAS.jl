# check_partition: test/helpers/checks.jl
@testset "Multilevel partition and separators: structural references" begin
    for T in TYPES
        n=36; A=zeros(T,n,n)
        for i in 1:n-1
            A[i,i+1]=T(i%3+1)
        end
        # Grid chords create a nontrivial multilevel graph; one-triangle input.
        for i in 1:n-6
            A[i,i+6]=T(2)
        end
        C=CSRMatrix(A); saved=copy(C.values)
        for k in (1,2,3,5,n), weighted in (false,true)
            result=multilevel_partition(C,k;coarse_size=4,refine_passes=4,edge_weights=weighted ? :abs : :unit)
            check_partition(result,A,k;weighted)
            @test result.labels==multilevel_partition(C,k;coarse_size=4,refine_passes=4,edge_weights=weighted ? :abs : :unit).labels
        end
        weights=Float64.(1:n)
        result=multilevel_partition(C,3;vertex_weights=weights,coarse_size=4)
        check_partition(result,A,3;vertex_weights=weights)
        heavy=ones(Float64,n); heavy[1]=1000
        result=multilevel_partition(C,2;vertex_weights=heavy,imbalance=0.01,coarse_size=4)
        @test !result.balanced
        for leafsize in (1,8,100)
            tree=multilevel_nested_dissection(C;leafsize,coarse_size=4,return_tree=true)
            @test sort(tree.permutation)==collect(1:n)
            position=invperm(tree.permutation)
            for node in tree.nodes
                node.leaf && continue
                @test sort(vcat(node.left,node.right,node.separator))==sort(node.vertices)
                @test all(iszero(A[i,j]) && iszero(A[j,i]) for i in node.left for j in node.right)
                @test maximum(position[v] for v in vcat(node.left,node.right))<minimum((position[v] for v in node.separator);init=n+1)
            end
        end
        @test C.values==saved
        disconnected=zeros(T,9,9); disconnected[1,2]=one(T); disconnected[5,6]=one(T)
        check_partition(multilevel_partition(CSRMatrix(disconnected),3;coarse_size=2),disconnected,3)
        empty=multilevel_partition(CSRMatrix(zeros(T,0,0)),1)
        @test isempty(empty.labels) && empty.cut==0 && empty.balanced
        @test isempty(multilevel_nested_dissection(CSRMatrix(zeros(T,0,0))))
        @test multilevel_partition(CSRMatrix(ones(T,1,1)),1).labels==[1]
        GC.gc(); first=multilevel_partition(C,3;coarse_size=4)
        results=Vector{Any}(undef,2)
        @sync for j in 1:2
            Threads.@spawn results[j]=multilevel_partition(C,3;coarse_size=4)
        end
        @test results[1].labels==results[2].labels==first.labels
        @test_throws ArgumentError multilevel_partition(C,0)
        @test_throws ArgumentError multilevel_partition(C,n+1)
        @test_throws ArgumentError multilevel_partition(C,2;coarse_size=1)
        @test_throws ArgumentError multilevel_partition(C,2;imbalance=NaN)
        @test_throws ArgumentError multilevel_partition(C,2;vertex_weights=zeros(n))
        extreme=ones(Float64,n); extreme[1]=floatmax(Float64); extreme[2]=nextfloat(0.0)
        @test_throws ArgumentError multilevel_partition(C,2;vertex_weights=extreme)
        @test_throws DimensionMismatch multilevel_partition(C,2;vertex_weights=ones(n-1))
        @test_throws ArgumentError multilevel_nested_dissection(C;leafsize=0)
    end
end
