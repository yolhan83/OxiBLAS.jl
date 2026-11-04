function matching_oracle(A,objective,mode)
    m,n=size(A); used=falses(n); best=(-Inf,-Inf); bestweight=0.0; bestcardinality=0
    transform(x)=mode===:value ? Float64(real(x)) : mode===:abs ? Float64(abs(x)) : log(Float64(abs(x)))
    function visit(row,cardinality,weight)
        if row>m
            score=objective===:cardinality ? (Float64(cardinality),weight) : (weight,Float64(cardinality))
            if isless(best,score)
                best=score; bestweight=weight; bestcardinality=cardinality
            end
            return
        end
        visit(row+1,cardinality,weight)
        for column in 1:n
            (used[column] || iszero(A[row,column])) && continue
            used[column]=true; visit(row+1,cardinality+1,weight+transform(A[row,column])); used[column]=false
        end
    end
    visit(1,0,0.0)
    (;cardinality=bestcardinality,total_weight=bestweight)
end
function check_weighted_matching(result,A)
    @test result.cardinality==length(result.edges)==count(!iszero,result.rows)==count(!iszero,result.columns)
    @test all(A[i,j]!=0 && result.rows[i]==j && result.columns[j]==i for (i,j) in result.edges)
    @test length(unique(first.(result.edges)))==result.cardinality
    @test length(unique(last.(result.edges)))==result.cardinality
end
@testset "Sparse weighted matching: exhaustive independent reference" begin
    for T in TYPES
        modes=T<:Real ? (:value,:abs,:logabs) : (:abs,:logabs)
        for (m,n) in ((0,3),(3,0),(1,1),(3,4),(4,3)), sample in 1:4
            A=T.(rand(-4:4,m,n)); T<:Complex && (A .*= T(im))
            C=CSRMatrix(A); saved=Matrix(C)
            for objective in (:cardinality,:weight), mode in modes
                result=weighted_bipartite_matching(C;objective,weight=mode)
                expected=matching_oracle(A,objective,mode)
                check_weighted_matching(result,A)
                objective===:cardinality && @test result.cardinality==expected.cardinality
                @test result.total_weight≈expected.total_weight rtol=1e-6 atol=1e-6
                @test result.augmentations==result.cardinality
                @test Matrix(C)==saved
            end
        end
        # A locally maximal edge would prevent the global weight optimum.
        A=T[9 8;8 1]; result=weighted_bipartite_matching(CSRMatrix(A);weight=:abs)
        @test result.rows==[2,1] && result.total_weight==16
        A=T[-5 -1;-2 -4]
        if T<:Real
            @test weighted_bipartite_matching(CSRMatrix(A);objective=:weight,weight=:value).cardinality==0
            result=weighted_bipartite_matching(CSRMatrix(A);weight=:value)
            @test result.rows==[2,1]
            @test result.total_weight≈-3
        else
            @test_throws ArgumentError weighted_bipartite_matching(CSRMatrix(A);weight=:value)
        end
        @test weighted_bipartite_matching(CSRMatrix(T[1 0;0 1]);weight=:logabs).cardinality==2
        # Graph API uses an undirected vertex bipartition, distinct from rows/cols.
        graph=zeros(T,6,6)
        for (i,j,w) in ((1,2,3),(2,3,5),(4,5,2))
            graph[i,j]=T(w)
        end
        coloring=bipartite_coloring(CSRMatrix(graph))
        @test coloring.is_bipartite
        @test all(coloring.colors[i]!=coloring.colors[j] for i in 1:6,j in 1:6 if !iszero(graph[i,j]))
        result=weighted_graph_matching(CSRMatrix(graph);weight=:abs)
        @test result.cardinality==2 && result.total_weight==7
        @test all(result.mate[result.mate[i]]==i for i in 1:6 if result.mate[i]!=0)
        @test weighted_graph_matching(CSRMatrix(T[0 1 1;1 0 1;1 1 0]))===nothing
        @test weighted_graph_matching(CSRMatrix(T[0 1;1 0]);weight=:logabs).cardinality==1
        @test isempty(weighted_graph_matching(CSRMatrix(zeros(T,0,0))).mate)
        @test degree_sequence(CSRMatrix(graph))==[1,2,1,1,1,0]
        @test !is_structurally_symmetric(CSRMatrix(graph))
        @test is_structurally_symmetric(CSRMatrix(graph+graph'))
        path=zeros(T,7,7)
        for i in 1:6
            path[i,i+1]=one(T)
        end
        @test pseudo_peripheral_vertex(CSRMatrix(path);start=4) in (1,7)
        @test pseudo_peripheral_vertex(CSRMatrix(zeros(T,0,0)))==0
        @test_throws ArgumentError pseudo_peripheral_vertex(CSRMatrix(path);start=0)
        # Reproducibility/GC/concurrency; no shared work state between calls.
        C=CSRMatrix(T[9 8;8 1]); first=weighted_bipartite_matching(C;weight=:abs)
        GC.gc(); @test isequal(first,weighted_bipartite_matching(C;weight=:abs))
        results=Vector{Any}(undef,2)
        @sync for j in 1:2
            Threads.@spawn results[j]=weighted_bipartite_matching(C;weight=:abs)
        end
        @test isequal(results[1],results[2])
        @test_throws ArgumentError weighted_bipartite_matching(C;objective=:unknown)
        @test_throws ArgumentError weighted_bipartite_matching(C;weight=:unknown)
        @test_throws ArgumentError weighted_bipartite_matching(CSRMatrix(fill(T(Inf),1,1)))
    end
    largest=floatmax(Float64)
    overflow=weighted_bipartite_matching(CSRMatrix([largest 0;0 largest]))
    @test overflow.rows==[1,2] && overflow.weight_overflow
end
