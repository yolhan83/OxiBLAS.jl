# Prepared only: independent exhaustive maximum cardinality on small patterns.
function graph_cardinality_oracle(A)
    n=size(A,1); occupied=falses(n)
    function visit(v)
        v>n && return 0
        occupied[v] && return visit(v+1)
        best=visit(v+1); occupied[v]=true
        for u in v+1:n
            occupied[u] && continue
            (!iszero(A[v,u]) || !iszero(A[u,v])) || continue
            occupied[u]=true; best=max(best,1+visit(v+1)); occupied[u]=false
        end
        occupied[v]=false; best
    end
    visit(1)
end
function matrix_cardinality_oracle(A)
    used=falses(size(A,2))
    function visit(row)
        row>size(A,1) && return 0
        best=visit(row+1)
        for col in axes(A,2)
            (used[col] || iszero(A[row,col])) && continue
            used[col]=true; best=max(best,1+visit(row+1)); used[col]=false
        end
        best
    end
    visit(1)
end
@testset "Layered matching: matrix and graph interpretations" begin
    for T in TYPES, (m,n) in ((0,0),(0,4),(4,0),(1,1),(3,5),(5,3),(5,5)), sample in 1:5
        A=T.(rand(0:1,m,n)); C=CSRMatrix(A); saved=copy(C.values)
        for storage in (C,CSCMatrix(C),COOMatrix(C))
            result=bipartite_matching(storage)
            @test result.cardinality==matrix_cardinality_oracle(A)
            @test count(!iszero,result.rows)==count(!iszero,result.columns)==result.cardinality
            @test all(j==0 || (!iszero(A[i,j]) && result.columns[j]==i) for (i,j) in enumerate(result.rows))
            @test all(i==0 || result.rows[i]==j for (j,i) in enumerate(result.columns))
        end
        @test C.values==saved
    end
    for T in TYPES, n in (0,1,2,7,9), sample in 1:6
        A=zeros(T,n,n)
        for i in 1:n, j in i+1:n
            isodd(i)!=isodd(j) && rand(Bool) && (A[i,j]=one(T))
        end
        C=CSRMatrix(A); result=bipartite_graph_matching(C)
        @test result!==nothing
        @test result.cardinality==graph_cardinality_oracle(A)==length(result.edges)
        @test all(i!=j && (!iszero(A[i,j]) || !iszero(A[j,i])) for (i,j) in result.edges)
        @test all(v==0 || result.mate[v]==i for (i,v) in enumerate(result.mate))
        @test result.covers_all_vertices==(2result.cardinality==n)
        @test result.saturated_smaller_side==(result.cardinality==min(length(result.left),length(result.right)))
        @test bipartite_graph_matching(CSCMatrix(C)).mate==result.mate
        @test bipartite_graph_matching(CSRMatrix(A+A')).mate==result.mate
        if n>0
            withdiag=A+T(7)*LA.I
            @test bipartite_graph_matching(CSRMatrix(withdiag)).mate==result.mate
        end
    end
    triangle=CSRMatrix([0.0 1 1;0 0 1;0 0 0])
    @test bipartite_graph_matching(triangle)===nothing
    @test bipartite_matching(CSRMatrix(Matrix{Float64}(LA.I,4,4))).cardinality==4
    @test bipartite_graph_matching(CSRMatrix(Matrix{Float64}(LA.I,4,4))).cardinality==0
    @test_throws DimensionMismatch bipartite_graph_matching(CSRMatrix(zeros(2,3)))
    # The final augmentation traverses thousands of matched vertices. A
    # recursive DFS implementation can exhaust the Julia stack here.
    n=4096; rr=Int[]; cc=Int[]
    for i in 1:n-1
        push!(rr,i,i); push!(cc,i,i+1)
    end
    push!(rr,n); push!(cc,1)
    C=CSRMatrix(COOMatrix(n,n,rr,cc,ones(length(rr))))
    result=bipartite_matching(C)
    @test result.cardinality==n
    @test result.rows[n]==1 && result.rows[1:n-1]==collect(2:n)
    GC.gc(); tasks=[Threads.@spawn bipartite_matching(C) for _ in 1:3]; GC.gc()
    @test all(r->r.rows==result.rows,fetch.(tasks))
end
