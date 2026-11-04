# Dense oracle deliberately recomputes every gain and scans all vertices:
# independent of heaps, version counters and incremental gain updates.
function dense_fm_oracle(A,initial;fraction=.5,imbalance=.1,passes=6,vertex_weights=ones(size(A,1)),weighted=false)
    n=size(A,1); W=zeros(Float64,n,n)
    for i in 1:n,j in i+1:n
        if !iszero(A[i,j]) || !iszero(A[j,i])
            W[i,j]=W[j,i]=weighted ? max(abs(A[i,j]),abs(A[j,i])) : 1.0
        end
    end
    edge_scale=maximum(W;init=0.0); iszero(edge_scale) && (edge_scale=1.0); W./=edge_scale
    weights=vertex_weights./maximum(vertex_weights;init=1.0); labels=copy(initial)
    total=sum(weights); target=fraction*total; slack=imbalance*min(fraction,1-fraction)*total
    lower,upper=target-slack,target+slack; tolerance=64eps(Float64)*max(1,total)
    violation(load)=max(0.0,lower-load,load-upper)
    cut(labels)=sum((W[i,j] for i in 1:n for j in i+1:n if labels[i]!=labels[j]);init=0.0)
    for _ in 1:passes
        original=copy(labels); load=sum((weights[i] for i in 1:n if labels[i]==1);init=0.0)
        best=copy(labels); bestcut=cut(labels); bestviolation=violation(load); startcut=bestcut; startviolation=bestviolation
        explore_lower=min(lower,load)-maximum(weights); explore_upper=max(upper,load)+maximum(weights)
        locked=falses(n); cuttol=64eps(Float64)*max(1,abs(startcut))
        for step in 1:n
            candidates=Tuple{Float64,Int}[]; countleft=count(==(1),labels)
            for v in 1:n
                locked[v] && continue
                (labels[v]==1 ? countleft>1 : n-countleft>1) || continue
                nextload=load+(labels[v]==1 ? -weights[v] : weights[v])
                explore_lower<=nextload<=explore_upper || continue
                gain=sum((labels[j]==labels[v] ? -W[v,j] : W[v,j] for j in 1:n);init=0.0)
                push!(candidates,(-gain,v))
            end
            isempty(candidates) && break
            _,v=minimum(candidates); load+=(labels[v]==1 ? -weights[v] : weights[v])
            labels[v]=3-labels[v]; locked[v]=true; currentcut=cut(labels); currentviolation=violation(load)
            if currentviolation<bestviolation-tolerance || (currentviolation<=bestviolation+tolerance && currentcut<bestcut-cuttol)
                best=copy(labels); bestcut=currentcut; bestviolation=currentviolation
            end
        end
        labels=best
        (bestviolation<startviolation-tolerance || (bestviolation<=startviolation+tolerance && bestcut<startcut-cuttol)) || break
    end
    labels
end
@testset "Best-prefix weighted partition refinement" begin
    for T in TYPES, n in (4,7,12), weighted in (false,true), passes in (0,1,4)
        A=zeros(T,n,n)
        for i in 1:n, j in i+1:n
            rand(Bool) && (A[i,j]=T(rand((.25,.5,1.0,2.0,4.0))))
        end
        C=CSRMatrix(A); saved=copy(C.values); labels=[i<=n÷2 ? 1 : 2 for i in 1:n]; before=copy(labels)
        weights=Float64[rand((1,2,4)) for _ in 1:n]
        for fraction in (.5,1/3), imbalance in (0.0,.2)
            result=refine_partition(C,labels;fraction,imbalance,passes,vertex_weights=weights,edge_weights=weighted ? :abs : :unit)
            expected=dense_fm_oracle(A,labels;fraction,imbalance,passes,vertex_weights=weights,weighted)
            @test result.labels==expected
            @test labels==before && C.values==saved
            @test sort(vcat(result.parts...))==collect(1:n)
            @test all(!isempty,result.parts)
            @test result.loads≈[sum(weights[result.parts[i]]) for i in 1:2]
            cut=sum((weighted ? max(abs(A[i,j]),abs(A[j,i])) : 1.0 for i in 1:n for j in i+1:n
                     if result.labels[i]!=result.labels[j] && (!iszero(A[i,j]) || !iszero(A[j,i])));init=0.0)
            @test result.cut≈cut
            @test result.method===:fm && !result.weight_overflow
            for entry in result.history
                @test 0<=entry.retained_moves<=entry.explored_moves<=n
                @test entry.violation_after<=entry.violation_before+1e-12
                if isapprox(entry.violation_after,entry.violation_before;atol=1e-12)
                    @test entry.cut_after<=entry.cut_before+1e-12
                end
            end
        end
    end
    # A pass explores a worse suffix and must retain only its best prefix.
    A=[0.0 3 4 0;3 0 0 4;4 0 0 3;0 4 3 0]
    labels=[1,1,2,2]; C=CSRMatrix(A)
    result=refine_partition(C,labels;edge_weights=:abs,imbalance=0,passes=1)
    @test result.cut==6 && result.balanced
    @test result.labels==[2,1,2,1]
    @test only(result.history).explored_moves==4
    @test only(result.history).retained_moves==2
    @test refine_partition(C,labels;method=:greedy,edge_weights=:abs,imbalance=0).cut<=8
    @test refine_partition(C,labels;passes=0).labels==labels
    GC.gc(); tasks=[Threads.@spawn refine_partition(C,labels;edge_weights=:abs) for _ in 1:3]
    @test all(r->r.labels==refine_partition(C,labels;edge_weights=:abs).labels,fetch.(tasks))
    @test isempty(refine_partition(CSRMatrix(zeros(0,0)),Int[]).labels)
    @test refine_partition(CSRMatrix([1.0;;]),[1]).labels==[1]
    @test_throws ArgumentError refine_partition(C,labels;method=:unknown)
    @test_throws ArgumentError refine_partition(C,labels;fraction=1)
    @test_throws ArgumentError refine_partition(C,labels;passes=-1)
    @test_throws ArgumentError refine_partition(C,[1,1,1,1])
    @test_throws ArgumentError refine_partition(C,[1,1,2,0])
    @test_throws DimensionMismatch refine_partition(C,[1,2])
    @test_throws DimensionMismatch refine_partition(CSRMatrix(ones(2,3)),[1,2])
    @test_throws ArgumentError refine_partition(C,labels;vertex_weights=[1,1,1,NaN])
    extreme=[0.0 floatmax(Float64) 0;0 0 nextfloat(0.0);0 0 0]
    @test_throws ArgumentError multilevel_partition(CSRMatrix(extreme);edge_weights=:abs)
    for refinement in (:greedy,:fm), k in (2,3)
        grid=zeros(36,36)
        for i in 1:35
            grid[i,i+1]=1
            i<=30 && (grid[i,i+6]=1)
        end
        result=multilevel_partition(CSRMatrix(grid),k;coarse_size=4,refinement)
        check_partition(result,grid,k)
        @test result.refinement===refinement
        tree=multilevel_nested_dissection(CSRMatrix(grid);leafsize=4,coarse_size=4,refinement,return_tree=true)
        @test sort(tree.permutation)==collect(1:36)
        for node in tree.nodes
            node.leaf && continue
            @test all(iszero(grid[i,j]) && iszero(grid[j,i]) for i in node.left for j in node.right)
        end
    end
    @test_throws ArgumentError multilevel_partition(C;refinement=:unknown)
    @test_throws ArgumentError multilevel_nested_dissection(C;refinement=:unknown)
end
