# Shared result checks (formerly in level3_blocked.jl, schur.jl and
# multilevel_partition.jl).
tri_indices(n,uplo) = [(i,j) for j in 1:n for i in 1:n if (uplo===:U ? i<=j : i>=j)]
function check_triangle(C,expected,old,uplo,T)
    n=size(C,1)
    @test all(isapprox(C[i,j],expected[i,j];rtol=5tol(T),atol=5tol(T)) for (i,j) in tri_indices(n,uplo))
    @test all(isequal(C[i,j],old[i,j]) for (i,j) in tri_indices(n,uplo===:U ? :L : :U) if i!=j)
end
function spectrum_error(x,y)
    isempty(x) && return 0.0
    available=collect(eachindex(y)); largest=0.0
    for λ in x
        pos=argmin([abs(λ-y[i]) for i in available])
        idx=available[pos]; largest=max(largest,abs(λ-y[idx])); deleteat!(available,pos)
    end
    largest
end
function check_partition(result,A,k;vertex_weights=ones(size(A,1)),weighted=false)
    n=size(A,1)
    @test length(result.labels)==n && result.nparts==k
    @test all(1<=label<=k for label in result.labels)
    @test sort(vcat(result.parts...))==collect(1:n)
    n>0 && @test all(!isempty, result.parts)
    expectedloads=[sum((vertex_weights[i] for i in part);init=0.0) for part in result.parts]
    @test result.loads≈expectedloads rtol=1e-12 atol=1e-12
    cut=sum((weighted ? max(abs(A[i,j]),abs(A[j,i])) : 1.0 for i in 1:n for j in i+1:n
             if result.labels[i]!=result.labels[j] && (!iszero(A[i,j]) || !iszero(A[j,i])));init=0.0)
    @test result.cut≈cut rtol=1e-6 atol=1e-6
    @test result.balanced==(result.achieved_imbalance<=result.requested_imbalance+64eps(Float64))
end
