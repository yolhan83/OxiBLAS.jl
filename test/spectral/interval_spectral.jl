@testset "Interval eigenpairs: count provenance and multiplicities" begin
    for T in TYPES
        R=real(T); diagonal=T[-4,-1,0,2,2,5]; D=Matrix(LA.Diagonal(diagonal))
        result=eigenvalues_in_interval(CSRMatrix(D),-1,2)
        @test result.values==R[-1,0,2,2] && result.count==4
        @test result.complete && result.certified_count && result.converged
        @test D*result.vectors==result.vectors*LA.Diagonal(result.values)
        @test count_eigenvalues_in_interval(D,-1,2)==(count=4,complete=true,certified=true,method=:diagonal)
        @test count_eigenvalues_in_interval(D,2,2).count==2
        if R===Float32
            @test !count_eigenvalues_in_interval(D,nextfloat(1.0),3.0).certified
            @test !eigenvalues_in_interval(D,nextfloat(1.0),3.0).certified_count
        end
        @test eigenvalues_in_interval(D,10,11).count==0
        @test eigenvalues_in_interval(D,-1,2;vectors=false).vectors===nothing
        initial=zeros(T,6,2); initial[2,1]=one(T); initial[4,2]=one(T)
        # Verified pairs in a strict subspace cannot certify an exhaustive count.
        partial=eigenvalues_in_interval(D,-1,2;method=:krylov,subspace=2,blocksize=2,maxblocks=1,initial)
        @test partial.count==2 && !partial.complete && !partial.certified_count && !partial.converged
        count=count_eigenvalues_in_interval(D,-1,2;method=:krylov,subspace=2,blocksize=2,maxblocks=1,initial)
        @test count.count==2 && !count.complete && !count.certified
        # Sturm endpoint side limits: [1,3] contains both exact eigenvalues.
        tri=T[2 -1;-1 2]
        @test count_eigenvalues_in_interval(tri,1,3).count==2
        @test count_eigenvalues_in_interval(tri,1,1).count==1
        @test !count_eigenvalues_in_interval(tri,1,3).certified
        n=9; d=fill(T(2),n); off=fill(T(-1),n-1)
        if T<:Complex
            off .*= T(im)
        end
        tri=Matrix(LA.Tridiagonal(conj.(off),d,off))
        expected=sort(LA.eigvals(LA.Hermitian(tri))); expected=filter(x->R(.4)<=x<=R(2.4),expected)
        result=eigenvalues_in_interval(CSRMatrix(tri),.4,2.4)
        @test result.values≈expected rtol=100tol(T) atol=100tol(T)
        @test result.complete && result.converged && !result.certified_count
        @test tri*result.vectors≈result.vectors*LA.Diagonal(result.values) rtol=100tol(T) atol=100tol(T)
        @test count_eigenvalues_in_interval(tri,.4,2.4).count==length(expected)
        # Genuine dense coupling forces the general projection path.
        Q=Matrix(LA.qr(randn(T,8,8)).Q); eigenvalues=R[-4,-2,.2,.2,.3,2,3,5]
        H=Matrix(LA.Hermitian(Q*LA.Diagonal(T.(eigenvalues))*Q'))
        full=eigenvalues_in_interval(CSRMatrix(H),-.5,.5;subspace=8,blocksize=3,maxblocks=8)
        @test full.values≈R[.2,.2,.3] rtol=100tol(T) atol=100tol(T)
        @test full.complete && full.converged && !full.certified_count
        filtered=polynomial_filtered_eigenvalues(H,-.5,.5,3;subspace=8,blocksize=3,degree=12)
        @test filtered.values≈R[.2,.2,.3] rtol=100tol(T) atol=100tol(T)
        @test filtered.converged && length(filtered.values)==3
        @test H*filtered.vectors≈filtered.vectors*LA.Diagonal(filtered.values) rtol=100tol(T) atol=100tol(T)
        @test filtered.vectors'*filtered.vectors≈Matrix{T}(LA.I,3,3) rtol=100tol(T) atol=100tol(T)
        @test filtered.matrix_products==filtered.filter_products+filtered.projection_products+filtered.verification_products
        GC.gc(); again=polynomial_filtered_eigenvalues(H,-.5,.5,3;subspace=8,blocksize=3,degree=12,vectors=false)
        @test again.vectors===nothing
        @test again.values≈filtered.values rtol=100tol(T) atol=100tol(T)
        # Constant filters fall back to the original operator for expansion.
        ordinary=polynomial_filtered_eigenvalues(H,-.5,.5,3;degree=0,subspace=8,blocksize=3)
        @test ordinary.converged && ordinary.filter_products==0
        @test ordinary.values≈R[.2,.2,.3] rtol=100tol(T) atol=100tol(T)
        spectrum=vcat(collect(range(-5,-1;length=10)),[-.2,.1,.2],collect(range(1,5;length=11)))
        large=CSRMatrix(Matrix(LA.Diagonal(T.(spectrum))))
        restarted=polynomial_filtered_eigenvalues(large,-.25,.25,3;subspace=10,keep=4,blocksize=3,degree=24,maxiter=20)
        @test restarted.converged
        @test restarted.values≈R[-.2,.1,.2] rtol=100tol(T) atol=100tol(T)
        @test all(h.dimension<=10 for h in restarted.history)
        @test !restarted.complete
        unavailable=polynomial_filtered_eigenvalues(D,10,11,2;subspace=6)
        @test isempty(unavailable.values) && unavailable.complete && !unavailable.converged
        @test unavailable.reason===:insufficient_interval_pairs
        budget=polynomial_filtered_eigenvalues(H,-.5,.5,3;maxiter=0)
        @test isempty(budget.values) && !budget.complete && !budget.converged && budget.matrix_products==0
        empty=eigenvalues_in_interval(zeros(T,0,0),-1,1)
        @test empty.complete && empty.converged && empty.count==0
        @test polynomial_filtered_eigenvalues(zeros(T,0,0),-1,1).complete
        @test count_eigenvalues_in_interval(zeros(T,0,0),-1,1).count==0
        @test_throws ArgumentError eigenvalues_in_interval(H,1,-1)
        @test_throws ArgumentError eigenvalues_in_interval(H,-1,1;method=:unknown)
        @test_throws ArgumentError eigenvalues_in_interval(H,-1,1;method=:tridiagonal)
        @test_throws ArgumentError count_eigenvalues_in_interval(H,-1,1;method=:tridiagonal)
        @test_throws ArgumentError polynomial_filtered_eigenvalues(H,-1,1,9)
        @test_throws ArgumentError polynomial_filtered_eigenvalues(H,-1,1,2;keep=0,subspace=4)
        @test_throws ArgumentError polynomial_filtered_eigenvalues(H,-1,1,2;deflation_tol=NaN)
        @test_throws ArgumentError eigenvalues_in_interval(H,-Inf,Inf)
        @test_throws DimensionMismatch eigenvalues_in_interval(zeros(T,2,3),-1,1)
    end
end
