# Supervised phase A: test definitions only, no execution implied.
struct FilterNoRead{T,N} <: AbstractArray{T,N}
    shape::NTuple{N,Int}
end
Base.size(A::FilterNoRead)=A.shape
Base.getindex(::FilterNoRead,I...)=error("filter input must not be read")

@testset "Chebyshev filter coefficients, recurrence and reusable workspace" begin
    for T in TYPES
        R=real(T); n=6; d=R[-1,-.5,0,.25,.5,1]; A=Matrix(LA.Diagonal(T.(d)))
        @test hermitian_spectral_bounds(CSRMatrix(A))==(-one(R),one(R))
        X=randn(T,n,3)
        # Half-window, degree one: independent analytic formulas.
        for (damping,slope) in ((:none,2/pi),(:jackson,1/pi))
            P=polynomial_filter(CSRMatrix(A),0,1;degree=1,bounds=(-1,1),damping)
            Y=fill(T(NaN),n,3); apply_filter!(Y,P,X)
            @test Y≈T(.5).*X+R(slope).*(A*X) rtol=20tol(T) atol=20tol(T)
            @test filter_response(P,R(.25))≈R(.5+slope*.25) rtol=10tol(T) atol=10tol(T)
        end
        # Independent spectral functional calculus uses cos(k acos(lambda)),
        # rather than the implementation's matrix three-term recurrence.
        Q=Matrix(LA.qr(randn(T,n,n)).Q); H=Matrix(LA.Hermitian(Q*A*Q'))
        E=LA.eigen(LA.Hermitian(H)); bounds=(-R(1.1),R(1.1))
        for damping in (:jackson,:none)
            P=polynomial_filter(H,-.3,.6;degree=12,bounds,damping)
            weights=[sum(P.coefficients[k+1]*cos(k*acos(clamp(value/R(1.1),-one(R),one(R)))) for k in 0:length(P.coefficients)-1) for value in E.values]
            expected=E.vectors*LA.Diagonal(weights)*E.vectors'*X
            workspace=FilterWorkspace(P;columns=1); Y=similar(X)
            apply_filter!(Y,P,X;workspace)
            @test Y≈expected rtol=100tol(T) atol=100tol(T)
            buffers=workspace.buffers
            apply_filter!(Y,P,X;workspace,threaded=false)
            @test all(buffers[i]===workspace.buffers[i] for i in 1:4)
            @test Y≈expected rtol=100tol(T) atol=100tol(T)
            alias=copy(X); apply_filter!(alias,P,alias;workspace)
            @test alias≈expected rtol=100tol(T) atol=100tol(T)
            previous=randn(T,n,3); Y=copy(previous)
            apply_filter!(Y,P,X;workspace,α=T(2),β=T(-.5))
            @test Y≈2expected-.5previous rtol=100tol(T) atol=100tol(T)
            vector=copy(X[:,1]); apply_filter!(vector,P,vector;workspace)
            @test vector≈expected[:,1] rtol=100tol(T) atol=100tol(T)
            reversed=view(X,n:-1:1,:); reference=similar(X)
            apply_filter!(reference,P,Matrix(reversed);workspace)
            apply_filter!(Y,P,reversed;workspace)
            @test Y≈reference rtol=tol(T) atol=tol(T)
            y1=similar(X); y2=similar(X)
            @sync begin
                Threads.@spawn apply_filter!(y1,P,X;workspace)
                Threads.@spawn apply_filter!(y2,P,X;workspace)
                Threads.@spawn GC.gc()
            end
            @test y1≈expected rtol=100tol(T) atol=100tol(T)
            @test y1≈y2 rtol=tol(T) atol=tol(T)
            unread=FilterNoRead{T,2}((n,3)); fill!(Y,T(NaN))
            apply_filter!(Y,P,unread;workspace,α=zero(T))
            @test all(iszero,Y)
            storage=view(workspace.buffers[1],:,1:3)
            @test_throws ArgumentError apply_filter!(Y,P,storage;workspace)
            @test_throws ArgumentError apply_filter!(storage,P,X;workspace)
            @test_throws ArgumentError apply_filter!(Y,P,fill(T(Inf),n,3);workspace)
            @test !workspace.active
            apply_filter!(Y,P,X;workspace)
            @test Y≈expected rtol=100tol(T) atol=100tol(T)
        end
        # Indicator moments checked by independent midpoint integration.
        P=polynomial_filter(A,-.3,.6;degree=6,bounds=(-1,1),damping=:none)
        points=8192
        for k in 0:6
            moment=sum((-.3<=cospi((j-.5)/points)<=.6 ? cospi(k*(j-.5)/points) : 0.0) for j in 1:points)/points
            @test abs(P.coefficients[k+1]-(k==0 ? moment : 2moment))<.002
        end
        full=polynomial_filter(A,-2,2); outside=polynomial_filter(A,2,3)
        Y=similar(X); apply_filter!(Y,full,X); @test Y==X
        apply_filter!(Y,outside,X); @test all(iszero,Y)
        scalar=polynomial_filter(T(3).*Matrix{T}(LA.I,n,n),2,4)
        apply_filter!(Y,scalar,X); @test Y==X
        scalar=polynomial_filter(T(3).*Matrix{T}(LA.I,n,n),4,5)
        apply_filter!(Y,scalar,X); @test all(iszero,Y)
        half=polynomial_filter(A,0,1;degree=0,bounds=(-1,1))
        apply_filter!(Y,half,X); @test Y≈T(.5).*X rtol=tol(T) atol=tol(T)
        # Cached scaling avoids forming the overflowing unscaled A*X.
        for scale in (R(2)^(-60),floatmax(R)/4)
            large=Matrix(LA.Diagonal(T.([-scale,scale])))
            P=polynomial_filter(large,zero(R),scale;degree=1,bounds=(-scale,scale),damping=:none)
            x=fill(T(8),2); y=similar(x); apply_filter!(y,P,x)
            @test y≈T.(R[8*(.5-2/pi),8*(.5+2/pi)]) rtol=40tol(T) atol=40tol(T)
        end
        P=polynomial_filter(CSRMatrix(A),-.3,.6)
        alias_plan=polynomial_filter(Matrix{T}(LA.I,n,n),0,2)
        @test_throws ArgumentError apply_filter!(alias_plan.matrix.values,alias_plan,ones(T,n))
        @test_throws ArgumentError FilterWorkspace(P;columns=-1)
        wrong_rows=FilterWorkspace(polynomial_filter(zeros(T,n+1,n+1),-1,1))
        @test_throws DimensionMismatch apply_filter!(zeros(T,n),P,zeros(T,n);workspace=wrong_rows)
        @test_throws DimensionMismatch apply_filter!(zeros(T,n-1),P,zeros(T,n-1))
        empty=polynomial_filter(zeros(T,0,0),-1,1)
        @test isempty(apply_filter!(T[],empty,T[]))
        @test size(apply_filter!(zeros(T,n,0),P,zeros(T,n,0)))==(n,0)
        @test_throws ArgumentError filter_response(P,NaN)
        @test_throws ArgumentError polynomial_filter(A,1,1)
        @test_throws ArgumentError polynomial_filter(A,-1,1;bounds=(1,-1))
        @test_throws ArgumentError polynomial_filter(A,-1,1;bounds=(-1,1,2))
        @test_throws ArgumentError polynomial_filter(A,-1,1;degree=-1)
        @test_throws ArgumentError polynomial_filter(A,-1,1;damping=:unknown)
        bad=copy(A); bad[1,3]=one(T)
        @test_throws ArgumentError polynomial_filter(bad,-1,1)
        @test_throws ArgumentError hermitian_spectral_bounds(bad)
        @test_throws ArgumentError polynomial_filter(fill(T(NaN),2,2),-1,1)
    end
end
