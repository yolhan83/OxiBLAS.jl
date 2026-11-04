# Prepared only: no tests/benchmarks run without explicit authorization.
function partial_svd_check(A,F,expected;factor=5000)
    T=eltype(A); R=real(eltype(F.S)); m,n=size(A); q=length(expected)
    bound=factor*max(1,m,n)*eps(R)*max(one(R),LA.norm(A))
    @test length(F.S)==q
    @test LA.norm(F.S-expected)<=bound
    if F.U===nothing
        @test F.Vt===nothing
    else
        @test size(F.U)==(m,q) && size(F.Vt)==(q,n)
        V=F.Vt'
        @test LA.norm(A*V-F.U*LA.Diagonal(F.S))<=bound
        @test LA.norm(A'*F.U-V*LA.Diagonal(F.S))<=bound
    end
end

@testset "Genuine selected SVD: tall/wide, values/vectors, both representations" begin
    for T in TYPES, (m,n) in ((1,1),(17,11),(11,17)), representation in (:dilation,:gram)
        A=randn(T,m,n); old=copy(A); r=min(m,n); reference=LA.svd(A)
        choices=(1:min(2,r),max(1,r÷2):min(r,max(1,r÷2)+1),max(1,r-1):r)
        for indices in choices, vectors in (false,true)
            info=Ref{Any}(); F=svd_selected(A;indices,vectors,representation,diagnostics=info)
            partial_svd_check(A,F,reference.S[indices]); @test A==old
            @test info[].bisection_targets==length(indices)
            @test info[].inverse_targets==(vectors ? length(indices) : 0)
            if vectors
                q=length(indices); bound=5000max(m,n)*eps(real(T))
                @test LA.norm(F.U'*F.U-Matrix{T}(LA.I,q,q))<=bound
                @test LA.norm(F.Vt*F.Vt'-Matrix{T}(LA.I,q,q))<=bound
            end
        end
        info=Ref{Any}(); F=svd_truncated(A,min(2,r);diagnostics=info)
        partial_svd_check(A,F,reference.S[1:min(2,r)])
        @test info[].bisection_targets==min(2,r)
        # Duplicate/unordered requests solve each distinct rank once.
        indices=r==1 ? [1,1] : [r,1,r]
        F=svd_selected(A;indices,diagnostics=info)
        partial_svd_check(A,F,reference.S[indices]); @test info[].distinct==length(unique(indices))
    end
end

@testset "Index/interval boundaries, clusters and structural nullspaces" begin
    for T in TYPES, wide in (false,true), representation in (:dilation,:gram)
        values=real(T)[16,8,8,4,1,.5,0,0,0]; A=zeros(T,12,9)
        for j in 1:9
            A[j,j]=T<:Complex ? T(im)*values[j] : T(values[j])
        end
        wide && (A=Matrix(adjoint(A)))
        for interval in ((0,8),(8,16),(-1,0),(3,3),(32,64),(-Inf,Inf)), vectors in (false,true)
            expected=filter(s->interval[1]<s<=interval[2],values)
            info=Ref{Any}(); F=svd_selected(A;interval,vectors,representation,diagnostics=info)
            partial_svd_check(A,F,expected)
            @test info[].bisection_targets==count(>(0),expected)
            if vectors
                q=length(expected); bound=5000eps(real(T))*max(size(A)...)
                @test LA.norm(F.U'*F.U-Matrix{T}(LA.I,q,q))<=bound
                @test LA.norm(F.Vt*F.Vt'-Matrix{T}(LA.I,q,q))<=bound
            end
        end
        # A zero singular value need not have coordinate basis null vectors.
        B=T[0 1 0 0;0 0 0 0;0 0 2 3;0 0 0 0]
        F=svd_selected(B;indices=3:4,representation)
        partial_svd_check(B,F,zeros(real(T),2))
        @test LA.norm(F.U'*F.U-Matrix{T}(LA.I,2,2))<=5000eps(real(T))
        @test LA.norm(F.Vt*F.Vt'-Matrix{T}(LA.I,2,2))<=5000eps(real(T))
        F=svd_selected(zeros(T,5,3);indices=2:3,representation)
        partial_svd_check(zeros(T,5,3),F,zeros(real(T),2))
    end
    # Repeated nonzero values in a rotated invariant subspace, independent
    # matrix construction and full LAPACK oracle. No singular-vector sign oracle.
    for T in TYPES
        Q=Matrix(LA.qr(randn(T,15,9)).Q)[:,1:9]; V=Matrix(LA.qr(randn(T,9,9)).Q)
        sigma=real(T)[9,9,9,5,3,3,1,.5,.25]; A=Q*LA.Diagonal(sigma)*V'
        F=svd_selected(A;indices=2:5); partial_svd_check(A,F,sigma[2:5])
        @test LA.norm(F.U'*F.U-Matrix{T}(LA.I,4,4))<=50000eps(real(T))
        @test LA.norm(F.Vt*F.Vt'-Matrix{T}(LA.I,4,4))<=50000eps(real(T))
    end
end

@testset "Small singular values without squared conditioning" begin
    for (R,exponents) in ((Float32,(-20,-60)),(Float64,(-30,-500)))
        for exponent in exponents
            delta=ldexp(one(R),exponent); A=R[1 1;0 delta]
            F=svd_selected(A;indices=2:2,vectors=false)
            expected=delta/sqrt(R(2))
            @test abs(F.S[1]-expected)<=R(1000)*eps(R)*expected
        end
    end
    # Demonstrable Gram loss: (2^-30)^2 is lost in 1+delta^2 in Float64.
    # The faithful explicit Gram variant must not fabricate singular triplets.
    A=[1.0 1.0;0.0 2.0^-30]
    @test_throws ConvergenceError svd_selected(A;indices=2:2,representation=:gram)
    for exponent in (-500,500)
        scale=ldexp(1.0,exponent); A=[3.0 .25 0;0 2 .125;0 0 1].*scale
        F=svd_selected(A;indices=2:3); partial_svd_check(A,F,LA.svdvals(A)[2:3])
    end
end

@testset "Partial SVD compact backtransforms and pivoted shifted solves" begin
    for T in TYPES
        A=randn(T,13,8); B=bidiagonal_factor(A); r=8; X=randn(T,r,3)
        U=OxiBLAS._selected_backtransform(B,X,true)
        V=OxiBLAS._selected_backtransform(B,X,false)
        @test LA.norm(U-B.U[:,1:r]*X)<=5000eps(real(T))*LA.norm(A)
        @test LA.norm(V-B.V*X)<=5000eps(real(T))*LA.norm(A)
    end
    for R in (Float32,Float64)
        # Zero diagonal forces neighboring pivoting; Thomas without pivoting
        # would need an artificial regularization before its very first divide.
        d=R[0,0,0,0]; e=R[1,.5,2]; lambda=R(.125); rhs=R[1,2,3,4]
        F=OxiBLAS._selected_shift_factor(d,e,lambda,OxiBLAS._selected_pivmin(R))
        x=copy(rhs); OxiBLAS._selected_shift_solve!(x,F)
        tri=Matrix(LA.SymTridiagonal(d,e))-lambda*Matrix{R}(LA.I,4,4)
        @test LA.norm(tri*x-rhs)<=5000eps(R)*LA.norm(rhs)
    end
end

@testset "Empty selections, invalid arguments, explicit budgets and views" begin
    for T in TYPES, (m,n) in ((0,0),(3,0),(0,4),(5,3))
        A=zeros(T,m,n)
        for vectors in (false,true)
            info=Ref{Any}(); F=svd_selected(A;indices=Int[],vectors,diagnostics=info)
            partial_svd_check(A,F,real(T)[])
            @test info[].bisection_targets==0 && info[].inverse_targets==0
            vectors && @test size(F.U)==(m,0) && size(F.Vt)==(0,n)
        end
    end
    for T in TYPES
        parent=randn(T,26,18); old=copy(parent); A=view(parent,26:-2:2,18:-2:2)
        F=svd_selected(A;indices=2:3); partial_svd_check(A,F,LA.svdvals(A)[2:3]); @test parent==old
        jobs=Task[]
        for _ in 1:2
            push!(jobs,Threads.@spawn svd_truncated(copy(A),2))
        end
        GC.gc()
        for task in jobs
            partial_svd_check(A,fetch(task),LA.svdvals(A)[1:2])
        end
    end
    A=randn(9,5)
    @test_throws ArgumentError svd_selected(A;indices=[0])
    @test_throws ArgumentError svd_selected(A;indices=[6])
    @test_throws ArgumentError svd_selected(A;indices=[true])
    @test_throws ArgumentError svd_selected(A;indices=1:2,interval=(0,1))
    @test_throws ArgumentError svd_selected(A;interval=(2,1))
    @test_throws ArgumentError svd_selected(A;interval=(NaN,1))
    @test_throws ArgumentError svd_selected(A;full=true)
    @test_throws ArgumentError svd_selected(A;representation=:unknown)
    @test_throws ArgumentError svd_selected(A;algorithm=:golub_kahan)
    @test_throws ArgumentError svd_selected(A;rtol=0)
    @test_throws ArgumentError svd_selected(A;atol=-1)
    @test_throws ArgumentError svd_selected([NaN 0.;0 1];indices=Int[])
    @test_throws ConvergenceError svd_selected(A;indices=1:1,maxiter=0)
    @test_throws ConvergenceError svd_selected(A;indices=1:1,maxinverse=0)
end

@testset "Partial SVD generic providers and half working precision" begin
    setprecision(BigFloat,256) do
        for T in EXT_TYPES
            A=T[4 .25 0;0 2 .125;0 0 1;0 0 0]; reference=svd_bidiagonal(A;vectors=false)
            for wide in (false,true), representation in (:dilation,:gram)
                M=wide ? adjoint(A) : A; info=Ref{Any}()
                F=svd_selected(M;indices=2:3,representation,diagnostics=info)
                check_extended_array(F.S,extref(reference.S[2:3]),T;factor=5000)
                U=extref(F.U); V=extref(F.Vt)'; MM=extref(M); s=extref(F.S)
                check_extended_array(MM*V,U*LA.Diagonal(s),T;factor=5000)
                check_extended_array(MM'*U,V*LA.Diagonal(s),T;factor=5000)
                @test info[].bisection_targets==2 && info[].inverse_targets==2
                if real(T)===Float16
                    @test eltype(F.U)===OxiBLAS._half_work(T)
                end
            end
        end
    end
end
