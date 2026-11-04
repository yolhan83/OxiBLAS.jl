# PREPARED ONLY. Regression coverage for the static priority audit.
# AuditTriangle: test/helpers/probes.jl
struct AuditNoReadVector{T} <: AbstractVector{T}
    owner::Matrix{T}
end
Base.size(x::AuditNoReadVector)=(size(x.owner,1),)
Base.dataids(x::AuditNoReadVector)=Base.dataids(x.owner)
Base.getindex(::AuditNoReadVector,i::Int)=error("alpha-zero input read")
@testset "Priority audit: column-major BLAS and physical views" begin
    for T in TYPES
        for (m,n,k) in ((17,19,13),(33,7,5))
            # Nontrivial leading dimensions and nonunit row/column strides.
            a=randn(T,2m+3,2k+3);b=randn(T,2k+3,2n+3);c=fill(T(NaN),2m+3,3n+3)
            A=view(a,2:2:2m,2:2:2k);B=view(b,2:2:2k,2:2:2n);C=view(c,2:2:2m,2:3:3n-1)
            expected=Matrix(A)*Matrix(B)
            for algorithm in (:reference,:blocked,:packed,:register,:simd)
                gemm!(C,A,B;algorithm,threaded=false)
                @test C≈expected rtol=30tol(T) atol=30tol(T)
                @test isnan(c[1,1])
            end
            padded=fill(T(NaN),m+5,n+2);out=view(padded,2:m+1,2:n+1)
            gemm!(out,A,B;algorithm=:simd)
            @test out≈expected rtol=30tol(T) atol=30tol(T)
            @test isnan(padded[1,1])
        end
        for upper in (true,false),hermitian in (false,true),threaded in (false,true)
            n=33;A=randn(T,n+5,n+2);M=view(A,2:n+1,2:n+1)
            if hermitian
                for j in 1:n;M[j,j]=T<:Complex ? complex(real(M[j,j]),NaN) : M[j,j];end
            end
            logical=Matrix(hermitian ? LA.Hermitian(M,upper ? :U : :L) : LA.Symmetric(M,upper ? :U : :L))
            x=randn(T,n);old=randn(T,n);y=copy(old);alpha=T(.5);beta=T(-.25)
            f=hermitian ? hemv! : symv!
            f(y,M,x,alpha,beta;uplo=upper ? :U : :L,threaded,parallel_threshold=0)
            @test y≈alpha*logical*x+beta*old rtol=40tol(T) atol=40tol(T)
            # Unknown layout fallback and selected-triangle alias snapshot.
            selected=AuditTriangle(Matrix(M),upper,false)
            x=copy(old);f(x,selected,x;uplo=upper ? :U : :L,threaded,parallel_threshold=0)
            @test x≈logical*old rtol=40tol(T) atol=40tol(T)
            destination=view(selected.data,:,1);expected=logical*old
            f(destination,selected,old;uplo=upper ? :U : :L,threaded,parallel_threshold=0)
            @test destination≈expected rtol=40tol(T) atol=40tol(T)
        end
        for uplo in (:U,:L),trans in (:N,:T,:C),unit in (false,true)
            n=19;M=randn(T,n,n);original=randn(T,n)
            D=Matrix(uplo===:U ? LA.UpperTriangular(M) : LA.LowerTriangular(M))
            unit && (D[LA.diagind(D)].=one(T))
            expected=op(D,trans)*original
            x=copy(original);trmv!(x,M;uplo,trans,unit)
            @test x≈expected rtol=30tol(T) atol=30tol(T)
            guarded=AuditTriangle(copy(M),uplo===:U,unit)
            # The vector shares A's storage: snapshot must not read unstored or unit entries.
            x=view(guarded.data,:,1);copyto!(x,original)
            selected=Matrix(uplo===:U ? LA.UpperTriangular(guarded.data) : LA.LowerTriangular(guarded.data))
            unit && (selected[LA.diagind(selected)].=one(T))
            expected=op(selected,trans)*original
            trmv!(x,guarded;uplo,trans,unit)
            @test x≈expected rtol=30tol(T) atol=30tol(T)
        end
        # A standard row-matrix output may wrap a vector, including Adjoint.
        a=randn(T,1,17);b=randn(T,17,9);row=zeros(T,9)
        for C in (transpose(row),adjoint(row))
            gemm!(C,a,b;algorithm=:register)
            @test C≈a*b rtol=30tol(T) atol=30tol(T)
        end
        duplicated_matrix=reshape(view(zeros(T,5),[1,1,2,3]),2,2)
        @test_throws ArgumentError gemm!(duplicated_matrix,ones(T,2,2),ones(T,2,2))
        A=randn(T,21,17)
        for trans in (:N,:T,:C)
            AA=op(A,trans);x=randn(T,size(AA,2));y=fill(T(NaN),size(AA,1))
            gemv!(y,A,x;trans,parallel_threshold=0)
            @test y≈AA*x rtol=30tol(T) atol=30tol(T)
        end
        ignored=ones(T,4,4);unread=AuditNoReadVector(ignored)
        syr2!(ignored,unread,unread,zero(T))
        @test all(==(one(T)),ignored)
        her2!(ignored,unread,unread,zero(T))
        @test all(==(one(T)),ignored)
        duplicate=view(zeros(T,4),[1,1,2,3]);saved=copy(duplicate)
        @test_throws ArgumentError trmv!(duplicate,ones(T,4,4))
        @test duplicate==saved
        @test_throws ArgumentError axpy!(duplicate,ones(T,4))
        @test_throws ArgumentError scal!(duplicate,T(2))
    end
end

@testset "Priority audit: packing buffers and aliases rejected before raw access" begin
    U=Union{Float32,Float64};mixed=fill!(Matrix{U}(undef,4,4),1.0)
    config=GemmConfig(Float64)
    @test_throws ArgumentError gemm!(mixed,mixed,mixed;config)
    @test_throws ArgumentError GemmWorkspace(U,4,4,4;config)
    @test_throws ArgumentError CSRMatrix(1,1,[1,2],[1],U[1.0])
    @test_throws ArgumentError CSCMatrix(1,1,[1,2],[1],U[1.0])
    for T in TYPES
        n=32;A=randn(T,n,n);B=randn(T,n,n);C=fill(T(7),n,n)
        config=GemmConfig(T;small_threshold=0,parallel_threshold=0)
        w=GemmWorkspace(T,n,n,n;workers=2,config)
        own=reshape(view(w.buffers[1].a,w.buffers[1].ai:w.buffers[1].ai+n*n-1),n,n)
        @test_throws ArgumentError gemm!(C,own,B;workspace=w,config)
        @test all(==(T(7)),C)
        @test_throws ArgumentError gemm!(own,A,B;workspace=w,config)
        damaged=GemmWorkspace(T,n,n,n;workers=1,config)
        damaged.buffers[1]=OxiBLAS._PackBuffer(T[0],T[0],1,1)
        @test_throws ArgumentError gemm!(C,A,B;workspace=damaged,config)
        @test all(==(T(7)),C)
        w.buffers[2]=w.buffers[1]
        @test_throws ArgumentError gemm!(C,A,B;workspace=w,config)
        @test all(==(T(7)),C)
        # Advanced indexing already owns its result: solves must not mutate RHS.
        rhs=randn(T,n,2);saved=copy(rhs)
        for make in (lu_factor,lu_complete,lu_rook)
            F=make(A);solution=solve(F,rhs)
            @test rhs==saved
            @test A*solution≈rhs rtol=60tol(T) atol=60tol(T)
        end
        H=A*A'+Matrix{T}(LA.I,n,n)
        guarded=AuditTriangle(H,false,false)
        ch=cholesky_factor(guarded);ld=ldlt_factor(H)
        @test H*solve(ch,rhs)≈rhs rtol=60tol(T) atol=60tol(T)
        @test H*solve(ld,rhs)≈rhs rtol=60tol(T) atol=60tol(T)
        @test rhs==saved
        GC.gc()
        normal=GemmWorkspace(T,n,n,n;workers=2,config)
        gemm!(C,A,B;workspace=normal,config)
        @test C≈A*B rtol=30tol(T) atol=30tol(T)
    end
end

@testset "Priority audit: tensor physical-layout cost model" begin
    for T in TYPES
        parent=randn(T,18,20);A=view(parent,2:2:16,2:2:18);B=randn(T,9,7);D=randn(T,7,5)
        for optimize in (:greedy,:beam,:optimal),model in (:work,:layout)
            plan=plan_einsum("ij,jk,kl->il",A,B,D;optimize,cost_model=model,layout_weight=3)
            report=einsum_layout_report(plan)
            @test report.input_strides[1]==collect(strides(A))
            @test report.estimated_work>0 && report.estimated_traffic>0 && report.model_only
            @test einsum(plan,A,B,D)≈A*B*D rtol=40tol(T) atol=40tol(T)
            copied=Matrix(A)
            @test einsum(plan,copied,B,D)≈copied*B*D rtol=40tol(T) atol=40tol(T)
            sliced=plan_einsum_sliced("ij,jk,kl->il",A,B,D;chunks=(j=3,),optimize,cost_model=model)
            @test einsum(sliced,A,B,D)≈A*B*D rtol=60tol(T) atol=60tol(T)
        end
        @test_throws ArgumentError plan_einsum("ij,jk->ik",A,B;cost_model=:invalid)
        @test_throws ArgumentError plan_einsum("ij,jk->ik",A,B;layout_weight=-1)
    end
    unread=TensorNoRead{Float64,2}((8,8))
    meta=plan_einsum("ij,jk->ik",unread,unread;optimize=:optimal,cost_model=:layout)
    @test all(isnothing,einsum_layout_report(meta).input_strides)
    output=fill(NaN,8,8);einsum!(output,meta,unread,unread;α=0)
    @test all(iszero,output)
    setprecision(BigFloat,256) do
        T=BigFloat;delta=ldexp(one(T),-180);A=T[1+delta 2;3 4+delta];x=T[1,2];y=zeros(T,2)
        symv!(y,A,x;threaded=true,parallel_threshold=0)
        @test y[1]==5+delta && y[2]==10+2delta
        z=copy(x);trmv!(z,A)
        @test z==T[5+delta,8+2delta]
        @test all(precision(v)==256 for v in z)
    end
end

@testset "Priority audit: CSC physical columns without assuming sorted rows" begin
    for T in TYPES
        # Deliberately unsorted row indices in each column; constructor permits it.
        A=CSCMatrix(3,3,[1,3,5,7],[3,1,2,1,3,2],T[.5,3,4,.25,5,.125])
        if T<:Complex
            A.values[1]+=T(im/4);A.values[2]-=T(im/8)
        end
        D=Matrix(A)
        for trans in (:N,:T,:C),threaded in (false,true)
            x=T[1,2,3];y=copy(x);saved=copy(y)
            sparse_mul!(y,A,x,T(.5),T(-.25);trans,threaded,parallel_threshold=0)
            @test y≈.5op(D,trans)*x-.25saved rtol=30tol(T) atol=30tol(T)
            z=copy(x);sparse_mul!(z,A,z;trans,threaded,parallel_threshold=0)
            @test z≈op(D,trans)*x rtol=30tol(T) atol=30tol(T)
        end
        @test_throws ArgumentError sparse_mul!(view(A.values,1:3),A,ones(T,3))
        y=fill(T(NaN),3);sparse_mul!(y,A,ones(T,3);threaded=false)
        @test all(isfinite,y)
    end
end
