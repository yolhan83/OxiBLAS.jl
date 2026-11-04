# Prepared for supervised execution; no test has been run by writing this file.
@testset "Golub-Kahan sparse SVD, retained memory and reusable contexts" begin
    for T in TYPES, shape in ((23,17),(17,23)), b in (1,2,3)
        m,n=shape; R=real(T); A=zeros(T,m,n)
        for j in 1:min(m,n)
            A[j,j]=T(j<=3 ? 40-5j : j/32)
            T<:Complex && (A[j,j]*=T(cis(R(j)/7)))
            j<min(m,n) && (A[j,j+1]=T(1)/64)
        end
        input=CSRMatrix(A); old=copy(input.values); cap=12; rank=3; keep=6
        w=SparseSVDWorkspace(input,rank;subspace=cap,blocksize=b,rowblock=5)
        ids=map(objectid,OxiBLAS._gk_arrays(w)); expected=LA.svdvals(A)[1:rank]
        F=sparse_svd_lanczos(input,rank;subspace=cap,blocksize=b,keep,workspace=w,return_basis=true)
        @test F.converged
        @test F.S≈expected rtol=100tol(T) atol=100tol(T)
        @test LA.norm(A*F.Vt'-F.U*LA.Diagonal(F.S))<=100tol(T)*LA.norm(A)
        @test LA.norm(A'*F.U-F.Vt'*LA.Diagonal(F.S))<=100tol(T)*LA.norm(A)
        @test F.operator_matvecs==sum(h.new_matvecs for h in F.history)
        @test F.verification_matvecs==2rank*length(F.history)
        @test F.left_basis_dimension<=cap && F.right_basis_dimension<=min(n,cap+max(b,rank))
        @test F.krylov.left'*A*F.krylov.right≈F.krylov.projected rtol=100tol(T) atol=100tol(T)*LA.norm(A)
        @test F.workspace_bytes==sparse_svd_workspace_bytes(w)
        savedU=copy(F.U); savedS=copy(F.S); savedVt=copy(F.Vt)
        # Values may change while sparsity structure is retained: no cached
        # normal matrix or stale operator image may survive reuse.
        input.values .*= R(2)
        G=sparse_svd_lanczos(input,rank;subspace=cap,blocksize=b,keep,workspace=w)
        @test G.S≈2expected rtol=100tol(T) atol=100tol(T)
        @test G.krylov===nothing
        @test F.U==savedU && F.S==savedS && F.Vt==savedVt
        @test map(objectid,OxiBLAS._gk_arrays(w))==ids
        @test input.values==2old
        # A shared context is serialized; results never alias its mutable data.
        jobs=Task[]
        for _ in 1:2
            push!(jobs,Threads.@spawn sparse_svd_lanczos(input,rank;subspace=cap,blocksize=b,keep,workspace=w))
        end
        GC.gc()
        for task in jobs
            @test fetch(task).S≈G.S rtol=100tol(T)
        end
    end
    for T in TYPES
        R=real(T); A=CSRMatrix(Matrix(LA.Diagonal(T.(1:9))))
        old=sparse_svd_lanczos(A,2;algorithm=:dilation,subspace=18,maxrestarts=0)
        new=sparse_svd_lanczos(A,2;subspace=9,maxrestarts=0)
        @test old.S≈new.S rtol=100tol(T) atol=100tol(T)
        @test new.krylov===nothing && old.krylov!==nothing
        bad=SparseSVDWorkspace(T,8,9,2;subspace=8)
        @test_throws DimensionMismatch sparse_svd_lanczos(A,2;workspace=bad)
        w=SparseSVDWorkspace(A,2;subspace=8)
        @test_throws DimensionMismatch sparse_svd_lanczos(A,3;subspace=8,workspace=w)
        @test_throws ArgumentError sparse_svd_lanczos(A,2;algorithm=:dilation,workspace=w)
        @test_throws ArgumentError sparse_svd_lanczos(A,2;algorithm=:unknown)
        @test_throws ArgumentError sparse_svd_lanczos(A,2;subspace=2,keep=1)
        @test_throws ArgumentError sparse_svd_lanczos(A,2;initial=view(w.V,:,1:2),subspace=8,workspace=w)
        # Payload growth is linear in ambient dimensions at fixed capacities.
        small=SparseSVDWorkspace(T,1000,800,3;subspace=20,blocksize=2)
        big=SparseSVDWorkspace(T,2000,1600,3;subspace=20,blocksize=2)
        fixed=sparse_svd_workspace_bytes(small)-sum(sizeof,(small.U,small.V,small.left,small.right,small.ritzU,small.ritzV))
        @test sparse_svd_workspace_bytes(big)<=2sparse_svd_workspace_bytes(small)
        @test length(small.U)+length(small.V)<2*(1000+800)*20 # old basis+images alone
        @test all(size(B,1)<=20 && size(B,2)<=23 for B in (small.projected,))
    end
end

@testset "Native triangular leaves, all logical triangles and views" begin
    for T in TYPES, side in (:L,:R), upper in (false,true), trans in (:N,:T,:C), unit in (false,true)
        order=7; rows,cols=side===:L ? (order,11) : (19,order)
        A=randn(T,order,order)/32
        for i in 1:order
            A[i,i]=T(2)
        end
        logical=upper ? Matrix(LA.UpperTriangular(A)) : Matrix(LA.LowerTriangular(A))
        unit && (logical[LA.diagind(logical)].=one(T))
        operand=trans===:N ? logical : trans===:T ? transpose(logical) : adjoint(logical)
        # Throw on any read of the unused triangle or implicit unit diagonal.
        selected=AuditTriangle(copy(A),upper,unit)
        base=randn(T,rows,cols); alpha=T(3)/4
        for operation in (trmm!,trsm!), layout in (:dense,:padded,:strided,:negative)
            owner=layout===:dense ? copy(base) : zeros(T,2rows+3,2cols+3)
            B=layout===:dense ? owner : layout===:padded ? view(owner,2:rows+1,2:cols+1) :
              layout===:strided ? view(owner,2:2:2rows,2:2:2cols) : view(owner,2:rows+1,2cols:-2:2)
            copyto!(B,base)
            expected=operation===trmm! ? (side===:L ? alpha*operand*base : alpha*base*operand) :
                (side===:L ? operand\(alpha*base) : (alpha*base)/operand)
            operation(B,selected,alpha;side,uplo=upper ? :U : :L,trans,unit,blocksize=8)
            @test B≈expected rtol=100tol(T) atol=100tol(T)
            copyto!(B,base)
            operation(B,selected,alpha;side,uplo=upper ? :U : :L,trans,unit,blocksize=8,algorithm=:reference,kernel=:scalar)
            @test B≈expected rtol=100tol(T) atol=100tol(T)
        end
        w=triangular_workspace(T,rows,cols;side,workers=1)
        result=copy(base); trsm!(result,selected;side,uplo=upper ? :U : :L,trans,unit,workspace=w)
        @test result≈(side===:L ? operand\base : base/operand) rtol=100tol(T) atol=100tol(T)
    end
end

@testset "Triangular context ownership, aliasing, GC and generic fallback" begin
    for T in TYPES, operation in (trmm!,trsm!)
        A=randn(T,17,17)/32
        for i in 1:17
            A[i,i]=T(2)
        end
        original=copy(A); expected=operation===trmm! ? LA.UpperTriangular(A)*A : LA.UpperTriangular(A)\A
        operation(A,A;uplo=:U,blocksize=4)
        @test A≈expected rtol=100tol(T) atol=100tol(T)
        w=triangular_workspace(T,17,11;workers=2); base=randn(T,17,11)
        jobs=Task[]
        for _ in 1:2
            push!(jobs,Threads.@spawn begin
                out=copy(base); operation(out,original;blocksize=4,workspace=w); out
            end)
        end
        GC.gc()
        expected=operation===trmm! ? LA.UpperTriangular(original)*base : LA.UpperTriangular(original)\base
        for task in jobs
            @test fetch(task)≈expected rtol=100tol(T) atol=100tol(T)
        end
        # Even a scalar-size leaf validates buffer aliasing before mutation.
        tiny=GemmWorkspace(T,4,4,4;workers=1)
        operand=reshape(view(tiny.buffers[1].a,1:4),2,2); fill!(operand,one(T)); saved=copy(operand)
        @test_throws ArgumentError operation(operand,Matrix{T}(LA.I,2,2);workspace=tiny)
        @test operand==saved
        B=copy(base); bad=copy(original); bad[3,3]=zero(T)
        operation===trsm! && @test_throws OxiBLAS.SingularException operation(B,bad;workspace=w)
        operation===trsm! && @test B==base
    end
    setprecision(BigFloat,192) do
        for T in EXT_TYPES
            T in TYPES && continue
            A=T[2 .125 0;0 3 .25;0 0 4]; B=T[1 2 3 4 5;2 3 4 5 6;3 4 5 6 7]
            for operation in (trmm!,trsm!)
                C=copy(B); operation(C,A;algorithm=:reference,kernel=:scalar)
                expected=operation===trmm! ? extref(A)*extref(B) : extref(A)\extref(B)
                check_extended_array(C,expected,T;factor=1000)
            end
        end
    end
end
