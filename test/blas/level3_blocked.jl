# Recursive triangular BLAS 3 against LinearAlgebra references and residuals.
struct StoredTriangle{T} <: AbstractMatrix{T}
    data::Matrix{T}
    upper::Bool
    unit::Bool
end
Base.size(A::StoredTriangle)=size(A.data)
function Base.getindex(A::StoredTriangle,i::Int,j::Int)
    (A.upper ? i<=j : i>=j) || error("unused triangle read")
    A.unit && i==j && error("unit diagonal read")
    A.data[i,j]
end
struct WriteOnlyMatrix{T} <: AbstractMatrix{T}
    data::Matrix{T}
end
Base.size(A::WriteOnlyMatrix)=size(A.data)
Base.getindex(::WriteOnlyMatrix,i::Int,j::Int)=error("old output read")
Base.setindex!(A::WriteOnlyMatrix,v,i::Int,j::Int)=(A.data[i,j]=v)
# tri_indices, check_triangle: test/helpers/checks.jl

@testset "Recursive triangular BLAS 3" begin
    for T in TYPES, n in (0,1,17,35,65), side in (:L,:R), uplo in (:U,:L), trans in (:N,:T,:C), unit in (false,true)
        A=randn(T,n,n)/max(1,n); A[LA.diagind(A)].+=T(2)
        tri=uplo===:U ? (unit ? LA.UnitUpperTriangular(A) : LA.UpperTriangular(A)) : (unit ? LA.UnitLowerTriangular(A) : LA.LowerTriangular(A))
        S=op(tri,trans)
        α=T<:Real ? T(-0.7) : T(0.7-0.3im)
        B=randn(T,side===:L ? (n,13) : (13,n)); old=copy(B)
        guard=StoredTriangle(A,uplo===:U,unit)
        expected=α*(side===:L ? S*old : old*S)
        @test trmm!(B,guard,α;side,uplo,trans,unit,blocksize=8,threaded=false)===B
        @test B≈expected rtol=5tol(T) atol=5tol(T)
        B=copy(old)
        expected=side===:L ? S\(α*old) : (α*old)/S
        @test trsm!(B,guard,α;side,uplo,trans,unit,blocksize=8,threaded=false)===B
        @test B≈expected rtol=5tol(T) atol=5tol(T)
        @test (side===:L ? S*B : B*S)≈α*old rtol=5tol(T) atol=5tol(T)
    end
end

@testset "Symmetric packing and triangular tile updates" begin
    for T in TYPES, n in (0,1,9,17,35,65), uplo in (:U,:L), side in (:L,:R), herm in (false,true)
        A=randn(T,n,n); guard=StoredTriangle(A,uplo===:U,false)
        S=herm ? LA.Hermitian(A,uplo) : LA.Symmetric(A,uplo)
        B=randn(T,side===:L ? (n,19) : (19,n)); C=randn(T,size(B)); old=copy(C)
        α=T<:Real ? T(0.7) : T(0.7+0.3im); β=T(-0.2)
        f=herm ? hemm! : symm!
        @test f(C,guard,B,α,β;side,uplo)≈α*(side===:L ? S*B : B*S)+β*old rtol=5tol(T) atol=5tol(T)
    end
    for T in TYPES, n in (0,1,9,17,35,65), uplo in (:U,:L), herm in (false,true), rank2 in (false,true), trans in (:N,:T)
        tr=herm && trans===:T ? :C : trans
        A=randn(T,trans===:N ? (n,23) : (23,n)); B=randn(T,size(A)); C=randn(T,n,n); old=copy(C)
        α=herm && !rank2 ? real(T)(0.7) : (T<:Real ? T(0.7) : T(0.7+0.3im))
        β=herm ? real(T)(-0.2) : T(-0.2)
        aa,bb=op(A,tr),op(B,tr)
        rhs(X)=herm ? adjoint(X) : transpose(X)
        expected=rank2 ? α*aa*rhs(bb)+(herm ? conj(α) : α)*bb*rhs(aa) : α*aa*rhs(aa)
        expected+=β*(herm ? Matrix(LA.Hermitian(old,uplo)) : old)
        f=herm ? (rank2 ? her2k! : herk!) : (rank2 ? syr2k! : syrk!)
        rank2 ? f(C,A,B,α,β;uplo,trans=tr,blocksize=8) : f(C,A,α,β;uplo,trans=tr,blocksize=8)
        check_triangle(C,expected,old,uplo,T)
        herm && @test all(iszero(imag(C[i,i])) for i in 1:n)
    end
end

@testset "GEMMT all operations and no-read contracts" begin
    for T in TYPES, n in (0,1,9,17,35), k in (0,13), uplo in (:U,:L), ta in (:N,:T,:C), tb in (:N,:T,:C)
        A=randn(T,ta===:N ? (n,k) : (k,n)); B=randn(T,tb===:N ? (k,n) : (n,k)); C=randn(T,n,n); old=copy(C)
        α=T<:Real ? T(0.7) : T(0.7+0.3im); β=T(-0.2)
        expected=α*op(A,ta)*op(B,tb)+β*old
        @test gemmt!(C,A,B,α,β;uplo,transA=ta,transB=tb,blocksize=8)===C
        check_triangle(C,expected,old,uplo,T)
    end
    for T in TYPES, n in (9,35), uplo in (:U,:L)
        for f in (syrk!,herk!)
            C=WriteOnlyMatrix(fill(T(NaN),n,n)); A=randn(T,n,11)
            f(C,A;uplo,blocksize=8)
            expected=f===herk! ? A*A' : A*transpose(A)
            @test all(isapprox(C.data[i,j],expected[i,j];atol=5tol(T),rtol=5tol(T)) for (i,j) in tri_indices(n,uplo))
            f(C,UnreadableMatrix{T}((n,11)),zero(T),zero(T);uplo)
            @test all(iszero(C.data[i,j]) for (i,j) in tri_indices(n,uplo))
        end
        C=WriteOnlyMatrix(fill(T(NaN),n,n)); A=randn(T,n,11); B=randn(T,11,n)
        gemmt!(C,A,B;uplo,blocksize=8)
        P=A*B
        @test all(isapprox(C.data[i,j],P[i,j];atol=5tol(T),rtol=5tol(T)) for (i,j) in tri_indices(n,uplo))
        gemmt!(C,UnreadableMatrix{T}((n,11)),UnreadableMatrix{T}((11,n)),zero(T),zero(T);uplo)
        @test all(iszero(C.data[i,j]) for (i,j) in tri_indices(n,uplo))
        for f in (trmm!,trsm!)
            C=fill(T(NaN),n,11)
            @test f(C,UnreadableMatrix{T}((n,n)),zero(T);uplo,blocksize=8)==zeros(T,n,11)
        end
    end
end

@testset "BLAS 3 aliasing, views, errors and worker workspaces" begin
    for T in TYPES, uplo in (:U,:L), side in (:L,:R)
        n=35; α=T(0.7); β=T(-0.2)
        A=randn(T,n,n)/n; A[LA.diagind(A)].+=T(2)
        for f in (symm!,hemm!)
            X=copy(A); S=f===hemm! ? LA.Hermitian(A,uplo) : LA.Symmetric(A,uplo)
            expected=α*(side===:L ? S*A : A*S)+β*A
            f(X,X,X,α,β;side,uplo)
            @test X≈expected rtol=5tol(T) atol=5tol(T)
        end
        for f in (trmm!,trsm!), trans in (:N,:T,:C), unit in (false,true)
            X=copy(A)
            tri=uplo===:U ? (unit ? LA.UnitUpperTriangular(A) : LA.UpperTriangular(A)) : (unit ? LA.UnitLowerTriangular(A) : LA.LowerTriangular(A))
            S=op(tri,trans)
            expected=f===trmm! ? α*(side===:L ? S*A : A*S) : (side===:L ? S\(α*A) : (α*A)/S)
            f(X,X,α;side,uplo,trans,unit,blocksize=8)
            @test X≈expected rtol=5tol(T) atol=5tol(T)
        end
        for f in (syrk!,syr2k!,herk!,her2k!,gemmt!)
            X=copy(A); herm=f in (herk!,her2k!); r2=f in (syr2k!,her2k!)
            P=f===gemmt! ? A*A : (herm ? A*A' : A*transpose(A))
            expected=α*(r2 ? 2 : 1)*P+β*(herm ? Matrix(LA.Hermitian(A,uplo)) : A)
            f in (syr2k!,her2k!,gemmt!) ? f(X,X,X,α,β;uplo,blocksize=8) : f(X,X,α,β;uplo,blocksize=8)
            check_triangle(X,expected,A,uplo,T)
        end
        storage=fill(T(99),75,41); C=view(storage,71:-2:3,3:37)
        A=randn(T,n,23); B=randn(T,23,n); old=copy(C)
        config=GemmConfig(T;mc=12,kc=7,nc=16,parallel_threshold=0)
        w=GemmWorkspace(T,8,8,23;config)
        gemmt!(C,A,B,α,β;uplo,blocksize=8,config,workspace=w)
        check_triangle(C,α*A*B+β*old,old,uplo,T)
        @test all(storage[i,j]==99 for j in 1:41 for i in 1:75 if !(i in 71:-2:3 && j in 3:37))
    end
    for f in (trmm!,trsm!)
        A=Matrix{Float64}(LA.I,35,35); A[19,19]=0; B=ones(35,7); old=copy(B)
        if f===trsm!
            @test_throws OxiBLAS.SingularException f(B,A;blocksize=8)
            @test B==old
        end
        @test_throws ArgumentError f(B,A;blocksize=0)
        @test_throws ArgumentError f(B,A;algorithm=:bad)
        A[19,19]=1
        @test_throws ArgumentError f(B,A;workspace=GemmWorkspace(Float64,1,1,1),blocksize=8)
        @test B==old
    end
    @test_throws DimensionMismatch gemmt!(zeros(3,4),ones(3,2),ones(2,4))
    @test_throws DimensionMismatch gemmt!(zeros(3,3),ones(3,2),ones(4,3))
    @test_throws ArgumentError herk!(zeros(ComplexF64,3,3),ones(ComplexF64,3,2),1im)
    @test_throws ArgumentError her2k!(zeros(ComplexF64,3,3),ones(ComplexF64,3,2),ones(ComplexF64,3,2),1,1im)
end

@testset "Available ISA execution and structured parallel updates" begin
    for T in TYPES, kind in (:scalar,:neon,:avx2,:avx512)
        available=try OxiBLAS._kernel_kind(kind); true catch e; e isa ArgumentError ? false : rethrow() end
        available || continue
        info=kernel_info(T;kernel=kind)
        n=2max(info.mr,info.nr)+1
        A=randn(T,n,n); B=randn(T,n,n); C=zeros(T,n,n)
        @test gemm!(C,A,B;kernel=kind,algorithm=:simd,threaded=false)≈A*B rtol=tol(T) atol=n*tol(T)
    end
    for T in TYPES
        n=129; A=randn(T,n,n)/n; A[LA.diagind(A)].+=T(2); B=randn(T,n,97)
        config=GemmConfig(T;mc=32,kc=17,nc=48,parallel_threshold=0)
        w=GemmWorkspace(T,n,97,n;config)
        X=copy(B); Y=copy(B)
        @sync begin
            Threads.@spawn trsm!(X,A;uplo=:U,blocksize=16,config,workspace=w)
            Threads.@spawn trmm!(Y,A;uplo=:U,blocksize=16,config,workspace=w)
        end
        @test LA.UpperTriangular(A)*X≈B rtol=5tol(T) atol=5tol(T)
        @test Y≈LA.UpperTriangular(A)*B rtol=5tol(T) atol=5tol(T)
        C=zeros(T,n,n)
        gemmt!(C,A,A;uplo=:L,blocksize=64,config)
        P=A*A
        @test all(isapprox(C[i,j],P[i,j];rtol=5tol(T),atol=5tol(T)) for (i,j) in tri_indices(n,:L))
    end
end

@testset "Hermitian diagonal and skipped data" begin
    for T in (ComplexF32,ComplexF64), n in (9,35), uplo in (:U,:L), rank2 in (false,true)
        A=randn(T,n,11); B=randn(T,n,11); C=randn(T,n,n)
        for i in 1:n
            C[i,i]=complex(real(C[i,i]),real(T)(NaN))
        end
        old=copy(C); α=rank2 ? T(0.7+0.3im) : T(0.7); β=T(0.2)
        P=rank2 ? α*A*B'+conj(α)*B*A' : α*A*A'
        expected=P+β*Matrix(LA.Hermitian(old,uplo))
        rank2 ? her2k!(C,A,B,α,β;uplo,blocksize=8) : herk!(C,A,α,β;uplo,blocksize=8)
        check_triangle(C,expected,old,uplo,T)
        @test all(iszero(imag(C[i,i])) for i in 1:n)
    end
    for T in TYPES, side in (:L,:R), f in (symm!,hemm!)
        B=UnreadableMatrix{T}((35,35)); A=UnreadableMatrix{T}((35,35)); C=WriteOnlyMatrix(fill(T(NaN),35,35))
        @test f(C,A,B,zero(T),zero(T);side)===C
        @test C.data==zeros(T,35,35)
    end
end

@testset "Triangular reversed views and output adjoints" begin
    for T in TYPES, uplo in (:U,:L), side in (:L,:R), trans in (:N,:T,:C), wrap in (false,true)
        n=35
        abacking=randn(T,75,75)/n
        A=view(abacking,73:-2:5,2:2:70)
        for i in 1:n
            A[i,i]+=T(2)
        end
        S=op(uplo===:U ? LA.UpperTriangular(A) : LA.LowerTriangular(A),trans)
        m,p=side===:L ? (n,17) : (17,n)
        if wrap
            backing=randn(T,p,m); B=adjoint(backing)
        else
            backing=fill(T(99),2m+3,2p+3); B=view(backing,2m:-2:2,3:2:2p+1)
            B.=randn(T,m,p)
        end
        old=copy(B); guard=copy(backing); α=T(.7)
        trmm!(B,A,α;side,uplo,trans,blocksize=8)
        @test B≈α*(side===:L ? S*old : old*S) rtol=5tol(T) atol=5tol(T)
        B.=old
        trsm!(B,A,α;side,uplo,trans,blocksize=8)
        @test (side===:L ? S*B : B*S)≈α*old rtol=5tol(T) atol=5tol(T)
        if !wrap
            @test all(isequal(backing[i,j],guard[i,j]) for j in axes(backing,2) for i in axes(backing,1) if !(i in 2m:-2:2 && j in 3:2:2p+1))
        end
    end
end
