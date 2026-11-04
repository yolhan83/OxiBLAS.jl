# Trace task ownership without assuming tasks remain on one physical thread.
struct TileTrace{T} <: AbstractMatrix{T}
    data::Matrix{T}
    tasks::Set{Task}
    guard::ReentrantLock
end
Base.size(C::TileTrace)=size(C.data)
Base.getindex(C::TileTrace,i::Int,j::Int)=C.data[i,j]
function Base.setindex!(C::TileTrace,v,i::Int,j::Int)
    lock(C.guard) do
        push!(C.tasks,current_task())
    end
    C.data[i,j]=v
end

@testset "Native inter-tile Level 3 task ownership" begin
    for T in TYPES
        n,k=65,43; A=randn(T,n,k); B=randn(T,k,n)
        C=TileTrace(zeros(T,n,n),Set{Task}(),ReentrantLock())
        config=GemmConfig(T;mc=16,kc=11,nc=16,parallel_threshold=0)
        w=GemmWorkspace(T,16,16,k;config)
        gemmt!(C,A,B;uplo=:L,blocksize=16,config,workspace=w)
        P=A*B
        @test all(isapprox(C[i,j],P[i,j];rtol=5tol(T),atol=5tol(T)) for j in 1:n for i in j:n)
        @test all(iszero(C[i,j]) for j in 1:n for i in 1:j-1)
        @test length(C.tasks)==min(Threads.nthreads(),length(w.buffers),15)
    end
end

@testset "Parallel triangular tile products, aliases and shared workspace" begin
    for T in TYPES, uplo in (:U,:L), herm in (false,true), rank2 in (false,true), trflag in (:N,:T)
        tr=herm && trflag===:T ? :C : trflag; n,k=65,43
        A=randn(T,tr===:N ? (n,k) : (k,n)); B=randn(T,size(A)); C=randn(T,n,n); old=copy(C)
        α=herm && !rank2 ? T(.7) : T<:Real ? T(.7) : T(.7+.3im); β=T(.2)
        aa,bb=op(A,tr),op(B,tr); rhs(X)=herm ? adjoint(X) : transpose(X)
        expected=(rank2 ? α*aa*rhs(bb)+(herm ? conj(α) : α)*bb*rhs(aa) : α*aa*rhs(aa))+β*(herm ? Matrix(LA.Hermitian(C,uplo)) : C)
        f=herm ? (rank2 ? her2k! : herk!) : (rank2 ? syr2k! : syrk!)
        config=GemmConfig(T;mc=16,kc=11,nc=16,parallel_threshold=0)
        w=GemmWorkspace(T,16,16,k;config)
        rank2 ? f(C,A,B,α,β;uplo,trans=tr,blocksize=16,config,workspace=w) : f(C,A,α,β;uplo,trans=tr,blocksize=16,config,workspace=w)
        check_triangle(C,expected,old,uplo,T)
    end
    for T in TYPES, uplo in (:U,:L), method in (:simd,:blocked,:reference)
        n=65; X=randn(T,n,n); old=copy(X); α=T(.7); β=T(.2)
        config=GemmConfig(T;mc=16,kc=11,nc=16,parallel_threshold=0)
        gemmt!(X,X,X,α,β;uplo,blocksize=16,config,algorithm=method)
        check_triangle(X,α*old*old+β*old,old,uplo,T)
    end
    for T in TYPES
        n,k=129,71; A=randn(T,n,k); B=randn(T,k,n); C=zeros(T,n,n); D=similar(C)
        config=GemmConfig(T;mc=32,kc=17,nc=32,parallel_threshold=0)
        w=GemmWorkspace(T,32,32,k;config)
        @sync begin
            Threads.@spawn gemmt!(C,A,B;blocksize=32,config,workspace=w)
            Threads.@spawn syrk!(D,A;blocksize=32,config,workspace=w)
        end
        P=A*B; Q=A*transpose(A)
        @test all(isapprox(C[i,j],P[i,j];rtol=5tol(T),atol=5tol(T)) for j in 1:n for i in 1:j)
        @test all(isapprox(D[i,j],Q[i,j];rtol=5tol(T),atol=5tol(T)) for j in 1:n for i in 1:j)
        # A one-buffer workspace limits active tile tasks and cannot deadlock.
        w=GemmWorkspace(T,32,32,k;config,workers=1)
        gemmt!(C,A,B;blocksize=32,config,workspace=w)
        @test all(isapprox(C[i,j],P[i,j];rtol=5tol(T),atol=5tol(T)) for j in 1:n for i in 1:j)
    end
end
