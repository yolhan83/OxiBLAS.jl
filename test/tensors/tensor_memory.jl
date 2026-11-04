@testset "Einsum plans with bounded intermediate payload (not RSS)" begin
    for T in TYPES
        A=randn(T,2,5); B=randn(T,5,4); C=randn(T,4,3)
        reference=A*B*C; outputbytes=sizeof(T)*6
        for optimize in (:greedy,:left)
            direct=plan_einsum("ij,jk,kl->il",A,B,C;optimize,memory_limit=outputbytes)
            @test direct.direct && isempty(direct.steps)
            @test direct.peak_intermediate_bytes==outputbytes
            @test einsum(direct,A,B,C)≈reference rtol=20tol(T) atol=20tol(T)
            roomy=plan_einsum("ij,jk,kl->il",A,B,C;optimize,memory_limit=4096)
            @test !roomy.direct && length(roomy.steps)==2
            @test roomy.peak_intermediate_bytes<=4096
            @test einsum(roomy,A,B,C)≈reference rtol=20tol(T) atol=20tol(T)
        end
        @test einsum("ij,jk,kl->il",A,B,C;memory_limit=outputbytes)≈reference rtol=20tol(T) atol=20tol(T)
        result=fill(T(NaN),2,3)
        einsum!(result,"ij,jk,kl->il",A,B,C;memory_limit=outputbytes)
        @test result≈reference rtol=20tol(T) atol=20tol(T)
        @test_throws ArgumentError plan_einsum("ij,jk,kl->il",A,B,C;memory_limit=outputbytes-1)
        # Final scalar needs one scalar of storage, even with no free labels.
        X=randn(T,3,3); Y=randn(T,3,3)
        p=plan_einsum("ij,ji->",X,Y;memory_limit=sizeof(T))
        @test p.direct
        @test einsum(p,X,Y)[]≈sum(X[i,j]*Y[j,i] for i in 1:3,j in 1:3) rtol=10tol(T) atol=10tol(T)
        # Broadcast ellipsis, diagonals, empty reduction, and alias snapshot.
        X=randn(T,2,1,3); Y=randn(T,1,4,3)
        p=plan_einsum("...i,...i->...",X,Y;memory_limit=8sizeof(T))
        expected=[sum(X[i,1,k]*Y[1,j,k] for k in 1:3) for i in 1:2,j in 1:4]
        @test einsum(p,X,Y)≈expected rtol=10tol(T) atol=10tol(T)
        emptyA=zeros(T,3,0); emptyB=zeros(T,0,2)
        @test all(iszero,einsum("ij,jk->ik",emptyA,emptyB;memory_limit=6sizeof(T)))
        @test size(einsum("ij,jk->ik",zeros(T,0,3),zeros(T,3,2);memory_limit=0))==(0,2)
        X=randn(T,3,3); Y=randn(T,3,3); expected=X*Y
        p=plan_einsum("ij,jk->ik",X,Y;memory_limit=9sizeof(T))
        einsum!(X,p,X,Y)
        @test X≈expected rtol=10tol(T) atol=10tol(T)
        saved=copy(X); GC.gc()
        @test einsum(p,X,Y)≈saved*Y rtol=10tol(T) atol=10tol(T)
        r1=similar(X); r2=similar(X)
        @sync begin
            Threads.@spawn einsum!(r1,p,X,Y)
            Threads.@spawn einsum!(r2,p,X,Y)
        end
        @test r1≈r2 rtol=tol(T) atol=tol(T)
        other=T===Float64 ? ComplexF64 : Float64
        @test_throws ArgumentError einsum(p,other.(real.(X)),other.(real.(Y)))
        unread=TensorNoRead{T,2}((3,3)); target=fill(T(NaN),3,3)
        einsum!(target,"ij,jk->ik",unread,unread;memory_limit=9sizeof(T),α=zero(T))
        @test all(iszero,target)
    end
    @test_throws ArgumentError plan_einsum("i->i",zeros(2);memory_limit=-1)
    @test_throws ArgumentError plan_einsum("i->i",zeros(2);memory_limit=1.5)
    @test_throws ArgumentError plan_einsum("i->i",zeros(2);optimize=:invalid)
end

@testset "Reusable einsum intermediates and shared-workspace safety" begin
    for T in TYPES
        A=randn(T,3,5); B=randn(T,5,4); C=randn(T,4,2); D=randn(T,2,3)
        plan=plan_einsum("ij,jk,kl,lm->mi",A,B,C,D)
        expected=permutedims(A*B*C*D)
        workspace=EinsumWorkspace(T); output=similar(expected)
        einsum!(output,plan,A,B,C,D;workspace)
        @test output≈expected rtol=30tol(T) atol=30tol(T)
        @test !isempty(workspace.buffers) && !any(workspace.leased) && !workspace.active
        buffers=copy(workspace.buffers)
        einsum!(output,plan,A,B,C,D;workspace)
        @test length(buffers)==length(workspace.buffers)
        @test all(buffers[i]===workspace.buffers[i] for i in eachindex(buffers))
        # An out-of-place result must survive subsequent reuse and empty!.
        owned=einsum(plan,A,B,C,D;workspace); saved=copy(owned)
        einsum!(output,plan,2A,B,C,D;workspace)
        @test owned==saved
        @test output≈2expected rtol=30tol(T) atol=30tol(T)
        @test all(!Base.mightalias(owned,b) for b in workspace.buffers)
        GC.gc(); einsum!(output,plan,A,B,C,D;workspace)
        @test output≈expected rtol=30tol(T) atol=30tol(T)
        r1=similar(output); r2=similar(output)
        @sync begin
            Threads.@spawn einsum!(r1,plan,A,B,C,D;workspace)
            Threads.@spawn einsum!(r2,plan,2A,B,C,D;workspace)
        end
        @test r1≈expected rtol=30tol(T) atol=30tol(T)
        @test r2≈2expected rtol=30tol(T) atol=30tol(T)
        # Reject direct access to the owned pool as either input or output.
        buffer=workspace.buffers[1]
        identity_plan=plan_einsum("i->i",buffer)
        @test_throws ArgumentError einsum(identity_plan,buffer;workspace)
        @test_throws ArgumentError einsum!(buffer,identity_plan,zeros(T,length(buffer));workspace)
        # A getter failure after leasing scratch must release all leases.
        unread=TensorNoRead{T,2}((3,3)); id=plan_einsum("ij->ji",unread)
        @test_throws ErrorException einsum!(zeros(T,3,3),id,unread;workspace)
        @test !any(workspace.leased) && !workspace.active
        einsum!(output,plan,A,B,C,D;workspace)
        @test output≈expected rtol=30tol(T) atol=30tol(T)
        alias=copy(A); partner=randn(T,5,5); reference=alias*partner
        aliasplan=plan_einsum("ij,jk->ik",alias,partner;memory_limit=length(alias)*sizeof(T))
        einsum!(alias,aliasplan,alias,partner;workspace)
        @test alias≈reference rtol=20tol(T) atol=20tol(T)
        empty!(workspace)
        @test isempty(workspace.buffers) && isempty(workspace.leased)
        @test owned==saved
        other=T===Float64 ? ComplexF64 : Float64
        @test_throws ArgumentError einsum(plan,A,B,C,D;workspace=EinsumWorkspace(other))
    end
end
