# Prepared only: no numerical execution in the supervised implementation phase.
function tensor_plan_work(plan)
    labels=copy.(plan.labels);total=BigInt(0)
    for (i,j,out) in plan.steps
        total+=prod(BigInt(plan.dimensions[l]) for l in union(labels[i],labels[j]))
        deleteat!(labels,j);deleteat!(labels,i);push!(labels,out)
    end
    total
end
@testset "Advanced contraction path search" begin
    for T in TYPES
        # Independent enumeration of the five parenthesizations of a chain.
        a,b,c,d,e=10,100,5,50,2
        A=zeros(T,a,b);B=zeros(T,b,c);C=zeros(T,c,d);D=zeros(T,d,e)
        costs=(a*b*c+a*c*d+a*d*e,a*b*c+c*d*e+a*c*e,
               b*c*d+a*b*d+a*d*e,b*c*d+b*d*e+a*b*e,
               c*d*e+b*c*e+a*b*e)
        greedy=plan_einsum("ab,bc,cd,de->ae",A,B,C,D)
        optimal=plan_einsum("ab,bc,cd,de->ae",A,B,C,D;optimize=:optimal)
        @test tensor_plan_work(optimal)==minimum(costs)
        @test tensor_plan_work(optimal)<=tensor_plan_work(greedy)
        X=randn(T,3,5);Y=randn(T,5,4);Z=randn(T,4,2)
        for optimize in (:beam,:optimal)
            plan=plan_einsum("ij,jk,kl->il",X,Y,Z;optimize,beam_width=8)
            @test einsum(plan,X,Y,Z)≈X*Y*Z rtol=30tol(T) atol=30tol(T)
            @test einsum("ij,jk,kl->il",X,Y,Z;optimize)≈X*Y*Z rtol=30tol(T) atol=30tol(T)
            result=similar(X*Y*Z)
            einsum!(result,"ij,jk,kl->il",X,Y,Z;optimize)
            @test result≈X*Y*Z rtol=30tol(T) atol=30tol(T)
            bounded=plan_einsum("ij,jk,kl->il",X,Y,Z;optimize,memory_limit=6sizeof(T))
            @test bounded.direct && bounded.peak_intermediate_bytes==6sizeof(T)
        end
        @test_throws ArgumentError plan_einsum("ab,bc,cd,de->ae",A,B,C,D;optimize=:optimal,search_limit=1)
        @test_throws ArgumentError plan_einsum("ij,jk->ik",X,Y;optimize=:beam,beam_width=0)
        @test_throws ArgumentError plan_einsum("ij,jk->ik",X,Y;optimize=:beam,search_limit=0)
        # Repeated labels and a hyperedge shared by all three operands.
        X=randn(T,3,3);Y=randn(T,3);Z=randn(T,3)
        expected=sum(X[i,i]*Y[i]*Z[i] for i in 1:3)
        for optimize in (:beam,:optimal)
            @test einsum("ii,i,i->",X,Y,Z;optimize)[]≈expected rtol=20tol(T) atol=20tol(T)
        end
    end
end

@testset "Sliced contractions, aliases and workspace ownership" begin
    for T in TYPES
        A=randn(T,7,9);B=randn(T,9,5);expected=A*B
        for optimize in (:greedy,:left,:beam,:optimal), chunks in ((i=3,), (k=4,), (i=3,j=2,k=4))
            plan=plan_einsum_sliced("ik,kj->ij",A,B;chunks,optimize)
            @test einsum(plan,A,B)≈expected rtol=40tol(T) atol=40tol(T)
            out=fill(T(NaN),7,5);workspace=EinsumWorkspace(T)
            einsum!(out,plan,A,B;workspace)
            @test out≈expected rtol=40tol(T) atol=40tol(T)
            @test !workspace.active && !any(workspace.leased)
            saved=copy(out);einsum!(out,plan,A,B;α=T(2),β=T(-1),workspace)
            @test out≈2expected-saved rtol=60tol(T) atol=60tol(T)
        end
        # Caller-owned full output is larger than the allowed scratch payload.
        plan=plan_einsum_sliced("ik,kj->ij",A,B;chunks=(i=2,j=2,k=3),scratch_limit=8sizeof(T))
        @test plan.peak_intermediate_bytes<=8sizeof(T)
        @test plan.slice_count==4*3*3
        @test einsum(plan,A,B)≈expected rtol=60tol(T) atol=60tol(T)
        @test_throws ArgumentError plan_einsum_sliced("ik,kj->ij",A,B;chunks=(i=2,j=2),scratch_limit=sizeof(T))
        @test_throws ArgumentError plan_einsum_sliced("ik,kj->ij",A,B;chunks=(q=1,))
        @test_throws ArgumentError plan_einsum_sliced("ik,kj->ij",A,B;chunks=(i=0,))
        X=randn(T,4,4);Y=randn(T,4,4);expected=X*Y
        plan=plan_einsum_sliced("ik,kj->ij",X,Y;chunks=(i=2,k=2))
        workspace=EinsumWorkspace(T);einsum!(X,plan,X,Y;workspace)
        @test X≈expected rtol=40tol(T) atol=40tol(T)
        GC.gc();left=similar(X);right=similar(X)
        @sync begin
            Threads.@spawn einsum!(left,plan,X,Y;workspace)
            Threads.@spawn einsum!(right,plan,X,Y;workspace)
        end
        @test left≈right rtol=tol(T) atol=tol(T)
        unread=TensorNoRead{T,2}((4,4));fill!(left,T(NaN))
        einsum!(left,plan,unread,unread;α=zero(T),workspace)
        @test all(iszero,left)
        @test_throws DimensionMismatch einsum(plan,zeros(T,3,4),Y)
        # Repeated labels and scalar result, including empty reductions.
        X=randn(T,7,7)
        plan=plan_einsum_sliced("ii->",X;chunks=(i=3,))
        @test einsum(plan,X)[]≈sum(X[i,i] for i in 1:7) rtol=20tol(T) atol=20tol(T)
        X=zeros(T,3,0);Y=zeros(T,0,2)
        plan=plan_einsum_sliced("ik,kj->ij",X,Y;chunks=(k=1,))
        @test plan.slice_count==0 && all(iszero,einsum(plan,X,Y))
        # Broadcasting retains singleton ellipsis axes; named labels are tiled.
        X=randn(T,1,3,5);Y=randn(T,2,5,4)
        plan=plan_einsum_sliced("...ik,...kj->...ij",X,Y;chunks=(i=2,k=2))
        expected=cat((reshape(X[1,:,:]*Y[b,:,:],1,3,4) for b in 1:2)...;dims=1)
        @test einsum(plan,X,Y)≈expected rtol=40tol(T) atol=40tol(T)
        @test einsum(plan,reverse(X;dims=3),reverse(Y;dims=2))≈expected rtol=40tol(T) atol=40tol(T)
        # Leases are released after a numerical read error, then reusable.
        plan=plan_einsum_sliced("ik,kj->ij",zeros(T,4,4),zeros(T,4,4);chunks=(k=2,))
        @test_throws ErrorException einsum!(left,plan,unread,unread;workspace)
        @test !workspace.active && !any(workspace.leased)
        einsum!(left,plan,ones(T,4,4),ones(T,4,4);workspace)
        @test all(==(T(4)),left)
        # The context's own storage cannot be an operand, even for alpha=0.
        whole=plan_einsum("ik,kj->ij",ones(T,4,4),ones(T,4,4))
        einsum!(left,whole,ones(T,4,4),ones(T,4,4);workspace)
        slot=findfirst(buffer->length(buffer)>=16,workspace.buffers)
        owned=reshape(view(workspace.buffers[slot],1:16),4,4)
        @test_throws ArgumentError einsum!(left,plan,owned,owned;α=zero(T),workspace)
        reversed=view(left,4:-1:1,:)
        einsum!(reversed,plan,ones(T,4,4),ones(T,4,4))
        @test all(==(T(4)),left)
    end
    setprecision(BigFloat,256) do
        delta=ldexp(one(BigFloat),-160);A=fill(one(BigFloat)+delta,2,2);B=ones(BigFloat,2,2)
        plan=plan_einsum_sliced("ik,kj->ij",A,B;chunks=(k=1,))
        result=einsum(plan,A,B)
        @test eltype(result)===BigFloat && result[1]==2+2delta && result[1]!=2
        @test_throws ArgumentError plan_einsum_sliced("ik,kj->ij",A,B;chunks=(k=1,),scratch_limit=10000)
        for T in EXT_TYPES
            R=real(T);bits=precision(R);delta=ldexp(one(R),-max(4,bits÷2))
            A=fill(T(one(R)+delta),2,2);B=ones(T,2,2)
            plan=plan_einsum_sliced("ik,kj->ij",A,B;chunks=(k=1,))
            result=einsum(plan,A,B;workspace=EinsumWorkspace(T))
            @test eltype(result)===T
            check_extended_array(result,extref(A)*extref(B),T;factor=20)
        end
    end
end
