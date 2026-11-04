@testset "Generic tensor contractions and decompositions" begin
    setprecision(BigFloat,512) do
        for T in EXT_TYPES
            A=T.([1.0 .25 2;.125 3 .5]); B=T.([2.0 .125;.25 1;3 .5])
            expected=extref(A)*extref(B); workspace=EinsumWorkspace(T)
            plan=plan_einsum("ik,kj->ij",A,B)
            @test plan.memory_model===(isbitstype(T) ? :inline : :reference_slots)
            check_extended_array(einsum(plan,A,B;workspace),expected,T;factor=100)
            out=fill(T(NaN),2,2);einsum!(out,plan,A,B;workspace)
            check_extended_array(out,expected,T;factor=100)
            check_extended_array(contract(A,B,(2,),(1,)),expected,T;factor=100)
            rhs=T.([1.0 .25;.5 2]); expected3=expected*extref(rhs)
            check_extended_array(einsum("ab,bc,cd->ad",A,B,rhs;optimize=:none),expected3,T;factor=200)
            cube=reshape(T.([mod(7i+3,11)/16 for i in 1:12]),2,3,2)
            for make in (() ->hosvd(cube),() ->tensor_train(cube))
                factor=make()
                check_extended_array(reconstruct(factor),extref(cube),T;factor=5000)
            end
            # Exact rank-one initial factors isolate generic CP arithmetic from
            # a convergence claim for arbitrary/nonconvex initializations.
            initial=[reshape(T[1,2],2,1),reshape(T[1,.5,2],3,1),reshape(T[2,1],2,1)]
            rankone=T[initial[1][i]*initial[2][j]*initial[3][k] for i in 1:2,j in 1:3,k in 1:2]
            cp=cp_als(rankone,1;initial,maxiter=3,regularization=zero(real(T)))
            check_extended_array(reconstruct(cp),extref(rankone),T;factor=5000)
            acube=repeat(reshape(A,2,3,1),1,1,2);bcube=repeat(reshape(B,3,2,1),1,1,2)
            result=batched_matmul(acube,bcube)
            check_extended_array(result[:,:,1],expected,T;factor=200)
            if !isbitstype(T)
                @test_throws ArgumentError plan_einsum("ik,kj->ij",A,B;memory_limit=100000)
            end
            saved=copy(A);einsum!(out,plan,A,B;workspace);@test A==saved
            empty!(workspace);@test isempty(workspace.buffers)
        end
        A=reshape(BigFloat[one(BigFloat)+ldexp(one(BigFloat),-200)],1,1)
        result=einsum("ik,kj->ij",A,ones(BigFloat,1,1))
        @test result[1]==A[1] && result[1]!=one(BigFloat)
        @test precision(result[1])==precision(BigFloat)
        for algorithm in (:jacobi,:parallel_jacobi)
            A=ones(BigFloat,4,2); F=svd_factor(A;algorithm)
            check_extended_array(extref(F.U)*LA.Diagonal(extref(F.S))*extref(F.Vt),A,BigFloat;factor=1000)
        end
    end
end
