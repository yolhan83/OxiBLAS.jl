# TensorNoRead: test/helpers/probes.jl

@testset "v0.6 Julia-array contractions and plans" begin
    for T in TYPES
        A=randn(T,5,7); B=randn(T,7,3); D=randn(T,3,4)
        P=plan_einsum("ij,jk,kl->il",A,B,D)
        @test einsum(P,A,B,D)≈A*B*D rtol=20tol(T) atol=20tol(T)
        @test einsum("ij,jk->ik",A,B)≈A*B rtol=10tol(T) atol=10tol(T)
        C=fill(T(NaN),5,3); einsum!(C,"ij,jk->ik",A,B)
        @test C≈A*B rtol=10tol(T) atol=10tol(T)
        M=randn(T,5,5)
        @test einsum("ii->",M)[]≈sum(M[i,i] for i in 1:5) rtol=tol(T)
        @test einsum("ij",M)≈M
        @test einsum("ij->ji",M)≈transpose(M)
        @test einsum("ij->i",M)≈vec(sum(M;dims=2)) rtol=tol(T) atol=tol(T)
        X=randn(T,3,4,3)
        @test einsum("iji->j",X)≈[sum(X[i,j,i] for i in 1:3) for j in 1:4] rtol=tol(T)
        X=randn(T,2,3,4); Y=randn(T,4,5,2)
        expected=[sum(X[a,b,k]*Y[k,d,e] for k in 1:4) for a in 1:2,b in 1:3,d in 1:5,e in 1:2]
        @test contract(X,Y,(3,),(1,))≈expected rtol=10tol(T) atol=10tol(T)
        @test einsum("abc,cde->abde",X,Y)≈expected rtol=10tol(T) atol=10tol(T)
        @test contract(X,Y,(3,),(1,);conjugateA=true)≈contract(conj.(X),Y,(3,),(1,)) rtol=tol(T)
        X=randn(T,2,1,3,4); Y=randn(T,1,5,4,6)
        Z=einsum("...ij,...jk->...ik",X,Y)
        @test size(Z)==(2,5,3,6)
        for i in 1:2,j in 1:5
            @test Z[i,j,:,:]≈X[i,1,:,:]*Y[1,j,:,:] rtol=10tol(T) atol=10tol(T)
        end
        AA=randn(T,3,4,2,1); BB=randn(T,4,5,1,3); CC=batched_matmul(AA,BB;batch_parallel_threshold=0)
        @test size(CC)==(3,5,2,3)
        for i in 1:2,j in 1:3
            @test CC[:,:,i,j]≈AA[:,:,i,1]*BB[:,:,1,j] rtol=10tol(T) atol=10tol(T)
        end
        AA=randn(T,3,3); BB=randn(T,3,3); expected=AA*BB
        einsum!(AA,"ij,jk->ik",AA,BB)
        @test AA≈expected rtol=10tol(T) atol=10tol(T)
        X=randn(T,9,4,7); V=view(X,9:-2:1,:,1:2:7); U=randn(T,4,3)
        @test contract(V,U,(2,),(1,))≈einsum("abc,bd->acd",V,U) rtol=10tol(T) atol=10tol(T)
        X=randn(T,3,4,2)
        F=hosvd(X); @test reconstruct(F)≈X rtol=50tol(T) atol=50tol(T)
        F=tensor_train(X); @test reconstruct(F)≈X rtol=50tol(T) atol=50tol(T)
        @test all(size(F.cores[d],3)==size(F.cores[d+1],1) for d in 1:2)
        U=[randn(T,3,1),randn(T,4,1),randn(T,2,1)]
        X=[U[1][i,1]*U[2][j,1]*U[3][k,1] for i in 1:3,j in 1:4,k in 1:2]
        F=cp_als(X,1;initial=U,maxiter=3,rtol=100eps(real(T)))
        @test reconstruct(F)≈X rtol=50tol(T) atol=50tol(T)
        @test F.relative_error<50tol(T)
        # MTTKRP must flatten Cartesian mode indices in column-major order,
        # including multiple components and a noncontiguous four-mode input.
        U4=[randn(T,n,2) for n in (2,3,4,3)]
        X4=[sum(U4[1][i,r]*U4[2][j,r]*U4[3][k,r]*U4[4][l,r] for r in 1:2)
            for i in 1:2,j in 1:3,k in 1:4,l in 1:3]
        view4=PermutedDimsArray(X4,(3,2,1,4)); initial4=U4[[3,2,1,4]]
        saved4=copy.(initial4); original4=copy(view4)
        F4=cp_als(view4,2;initial=initial4,maxiter=3,rtol=100eps(real(T)))
        @test reconstruct(F4)≈view4 rtol=100tol(T) atol=100tol(T)
        @test F4.relative_error<100tol(T)
        @test view4==original4 && initial4==saved4
        # α=0/β=0 does not read either operand or the previous output.
        unread=TensorNoRead{T,2}((3,3)); C=fill(T(NaN),3,3)
        einsum!(C,"ij,jk->ik",unread,unread;α=zero(T))
        @test all(iszero,C)
        fill!(C,T(NaN)); contract!(C,unread,unread,(2,),(1,);α=zero(T))
        @test all(iszero,C)
        @test all(iszero,einsum("ij,jk->ik",zeros(T,3,0),zeros(T,0,4)))
    end
    @test_throws DimensionMismatch plan_einsum("ij,jk->ik",zeros(2,3),zeros(4,5))
    @test_throws ArgumentError plan_einsum("ij->ii",zeros(2,3))
    @test_throws ArgumentError plan_einsum("ij->k",zeros(2,3))
    @test_throws DimensionMismatch plan_einsum("ii->",zeros(2,3))
    @test_throws ArgumentError contract(zeros(2,3),zeros(3,2),(2,2),(1,1))
    P=plan_einsum("ij,jk->ik",zeros(2,3),zeros(3,4))
    @test_throws DimensionMismatch einsum(P,zeros(2,5),zeros(5,4))
    C=zeros(3,3); @test_throws ArgumentError einsum!(view(C,[1,1,3],:),"ij->ij",C)
end
