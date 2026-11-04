# Supervised phase A: these reference cases are written, not executed.
function check_block_projection(F,A,T)
    R=real(T); n=size(A,1); scale=max(LA.norm(A),one(R)); tolerance=300eps(R)*max(1,n)
    @test F.images≈A*F.basis rtol=tolerance atol=tolerance*scale
    @test F.basis'*F.basis≈Matrix{T}(LA.I,size(F.basis,2),size(F.basis,2)) rtol=tolerance atol=tolerance
    @test F.projected≈F.basis'*A*F.basis rtol=tolerance atol=tolerance*scale
    @test F.images≈F.basis*F.projected+F.residual_basis*F.residual_coupling rtol=4tolerance atol=4tolerance*scale
    @test LA.norm(F.basis'*F.residual_basis)<=tolerance
    @test all(length(block)>0 for block in F.blocks)
    @test reduce(vcat,collect.(F.blocks))==collect(1:size(F.basis,2))
    @test F.iterations==length(F.blocks) && F.matvecs==size(F.basis,2)
    for i in eachindex(F.blocks), j in eachindex(F.blocks)
        absent=F.hermitian ? abs(i-j)>1 : i>j+1
        absent && @test LA.norm(F.projected[F.blocks[i],F.blocks[j]])<=tolerance*scale
    end
end

@testset "Block Arnoldi/Lanczos projections and actual Ritz residuals" begin
    for T in TYPES
        R=real(T); n=11; A=randn(T,n,n); H=A+A'; initial=randn(T,n,3); saved=copy(initial)
        for threaded in (false,true), partial in (false,true)
            for (operator,hermitian) in ((A,false),(H,true))
                builder=hermitian ? block_lanczos_factor : block_arnoldi_factor
                F=builder(CSRMatrix(operator);blocksize=3,initial,subspace=partial ? 8 : n,
                          maxblocks=partial ? 2 : n,threaded)
                @test F isa BlockKrylovFactor{T}
                check_block_projection(F,operator,T)
                @test initial==saved
            end
        end
        reference=sort(LA.eigvals(LA.Hermitian(H));rev=true)
        for input in (H,CSRMatrix(H),CSCMatrix(CSRMatrix(H)),COOMatrix(CSRMatrix(H)))
            result=block_lanczos(input;nev=3,which=:largest,blocksize=3,initial,maxblocks=n,subspace=n)
            @test result.values≈reference[1:3] rtol=30tol(T) atol=30tol(T)
            @test result.converged && all(result.converged_pairs)
            @test H*result.vectors≈result.vectors*LA.Diagonal(result.values) rtol=30tol(T) atol=30tol(T)
            independent=[LA.norm(H*result.vectors[:,j]-result.values[j]*result.vectors[:,j]) for j in 1:3]
            @test result.residuals≈independent atol=50eps(R)*max(1,LA.norm(H))
            @test result.vectors'*result.vectors≈Matrix{T}(LA.I,3,3) rtol=30tol(T) atol=30tol(T)
        end
        D=Matrix(LA.Diagonal(T.([-9,-5,-2,1,3,4,7,8,12,15,20])))
        for (which,expected) in ((:smallest,[-9,-5,-2]),(:largest_magnitude,[20,15,12]),
                                 (:smallest_magnitude,[1,-2,3]),(:bothends,[20,-9,15]),(:closest,[4,3,7]))
            result=block_lanczos(CSRMatrix(D);nev=3,which,target=R(4.5),blocksize=3,maxblocks=n,subspace=n)
            @test result.values≈R.(expected) rtol=30tol(T) atol=30tol(T)
        end
        # Non-Hermitian real operators must recover complex conjugate eigenpairs.
        rotation=T[1 -2;2 1]
        result=block_arnoldi(CSRMatrix(rotation);nev=2,blocksize=2,subspace=2)
        @test result.converged
        @test real.(result.values)≈ones(R,2) rtol=20tol(T) atol=20tol(T)
        @test sort(imag.(result.values))≈R[-2,2] rtol=20tol(T) atol=20tol(T)
        @test rotation*result.vectors≈result.vectors*LA.Diagonal(result.values) rtol=20tol(T) atol=20tol(T)
        @test result.verification_matvecs==(T<:Real ? 4 : 2)
        for scale in (R(2)^(-60),R(2)^60)
            scaled=scale.*Matrix(LA.Diagonal(T[1,2,4,8]))
            result=block_lanczos(CSRMatrix(scaled);nev=2,which=:largest,blocksize=2,maxblocks=4,subspace=4)
            @test result.converged
            @test result.values./scale≈R[8,4] rtol=40tol(T) atol=40tol(T)
        end
        # Deflation preserves an invariant subspace without inventing directions.
        diagonal=Matrix(LA.Diagonal(T.(1:n))); exact=zeros(T,n,3)
        exact[1,1]=one(T); exact[2,2]=one(T); exact[:,3]=exact[:,1]
        F=block_lanczos_factor(diagonal;blocksize=3,initial=exact,maxblocks=n,subspace=n)
        @test size(F.basis,2)==2 && length(F.blocks[1])==2
        @test F.reason===:invariant_subspace
        check_block_projection(F,diagonal,T)
        missing=block_lanczos(diagonal;nev=3,blocksize=3,initial=exact,maxblocks=n,subspace=n)
        @test !missing.converged && length(missing.values)==2
        @test missing.reason===:invariant_subspace
        # Repeated eigenvalues need multiple directions, retained as a block.
        repeated=Matrix(LA.Diagonal(T.(vcat(fill(2,4),fill(5,4)))))
        result=block_lanczos(repeated;nev=3,which=:largest,blocksize=3,maxblocks=8,subspace=8)
        @test result.values≈fill(R(5),3) rtol=30tol(T) atol=30tol(T)
        @test result.vectors'*result.vectors≈Matrix{T}(LA.I,3,3) rtol=30tol(T) atol=30tol(T)
        # One block reports approximation status and its genuine residual.
        limited=block_lanczos(CSRMatrix(H);nev=3,blocksize=3,initial,maxblocks=1,subspace=n)
        @test limited.iterations==1 && limited.matvecs==3
        @test limited.reason in (:block_limit,:converged)
        @test limited.residuals≈[LA.norm(H*limited.vectors[:,j]-limited.values[j]*limited.vectors[:,j]) for j in 1:3] atol=100eps(R)*max(1,LA.norm(H))
        for factor in (block_arnoldi_factor,block_lanczos_factor)
            @test_throws ArgumentError factor(zeros(T,0,0))
            @test_throws ArgumentError factor(H;blocksize=0)
            @test_throws ArgumentError factor(H;maxblocks=0)
            @test_throws ArgumentError factor(H;subspace=n+1)
            @test_throws ArgumentError factor(H;deflation_tol=NaN)
            @test_throws ArgumentError factor(H;blocksize=2,initial=zeros(T,n,2))
            @test_throws ArgumentError factor(H;blocksize=2,initial=fill(T(Inf),n,2))
            @test_throws DimensionMismatch factor(H;initial=zeros(T,n,1),blocksize=2)
            @test_throws DimensionMismatch factor(zeros(T,2,3))
        end
        nonhermitian=copy(H); nonhermitian[1,2]+=T(1)
        @test_throws ArgumentError block_lanczos(nonhermitian)
        @test_throws ArgumentError block_lanczos(CSRMatrix(nonhermitian))
        @test_throws ArgumentError block_arnoldi(H;which=:bothends)
        @test_throws ArgumentError block_arnoldi(H;which=:unknown)
        @test_throws ArgumentError block_lanczos(H;nev=0)
        @test_throws ArgumentError block_lanczos(H;rtol=-1)
        # Read-only strided operands, deterministic seeds, independent tasks, GC.
        V=view(H,n:-1:1,n:-1:1); first=block_lanczos(V;nev=2,maxblocks=n,subspace=n)
        GC.gc(); second=block_lanczos(V;nev=2,maxblocks=n,subspace=n)
        @test first.values==second.values
        @test first.vectors≈second.vectors rtol=tol(T) atol=tol(T)
        results=Vector{Any}(undef,2)
        @sync for j in 1:2
            Threads.@spawn results[j]=block_lanczos(V;nev=2,maxblocks=n,subspace=n)
        end
        @test results[1].values==results[2].values
        # Default iteration/storage budget must stay bounded on a larger operator.
        larger=CSRMatrix(Matrix(LA.Diagonal(T.(1:64))))
        bounded=block_lanczos_factor(larger)
        @test bounded.matvecs<=20 && size(bounded.basis,2)<=20
        many=block_lanczos(larger;nev=12)
        @test many.basis_dimension<=24 && length(many.values)==12
    end
end

@testset "Thick restart retains multiple Ritz vectors" begin
    for T in TYPES
        R=real(T); n=20; d=T.(vcat(collect(range(1,5;length=n-2)),30,40)); A=Matrix(LA.Diagonal(d))
        result=thick_restart_lanczos(CSRMatrix(A);nev=2,subspace=5,keep=2,blocksize=1,maxrestarts=50,which=:largest)
        @test result.converged
        @test result.values≈R[40,30] rtol=30tol(T) atol=30tol(T)
        @test result.restarts>=1 && length(result.history)==result.restarts+1
        @test all(h.dimension<=5 for h in result.history)
        @test all(first(h.block_widths)>=2 for h in result.history[2:end])
        @test result.iterations==sum(length(h.block_widths) for h in result.history)
        @test result.matvecs==sum(h.dimension for h in result.history)
        @test all(LA.norm(A*result.vectors[:,j]-result.values[j]*result.vectors[:,j])<=2sqrt(eps(R))*max(one(R),abs(result.values[j])) for j in 1:2)
        no_restart=thick_restart_lanczos(A;nev=2,subspace=5,keep=2,blocksize=1,maxrestarts=0,which=:largest)
        @test no_restart.restarts==0 && length(no_restart.history)==1
        @test no_restart.reason in (:maxrestarts,:converged,:invariant_subspace)
        full=thick_restart_lanczos(A;nev=2,subspace=n,which=:smallest)
        @test full.restarts==0 && full.converged
        @test full.values≈real.(d[1:2]) rtol=30tol(T) atol=30tol(T)
        scalar=thick_restart_lanczos(reshape(T[3],1,1))
        @test scalar.values≈R[3] && scalar.converged
        @test_throws ArgumentError thick_restart_lanczos(A;nev=3,subspace=5,keep=2)
        @test_throws ArgumentError thick_restart_lanczos(A;keep=20)
        @test_throws ArgumentError thick_restart_lanczos(A;maxrestarts=-1)
        @test_throws ArgumentError thick_restart_lanczos(A;target=Inf)
    end
end
