# Supervised preparation only. No execution implied.
@testset "Restarted block Arnoldi: general real/complex and cached images" begin
    for T in TYPES, reorthogonalization in (:full,:selective), locking in (false,true)
        R=real(T); n=24; A=zeros(T,n,n)
        for j in 1:n
            A[j,j]=T(j<=3 ? 110-10j : j/32)
            T<:Complex && (A[j,j]+=T(im)/16)
            j<n && (A[j,j+1]=T(1)/64)
        end
        old=copy(A); result=thick_restart_arnoldi(CSRMatrix(A);nev=3,blocksize=2,
            subspace=12,keep=6,maxrestarts=30,rtol=R(200)*eps(R),atol=R(200)*eps(R),reorthogonalization,locking)
        @test A==old
        @test result.converged
        @test spectrum_error(result.values,LA.eigvals(A)[sortperm(abs.(LA.eigvals(A));rev=true)[1:3]])<=100tol(T)
        independent=[LA.norm(A*result.vectors[:,j]-result.values[j]*result.vectors[:,j]) for j in 1:3]
        @test LA.norm(independent-result.residuals)<=5000eps(R)*LA.norm(A)
        F=result.factor
        @test LA.norm(F.images-A*F.basis)<=5000eps(R)*LA.norm(A)
        @test LA.norm(F.basis'*F.basis-Matrix{T}(LA.I,size(F.basis,2),size(F.basis,2)))<=50000eps(R)
        @test LA.norm(F.images-F.basis*F.projected-F.residual_basis*F.residual_coupling)<=50000eps(R)*LA.norm(A)
        @test result.matvecs==sum(h.new_matvecs for h in result.history)
        @test all(h.dimension<=12 for h in result.history)
        @test all(h.new_matvecs<12 for h in result.history[2:end]) # retained images cached
        @test result.soft_locked==(locking ? count(identity,result.converged_pairs) : 0)
    end
    for T in TYPES
        A=randn(T,9,9)
        for input in (A,CSRMatrix(A),CSCMatrix(CSRMatrix(A)),COOMatrix(CSRMatrix(A)))
            F=thick_restart_arnoldi(input;nev=3,subspace=9,blocksize=2)
            @test F.converged
            reference=LA.eigvals(A)
            # At a real conjugate-pair cutoff, either member is a valid choice.
            # Check membership without replacement AND the extremal magnitudes.
            @test spectrum_error(F.values,reference)<=200tol(T)*LA.norm(A)
            @test sort(abs.(F.values);rev=true)≈sort(abs.(reference);rev=true)[1:3] rtol=200tol(T) atol=200tol(T)*LA.norm(A)
        end
        # Real conjugate pairs are retained together even when nev is odd.
        B=zeros(T,12,12); B[1:2,1:2]=T[5 -8;8 5]
        for i in 3:12
            B[i,i]=T(i)/16
        end
        F=thick_restart_arnoldi(B;nev=1,subspace=7,blocksize=2,keep=3,maxrestarts=20)
        @test F.converged
        @test abs(abs(F.values[1])-sqrt(89))<=100tol(T)
        start=ones(T,12,2); start[:,2]=start[:,1]
        G=thick_restart_arnoldi(B;nev=1,subspace=7,blocksize=2,keep=3,initial=start,maxrestarts=0,rtol=0,atol=0)
        @test !G.converged && G.reason in (:maxrestarts,:invariant_subspace)
        @test first(G.block_widths)==1
    end
    @test_throws ArgumentError thick_restart_arnoldi(ones(8,8);nev=2,subspace=4,keep=3,blocksize=2)
    @test_throws ArgumentError thick_restart_arnoldi(ones(8,8);reorthogonalization=:unknown)
    @test_throws ArgumentError thick_restart_arnoldi(ones(8,8);blocksize=0)
end

@testset "XTrace inverse-triangular and rank-deficient paths vs explicit LOO" begin
    for T in TYPES, probe_type in (:rademacher,:gaussian,:spherical), p in (1,4,12)
        n=9; A=randn(T,n,n); old=copy(A); config=StochasticConfig(;num_probes=p,seed=123,probe_type)
        # General nonsymmetric matrices check both conjugations in complex mode.
        fast=trace_xtrace(A;config,normalization=:none,threaded=false,return_samples=true)
        reference=trace_xtrace_loo(A;config)
        @test abs(fast.estimate-reference.estimate)<=50000eps(real(T))*max(1,LA.norm(A))
        @test fast.matvecs==p+fast.sketch_rank && fast.matvecs<=2p
        @test length(fast.samples)==p && isnan(fast.std_error)
        @test fast.error_kind===:unavailable_correlated_samples
        @test trace_xtrace(A;config,normalization=:none,threaded=false).estimate==fast.estimate
        @test A==old
    end
    for T in TYPES, p in (2,5,16), kind in (:zero,:rankone,:full)
        A=zeros(T,12,12)
        if kind===:rankone
            A[1,1]=T(7); A[1,2]=T(.5)
        elseif kind===:full
            for i in 1:12
                A[i,i]=T(i)
            end
        end
        config=StochasticConfig(num_probes=p,seed=41)
        F=trace_xtrace(CSRMatrix(A);config,normalization=:none)
        reference=trace_xtrace_loo(CSRMatrix(A);config)
        @test abs(F.estimate-reference.estimate)<=50000eps(real(T))*max(1,LA.norm(A))
        @test F.matvecs<=2p
        kind!==:full && @test F.path===:rank_deficient_sketch
    end
    for T in TYPES
        I=Matrix{T}(LA.I,20,20); config=StochasticConfig(num_probes=6,seed=74,probe_type=:spherical)
        F=trace_xtrace(I;config,return_samples=true)
        @test F.path===:inverse_triangular && F.normalization===:resphere
        @test maximum(abs.(F.samples.-T(20)))<=50000eps(real(T))
        @test F.matvecs==12
        # Independent explicit spherical-residual oracle on the exact probes.
        A=randn(T,20,20); state=Ref(config.seed); Omega=zeros(T,20,6)
        for j in 1:6
            OxiBLAS._probe!(view(Omega,:,j),state,:spherical)
        end
        samples=T[]
        for j in 1:6
            keep=[i for i in 1:6 if i!=j]
            Q=Matrix(LA.qr(A*Omega[:,keep]).Q)[:,1:5]
            z=Omega[:,j]-Q*(Q'*Omega[:,j]); dim=20-size(Q,2)
            push!(samples,LA.tr(Q'*A*Q)+dim*LA.dot(z,A*z)/sum(abs2,z))
        end
        F=trace_xtrace(A;config,return_samples=true)
        @test LA.norm(F.samples-samples)<=50000eps(real(T))*LA.norm(A)
        for S in (CSCMatrix(CSRMatrix(A)),COOMatrix(CSRMatrix(A)))
            G=trace_xtrace(S;config)
            @test abs(G.estimate-F.estimate)<=50000eps(real(T))*LA.norm(A)
        end
    end
    @test_throws ArgumentError trace_xtrace(ones(5,5);normalization=:resphere)
    @test_throws ArgumentError trace_xtrace(ones(5,5);normalization=:unknown)
    @test_throws ArgumentError trace_xtrace(ones(5,5);ranktol=-1)
    @test_throws DimensionMismatch trace_xtrace(ones(4,5))
end
