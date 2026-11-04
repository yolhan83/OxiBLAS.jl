# Whole-operation comparisons, single supervised report. No work at include.
function full_arnoldi_xtrace_case(operation,T,n,data;reference=false)
    R=real(T); setup=()->nothing; symbols=String[]; f=nothing; flops=NaN
    note="native block/stochastic algorithm; no equivalent BLAS/LAPACK whole-operation reference"
    if operation in (:arnoldi_restart,:arnoldi_restart_selective,:arnoldi_restart_fullspace)
        # Explicitly gapped nonsymmetric operator, derived deterministically.
        A=copy(data.B)
        for j in 1:n
            A[j,j]+=T(j<=3 ? 110-10j : j/(32n))
        end
        S=CSRMatrix(A); space=operation===:arnoldi_restart_fullspace ? n : min(n,16)
        mode=operation===:arnoldi_restart_selective ? :selective : :full
        keep=min(space-2,max(3,space÷2)); keep=max(0,keep)
        f=reference ? nothing : ()->thick_restart_arnoldi(S;nev=min(3,n),blocksize=min(2,n),subspace=space,keep,maxrestarts=30,reorthogonalization=mode)
        verify=F->length(F.values)==min(3,n) && all(isfinite,F.residuals) &&
            LA.norm(A*F.vectors-F.vectors*LA.Diagonal(F.values))<=5000n*sqrt(eps(R))*max(one(R),LA.norm(A))
        note="true general block Krylov-Schur restart; mode=$(mode); dimension=$(space); retained images cached; actual operator residual verification; no dense eigensolve substituted as equivalent timing"
    else
        A=operation===:xtrace_rank_deficient ? zeros(T,n,n) : copy(data.H)
        operation===:xtrace_rank_deficient && (A[1,1]=T(7))
        p=min(12,max(2,n÷4)); config=StochasticConfig(num_probes=p,seed=824,
            probe_type=operation===:xtrace_resphere ? :spherical : :rademacher)
        S=CSRMatrix(A)
        if !reference
            if operation===:xtrace_loo
                f=()->trace_xtrace_loo(S;config)
            else
                f=()->trace_xtrace(S;config,normalization=operation===:xtrace_resphere ? :resphere : :none)
            end
        end
        expected=LA.tr(A); bound=max(R(.5)*abs(expected),R(10000)*eps(R))
        verify=F->isfinite(F.estimate) && abs(F.estimate-expected)<=bound &&
            (operation===:xtrace_loo || F.matvecs<=2p)
        note="same input probes p=$(p), explicit operator application counts returned; optimized XTrace vs separate LOO case; rank-deficient small-space downdates; correlated SE not certified; no different trace algorithm substituted"
    end
    (;f,setup,verify,symbols,flops,note)
end
