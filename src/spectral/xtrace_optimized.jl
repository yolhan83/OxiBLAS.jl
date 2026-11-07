# SPDX-License-Identifier: Apache-2.0
# Exchangeable XTrace: one range QR, normalized inverse-triangular downdates,
# and cached A*Q. Rank-deficient deletions are resolved only in sketch space.
# The Rust stochastic.rs formula is NOT the published XTrace; see lot document.
function _xtrace_core(A,Omega,Y;normalization=:none,threaded=true,ranktol=64eps(real(eltype(Y)))*max(size(Y)...))
    T=eltype(Y); R=real(T); n,p=size(Y); fullrank=false
    if p<=n
        F=qr_factor(Y;threaded); S=zeros(T,p,p)
        for j in 1:p, i in 1:j
            S[i,j]=F.factors[i,j]
        end
        scale=maximum(abs,S;init=zero(R))
        # F.factors holds the same diagonal; S is rebound below and the
        # generator captured it.
        fullrank=scale>0 && all(abs(F.factors[i,i])>ranktol*scale for i in 1:p)
        if fullrank
            Q=qr_q(F;columns=p); inverse=_identity(T,p); trsm!(inverse,S;uplo=:U)
            fullrank=all(isfinite,inverse) && scale*maximum(abs,inverse)*p<Base.inv(sqrt(eps(R)))
            if fullrank
                S=Matrix(adjoint(inverse))
                for j in 1:p
                    view(S,:,j)./=nrm2(view(S,:,j))
                end
            end
        end
    end
    if !fullrank
        Q=_block_basis(Y,zeros(T,n,0);limit=min(n,p),rtol=ranktol,threaded)
    end
    rank=size(Q,2); AQ=Matrix{T}(undef,n,rank); _block_apply!(AQ,A,Q;threaded)
    W=_mm(adjoint(Q),Omega;threaded); H=_mm(adjoint(Q),AQ;threaded)
    B=fullrank ? nothing : _mm(adjoint(Q),Y;threaded)
    coefficients=zeros(T,rank,p); low=zeros(T,p); dimensions=zeros(Int,p); traceH=sum((H[i,i] for i in 1:rank);init=zero(T))
    hs=zeros(T,rank)   # gemv! overwrites it for every probe
    for j in 1:p
        if fullrank
            s=view(S,:,j); w=view(W,:,j); c=dotc(s,w)
            for i in 1:rank
                coefficients[i,j]=w[i]-s[i]*c
            end
            gemv!(hs,H,s)
            low[j]=traceH-dotc(s,hs); dimensions[j]=n-rank+1
        else
            # This QR is at most rank-by-(p-1), never n-by-(p-1).
            keep=[i for i in 1:p if i!=j]
            C=_block_basis(view(B,:,keep),zeros(T,rank,0);limit=rank,rtol=ranktol,threaded)
            c=zeros(T,size(C,2)); gemv!(c,C,view(W,:,j);trans=:C)
            gemv!(view(coefficients,:,j),C,c)
            HC=_mm(H,C;threaded); low[j]=sum((dotc(view(C,:,i),view(HC,:,i)) for i in axes(C,2));init=zero(T))
            dimensions[j]=n-size(C,2)
        end
    end
    # Batched residuals avoid cancellation in small-matrix expanded formulas,
    # while requiring no additional operator products for leave-one-out probes.
    residual=copy(Omega); images=copy(Y)
    gemm!(residual,Q,coefficients,-one(T),one(T);threaded)
    gemm!(images,AQ,coefficients,-one(T),one(T);threaded)
    samples=zeros(T,p)
    for j in 1:p
        if dimensions[j]==0
            samples[j]=low[j]; continue
        end
        value=dotc(view(residual,:,j),view(images,:,j))
        if normalization===:resphere
            norm=nrm2(view(residual,:,j))
            norm>0 || throw(ConvergenceError(:xtrace_resphering,0,j:j))
            # Divide sequentially to avoid squaring a subnormal/huge norm.
            value=(value/norm)*(R(dimensions[j])/norm)
        end
        samples[j]=low[j]+value
    end
    (;samples,rank,additional_matvecs=rank,path=fullrank ? :inverse_triangular : :rank_deficient_sketch)
end

"""
    trace_xtrace(A; config=StochasticConfig(), normalization=:auto, kwargs...)

Optimized exchangeable XTrace, dense or native sparse, real or complex.
`num_probes` retains its existing meaning: p independent input probes, requiring
p+rank(A*Omega) <= 2p operator products. `normalization=:auto` respheres spherical
probes; :resphere also accepts Gaussian probes, and :none matches the explicit
leave-one-out estimator. Rademacher resphering is rejected because it is biased.
The empirical exchangeable error estimate is heuristic; std_error remains NaN.
"""
function trace_xtrace(A::AbstractMatrix{T};config=StochasticConfig(),normalization=:auto,
        threaded=true,ranktol=64eps(real(T))*max(1,size(A,1),config.num_probes),return_samples=false) where {T<:NativeFloat}
    normalization in (:auto,:none,:resphere) || throw(ArgumentError("unknown XTrace normalization"))
    mode=normalization===:auto ? (config.probe_type===:spherical ? :resphere : :none) : normalization
    mode===:resphere && config.probe_type===:rademacher && throw(ArgumentError("resphering requires spherical or Gaussian probes"))
    isfinite(ranktol) && 0<=ranktol<1 || throw(ArgumentError("invalid XTrace rank tolerance"))
    A=_stochastic_setup(A); n=size(A,1); p=config.num_probes; Omega=Matrix{T}(undef,n,p); Y=similar(Omega); state=Ref(config.seed)
    for j in 1:p
        _probe!(view(Omega,:,j),state,config.probe_type)
    end
    _block_apply!(Y,A,Omega;threaded); all(isfinite,Y) || throw(ArgumentError("nonfinite XTrace sketch"))
    result=_xtrace_core(A,Omega,Y;normalization=mode,threaded,ranktol)
    summary=_stochastic_summary(result.samples,p+result.additional_matvecs,config;independent=false)
    R=real(T); empirical=p>1 ? sqrt(sum(abs2,result.samples.-summary.estimate)/(p*(p-1))) : R(Inf)
    merge(summary,(;sketch_rank=result.rank,path=result.path,normalization=mode,empirical_error=empirical,
        empirical_error_kind=:uncertified_exchangeable_heuristic,samples=return_samples ? result.samples : nothing))
end
