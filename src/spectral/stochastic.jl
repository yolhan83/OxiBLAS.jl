# SPDX-License-Identifier: Apache-2.0
"""Reproducible stochastic estimates with a private random stream per call.
`probe_type` is :rademacher, :gaussian or :spherical (covariance I).
`confidence` is metadata, not a certified confidence interval.
"""
struct StochasticConfig
    num_probes::Int
    seed::UInt64
    probe_type::Symbol
    confidence::Float64
    function StochasticConfig(;num_probes::Integer=30,seed::Integer=42,
                              probe_type::Symbol=:rademacher,confidence=0.95)
        num_probes>0 && seed>=0 || throw(ArgumentError("positive probe count and nonnegative seed required"))
        probe_type in (:rademacher,:gaussian,:spherical) || throw(ArgumentError("unknown probe distribution"))
        isfinite(confidence) && 0<confidence<1 || throw(ArgumentError("confidence must lie in (0,1)"))
        new(Int(num_probes),UInt64(seed),probe_type,Float64(confidence))
    end
end

# SplitMix64: use the high bits for sampling, never the low alternating LCG bit.
@inline function _probe_bits!(state)
    state[]+=UInt64(0x9e3779b97f4a7c15)
    z=state[]; z=(z ⊻ (z>>30))*UInt64(0xbf58476d1ce4e5b9)
    z=(z ⊻ (z>>27))*UInt64(0x94d049bb133111eb)
    z ⊻ (z>>31)
end
@inline _probe_uniform!(state)=(Float64(_probe_bits!(state)>>12)+0.5)*0x1.0p-52
function _probe!(z::AbstractVector{T},state,kind) where {T<:NativeFloat}
    R=real(T)
    for i in eachindex(z)
        if kind===:rademacher
            bits=_probe_bits!(state)
            z[i]=T<:Complex ? (one(T),-one(T),T(im),-T(im))[Int(bits>>62)+1] : (bits>>63==0 ? one(T) : -one(T))
        else
            radius=sqrt(-2log(_probe_uniform!(state))); angle=2pi*_probe_uniform!(state)
            z[i]=T<:Complex ? T(complex(radius*cos(angle),radius*sin(angle))/sqrt(2)) : T(radius*cos(angle))
        end
    end
    if kind===:spherical && !isempty(z)
        scale=sqrt(R(length(z)))/nrm2(z)
        for i in eachindex(z)
            z[i]*=scale
        end
    end
    z
end
_stochastic_mv!(y,A,x)=_native_mv!(y,A,x)
_stochastic_mv!(y,A::NativeSparse,x)=sparse_mul!(y,A,x;threaded=false)
_stochastic_operator(A)=A
_stochastic_operator(A::Union{CSCMatrix,COOMatrix})=CSRMatrix(A)
function _stochastic_setup(A,square=true)
    Base.require_one_based_indexing(A)
    square && size(A,1)!=size(A,2) && throw(DimensionMismatch("square stochastic operator required"))
    _stochastic_operator(A) # canonicalize once, never once per probe
end
function _stochastic_summary(samples,matvecs,config;independent=true)
    T=eltype(samples); R=real(T); mean=zero(T); m2=zero(R)
    for (k,x) in enumerate(samples)
        isfinite(x) || throw(ArgumentError("nonfinite stochastic sample"))
        delta=x-mean; mean+=delta/k; m2+=real(conj(delta)*(x-mean))
    end
    p=length(samples)
    # Correlated leave-one-out samples do not admit this independent-sample SE.
    se=independent ? (p>1 ? sqrt(max(zero(R),m2)/(p*(p-1))) : R(Inf)) : R(NaN)
    (;estimate=mean,std_error=se,n_probes_used=p,matvecs,confidence=config.confidence,
      error_kind=independent ? :sampling_only : :unavailable_correlated_samples)
end

"""Hutchinson trace estimate for real/complex square operators.
Complex probes use four unit phases; std_error measures E|error|², not separate
real/imaginary confidence intervals. A single probe has unknown (Inf) error.
"""
function trace_hutchinson(A::AbstractMatrix{T};config=StochasticConfig()) where {T<:NativeFloat}
    A=_stochastic_setup(A); n=size(A,1); z=zeros(T,n); w=similar(z)
    state=Ref(config.seed); samples=Vector{T}(undef,config.num_probes)
    for j in eachindex(samples)
        _probe!(z,state,config.probe_type); _stochastic_mv!(w,A,z); samples[j]=dotc(z,w)
    end
    _stochastic_summary(samples,length(samples),config)
end

function _project_complement!(z,Q)
    for _ in 1:2, j in axes(Q,2)
        h=dotc(view(Q,:,j),z); axpy!(z,view(Q,:,j),-h)
    end
    z
end
function _basis_trace(A,Q,w)
    value=zero(eltype(Q))
    for j in axes(Q,2)
        _stochastic_mv!(w,A,view(Q,:,j)); value+=dotc(view(Q,:,j),w)
    end
    value
end
"""Hutch++ with independent range sketch and residual probes.
num_probes budgets operator applications (unused rank-deficient sketch columns
can reduce that count). The returned SE is conditional sampling error only.
"""
function trace_hutch_plus_plus(A::AbstractMatrix{T};config=StochasticConfig()) where {T<:NativeFloat}
    A=_stochastic_setup(A); config.num_probes>=3 || throw(ArgumentError("Hutch++ requires at least three probes"))
    n=size(A,1); s=fld(config.num_probes,3); r=config.num_probes-2s
    state=Ref(config.seed); z=zeros(T,n); w=similar(z); Y=Matrix{T}(undef,n,s)
    for j in 1:s
        _probe!(z,state,config.probe_type); _stochastic_mv!(view(Y,:,j),A,z)
    end
    Q=_orthogonal_basis(Y); low=_basis_trace(A,Q,w); samples=Vector{T}(undef,r)
    for j in 1:r
        _probe!(z,state,config.probe_type); _project_complement!(z,Q)
        _stochastic_mv!(w,A,z); samples[j]=dotc(z,w)+low
    end
    _stochastic_summary(samples,s+size(Q,2)+r,config)
end

"""Exchangeable leave-one-out trace control variate (XTrace family).
Each probe is independent of the range basis formed from all other probes.
This implementation rebuilds those bases; it does not implement the optimized
inverse-triangular XTrace formula. Correlated samples: std_error is NaN.
num_probes counts input probes, not the more numerous operator applications.
"""
function trace_xtrace_loo(A::AbstractMatrix{T};config=StochasticConfig()) where {T<:NativeFloat}
    A=_stochastic_setup(A); n=size(A,1); p=config.num_probes; state=Ref(config.seed)
    Z=Matrix{T}(undef,n,p); Y=similar(Z); z=zeros(T,n); w=similar(z)
    for j in 1:p
        _probe!(view(Z,:,j),state,config.probe_type); _stochastic_mv!(view(Y,:,j),A,view(Z,:,j))
    end
    samples=Vector{T}(undef,p); matvecs=p
    for j in 1:p
        keep=[k for k in 1:p if k!=j]; Q=_orthogonal_basis(view(Y,:,keep))
        low=_basis_trace(A,Q,w); copyto!(z,view(Z,:,j)); _project_complement!(z,Q)
        _stochastic_mv!(w,A,z); samples[j]=low+dotc(z,w); matvecs+=size(Q,2)+1
    end
    _stochastic_summary(samples,matvecs,config;independent=false)
end

"""Unbiased diagonal estimate; entrywise standard errors use complex |.|²."""
function stochastic_diagonal(A::AbstractMatrix{T};config=StochasticConfig()) where {T<:NativeFloat}
    A=_stochastic_setup(A); n=size(A,1); R=real(T); z=zeros(T,n); w=similar(z)
    diagonal=zeros(T,n); m2=zeros(R,n); state=Ref(config.seed)
    for k in 1:config.num_probes
        _probe!(z,state,config.probe_type); _stochastic_mv!(w,A,z)
        for i in 1:n
            x=conj(z[i])*w[i]; isfinite(x) || throw(ArgumentError("nonfinite diagonal sample"))
            delta=x-diagonal[i]; diagonal[i]+=delta/k; m2[i]+=real(conj(delta)*(x-diagonal[i]))
        end
    end
    p=config.num_probes; std_error=p>1 ? sqrt.(max.(zero(R),m2)./(p*(p-1))) : fill(R(Inf),n)
    (;diagonal,std_error,n_probes_used=p,matvecs=p,confidence=config.confidence,error_kind=:sampling_only)
end

"""Rectangular Frobenius estimate. squared_estimate is unbiased; its square
root estimate is generally biased. squared_std_error applies to the square.
"""
function stochastic_frobenius(A::AbstractMatrix{T};config=StochasticConfig()) where {T<:NativeFloat}
    A=_stochastic_setup(A,false); R=real(T); z=zeros(T,size(A,2)); w=zeros(T,size(A,1))
    state=Ref(config.seed); samples=Vector{R}(undef,config.num_probes)
    for j in eachindex(samples)
        _probe!(z,state,config.probe_type); _stochastic_mv!(w,A,z); samples[j]=nrm2(w)^2
    end
    result=_stochastic_summary(samples,length(samples),config)
    (;estimate=sqrt(result.estimate),squared_estimate=result.estimate,squared_std_error=result.std_error,
      n_probes_used=result.n_probes_used,matvecs=result.matvecs,confidence=config.confidence,error_kind=:sampling_only)
end

function _stochastic_hermitian(A::NativeSparse)
    D=sparse_add(A,sparse_transpose(A;conjugate=true),one(eltype(A)),-one(eltype(A)))
    all(iszero,D.values)
end
function _stochastic_hermitian(A::AbstractMatrix)
    all(A[i,j]==conj(A[j,i]) for j in axes(A,2) for i in 1:j)
end
function _log_quadrature!(A,z,V,w,d,e)
    T=eltype(z); R=real(T); norm=nrm2(z); fill!(d,zero(R)); fill!(e,zero(R))
    iszero(norm) && return zero(R),0
    for i in eachindex(z)
        V[i,1]=z[i]/norm
    end
    used=0; beta=zero(R)
    for j in axes(V,2)
        _stochastic_mv!(w,A,view(V,:,j)); alpha=real(dotc(view(V,:,j),w)); d[j]=alpha
        axpy!(w,view(V,:,j),-alpha); j>1 && axpy!(w,view(V,:,j-1),-beta)
        for _ in 1:2, k in 1:j
            h=dotc(view(V,:,k),w); axpy!(w,view(V,:,k),-h)
        end
        beta=nrm2(w); used=j
        (beta<=8eps(R)*max(abs(alpha),j>1 ? abs(e[j-1]) : zero(R)) || j==size(V,2)) && break
        e[j]=beta
        for i in eachindex(w)
            V[i,j+1]=w[i]/beta
        end
    end
    E=eigen_tridiagonal(view(d,1:used),view(e,1:max(0,used-1)))
    all(x->isfinite(x) && x>0,E.values) || throw(ArgumentError("log determinant requires positive Lanczos Ritz values"))
    value=sum(abs2(E.vectors[1,k])*log(E.values[k]) for k in 1:used)
    norm^2*value,used
end
"""Stochastic Lanczos quadrature logdet for Hermitian positive-definite A.
Hermitian symmetry is checked exactly by default; check_hermitian=false is a
caller assertion. Positive Ritz values do not certify global definiteness.
std_error excludes Lanczos truncation bias; no quadrature error bound is claimed.
"""
function stochastic_logdet(A::AbstractMatrix{T};config=StochasticConfig(),
                          lanczos_steps::Integer=min(30,size(A,1)),check_hermitian=true) where {T<:NativeFloat}
    A=_stochastic_setup(A); n=size(A,1); 0<=lanczos_steps<=n || throw(ArgumentError("invalid Lanczos length"))
    n>0 && lanczos_steps==0 && throw(ArgumentError("nonempty operator requires Lanczos steps"))
    check_hermitian && !_stochastic_hermitian(A) && throw(ArgumentError("Hermitian operator required"))
    R=real(T); z=zeros(T,n); w=similar(z); V=Matrix{T}(undef,n,lanczos_steps)
    d=zeros(R,lanczos_steps); e=zeros(R,max(0,lanczos_steps-1)); state=Ref(config.seed)
    samples=zeros(R,config.num_probes); matvecs=0
    for j in eachindex(samples)
        n==0 && continue
        _probe!(z,state,config.probe_type); value,used=_log_quadrature!(A,z,V,w,d,e)
        samples[j]=value; matvecs+=used
    end
    merge(_stochastic_summary(samples,matvecs,config),(;lanczos_steps,error_kind=:sampling_excludes_quadrature_bias))
end
