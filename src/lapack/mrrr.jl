# SPDX-License-Identifier: Apache-2.0
# Relatively robust LDL representations, differential qd and twisted vectors,
# adapting upstream evd/mrrr.rs. QR is an explicit safety fallback, never a
# fabricated eigenvector when a representation loses numerical robustness.
struct _RRR{R}
    d::Vector{R}
    l::Vector{R}
    ld::Vector{R}
    lld::Vector{R}
end
@inline _pivot_floor(x,p)=abs(x)<p ? copysign(p,x) : x
function _rrr_factor(d,e,sigma,pivmin)
    n=length(d); R=eltype(d); D=zeros(R,n); L=zeros(R,max(0,n-1)); LD=similar(L); LLD=similar(L)
    D[1]=_pivot_floor(d[1]-sigma,pivmin)
    for i in 1:n-1
        L[i]=e[i]/D[i]; LD[i]=e[i]; LLD[i]=L[i]*LD[i]
        D[i+1]=_pivot_floor((d[i+1]-sigma)-LLD[i],pivmin)
    end
    _RRR(D,L,LD,LLD)
end
function _rrr_child(F,tau,pivmin)
    n=length(F.d); R=eltype(F.d); D=zeros(R,n); L=zeros(R,max(0,n-1)); LD=similar(L); LLD=similar(L); s=-tau
    for i in 1:n-1
        D[i]=_pivot_floor(F.d[i]+s,pivmin); L[i]=F.ld[i]/D[i]; LD[i]=L[i]*D[i]; LLD[i]=L[i]*LD[i]
        s=F.lld[i]*(s/D[i])-tau
    end
    D[n]=_pivot_floor(F.d[n]+s,pivmin); _RRR(D,L,LD,LLD)
end
function _rrr_count(F,mu,pivmin)
    count=0; s=-mu
    for i in 1:length(F.d)-1
        pivot=F.d[i]+s; abs(pivot)<pivmin && (pivot=-pivmin)
        pivot<0 && (count+=1); s=F.lld[i]*(s/pivot)-mu
        isfinite(s) || throw(ConvergenceError(:rrr_negcount,0,i:i))
    end
    pivot=F.d[end]+s; abs(pivot)<pivmin && (pivot=-pivmin)
    count+(pivot<0)
end
function _tridiagonal_bounds(d,e)
    R=eltype(d); lower=R(Inf); upper=-R(Inf)
    for i in eachindex(d)
        radius=(i>1 ? abs(e[i-1]) : zero(R))+(i<length(d) ? abs(e[i]) : zero(R))
        lower=min(lower,d[i]-radius); upper=max(upper,d[i]+radius)
    end
    margin=8eps(R)*max(abs(lower),abs(upper))*max(1,length(d))+floatmin(R)
    lower-margin,upper+margin
end
function _sturm_count(d,e,x,pivmin)
    isempty(d) && return 0
    pivot=d[1]-x; abs(pivot)<pivmin && (pivot=-pivmin); count=Int(pivot<0)
    for i in 2:length(d)
        pivot=(d[i]-x)-(e[i-1]/pivot)*e[i-1]
        abs(pivot)<pivmin && (pivot=-pivmin); pivot<0 && (count+=1)
    end
    count
end
function _bisect_values(d,e;threaded=true,maxiter=_floating_iterations(eltype(d),256))
    R=eltype(d); n=length(d); result=Vector{R}(undef,n); n==0 && return result
    lower,upper=_tridiagonal_bounds(d,e); pivmin=floatmin(R)*max(one(R),maximum(abs2,e;init=zero(R)))
    _foreach_range(n,threaded ? Threads.nthreads() : 1) do range,_
        for k in range
            lo,hi=lower,upper; converged=false
            for _ in 1:maxiter
                middle=lo+(hi-lo)/2
                if middle==lo || middle==hi || hi-lo<=2eps(R)*max(abs(lo),abs(hi),abs(lower),abs(upper),floatmin(R))
                    converged=true; break
                end
                _sturm_count(d,e,middle,pivmin)<k ? (lo=middle) : (hi=middle)
            end
            converged || throw(ConvergenceError(:sturm_bisection,maxiter,k:k))
            result[k]=lo+(hi-lo)/2
        end
    end
    result
end
function _rrr_refine(F,index,mu,error,diameter,pivmin)
    R=eltype(F.d); floor=eps(R)*diameter+floatmin(R); lo=mu-error-floor; hi=mu+error+floor
    for side in 1:2
        step=error+floor
        for _ in 1:80
            done=side==1 ? _rrr_count(F,lo,pivmin)<index : _rrr_count(F,hi,pivmin)>=index
            done && break
            side==1 ? (lo-=step) : (hi+=step); step*=2
        end
    end
    _rrr_count(F,lo,pivmin)<index<=_rrr_count(F,hi,pivmin) || throw(ConvergenceError(:rrr_bracket,80,index:index))
    for _ in 1:_floating_iterations(eltype(F.d),160)
        mid=lo+(hi-lo)/2
        (mid==lo || mid==hi || hi-lo<=2eps(R)*max(abs(mid),floatmin(R))) && break
        _rrr_count(F,mid,pivmin)<index ? (lo=mid) : (hi=mid)
    end
    lo+(hi-lo)/2,max((hi-lo)/2,floatmin(R))
end
function _rrr_twist(F,mu,pivmin)
    R=eltype(F.d); n=length(F.d); s=zeros(R,n); p=zeros(R,n); lp=zeros(R,max(0,n-1)); um=similar(lp)
    s[1]=-mu
    for i in 1:n-1
        dp=_pivot_floor(F.d[i]+s[i],pivmin); lp[i]=F.ld[i]/dp; s[i+1]=F.lld[i]*(s[i]/dp)-mu
    end
    p[n]=F.d[n]-mu
    for i in n-1:-1:1
        dm=_pivot_floor(F.lld[i]+p[i+1],pivmin); tmp=F.d[i]/dm; um[i]=F.l[i]*tmp; p[i]=p[i+1]*tmp-mu
    end
    all(isfinite,s) && all(isfinite,p) || throw(ConvergenceError(:rrr_twist,0,1:n))
    twist=argmin(abs.(s.+p.+mu)); gamma=s[twist]+p[twist]+mu; z=zeros(R,n); z[twist]=one(R)
    for i in twist-1:-1:1
        abs(lp[i])>1 && abs(z[i+1])>floatmax(R)/abs(lp[i]) && (z ./= abs(lp[i]))
        z[i]=-lp[i]*z[i+1]
    end
    for i in twist:n-1
        abs(um[i])>1 && abs(z[i])>floatmax(R)/abs(um[i]) && (z ./= abs(um[i]))
        z[i+1]=-um[i]*z[i]
    end
    norm=nrm2(z); isfinite(norm) && norm>0 || throw(ConvergenceError(:rrr_vector,0,1:n))
    correction=gamma*(z[twist]/norm)^2
    z./norm,correction
end
function _mrrr_block(d,e,maxdepth)
    R=eltype(d); n=length(d)
    n<=1 && return (copy(d),_identity(R,n))
    lower,upper=_tridiagonal_bounds(d,e); diameter=upper-lower; pivmin=diameter*eps(R)^2+floatmin(R)
    values=_bisect_values(d,e;threaded=false); sigma=lower
    representation=_rrr_factor(d,e,sigma,pivmin)
    for attempt in 1:40
        all(>(zero(R)),representation.d) && break
        sigma-=diameter*(one(R)/20)*attempt+pivmin; representation=_rrr_factor(d,e,sigma,pivmin)
    end
    all(x->isfinite(x) && x>0,representation.d) || throw(ConvergenceError(:rrr_base,40,1:n))
    w=values.-sigma; errors=fill(eps(R)*diameter+pivmin,n); vectors=zeros(R,n,n)
    stack=[(representation,sigma,1,n,0)]
    while !isempty(stack)
        F,shift,first,last,depth=pop!(stack)
        for i in first:last
            w[i],errors[i]=_rrr_refine(F,i,w[i],errors[i],diameter,pivmin)
        end
        j=first
        while j<=last
            k=j
            while k<last && w[k+1]-w[k]<(one(R)/1000)*(max(abs(w[k]),abs(w[k+1]))+eps(R)*diameter)
                k+=1
            end
            if j<k
                spread=w[k]-w[j]; offset=spread+max(errors[j],errors[k])+eps(R)*max(abs(w[j]),abs(w[k]))+pivmin
                child=nothing; tau=zero(R)
                if depth<maxdepth && spread>eps(R)*diameter*n
                    for _ in 1:16
                        for candidate in (w[j]-offset,w[k]+offset)
                            trial=_rrr_child(F,candidate,pivmin)
                            if all(isfinite,trial.d) && all(isfinite,trial.l) && minimum(abs,trial.d)>pivmin && maximum(abs,trial.d)<diameter/eps(R)
                                child=trial; tau=candidate; break
                            end
                        end
                        child!==nothing && break
                        offset/=2
                    end
                end
                if child!==nothing
                    w[j:k].-=tau; push!(stack,(child,shift+tau,j,k,depth+1)); j=k+1; continue
                end
                # Coincident-to-roundoff cluster: use an independent native QR
                # block solution rather than inventing dependent twisted vectors.
                fallback=eigen_tridiagonal(d,e)
                vectors[:,j:k]=fallback.vectors[:,j:k]; values[j:k]=fallback.values[j:k]
            else
                mu=w[j]; error=errors[j]; vector,correction=_rrr_twist(F,mu,pivmin)
                for _ in 1:8
                    abs(correction)<=2eps(R)*max(abs(mu),floatmin(R)) && break
                    next=mu+correction
                    abs(next-w[j])<=error+eps(R)*diameter || break
                    mu=next; vector,correction=_rrr_twist(F,mu,pivmin)
                end
                vectors[:,j]=vector; values[j]=shift+mu
            end
            j=k+1
        end
    end
    # Guard the representation path with actual residual and orthogonality
    # checks; fallback is numerical computation, not an assertion of success.
    threshold=64eps(R)*max(one(R),diameter)*n
    for j in 1:n
        residual=zero(R)
        for i in 1:n
            v=(d[i]-values[j])*vectors[i,j]+(i>1 ? e[i-1]*vectors[i-1,j] : zero(R))+(i<n ? e[i]*vectors[i+1,j] : zero(R))
            residual=hypot(residual,v)
        end
        residual<=threshold || throw(ConvergenceError(:mrrr_residual,0,j:j))
        for k in 1:j-1
            abs(dotc(view(vectors,:,j),view(vectors,:,k)))<=64eps(R)*n || throw(ConvergenceError(:mrrr_orthogonality,0,k:j))
        end
    end
    values,vectors
end
"""MRRR eigenpairs with splitting, robust shifted LDL representations,
differential qd and twisted vectors. A failed representation falls back to the
native implicit-QR block solver. Selected index/range output is supported.
"""
function eigen_tridiagonal_mrrr(d::AbstractVector{R},e::AbstractVector{R};vectors=true,indices=nothing,interval=nothing,threaded=true,maxdepth=64) where {R<:AbstractFloat}
    threaded=threaded && generic_threadsafe(R)
    Base.require_one_based_indexing(d,e); n=length(d)
    length(e)==max(0,n-1) && maxdepth>=0 || throw(ArgumentError("invalid MRRR dimensions/options"))
    indices!==nothing && interval!==nothing && throw(ArgumentError("select eigenvalues by indices or interval"))
    _finite_scale(d); _finite_scale(e)
    scale=max(maximum(abs,d;init=zero(R)),maximum(abs,e;init=zero(R))); iszero(scale) && (scale=one(R))
    D=d./scale; E=e./scale; blocks=UnitRange{Int}[]; first=1
    for i in 1:n-1
        if abs(E[i])<=eps(R)*sqrt(abs(D[i]))*sqrt(abs(D[i+1]))
            push!(blocks,first:i); first=i+1
        end
    end
    n>0 && push!(blocks,first:n)
    # raw: filled by the parallel closure; `values` is rebound after it.
    raw=zeros(R,n); Q=vectors ? zeros(R,n,n) : nothing
    _foreach_range(length(blocks),threaded ? Threads.nthreads() : 1) do range,_
        for k in range
            block=blocks[k]; dd=D[block]; ee=E[Base.first(block):Base.last(block)-1]
            if vectors
                result=try
                    _mrrr_block(dd,ee,maxdepth)
                catch error
                    error isa ConvergenceError || rethrow()
                    fallback=eigen_tridiagonal(dd,ee); (fallback.values,fallback.vectors)
                end
                raw[block]=result[1]; Q[block,block]=result[2]
            else
                raw[block]=_bisect_values(dd,ee;threaded=false)
            end
        end
    end
    raw .*= scale; order=sortperm(raw); values=raw[order]
    selected=indices===nothing ? (interval===nothing ? collect(1:n) : findall(x->interval[1]<x<=interval[2],values)) : collect(indices)
    checkbounds(values,selected)
    EigenFactor(values[selected],vectors ? Q[:,order[selected]] : nothing,0)
end
function eigen_hermitian_mrrr(A::AbstractMatrix{T};uplo=:U,vectors=true,kwargs...) where {T<:GenericFloat}
    F=tridiagonal_factor(A;uplo); E=eigen_tridiagonal_mrrr(F.diagonal,F.offdiagonal;vectors,kwargs...)
    !vectors && return EigenFactor{T,real(T)}(E.values,nothing,E.iterations)
    Q=tridiagonal_q(F); V=Matrix{T}(undef,size(A,1),length(E.values)); gemm!(V,Q,T.(E.vectors))
    EigenFactor(E.values,V,E.iterations)
end
