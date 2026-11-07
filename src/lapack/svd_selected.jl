# SPDX-License-Identifier: Apache-2.0
# Selected bidiagonal SVD: upstream svd/selective.rs/truncated.rs strategy,
# with actual selected bisection, compact backtransforms and pivoted inverse
# iteration. See docs/src/dev/LOT_SVD_PARTIELLE.md for deliberate numerical corrections.

_selected_pivmin(::Type{R}) where R=floatmin(R)
_selected_pivmin(::Type{Float32})=nextfloat(0f0)
_selected_pivmin(::Type{Float64})=nextfloat(0.0)
function _selected_sturm(d,e,x,pivmin)
    isempty(d) && return 0
    p=d[1]-x; abs(p)<pivmin && (p=-pivmin); count=Int(p<=0)
    @inbounds for i in 2:length(d)
        z=e[i-1]; term=(z/p)*z
        # Division-first protects a small e^2; multiplication-first protects
        # the intermediate ratio when e/p overflows but e^2/p still fits.
        !isfinite(term) && (term=(z*z)/p)
        p=(d[i]-x)-term
        isnan(p) && throw(ConvergenceError(:selected_sturm,0,i:i))
        abs(p)<pivmin && (p=-pivmin); count+=Int(p<=0)
    end
    count
end

# Zero diagonal dilation is a bipartite path. Every odd component has one
# structural zero vector supported on just V or just U. Pair those components
# to obtain true left/right null vectors, rather than arbitrary basis columns.
function _selected_components(off,n)
    components=UnitRange{Int}[]; start=1
    for j in 1:length(off)
        if iszero(off[j])
            push!(components,start:j); start=j+1
        end
    end
    n>0 && push!(components,start:2n)
    v=[b for b in components if isodd(length(b)) && isodd(first(b))]
    u=[b for b in components if isodd(length(b)) && iseven(first(b))]
    length(v)==length(u) || throw(ArgumentError("invalid bidiagonal path partition"))
    v,u
end
function _selected_zero!(x,off,block)
    fill!(x,zero(eltype(x))); x[first(block)]=one(eltype(x))
    # Maintain logarithmic magnitudes to avoid products of long ratios. The
    # alternating signs are restored explicitly after normalizing the largest.
    R=eltype(x); logs=Vector{R}(undef,cld(length(block),2)); logs[1]=zero(R)
    for (j,i) in enumerate(first(block):2:last(block)-2)
        logs[j+1]=logs[j]+log(abs(off[i]))-log(abs(off[i+1]))
    end
    largest=maximum(logs)
    for (j,i) in enumerate(first(block):2:last(block))
        x[i]=isodd(j) ? exp(logs[j]-largest) : -exp(logs[j]-largest)
    end
    x./=nrm2(x)
    x
end

function _selected_tridiagonal(d,e,representation)
    R=eltype(d); n=length(d)
    if representation===:dilation
        diag=zeros(R,2n); off=zeros(R,max(0,2n-1))
        for i in 1:n
            off[2i-1]=d[i]; i<n && (off[2i]=e[i])
        end
        return diag,off
    end
    diag=zeros(R,n); off=zeros(R,max(0,n-1))
    for i in 1:n
        diag[i]=d[i]*d[i]+(i>1 ? e[i-1]*e[i-1] : zero(R))
        i<n && (off[i]=d[i]*e[i])
    end
    diag,off
end
function _selected_upper(d,e)
    R=eltype(d); bound=zero(R)
    for i in eachindex(d)
        bound=max(bound,abs(d[i])+(i>1 ? abs(e[i-1]) : zero(R))+(i<length(d) ? abs(e[i]) : zero(R)))
    end
    bound+16eps(R)*bound+_selected_pivmin(R)
end
function _selected_root(d,e,index,upper,pivmin,maxiter,rtol,atol)
    R=eltype(d); lo=zero(R); hi=upper; evaluations=0
    for iteration in 1:maxiter
        mid=lo+(hi-lo)/2
        if mid==lo || mid==hi || hi-lo<=max(atol,rtol*max(abs(lo),abs(hi)))
            return lo+(hi-lo)/2,evaluations
        end
        count=_selected_sturm(d,e,mid,pivmin); evaluations+=1
        count<index ? (lo=mid) : (hi=mid)
    end
    throw(ConvergenceError(:selected_bisection,maxiter,index:index))
end

# GTTRF-style neighboring partial pivoting, including the second superdiagonal.
# No dense shifted matrix and no Thomas fallback with arbitrary basis vectors.
function _selected_shift_factor(d,e,lambda,pivmin)
    R=eltype(d); n=length(d); a=copy(e); b=d.-lambda; c=copy(e); c2=zeros(R,max(0,n-2)); swaps=falses(max(0,n-1))
    floor=max(pivmin,eps(R)*abs(lambda))
    for i in 1:n-1
        abs(b[i])<floor && iszero(a[i]) && (b[i]=copysign(floor,b[i]))
        if abs(b[i])>=abs(a[i])
            iszero(b[i]) && (b[i]=-floor)
            factor=a[i]/b[i]; a[i]=factor; b[i+1]-=factor*c[i]
        else
            factor=b[i]/a[i]; b[i]=a[i]; a[i]=factor
            temp=b[i+1]; b[i+1]=c[i]-factor*temp; c[i]=temp; swaps[i]=true
            if i<n-1
                c2[i]=c[i+1]; c[i+1]=-factor*c[i+1]
            end
        end
    end
    for i in 1:n
        abs(b[i])<floor && (b[i]=iszero(b[i]) ? -floor : copysign(floor,b[i]))
    end
    (;a,b,c,c2,swaps)
end
function _selected_shift_solve!(x,F)
    R=eltype(x); n=length(x); limit=sqrt(floatmax(R))/max(1,8n)
    for i in 1:n-1
        if F.swaps[i]
            old=x[i]; x[i]=x[i+1]; x[i+1]=old-F.a[i]*x[i]
        else
            x[i+1]-=F.a[i]*x[i]
        end
    end
    for i in n:-1:1
        i<n && (x[i]-=F.c[i]*x[i+1])
        i<n-1 && (x[i]-=F.c2[i]*x[i+2])
        if abs(x[i])>limit*abs(F.b[i])
            factor=(limit*abs(F.b[i]))/abs(x[i]); x.*=factor
        end
        x[i]/=F.b[i]
    end
    all(isfinite,x) || throw(ConvergenceError(:selected_inverse_overflow,0,1:n))
    x
end
function _selected_reorth!(x,Z,columns)
    for _ in 1:2, j in columns
        s=dotu(view(Z,:,j),x)
        @inbounds @simd for i in eachindex(x)
            x[i]-=s*Z[i,j]
        end
    end
    x
end
function _selected_eigenvectors(d,e,values,maxinverse,rtol,atol)
    R=eltype(d); n=length(d); q=length(values); Z=zeros(R,n,q); residual=zeros(R,n); iterations=0
    pivmin=_selected_pivmin(R); norm=_selected_upper(d,e)
    for col in 1:q
        lambda=values[col]; F=_selected_shift_factor(d,e,lambda,pivmin)
        # The local generator does not touch Julia's global RNG. Independent
        # starts span repeated eigenspaces; reorthogonalization happens inside
        # inverse iteration, not after convergence to the same repeated vector.
        x=vec(_random_projection(R,n,1,UInt64(col)+0x051ec7))
        cluster=[j for j in 1:col-1 if abs(values[j]-lambda)<=sqrt(eps(R))*max(norm,abs(lambda))]
        converged=false
        for step in 1:maxinverse
            _selected_reorth!(x,Z,cluster)
            t=nrm2(x)
            if iszero(t)
                fill!(x,zero(R)); x[mod1(col+step,n)]=one(R)
            else
                x./=t
            end
            _selected_shift_solve!(x,F); _selected_reorth!(x,Z,cluster)
            t=nrm2(x); iszero(t) && continue; x./=t
            for i in 1:n
                residual[i]=(d[i]-lambda)*x[i]+(i>1 ? e[i-1]*x[i-1] : zero(R))+(i<n ? e[i]*x[i+1] : zero(R))
            end
            iterations+=1
            if nrm2(residual)<=max(atol,16rtol*max(norm,abs(lambda)))
                converged=true; break
            end
        end
        converged || throw(ConvergenceError(:selected_inverse,maxinverse,col:col))
        Z[:,col]=x
    end
    Z,iterations
end

# Apply compact Householder products directly to q requested columns, never
# materializing the full U/V properties of BidiagonalFactor.
function _selected_backtransform(F,X,left)
    T=eltype(F.factors); A=F.factors; m,n=size(A); rows=left ? m : n
    q=size(X,2); Y=zeros(T,rows,q); Y[1:size(X,1),:]=X
    scratch=zeros(T,rows); count=left ? n : max(0,n-1)
    for k in count:-1:1
        start=left ? k : k+1; v=view(scratch,1:rows-start+1); v[1]=one(T)
        for i in 2:length(v)
            v[i]=left ? A[start+i-1,k] : A[k,start+i-1]
        end
        tau=left ? F.tauleft[k] : F.tauright[k]
        _reflect_left!(Y,v,tau,start,1:q)
    end
    Y
end

function _selected_empty(::Type{T},m,n,vectors) where T
    R=real(T)
    SVDFactor{T,R}(vectors ? zeros(T,m,0) : nothing,R[],vectors ? zeros(T,0,n) : nothing,0)
end

"""
    svd_selected(A::AbstractMatrix{T}; indices=nothing, interval=nothing, vectors=true, algorithm=:selected, representation=:dilation, full=false, maxiter=..., maxinverse=100, rtol=..., atol=..., threaded=true, diagnostics=nothing) -> SVDFactor

Genuine selected dense SVD, with 1-based descending `indices`, or the
half-open value interval `(low,high]`. Only the requested distinct ranks are
bisected and only their vectors are computed/backtransformed. Index order and
duplicates are preserved. With neither selector, all min(m,n) values are requested.

`representation=:dilation` uses the symmetric bidiagonal linearization, avoiding
B'B's squared conditioning. `:gram` ports Rust's tridiagonal B'B representation
explicitly. No external solver or full-spectrum safety fallback is used.
`diagnostics=Ref{Any}()` receives work counts, without changing SVDFactor.
"""
function svd_selected(A::AbstractMatrix{T};indices=nothing,interval=nothing,
        vectors=true,algorithm=:selected,representation=:dilation,full=false,
        maxiter=_floating_iterations(real(T),2048),maxinverse=100,
        rtol=8eps(real(T)),atol=zero(real(T)),threaded=true,diagnostics=nothing) where {T<:GenericFloat}
    Base.require_one_based_indexing(A); m,n=size(A); r=min(m,n); R=real(T)
    indices!==nothing && interval!==nothing && throw(ArgumentError("select by indices or interval"))
    algorithm in (:selected,:bisection) || throw(ArgumentError("selected SVD requires :selected/:bisection; use svd_bidiagonal for full-spectrum algorithms"))
    representation in (:dilation,:gram) || throw(ArgumentError("unknown selected SVD representation"))
    full && throw(ArgumentError("selected SVD returns requested thin vectors, not full bases"))
    maxiter isa Integer && maxiter>=0 && maxinverse isa Integer && maxinverse>=0 && isfinite(rtol) && 0<rtol<1 && isfinite(atol) && atol>=0 || throw(ArgumentError("invalid selected SVD budget/tolerance"))
    diagnostics===nothing || diagnostics isa Base.RefValue || throw(ArgumentError("diagnostics must be a Ref"))
    requested=indices===nothing ? collect(1:r) : collect(indices)
    if indices!==nothing
        all(j->j isa Integer && !(j isa Bool) && 1<=j<=r,requested) || throw(ArgumentError("singular indices must lie in 1:min(size(A)...)"))
        requested=Int.(requested)
    end
    bounds=interval===nothing ? nothing : _selected_interval(interval,R)
    _finite_scale(A)
    if r==0 || (isempty(requested) && interval===nothing)
        diagnostics===nothing || (diagnostics[]=(;representation,order=0,selected=0,distinct=0,bisection_targets=0,inverse_targets=0,sturm_evaluations=0,iterations=0))
        return _selected_empty(T,m,n,vectors)
    end
    F=bidiagonal_factor(A); d,e,left,right=_real_bidiagonal(F)
    # One assignment: bs is captured by the closures below.
    bmax=max(maximum(d;init=zero(R)),maximum(e;init=zero(R))); bs=iszero(bmax) ? one(R) : bmax
    d./=bs; e./=bs
    _,path=_selected_tridiagonal(d,e,:dilation); nullV,nullU=_selected_components(path,r); nullity=length(nullV); positive=r-nullity
    diag,off=_selected_tridiagonal(d,e,representation); order=length(diag); upper=_selected_upper(diag,off); pivmin=_selected_pivmin(R)
    counts=0
    # Two rank counts locate an interval directly. There is no all-spectrum
    # pass to infer an offset or post-filter a fully computed SVD.
    if bounds!==nothing
        low,high=bounds
        # Returns (rank count, Sturm evaluations); the former closure
        # incremented the captured `counts` (Core.Box).
        count(bound)=begin
            bound<0 && return 0,0
            iszero(bound) && return nullity,0
            scaled=(bound/F.scale)/bs
            scaled>=upper+one(R) && return r,0
            !isfinite(scaled) && return r,0
            x=representation===:gram ? scaled*scaled : scaled
            representation===:gram && scaled>0 && iszero(x) && throw(ConvergenceError(:selected_gram_underflow,0,1:r))
            clamp(_selected_sturm(diag,off,x,pivmin)-(representation===:dilation ? r : 0),nullity,r),1
        end
        chigh,ehigh=count(high); clow,elow=count(low); counts=ehigh+elow
        first=r-chigh+1; last=r-clow; requested=collect(first:last)
    end
    unique_ranks=sort!(unique(requested)); q=length(unique_ranks); values=zeros(R,q); roots=zeros(R,q); evals=zeros(Int,q)
    if q==0
        diagnostics===nothing || (diagnostics[]=(;representation,order,selected=0,distinct=0,bisection_targets=0,inverse_targets=0,sturm_evaluations=counts,iterations=0))
        return _selected_empty(T,m,n,vectors)
    end
    positive_cols=findall(j->unique_ranks[j]<=positive,1:q)
    workers=_operation_workers(length(positive_cols),Float64(order)*length(positive_cols)*64,threaded && generic_threadsafe(T),262144)
    _foreach_range(length(positive_cols),workers) do range,_
        for jj in range
            col=positive_cols[jj]; rank=unique_ranks[col]; index=representation===:dilation ? 2r-rank+1 : r-rank+1
            absolute=(R(atol)/F.scale)/bs
            representation===:gram && (absolute*=absolute)
            # Task-local: `used` is also assigned after this closure, so the
            # tasks shared (and raced on) one boxed variable.
            local root,steps=_selected_root(diag,off,index,upper,pivmin,maxiter,R(rtol),absolute)
            if representation===:gram && (iszero(root) || (root<=64eps(R)*upper && any(z->!iszero(z),off)))
                throw(ConvergenceError(:selected_gram_accuracy,steps,index:index))
            end
            roots[col]=root; values[col]=representation===:dilation ? root : sqrt(max(zero(R),root)); evals[col]=steps
        end
    end
    iterations=sum(evals)
    if vectors
        Z,used=_selected_eigenvectors(diag,off,roots[positive_cols],maxinverse,R(rtol),zero(R)); iterations+=used
        U=zeros(R,r,q); V=zeros(R,r,q)
        for (jj,col) in enumerate(positive_cols)
            if representation===:dilation
                V[:,col]=view(Z,1:2:2r,jj); U[:,col]=view(Z,2:2:2r,jj)
            else
                V[:,col]=view(Z,:,jj)
                for i in 1:r
                    U[i,col]=d[i]*V[i,col]+(i<r ? e[i]*V[i+1,col] : zero(R))
                end
            end
            un=nrm2(view(U,:,col)); vn=nrm2(view(V,:,col))
            min(un,vn)>0 || throw(ConvergenceError(:selected_singular_vectors,0,col:col))
            view(U,:,col)./=un; view(V,:,col)./=vn
        end
        z=zeros(R,2r)
        for col in 1:q
            unique_ranks[col]<=positive && continue
            which=unique_ranks[col]-positive
            _selected_zero!(z,path,nullV[which]); V[:,col]=view(z,1:2:2r)
            _selected_zero!(z,path,nullU[which]); U[:,col]=view(z,2:2:2r)
        end
        _selected_triplet_check(d,e,values,U,V,R(rtol),R(atol)/F.scale/bs)
        UU=T.(U); VV=T.(V)
        for j in 1:q, i in 1:r
            UU[i,j]*=left[i]; VV[i,j]*=right[i]
        end
        outU=_selected_backtransform(F,UU,true); outV=_selected_backtransform(F,VV,false)
    end
    positions=Dict(rank=>j for (j,rank) in enumerate(unique_ranks)); selected=[positions[j] for j in requested]
    sigma=[(values[j]*bs)*F.scale for j in selected]
    all(isfinite,sigma) || throw(OverflowError("selected singular values cannot be represented in the working scalar type"))
    diagnostics===nothing || (diagnostics[]=(;representation,order,selected=length(requested),distinct=q,bisection_targets=length(positive_cols),inverse_targets=vectors ? length(positive_cols) : 0,sturm_evaluations=counts+sum(evals),iterations))
    !vectors && return SVDFactor{T,R}(nothing,sigma,nothing,iterations)
    if F.wide
        SVDFactor(outV[:,selected],sigma,Matrix(adjoint(outU[:,selected])),iterations)
    else
        SVDFactor(outU[:,selected],sigma,Matrix(adjoint(outV[:,selected])),iterations)
    end
end

function _selected_interval(interval,::Type{R}) where R
    length(interval)==2 || throw(ArgumentError("interval must contain two real bounds"))
    low,high=interval
    low isa Real && high isa Real && !isnan(low) && !isnan(high) && low<=high || throw(ArgumentError("invalid singular-value interval"))
    R(low),R(high)
end
function _selected_triplet_check(d,e,s,U,V,rtol,atol)
    R=eltype(d); n=length(d); a=zeros(R,n); b=similar(a)
    tolerance=max(atol,128rtol*sqrt(R(max(1,n)))*max(maximum(d;init=zero(R)),maximum(e;init=zero(R))))
    for j in eachindex(s)
        for i in 1:n
            a[i]=d[i]*V[i,j]+(i<n ? e[i]*V[i+1,j] : zero(R))-s[j]*U[i,j]
            b[i]=d[i]*U[i,j]+(i>1 ? e[i-1]*U[i-1,j] : zero(R))-s[j]*V[i,j]
        end
        max(nrm2(a),nrm2(b))<=tolerance || throw(ConvergenceError(:selected_svd_residual,0,j:j))
    end
end
