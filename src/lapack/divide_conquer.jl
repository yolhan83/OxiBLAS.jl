# SPDX-License-Identifier: Apache-2.0
# Cuppen split followed by a secular rank-one merge. No dense Gram matrix.
function _orthonormalize_columns!(Q)
    T=eltype(Q); scratch=Vector{T}(undef,size(Q,1))
    for j in axes(Q,2)
        for _ in 1:2, k in 1:j-1
            h=dotc(view(Q,:,k),view(Q,:,j)); axpy!(view(Q,:,j),view(Q,:,k),-h)
        end
        s=nrm2(view(Q,:,j))
        if s<=eps(real(T))
            _complete_svd_column!(Q,j,scratch)
        else
            for i in axes(Q,1)
                Q[i,j]=_realdiv(Q[i,j],s)
            end
        end
    end
    Q
end

function _secular_value(d,z,rho,origin,eta)
    R=eltype(d); positive=one(R); negative=zero(R); derivative=zero(R)
    for i in eachindex(d)
        den=(d[i]-origin)-eta
        v=(rho*z[i])*(z[i]/den)
        v>0 ? (positive+=v) : (negative+=v)
        derivative+=(rho*z[i]/den)*(z[i]/den)
    end
    positive+negative,derivative
end
function _secular_root(d,z,rho,k,maxiter)
    R=eltype(d); n=length(d); origin=d[k]
    if k<n
        gap=d[k+1]-d[k]; middle=gap/2
        value,_=_secular_value(d,z,rho,origin,middle)
        if value<0
            origin=d[k+1]; lo=-gap; hi=zero(R)
        else
            lo=zero(R); hi=gap
        end
    else
        lo=zero(R); hi=rho*sum(abs2,z)
        hi>0 || return origin,zero(R)
        # Weyl's upper bound is inclusive; enlarge by one representable step.
        hi=nextfloat(hi)
    end
    eta=lo+(hi-lo)/2
    for _ in 1:maxiter
        value,derivative=_secular_value(d,z,rho,origin,eta)
        value<0 ? (lo=eta) : (hi=eta)
        if nextfloat(lo)>=hi || hi-lo<=8eps(R)*max(abs(eta),floatmin(R))
            return origin,lo+(hi-lo)/2
        end
        candidate=eta-value/derivative
        eta=isfinite(candidate) && lo<candidate<hi ? candidate : lo+(hi-lo)/2
    end
    throw(ConvergenceError(:secular_equation,maxiter,k:k))
end

function _secular_merge(d,z,rho;maxiter=160)
    R=eltype(d); n=length(d); perm=sortperm(d); poles=d[perm]; coupling=z[perm]
    basis=_identity(R,n)[:,perm]
    # Coincident poles admit an exact coupling-combining plane rotation.
    # The same step on adjacent poles within roundoff is numerical deflation.
    scale=maximum(abs,poles;init=zero(R))+rho*sum(abs2,coupling)
    tolerance=8eps(R)*scale
    first=1
    while first<=n
        last=first
        while last<n && abs(poles[last+1]-poles[first])<=tolerance
            last+=1
        end
        for j in first+1:last
            a,b=coupling[first],coupling[j]; h=hypot(a,b)
            iszero(h) && continue
            c=a/h; s=b/h
            for i in 1:n
                x,y=basis[i,first],basis[i,j]
                basis[i,first]=c*x+s*y; basis[i,j]=-s*x+c*y
            end
            coupling[first]=h; coupling[j]=zero(R); poles[j]=poles[first]
        end
        first=last+1
    end
    active=findall(i->rho*abs(coupling[i])>eps(R)*scale,1:n)
    inactive=setdiff(collect(1:n),active)
    values=R[]; X=zeros(R,n,n); column=0
    for i in inactive
        column+=1; push!(values,poles[i]); X[i,column]=one(R)
    end
    dd=poles[active]; zz=coupling[active]
    logs=Vector{R}(undef,length(active)); signs=similar(logs)   # rewritten for every root
    for k in eachindex(active)
        origin,eta=_secular_root(dd,zz,rho,k,maxiter)
        column+=1; push!(values,origin+eta)
        # Log scaling avoids overflow when a root is extremely close to a pole.
        maxlog=-R(Inf)
        exact=0
        for i in eachindex(active)
            den=(dd[i]-origin)-eta
            if iszero(den)
                exact=i; break
            end
            logs[i]=log(abs(zz[i]))-log(abs(den)); signs[i]=sign(zz[i])*sign(den)
            maxlog=max(maxlog,logs[i])
        end
        if exact!=0
            X[active[exact],column]=one(R)
        else
            for i in eachindex(active)
                X[active[i],column]=signs[i]*exp(logs[i]-maxlog)
            end
        end
    end
    _orthonormalize_columns!(X)
    vectors=similar(X); gemm!(vectors,basis,X;threaded=false)
    order=sortperm(values)
    values[order],vectors[:,order]
end

function _cuppen(d,e,leafsize,budget,maxiter)
    n=length(d)
    n<=leafsize && return eigen_tridiagonal(d,e;maxiter=maxiter*max(1,n))
    h=n÷2; rho=abs(e[h]); left=d[1:h]; right=d[h+1:n]
    if iszero(rho)
        # Zero coupling is a genuine independent split, with no secular solve.
        leftresult=_cuppen(left,e[1:h-1],leafsize,1,maxiter)
        rightresult=_cuppen(right,e[h+1:n-1],leafsize,1,maxiter)
        values=vcat(leftresult.values,rightresult.values)
        Q=zeros(eltype(d),n,n); Q[1:h,1:h]=leftresult.vectors; Q[h+1:n,h+1:n]=rightresult.vectors
        order=sortperm(values)
        return EigenFactor(values[order],Q[:,order],leftresult.iterations+rightresult.iterations)
    end
    left[end]-=rho; right[1]-=rho
    if budget>1
        task=Threads.@spawn _cuppen(left,e[1:h-1],leafsize,budget÷2,maxiter)
        rightresult=_cuppen(right,e[h+1:n-1],leafsize,budget-budget÷2,maxiter)
        leftresult=fetch(task)
    else
        leftresult=_cuppen(left,e[1:h-1],leafsize,1,maxiter)
        rightresult=_cuppen(right,e[h+1:n-1],leafsize,1,maxiter)
    end
    poles=vcat(leftresult.values,rightresult.values)
    z=vcat(vec(leftresult.vectors[end,:]),sign(e[h]).*vec(rightresult.vectors[1,:]))
    values,X=_secular_merge(poles,z,rho;maxiter)
    Q=zeros(eltype(d),n,n)
    gemm!(view(Q,1:h,:),leftresult.vectors,view(X,1:h,:);threaded=false)
    gemm!(view(Q,h+1:n,:),rightresult.vectors,view(X,h+1:n,:);threaded=false)
    EigenFactor(values,Q,leftresult.iterations+rightresult.iterations+1)
end

"""
    eigen_tridiagonal_dc(d::AbstractVector{R}, e::AbstractVector{R}; vectors=true, leafsize=16, threaded=true, maxiter=...) -> EigenFactor

Cuppen divide-and-conquer eigenpairs of a real symmetric tridiagonal.
Recursive blocks use independent storage; merges solve the secular rank-one
problem, with deflation of repeated poles. `vectors=false` discards the vectors.
"""
function eigen_tridiagonal_dc(d::AbstractVector{R},e::AbstractVector{R};vectors=true,leafsize=16,threaded=true,maxiter=_floating_iterations(R,160)) where {R<:AbstractFloat}
    threaded=threaded && generic_threadsafe(R)
    Base.require_one_based_indexing(d,e)
    length(e)==max(0,length(d)-1) || throw(DimensionMismatch("invalid tridiagonal offdiagonal length"))
    leafsize>=1 && maxiter>=1 || throw(ArgumentError("invalid divide-and-conquer budget"))
    _finite_scale(d); _finite_scale(e)
    scale=max(maximum(abs,d;init=zero(R)),maximum(abs,e;init=zero(R)))
    iszero(scale) && (scale=one(R))
    E=_cuppen(d./scale,e./scale,leafsize,threaded ? Threads.nthreads() : 1,maxiter)
    EigenFactor(E.values.*scale,vectors ? E.vectors : nothing,E.iterations)
end
function eigen_hermitian_dc(A::AbstractMatrix{T};uplo=:U,vectors=true,kwargs...) where {T<:GenericFloat}
    F=tridiagonal_factor(A;uplo)
    E=eigen_tridiagonal_dc(F.diagonal,F.offdiagonal;vectors,kwargs...)
    if !vectors
        return EigenFactor{T,real(T)}(E.values,nothing,E.iterations)
    end
    Q=tridiagonal_q(F); V=Matrix{T}(undef,size(A)); gemm!(V,Q,T.(E.vectors))
    EigenFactor(E.values,V,E.iterations)
end
