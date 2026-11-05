# SPDX-License-Identifier: Apache-2.0
_native_mm!(Y,A::AbstractMatrix,X)=gemm!(Y,A,X)
_native_mm!(Y,A::NativeSparse,X)=sparse_mul!(Y,A,X)
function _spectral_mv!(y,A,x)
    if eltype(A)===eltype(x)
        return _native_mv!(y,A,x)
    elseif eltype(A)<:RealFloat && eltype(x)<:Complex
        xr=real.(x); xi=imag.(x); yr=similar(xr,size(A,1)); yi=similar(yr)
        _native_mv!(yr,A,xr); _native_mv!(yi,A,xi)
        for i in eachindex(y)
            y[i]=complex(yr[i],yi[i])
        end
        return y
    end
    throw(ArgumentError("incompatible spectral operator scalar types"))
end
function _ritz_order(values,which)
    which===:largest_magnitude && return sortperm(abs.(values);rev=true)
    which===:smallest_magnitude && return sortperm(abs.(values))
    which===:largest_real && return sortperm(real.(values);rev=true)
    which===:smallest_real && return sortperm(real.(values))
    throw(ArgumentError("unknown Ritz target"))
end
function _arnoldi_expand!(V,H,A,start,stop,work)
    for j in start:stop
        _spectral_mv!(work,A,view(V,:,j))
        for _ in 1:2, k in 1:j
            h=dotc(view(V,:,k),work); H[k,j]+=h; axpy!(work,view(V,:,k),-h)
        end
        beta=nrm2(work); H[j+1,j]=beta
        iszero(beta) && return j
        for i in axes(V,1)
            V[i,j+1]=work[i]/beta
        end
    end
    stop
end
function _arnoldi_ritz(A,V,H,used,nev,which,rtol,work)
    E=eigen_general(view(H,1:used,1:used)); order=_ritz_order(E.values,which)
    selected=order[1:min(nev,used)]; values=E.values[selected]; vectors=Matrix{eltype(V)}(undef,size(V,1),length(selected))
    gemm!(vectors,view(V,:,1:used),E.vectors[:,selected]); residuals=real(eltype(V))[]
    for j in axes(vectors,2)
        norm=nrm2(view(vectors,:,j)); view(vectors,:,j)./=norm
        _spectral_mv!(work,A,view(vectors,:,j)); axpy!(work,view(vectors,:,j),-values[j]); push!(residuals,nrm2(work))
    end
    flags=[residuals[j]<=rtol*max(one(real(eltype(V))),abs(values[j])) for j in eachindex(values)]
    (;values,vectors,residuals,converged=length(values)==nev && all(flags)),E.values,order
end
"""Twice-reorthogonalized Arnoldi Ritz pairs for real/complex operators."""
function arnoldi(A::AbstractMatrix{T};nev=1,maxiter=min(size(A,1),max(20,4nev)),which=:largest_magnitude,
                 start=nothing,rtol=sqrt(eps(real(T)))) where {T<:NativeFloat}
    n=_square(A); 0<nev<=maxiter<=n && rtol>=0 || throw(ArgumentError("invalid Arnoldi options"))
    C=Complex{real(T)}; initial=start===nothing ? vec(_random_projection(C,n,1,0x0b1a5)) : C.(start)
    Base.require_one_based_indexing(initial); length(initial)==n || throw(DimensionMismatch("Arnoldi start dimensions differ"))
    norm=nrm2(initial); norm>0 && isfinite(norm) || throw(ArgumentError("finite nonzero Arnoldi start required"))
    V=zeros(C,n,maxiter+1); H=zeros(C,maxiter+1,maxiter); work=zeros(C,n); V[:,1]=initial./norm
    used=_arnoldi_expand!(V,H,A,1,maxiter,work)
    result,_,_=_arnoldi_ritz(A,V,H,used,nev,which,rtol,work)
    merge(result,(;iterations=used,reason=result.converged ? :converged : :maxiter))
end
"""Implicitly restarted Arnoldi. Unwanted Ritz shifts are chased through the
small Hessenberg matrix; the transformed Arnoldi basis and residual are retained.
No dense expansion of A and no external eigensolver are used.
"""
function iram(A::AbstractMatrix{T};nev=1,subspace=min(size(A,1),max(20,2nev+4)),maxrestarts=30,
              which=:largest_magnitude,start=nothing,rtol=sqrt(eps(real(T)))) where {T<:NativeFloat}
    n=_square(A); 0<nev<=subspace<=n && maxrestarts>=0 && rtol>=0 || throw(ArgumentError("invalid IRAM options"))
    C=Complex{real(T)}; initial=start===nothing ? vec(_random_projection(C,n,1,0x0b1a5)) : C.(start)
    Base.require_one_based_indexing(initial); length(initial)==n || throw(DimensionMismatch("IRAM start dimensions differ"))
    norm=nrm2(initial); norm>0 && isfinite(norm) || throw(ArgumentError("finite nonzero IRAM start required"))
    m=subspace; V=zeros(C,n,m+1); H=zeros(C,m+1,m); work=zeros(C,n); V[:,1]=initial./norm
    used=_arnoldi_expand!(V,H,A,1,m,work); iterations=used
    rotated=Matrix{C}(undef,n,m); oldresidual=zeros(C,n)   # reused by every restart
    for restart in 0:maxrestarts
        result,values,order=_arnoldi_ritz(A,V,H,used,nev,which,rtol,work)
        if result.converged || restart==maxrestarts || used<m || nev==m
            return merge(result,(;iterations,restarts=restart,reason=result.converged ? :converged : used<m ? :invariant_subspace : :maxrestarts))
        end
        keep=min(m-1,max(nev,cld(m,2))); shifts=values[order[keep+1:m]]
        B=H[1:m,1:m]; Q=_identity(C,m); hm=H[m+1,m]
        for i in 1:n
            oldresidual[i]=hm*V[i,m+1]
        end
        for shift in shifts
            for k in 1:m-1
                f=k==1 ? B[1,1]-shift : B[k,k-1]
                g=k==1 ? B[2,1] : B[k+1,k-1]
                c,s,_=rotg(f,g); _unitary2_similarity!(B,Q,k,c,conj(s))
                k>1 && (B[k+1,k-1]=zero(C))
            end
        end
        gemm!(rotated,view(V,:,1:m),Q)
        for i in 1:n
            work[i]=B[keep+1,keep]*rotated[i,keep+1]+Q[m,keep]*oldresidual[i]
        end
        fill!(H,zero(C)); copyto!(view(H,1:keep,1:keep),view(B,1:keep,1:keep)); copyto!(view(V,:,1:keep),view(rotated,:,1:keep))
        for _ in 1:2, j in 1:keep
            h=dotc(view(V,:,j),work); H[j,keep]+=h; axpy!(work,view(V,:,j),-h)
        end
        beta=nrm2(work); H[keep+1,keep]=beta
        if iszero(beta)
            result,_,_=_arnoldi_ritz(A,V,H,keep,nev,which,rtol,work)
            return merge(result,(;iterations,restarts=restart+1,reason=result.converged ? :converged : :invariant_subspace))
        end
        for i in 1:n
            V[i,keep+1]=work[i]/beta
        end
        used=_arnoldi_expand!(V,H,A,keep+1,m,work); iterations+=used-keep
    end
    error("unreachable IRAM state")
end

struct ShiftInvertOperator{T,F,B} <: AbstractMatrix{T}
    factor::F
    B::B
    n::Int
    work::Vector{T}
    guard::ReentrantLock
end
Base.size(A::ShiftInvertOperator)=(A.n,A.n)
Base.getindex(::ShiftInvertOperator,i::Int,j::Int)=throw(ArgumentError("shift-invert is an implicit operator"))
function _native_mv!(y,A::ShiftInvertOperator,x)
    lock(A.guard) do
        A.B===nothing ? copyto!(A.work,x) : _native_mv!(A.work,A.B,x)
        _precondition!(y,A.work,A.factor)
    end
    y
end
function shift_invert(A::NativeSparse;sigma=zero(eltype(A)),B=nothing,kwargs...)
    n=_square(A); T=eltype(A); shift=convert(T,sigma)
    BB=B===nothing ? sparse_diagonal(ones(T,n)) : CSRMatrix(B)
    size(BB)==(n,n) && eltype(BB)===T || throw(DimensionMismatch("shift-invert metric differs"))
    F=sparse_lu(sparse_add(A,BB,one(T),-shift))
    operator=ShiftInvertOperator(F,B===nothing ? nothing : BB,n,zeros(T,n),ReentrantLock())
    # T is inferred from the work vector by the outer constructor.
    result=iram(operator;kwargs...)
    values=[iszero(mu) ? oftype(mu,Inf) : shift+Base.inv(mu) for mu in result.values]
    residuals=real(T)[]; temp=zeros(Complex{real(T)},n); metric=similar(temp)
    for j in axes(result.vectors,2)
        _spectral_mv!(temp,A,view(result.vectors,:,j))
        B===nothing ? copyto!(metric,view(result.vectors,:,j)) : _spectral_mv!(metric,BB,view(result.vectors,:,j))
        if isfinite(values[j])
            axpy!(temp,metric,-values[j]); push!(residuals,nrm2(temp))
        else
            push!(residuals,nrm2(metric))
        end
    end
    tolerance=get(kwargs,:rtol,sqrt(eps(real(T))))
    converged=all(residuals[j]<=tolerance*max(one(real(T)),isfinite(values[j]) ? abs(values[j]) : one(real(T))) for j in eachindex(values))
    merge(result,(;values,residuals,converged,reason=converged ? :converged : result.reason))
end

function _orthogonal_basis(A;rtol=32eps(real(eltype(A))))
    T=eltype(A); Q=Matrix{T}(undef,size(A)); used=0; scratch=Vector{T}(undef,size(A,1))
    for j in axes(A,2)
        copyto!(scratch,view(A,:,j)); original=nrm2(scratch)
        for _ in 1:2, k in 1:used
            h=dotc(view(Q,:,k),scratch); axpy!(scratch,view(Q,:,k),-h)
        end
        norm=nrm2(scratch)
        norm>rtol*original && norm>0 || continue
        used+=1
        for i in eachindex(scratch)
            Q[i,used]=scratch[i]/norm
        end
    end
    Q[:,1:used]
end
"""Locally optimal block preconditioned conjugate-gradient eigenpairs.
Supports a Hermitian positive-definite metric B for A*X=B*X*Diagonal(values).
The Rayleigh–Ritz subspace is span(X, preconditioned residual, old direction).
"""
function lobpcg(A::AbstractMatrix{T};nev=1,B=nothing,maxiter=100,which=:smallest,initial=nothing,
                rtol=sqrt(eps(real(T))),preconditioner=nothing,seed=0x0b1a5) where {T<:NativeFloat}
    n=_square(A); 0<nev<=n && maxiter>=0 && rtol>=0 && which in (:smallest,:largest) || throw(ArgumentError("invalid LOBPCG options"))
    B!==nothing && (size(B)!=(n,n) || eltype(B)!==T) && throw(DimensionMismatch("LOBPCG metric differs"))
    _check_preconditioner(preconditioner,n)
    X=initial===nothing ? _random_projection(T,n,nev,seed) : Matrix{T}(initial)
    size(X)==(n,nev) || throw(DimensionMismatch("LOBPCG initial block differs"))
    X=_orthogonal_basis(X); size(X,2)==nev || throw(ArgumentError("rank-deficient initial block"))
    direction=zeros(T,n,0); values=zeros(real(T),nev); residuals=fill(real(T)(Inf),nev); iterations=0
    # n×nev blocks reused by every iteration: a rotation writes into `spare`,
    # which then swaps roles with its source.
    AX=Matrix{T}(undef,n,nev); BX=similar(AX); spare=similar(AX); residual=similar(AX); W=similar(AX)
    metric=similar(AX); directions=similar(AX)
    for iteration in 0:maxiter
        _native_mm!(AX,A,X); B===nothing ? copyto!(BX,X) : _native_mm!(BX,B,X)
        H=_mm(adjoint(X),AX); G=_mm(adjoint(X),BX)
        E=eigen_generalized(H,G); order=sortperm(E.values;rev=which===:largest); C=E.vectors[:,order]
        values=E.values[order]
        gemm!(spare,X,C); X,spare=spare,X
        gemm!(spare,AX,C); AX,spare=spare,AX
        gemm!(spare,BX,C); BX,spare=spare,BX
        copyto!(residual,AX)
        for j in 1:nev
            axpy!(view(residual,:,j),view(BX,:,j),-values[j]); residuals[j]=nrm2(view(residual,:,j))
        end
        iterations=iteration
        # A loop, not a generator: `values` is reassigned every iteration.
        done=true
        for j in 1:nev
            residuals[j]<=rtol*max(one(real(T)),abs(values[j])) || (done=false; break)
        end
        if done
            return (;values,vectors=X,residuals,converged=true,iterations,reason=:converged)
        end
        iteration==maxiter && break
        for j in 1:nev
            _precondition!(view(W,:,j),view(residual,:,j),preconditioner)
        end
        Q=_orthogonal_basis(hcat(X,W,direction)); AQ=Matrix{T}(undef,n,size(Q,2)); BQ=similar(AQ)
        _native_mm!(AQ,A,Q); B===nothing ? copyto!(BQ,Q) : _native_mm!(BQ,B,Q)
        H=_mm(adjoint(Q),AQ); G=_mm(adjoint(Q),BQ); E=eigen_generalized(H,G)
        order=sortperm(E.values;rev=which===:largest)[1:nev]; next=spare; gemm!(next,Q,E.vectors[:,order])
        B===nothing ? copyto!(metric,next) : _native_mm!(metric,B,next)
        # direction = next - X (Xᴴ metric), in the reused block.
        copyto!(directions,next); gemm!(directions,X,_mm(adjoint(X),metric),-one(T),one(T)); direction=directions
        X,spare=next,X
    end
    (;values,vectors=X,residuals,converged=false,iterations,reason=:maxiter)
end
