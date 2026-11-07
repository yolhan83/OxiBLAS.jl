# SPDX-License-Identifier: Apache-2.0
# Golub–Kahan bidiagonal QR, adapting upstream svd/qr_based.rs.
include("svd_dense_kernels.jl")
include("svd_block_jacobi.jl")
include("svd_dense_qr.jl")
@inline function _real_givens(a::R,b::R) where {R<:AbstractFloat}
    # Safe range of Reference LAPACK dlartg (3.10+): with sqrt(safmin) < max|.|
    # < sqrt(safmax/2), sqrt(a^2+b^2) neither overflows nor loses the larger
    # square; an underflowing smaller square is below eps relative to it.
    if R<:RealFloat
        big=max(abs(a),abs(b))
        if sqrt(floatmin(R))<big<sqrt(floatmax(R)/2)
            h=sqrt(muladd(a,a,b*b))
            return a/h,b/h,h
        end
    end
    h=hypot(a,b)
    iszero(h) && return (one(R),zero(R),zero(R))
    if h<floatmin(R)/eps(R) || !isfinite(h)
        # A representable subnormal h has poor relative precision. Dividing
        # by it directly destroys c^2+s^2=1 and hence the singular vectors.
        scale=max(abs(a),abs(b));x=a/scale;y=b/scale;unit=hypot(x,y)
        return x/unit,y/unit,unit*scale
    end
    a/h,b/h,h
end
function _svd_plane!(Q,i,j,c,s)
    Q===nothing && return
    if eltype(Q)!==BigFloat
        @inbounds @simd for row in axes(Q,1)
            x,y=Q[row,i],Q[row,j]
            Q[row,i]=c*x+s*y; Q[row,j]=-s*x+c*y
        end
    else
        @inbounds for row in axes(Q,1)
            x,y=Q[row,i],Q[row,j]
            (iszero(x) && iszero(y)) && continue
            Q[row,i]=c*x+s*y; Q[row,j]=-s*x+c*y
        end
    end
end
function _bidiagonal_shift(d,e,lo,hi,up)
    # Scale the 2x2 Gram corner locally; no full Gram matrix is constructed.
    if up
        a=d[lo]; b=e[lo]; c=d[lo+1]; tail=lo+1<hi ? e[lo+1] : zero(a)
        scale=max(abs(a),abs(b),abs(c),abs(tail))
        iszero(scale) && return zero(a)
        aa=(a/scale)^2+(b/scale)^2; bb=(b/scale)*(c/scale); cc=(c/scale)^2+(tail/scale)^2
        delta=(cc-aa)/2; den=delta+copysign(hypot(delta,bb),delta)
        mu=iszero(den) ? aa : aa-bb*(bb/den)
    else
        a=d[hi-1]; b=e[hi-1]; c=d[hi]; tail=hi-1>lo ? e[hi-2] : zero(a)
        scale=max(abs(a),abs(b),abs(c),abs(tail))
        iszero(scale) && return zero(a)
        aa=(a/scale)^2+(tail/scale)^2; bb=(a/scale)*(b/scale); cc=(c/scale)^2+(b/scale)^2
        delta=(aa-cc)/2; den=delta+copysign(hypot(delta,bb),delta)
        mu=iszero(den) ? cc : cc-bb*(bb/den)
    end
    # Return the shift's square root. Forming scale^2 first loses the shift
    # on tiny deflated blocks (or overflows it on large blocks), even when
    # every singular value and the normalized QR start are representable.
    sqrt(max(zero(mu),mu))*scale
end
function _bidiagonal_qr!(d,e,U,V,maxiter,rtol)
    R=eltype(d); n=length(d); iterations=0; scales=nothing
    batch=U!==nothing && R<:RealFloat && n>=16
    cu=Vector{R}(undef,batch ? n : 0);su=similar(cu)
    cv=similar(cu);sv=similar(cu)
    smin=sqrt(floatmin(R))
    hi=n
    @inbounds while true
        while hi>1
            if abs(e[hi-1])<=rtol*(abs(d[hi-1])+abs(d[hi]))
                e[hi-1]=zero(eltype(e)); hi-=1
            else
                break
            end
        end
        hi<=1 && break
        lo=hi-1
        while lo>1
            if abs(e[lo-1])<=rtol*(abs(d[lo-1])+abs(d[lo]))
                e[lo-1]=zero(eltype(e)); break
            end
            lo-=1
        end
        if max(abs(d[lo]),abs(d[hi]))<smin
            blockscale=max(maximum(abs,view(d,lo:hi)),maximum(abs,view(e,lo:hi-1)))
            if blockscale<smin
                # Scale the complete active block, not only its Givens inputs:
                # otherwise rounded subnormal d/e updates can cycle indefinitely.
                # Split blocks inherit a common scale, so no rotations mix units.
                scales===nothing && (scales=ones(R,n))
                for k in lo:hi
                    d[k]/=blockscale;scales[k]*=blockscale
                end
                for k in lo:hi-1
                    e[k]/=blockscale
                end
            end
        end
        iterations>=maxiter && throw(ConvergenceError(:golub_kahan,iterations,lo:hi))
        iterations+=1
        if iszero(d[lo])
            extra=e[lo]; e[lo]=zero(eltype(e))
            for k in lo+1:hi
                c,s,r=_real_givens(d[k],extra); d[k]=r
                _svd_plane!(U,k,lo,c,s)
                if k<hi
                    extra=-s*e[k]; e[k]*=c
                end
            end
            continue
        elseif iszero(d[hi])
            extra=e[hi-1]; e[hi-1]=zero(eltype(e))
            for k in hi-1:-1:lo
                c,s,r=_real_givens(d[k],extra); d[k]=r
                _svd_plane!(V,k,hi,c,s)
                if k>lo
                    extra=-s*e[k-1]; e[k-1]*=c
                end
            end
            continue
        end
        # Chase from the larger endpoint toward the smaller one. Starting at
        # the tiny endpoint with a shift from the large corner can round every
        # sine to zero, leaving the entire active block unchanged indefinitely.
        up=abs(d[lo])<abs(d[hi]); shift=_bidiagonal_shift(d,e,lo,hi,up)
        # A normalized start avoids overflowing/underflowing the two products.
        if up
            scale=max(abs(d[hi]),abs(e[hi-1]),shift)
            f=(d[hi]/scale)^2-(shift/scale)^2; g=(d[hi]/scale)*(e[hi-1]/scale)
            dk1=d[hi]
            for k in hi-1:-1:lo
                c,s,r=_real_givens(f,g)
                k<hi-1 && (e[k+1]=r)
                ek=e[k]; dk=d[k]
                f=muladd(c,dk1,s*ek); ek=muladd(c,ek,-s*dk1); g=s*dk; dk*=c
                if batch;cu[k]=c;su[k]=s;else;_svd_plane!(U,k+1,k,c,s);end
                c,s,r=_real_givens(f,g); d[k+1]=r
                f=muladd(c,ek,s*dk); dk1=muladd(c,dk,-s*ek)
                if k>lo
                    ekm1=e[k-1]; g=s*ekm1; e[k-1]=c*ekm1
                end
                if batch;cv[k]=c;sv[k]=s;else;_svd_plane!(V,k+1,k,c,s);end
            end
            d[lo]=dk1; e[lo]=f
        else
            scale=max(abs(d[lo]),abs(e[lo]),shift)
            f=(d[lo]/scale)^2-(shift/scale)^2; g=(d[lo]/scale)*(e[lo]/scale)
            dk=d[lo]
            for k in lo:hi-1
                c,s,r=_real_givens(f,g)
                k>lo && (e[k-1]=r)
                ek=e[k]; dk1=d[k+1]
                f=muladd(c,dk,s*ek); ek=muladd(c,ek,-s*dk); g=s*dk1; dk1*=c
                if batch;cv[k]=c;sv[k]=s;else;_svd_plane!(V,k,k+1,c,s);end
                c,s,r=_real_givens(f,g); d[k]=r
                f=muladd(c,ek,s*dk1); dk=muladd(c,dk1,-s*ek)
                if k<hi-1
                    ekp1=e[k+1]; g=s*ekp1; e[k+1]=c*ekp1
                end
                if batch;cu[k]=c;su[k]=s;else;_svd_plane!(U,k,k+1,c,s);end
            end
            d[hi]=dk; e[hi-1]=f
        end
        if batch
            _svd_dense_plane_batch!(U,cu,su,lo,hi,up)
            _svd_dense_plane_batch!(V,cv,sv,lo,hi,up)
        end
    end
    scales===nothing || (d .*= scales)
    for j in 1:n
        if d[j]<0
            d[j]=-d[j]
            U!==nothing && (view(U,:,j).*= -one(eltype(U)))
        end
    end
    iterations
end

@inline function _ensure_rot_cap!(cu,su,cv,sv,need::Int)
    if need>length(cu)
        newcap=max(need,cld(5*length(cu),4))
        resize!(cu,newcap); resize!(su,newcap); resize!(cv,newcap); resize!(sv,newcap)
    end
    nothing
end
@inline function _ensure_sw_cap!(sw_off,sw_lo,sw_hi,sw_kind,need::Int)
    if need>length(sw_off)
        newcap=max(need,2*length(sw_off))
        resize!(sw_off,newcap); resize!(sw_lo,newcap); resize!(sw_hi,newcap); resize!(sw_kind,newcap)
    end
    nothing
end
function _bidiagonal_qr_record!(d::Vector{R},e::Vector{R},maxiter,rtol) where {R<:RealFloat}
    n=length(d); iterations=0; scales=nothing
    cap=max(64, n>=256 ? n*n : div(5*n*n,4))
    cu_all=Vector{R}(undef,cap); su_all=Vector{R}(undef,cap)
    cv_all=Vector{R}(undef,cap); sv_all=Vector{R}(undef,cap)
    sw_cap=max(32,div(5*n,2))
    sw_off=Vector{Int}(undef,sw_cap); sw_lo=Vector{Int}(undef,sw_cap)
    sw_hi=Vector{Int}(undef,sw_cap); sw_kind=Vector{Int8}(undef,sw_cap)
    pos=0; nsw=0
    smin=sqrt(floatmin(R))
    hi=n
    @inbounds while true
        while hi>1
            if abs(e[hi-1])<=rtol*(abs(d[hi-1])+abs(d[hi]))
                e[hi-1]=zero(R); hi-=1
            else
                break
            end
        end
        hi<=1 && break
        lo=hi-1
        while lo>1
            if abs(e[lo-1])<=rtol*(abs(d[lo-1])+abs(d[lo]))
                e[lo-1]=zero(R); break
            end
            lo-=1
        end
        if max(abs(d[lo]),abs(d[hi]))<smin
            blockscale=max(maximum(abs,view(d,lo:hi)),maximum(abs,view(e,lo:hi-1)))
            if blockscale<smin
                scales===nothing && (scales=ones(R,n))
                for k in lo:hi
                    d[k]/=blockscale;scales[k]*=blockscale
                end
                for k in lo:hi-1
                    e[k]/=blockscale
                end
            end
        end
        iterations>=maxiter && throw(ConvergenceError(:golub_kahan,iterations,lo:hi))
        iterations+=1
        len=hi-lo
        _ensure_rot_cap!(cu_all,su_all,cv_all,sv_all,pos+len)
        nsw+=1; _ensure_sw_cap!(sw_off,sw_lo,sw_hi,sw_kind,nsw)
        sw_off[nsw]=pos; sw_lo[nsw]=lo; sw_hi[nsw]=hi
        if iszero(d[lo])
            sw_kind[nsw]=Int8(2)
            base=pos-lo; pos+=len
            extra=e[lo]; e[lo]=zero(R)
            for k in lo+1:hi
                c,s,r=_real_givens(d[k],extra); d[k]=r
                cu_all[base+k]=c; su_all[base+k]=s
                if k<hi
                    extra=-s*e[k]; e[k]*=c
                end
            end
            continue
        elseif iszero(d[hi])
            sw_kind[nsw]=Int8(-2)
            base=pos-lo+1; pos+=len
            extra=e[hi-1]; e[hi-1]=zero(R)
            for k in hi-1:-1:lo
                c,s,r=_real_givens(d[k],extra); d[k]=r
                cv_all[base+k]=c; sv_all[base+k]=s
                if k>lo
                    extra=-s*e[k-1]; e[k-1]*=c
                end
            end
            continue
        end
        up=abs(d[lo])<abs(d[hi]); shift=_bidiagonal_shift(d,e,lo,hi,up)
        base=pos-lo+1; pos+=len
        if up
            sw_kind[nsw]=Int8(-1)
            scale=max(abs(d[hi]),abs(e[hi-1]),shift)
            f=(d[hi]/scale)^2-(shift/scale)^2; g=(d[hi]/scale)*(e[hi-1]/scale)
            dk1=d[hi]
            for k in hi-1:-1:lo
                c,s,r=_real_givens(f,g)
                k<hi-1 && (e[k+1]=r)
                ek=e[k]; dk=d[k]
                f=muladd(c,dk1,s*ek); ek=muladd(c,ek,-s*dk1); g=s*dk; dk*=c
                cu_all[base+k]=c; su_all[base+k]=s
                c,s,r=_real_givens(f,g); d[k+1]=r
                f=muladd(c,ek,s*dk); dk1=muladd(c,dk,-s*ek)
                if k>lo
                    ekm1=e[k-1]; g=s*ekm1; e[k-1]=c*ekm1
                end
                cv_all[base+k]=c; sv_all[base+k]=s
            end
            d[lo]=dk1; e[lo]=f
        else
            sw_kind[nsw]=Int8(1)
            scale=max(abs(d[lo]),abs(e[lo]),shift)
            f=(d[lo]/scale)^2-(shift/scale)^2; g=(d[lo]/scale)*(e[lo]/scale)
            dk=d[lo]
            for k in lo:hi-1
                c,s,r=_real_givens(f,g)
                k>lo && (e[k-1]=r)
                ek=e[k]; dk1=d[k+1]
                f=muladd(c,dk,s*ek); ek=muladd(c,ek,-s*dk); g=s*dk1; dk1*=c
                cv_all[base+k]=c; sv_all[base+k]=s
                c,s,r=_real_givens(f,g); d[k]=r
                f=muladd(c,ek,s*dk1); dk=muladd(c,dk1,-s*ek)
                if k<hi-1
                    ekp1=e[k+1]; g=s*ekp1; e[k+1]=c*ekp1
                end
                cu_all[base+k]=c; su_all[base+k]=s
            end
            d[hi]=dk; e[hi-1]=f
        end
    end
    scales===nothing || (d .*= scales)
    sgn=Vector{R}(undef,n)
    @inbounds for j in 1:n
        if d[j]<zero(R)
            d[j]=-d[j]; sgn[j]=-one(R)
        else
            sgn[j]=one(R)
        end
    end
    cu_all,su_all,cv_all,sv_all,sw_off,sw_lo,sw_hi,sw_kind,nsw,sgn,iterations
end

function _bidiagonal_qr_fused!(::Type{T},d::Vector{R},e::Vector{R},
                               p_phase::Vector{T},q_phase::Vector{T},
                               maxiter,rtol;lower::Bool=false,want_vt::Bool=true,threaded::Bool=true) where {T,R<:RealFloat}
    n=length(d)
    cu_all,su_all,cv_all,sv_all,sw_off,sw_lo,sw_hi,sw_kind,nsw,sgn,iterations=_bidiagonal_qr_record!(d,e,maxiter,rtol)
    order=sortperm(d;rev=true); S=d[order]
    U_out=Matrix{T}(undef,n,n)
    V_out=Matrix{T}(undef,n,n)
    L=R===Float32 ? 32 : 16
    valL=R===Float32 ? Val(32) : Val(16)
    ntiles=cld(n,L)
    total_tiles=2*ntiles
    workers=(threaded && n>=128) ? min(Threads.nthreads(),total_tiles) : 1
    _foreach_range(workers,workers) do _,worker
        buf=zeros(R,L,n)
        tbuf=want_vt ? Matrix{T}(undef,32,L) : Matrix{T}(undef,0,0)
        @inbounds for tid in worker:workers:total_tiles
            is_u=tid<=ntiles
            t_idx=is_u ? tid : tid-ntiles
            r_lo=(t_idx-1)*L+1
            r_hi=min(n,r_lo+L-1)
            rem=r_hi-r_lo+1
            cs=is_u ? cu_all : cv_all
            ss=is_u ? su_all : sv_all
            nz_lo,nz_hi=_svd_apply_recorded_tile!(buf,r_lo,r_hi,cs,ss,sw_off,sw_lo,sw_hi,sw_kind,nsw,is_u,valL)
            # Determine whether this tile writes rows of U_out/V_out directly or transposed into Vt_out
            write_rows=want_vt ? (is_u ⊻ lower) : true
            target=want_vt ? (write_rows ? U_out : V_out) : (is_u ? U_out : V_out)
            if write_rows
                for j in 1:n
                    col=order[j]
                    if col<nz_lo || col>nz_hi
                        @simd for i in 1:rem
                            target[r_lo+i-1,j]=zero(T)
                        end
                    elseif is_u
                        sj=sgn[col]
                        @simd for i in 1:rem
                            r=r_lo+i-1
                            target[r,j]=p_phase[r]*T(sj*buf[i,col])
                        end
                    else
                        @simd for i in 1:rem
                            r=r_lo+i-1
                            ph=lower ? conj(q_phase[r]) : q_phase[r]
                            target[r,j]=ph*T(buf[i,col])
                        end
                    end
                end
            else
                for j_base in 1:32:n
                    j_len=min(32,n-j_base+1)
                    for jj in 1:j_len
                        col=order[j_base+jj-1]
                        if col<nz_lo || col>nz_hi
                            @simd for i in 1:rem
                                tbuf[jj,i]=zero(T)
                            end
                        elseif is_u
                            sj=sgn[col]
                            @simd for i in 1:rem
                                r=r_lo+i-1
                                tbuf[jj,i]=p_phase[r]*T(sj*buf[i,col])
                            end
                        else
                            @simd for i in 1:rem
                                r=r_lo+i-1
                                tbuf[jj,i]=conj(q_phase[r])*T(buf[i,col])
                            end
                        end
                    end
                    for i in 1:rem
                        r=r_lo+i-1
                        @simd for jj in 1:j_len
                            target[j_base+jj-1,r]=tbuf[jj,i]
                        end
                    end
                end
            end
            for j in nz_lo:nz_hi
                @simd for i in 1:L
                    buf[i,j]=zero(R)
                end
            end
        end
    end
    S,U_out,V_out,iterations
end

function _real_bidiagonal(F::BidiagonalFactor{T}) where T
    A=F.factors; n=size(A,2); R=real(T)
    d=zeros(R,n); e=zeros(R,max(0,n-1)); left=ones(T,n); right=ones(T,n)
    # B = Diagonal(left) * B_real * Diagonal(right)'.
    for j in 1:n
        d[j],left[j]=_svd_dense_phase_product(A[j,j],right[j])
        if j<n
            e[j],right[j+1]=_svd_dense_phase_product(conj(A[j,j+1]),left[j])
        end
    end
    d,e,left,right
end

function _bidiagonal_dc(d,e;leafsize=16,threaded=true,maxiter=160)
    R=eltype(d); n=length(d)
    # Symmetric dilation [0 B'; B 0], interleaved into a tridiagonal chain.
    # Its eigenvalues are +/-sigma. This avoids squaring the condition number.
    off=zeros(R,max(0,2n-1))
    for i in 1:n
        off[2i-1]=d[i]
        i<n && (off[2i]=e[i])
    end
    E=eigen_tridiagonal_dc(zeros(R,2n),off;leafsize,threaded,maxiter)
    V=E.vectors[1:2:2n,n+1:2n]; _orthonormalize_columns!(V)
    U=zeros(R,n,n); sigma=zeros(R,n); scratch=zeros(R,n)
    for j in 1:n
        for i in 1:n
            U[i,j]=d[i]*V[i,j]+(i<n ? e[i]*V[i+1,j] : zero(R))
        end
        sigma[j]=nrm2(view(U,:,j))
    end
    order=sortperm(sigma;rev=true); sigma=sigma[order]; U=U[:,order]; V=V[:,order]
    for j in 1:n
        if iszero(sigma[j])
            _complete_svd_column!(U,j,scratch)
        else
            for i in 1:n
                U[i,j]/=sigma[j]
            end
        end
    end
    U,sigma,V,E.iterations
end
function _full_svd(F::SVDFactor{T},m,n) where T
    F.U===nothing && return F
    q=length(F.S); U=zeros(T,m,m); V=zeros(T,n,n)
    U[:,1:q]=F.U; V[:,1:q]=adjoint(F.Vt)
    scratch=zeros(T,max(m,n))
    for j in q+1:m
        _complete_svd_column!(U,j,view(scratch,1:m))
    end
    for j in q+1:n
        _complete_svd_column!(V,j,view(scratch,1:n))
    end
    SVDFactor(U,F.S,Matrix(adjoint(V)),F.sweeps)
end

"""SVD through two-sided Householder reduction and bidiagonal Golub–Kahan QR.
`algorithm=:divide_conquer` instead solves the symmetric bidiagonal dilation by
Cuppen recursion and secular merges; no A'A is formed. Real/complex, tall/wide,
values-only and full null-space options are supported.
"""
function svd_bidiagonal(A::AbstractMatrix{T};algorithm=:golub_kahan,vectors=true,full=false,
                        maxiter=_floating_iterations(real(T),160)*max(1,min(size(A)...))^2,rtol=8eps(real(T)),leafsize=16,threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    Base.require_one_based_indexing(A); m,n=size(A); R=real(T)
    algorithm in (:golub_kahan,:divide_conquer) || throw(ArgumentError("unknown bidiagonal SVD algorithm"))
    maxiter>=0 && 0<rtol<1 || throw(ArgumentError("invalid bidiagonal SVD budget/tolerance"))
    if m<n
        F=svd_bidiagonal(adjoint(A);algorithm,vectors,full,maxiter,rtol,leafsize,threaded)
        return SVDFactor{T,R}(F.Vt===nothing ? nothing : Matrix(adjoint(F.Vt)),F.S,F.U===nothing ? nothing : Matrix(adjoint(F.U)),F.sweeps)
    end
    if T<:NativeFloat && algorithm===:golub_kahan && n>=32 && m>=2n
        Q,small,scale=_svd_dense_tall_qr(A,false,threaded)
        if _svd_dense_qr_eligible(Q)
            G=svd_bidiagonal(small;algorithm,vectors,full=false,maxiter,rtol,leafsize,threaded)
            return _svd_dense_qr_result(Q,G,scale,m,n,full,threaded)
        end
    end
    dense_native=T<:NativeFloat && algorithm===:golub_kahan
    F=dense_native ? _svd_dense_bidiagonal_factor(A;threaded) : bidiagonal_factor(A)
    d,e,left,right=_real_bidiagonal(F)
    if algorithm===:divide_conquer
        U,sigma,V,iterations=_bidiagonal_dc(d,e;leafsize,threaded,maxiter=max(1,maxiter))
        sigma .*= F.scale
        !vectors && return SVDFactor{T,R}(nothing,sigma,nothing,iterations)
        UU=T.(U); VV=T.(V)
        for j in 1:n, i in 1:n
            UU[i,j]*=left[i]; VV[i,j]*=right[i]
        end
    elseif !vectors
        iterations=_bidiagonal_qr!(d,e,nothing,nothing,maxiter,rtol)
        sigma=sort!(d;rev=true); sigma .*= F.scale
        return SVDFactor{T,R}(nothing,sigma,nothing,iterations)
    elseif R<:RealFloat && n>=128
        sigma,UU,VV,iterations=_bidiagonal_qr_fused!(T,d,e,left,right,maxiter,rtol;lower=false,want_vt=false,threaded)
        sigma .*= F.scale
    else
        U=_identity(R,n); V=_identity(R,n)
        iterations=_bidiagonal_qr!(d,e,U,V,maxiter,rtol)
        order=sortperm(d;rev=true); sigma=d[order]; sigma .*= F.scale
        UU=Matrix{T}(undef,n,n); VV=Matrix{T}(undef,n,n)
        @inbounds for j in 1:n
            col=order[j]
            for i in 1:n
                UU[i,j]=left[i]*T(U[i,col])
                VV[i,j]=right[i]*T(V[i,col])
            end
        end
    end
    if dense_native
        resultU=_svd_dense_backtransform(F,UU,true;threaded)
        resultV=_svd_dense_backtransform(F,VV,false;threaded)
    else
        resultU=Matrix{T}(undef,m,n); resultV=Matrix{T}(undef,n,n)
        gemm!(resultU,view(F.U,:,1:n),UU;threaded); gemm!(resultV,F.V,VV;threaded)
    end
    result=SVDFactor(resultU,sigma,Matrix(adjoint(resultV)),iterations)
    full ? _full_svd(result,m,n) : result
end
svd_divide_conquer(A;kwargs...)=svd_bidiagonal(A;algorithm=:divide_conquer,kwargs...)

"""Leading-rank SVD. The default uses genuine selected bisection/inverse
iteration. Explicit :golub_kahan/:divide_conquer retain the full-spectrum paths.
"""
function svd_truncated(A::AbstractMatrix{T},rank::Integer;algorithm=:selected,kwargs...) where {T<:GenericFloat}
    0<=rank<=min(size(A)...) || throw(ArgumentError("invalid truncated SVD rank"))
    if algorithm in (:selected,:bisection)
        return svd_selected(A;indices=1:Int(rank),algorithm,kwargs...)
    end
    F=svd_bidiagonal(A;algorithm,kwargs...)
    SVDFactor(F.U===nothing ? nothing : F.U[:,1:rank],F.S[1:rank],F.Vt===nothing ? nothing : F.Vt[1:rank,:],F.sweeps)
end

# Deterministic local generator: no Random dependency or global RNG side effects.
function _random_projection(::Type{T},m,n,seed) where T
    # The generator state is threaded through a pure step (the former closure
    # mutated a captured state: a Core.Box per draw). Same sequence.
    R=real(T); state=UInt64(seed); Q=Matrix{T}(undef,m,n)
    step(s)=s*UInt64(6364136223846793005)+UInt64(1442695040888963407)
    draw(s)=R(s>>11)*R(0x1.0p-53)-R(0.5)
    for i in eachindex(Q)
        state=step(state); x=draw(state)
        if T<:Complex
            state=step(state); Q[i]=complex(x,draw(state))
        else
            Q[i]=x
        end
    end
    Q
end
"""Randomized range-finder SVD with explicit local seed and reorthogonalized
power steps. Intended for low target rank; residual quality is caller-controlled.
"""
function svd_randomized(A::AbstractMatrix{T},rank::Integer;oversampling=8,power_iterations=2,seed=0x0b1a5,threaded=true,kwargs...) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    Base.require_one_based_indexing(A); m,n=size(A)
    0<=rank<=min(m,n) && min(oversampling,power_iterations)>=0 || throw(ArgumentError("invalid randomized SVD options"))
    q=min(min(m,n),rank+oversampling); omega=_random_projection(T,n,q,seed)
    Y=Matrix{T}(undef,m,q); gemm!(Y,A,omega;threaded); Q=qr_q(qr_factor(Y;threaded);columns=q)
    for _ in 1:power_iterations
        Z=Matrix{T}(undef,n,q); gemm!(Z,adjoint(A),Q;threaded)
        W=qr_q(qr_factor(Z;threaded);columns=q); gemm!(Y,A,W;threaded)
        Q=qr_q(qr_factor(Y;threaded);columns=q)
    end
    B=Matrix{T}(undef,q,n); gemm!(B,adjoint(Q),A;threaded)
    F=svd_bidiagonal(B;threaded,kwargs...); U=Matrix{T}(undef,m,rank)
    gemm!(U,Q,view(F.U,:,1:rank);threaded)
    SVDFactor(U,F.S[1:rank],F.Vt[1:rank,:],F.sweeps)
end

function _jacobi_pair!(B,V,i,j,rtol,allow)
    T=eltype(B); R=real(T); m=size(B,1)
    nx,ny,rho=_svd_dense_pair_stats(B,i,j)
    (iszero(nx) || iszero(ny)) && return false
    off=abs(rho); off<=rtol && return false
    allow || return true
    if min(nx,ny)<=_svd_deflation_floor(R,nx,ny,m)
        col=nx<=ny ? i : j
        fill!(view(B,:,col),zero(T)); return true
    end
    ratio=min(nx,ny)/max(nx,ny); g=ratio*off
    iszero(g) && return false
    a,b=nx>=ny ? (one(R),ratio*ratio) : (ratio*ratio,one(R)); delta=(b-a)/2
    tangent=g/(delta+copysign(hypot(delta,g),delta)); c=Base.inv(hypot(one(R),tangent)); s=(c*tangent)*_realdiv(rho,off)
    _svd_rotate_columns!(B,i,j,c,s); _svd_rotate_columns!(V,i,j,c,s); true
end
function _svd_jacobi_round!(B,V,rtol,allow,schedule,flags,pairs,workers,n,slots)
    # Capture immutable arguments, rather than the enclosing mutable sweep
    # counter. Accumulate locally and publish one flag per task per round.
    _foreach_range(pairs,workers) do range,worker
        changed=false
        @inbounds for k in range
            i,j=schedule[k],schedule[slots+1-k]
            (i>n || j>n) && continue
            changed=_jacobi_pair!(B,V,min(i,j),max(i,j),rtol,allow) | changed
        end
        @inbounds flags[worker]=UInt8(changed)
    end
    any(!iszero,flags)
end
function _jacobi_parallel_svd!(B,V,rtol,maxsweeps,threaded)
    m,n=size(B); slots=iseven(n) ? n : n+1; slots<=1 && return 0
    schedule=collect(1:slots); pairs=slots÷2; sweeps=0
    # Keep enough disjoint pairs per task to amortize synchronization. Work
    # includes V when present; the bound depends on array dimensions, not ISA.
    work=Float64(m+(V===nothing ? 0 : n))*n
    workers=threaded ? min(Threads.nthreads(),max(1,pairs÷32),max(1,floor(Int,work/32768))) : 1
    flags=zeros(UInt8,workers)
    while true
        rotated=false
        for _ in 1:slots-1
            rotated=_svd_jacobi_round!(B,V,rtol,sweeps<maxsweeps,schedule,flags,pairs,workers,n,slots) | rotated
            if sweeps>=maxsweeps && rotated
                throw(ConvergenceError(:parallel_jacobi_svd,sweeps,1:n))
            end
            last=schedule[end]
            for k in slots:-1:3
                schedule[k]=schedule[k-1]
            end
            schedule[2]=last
        end
        !rotated && return sweeps
        sweeps+=1
    end
end
