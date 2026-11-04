# SPDX-License-Identifier: Apache-2.0
# Private dense SVD kernels. No shared BLAS/LAPACK method is changed here.
function _svd_dense_phase_product(z,phase)
    value=z*phase;mag=abs(value)
    mag,iszero(value) ? one(value) : _realdiv(value,mag)
end
function _svd_dense_phase_product(z::Complex{R},phase) where {R<:RealFloat}
    scale=max(abs(real(z)),abs(imag(z)))
    if scale<floatmin(R)/eps(R)
        iszero(scale) && return zero(R),one(z)
        scaled=_realdiv(z,scale);mag=abs(scaled)
        # Compose phases at unit scale: a rounded subnormal magnitude cannot
        # be used as the denominator of a unitary phase correction.
        unit=_realdiv(scaled,mag)*phase
        return mag*scale,_realdiv(unit,abs(unit))
    end
    value=z*phase;mag=abs(value)
    mag,iszero(value) ? one(value) : _realdiv(value,mag)
end
function _svd_dense_plane_batch!(Q,cs,ss,lo,hi,up)
    Q===nothing && return
    order=up ? (hi-1:-1:lo) : (lo:hi-1)
    # Each row sees the same ordered rotations. Small row tiles keep the
    # columns of the active block resident while applying a complete packet.
    @inbounds for first in 1:32:size(Q,1)
        last=min(first+31,size(Q,1))
        for k in order
            i,j=up ? (k+1,k) : (k,k+1);c,s=cs[k],ss[k]
            @simd for row in first:last
                x,y=Q[row,i],Q[row,j]
                Q[row,i]=c*x+s*y;Q[row,j]=-s*x+c*y
            end
        end
    end
    nothing
end
# Chained packet on native real matrices. Consecutive rotations share one
# column, (k,k+1) then (k+1,k+2) (or the reverse order): per block of rows that
# column is carried in vector registers, so each element is loaded and stored
# once per packet instead of twice per rotation. Same rotations, same order per
# row; only the fused multiply-add rounding differs from the scalar loop.
@inline function _svd_dense_carry_block!(p::Ptr{T},ld,row,cs,ss,first,order,::Val{L}) where {T,L}
    x=_vload(p+((first-1)*ld+row-1)*sizeof(T),Val(L));cur=first
    @inbounds for k in order
        nxt=cur==k ? k+1 : k
        c=_splat(cs[k],Val(L));s=ss[k]
        y=_vload(p+((nxt-1)*ld+row-1)*sizeof(T),Val(L))
        _vstore(p+((cur-1)*ld+row-1)*sizeof(T),_vfma(c,x,_vmul(_splat(s,Val(L)),y)))
        x=_vfma(c,y,_vmul(_splat(-s,Val(L)),x));cur=nxt
    end
    _vstore(p+((cur-1)*ld+row-1)*sizeof(T),x)
    nothing
end
function _svd_dense_plane_batch!(Q::Matrix{T},cs,ss,lo,hi,up) where {T<:RealFloat}
    order=up ? (hi-1:-1:lo) : (lo:hi-1);first=up ? hi : lo
    # The chain is latency bound: keep 16 (Float64) or 32 (Float32) rows in
    # flight per block (one LLVM vector, split into independent registers).
    m=size(Q,1);row=1
    GC.@preserve Q begin
        p=pointer(Q)
        if T===Float32
            while row+31<=m
                _svd_dense_carry_block!(p,m,row,cs,ss,first,order,Val(32));row+=32
            end
        end
        while row+15<=m
            _svd_dense_carry_block!(p,m,row,cs,ss,first,order,Val(16));row+=16
        end
        while row+7<=m
            _svd_dense_carry_block!(p,m,row,cs,ss,first,order,Val(8));row+=8
        end
        while row+3<=m
            _svd_dense_carry_block!(p,m,row,cs,ss,first,order,Val(4));row+=4
        end
    end
    @inbounds for r in row:m
        x=Q[r,first];cur=first
        for k in order
            nxt=cur==k ? k+1 : k;c,s=cs[k],ss[k];y=Q[r,nxt]
            Q[r,cur]=c*x+s*y;x=-s*x+c*y;cur=nxt
        end
        Q[r,cur]=x
    end
    nothing
end
function _svd_dense_householder!(v)
    T=eltype(v);R=real(T);scale=_finite_scale(v)
    if scale<sqrt(floatmin(R))
        @inbounds @simd for i in eachindex(v)
            v[i]=_realdiv(v[i],scale)
        end
        beta,tau=_householder!(v)
        return beta*scale,tau
    end
    _householder!(v)
end
function _svd_dense_pair_stats_fallback(B,i,j)
    T=eltype(B);m=size(B,1)
    nx=nrm2(view(B,:,i));ny=nrm2(view(B,:,j));rho=zero(T)
    (iszero(nx)||iszero(ny)) && return nx,ny,rho
    @inbounds for row in 1:m
        rho+=conj(_realdiv(B[row,i],nx))*_realdiv(B[row,j],ny)
    end
    nx,ny,rho
end

# Apply one stored Householder to a thin matrix. Each column is independent;
# only large panels create tasks. The reflector is read-only during the call.
function _svd_dense_reflect_left!(A,v,tau,firstrow,cols;threaded=false)
    iszero(tau) && return A
    workers=_operation_workers(length(cols),Float64(length(v))*length(cols),threaded,524288)
    _foreach_range(length(cols),workers) do range,_
        @inbounds for index in range
            j=cols[index]; s=zero(eltype(A))
            @simd for t in eachindex(v)
                s+=conj(v[t])*A[firstrow+t-1,j]
            end
            s*=tau
            @simd for t in eachindex(v)
                A[firstrow+t-1,j]-=v[t]*s
            end
        end
    end
    A
end

# SVD-private reduction: storage and Householder construction agree with
# bidiagonal_factor, while its shared kernels and callers remain untouched.
function _svd_dense_bidiagonal_factor(A::AbstractMatrix{T};threaded=false,blocked=true) where {T<:NativeFloat}
    blocked && size(A,2)>=96 && return _svd_dense_bidiagonal_blocked(A;threaded)
    F,scale=_normalized_copy(A);m,n=size(F);R=real(T)
    tl=zeros(R,n);tr=zeros(R,max(0,n-1));scratch=Vector{T}(undef,m);work=similar(scratch)
    @inbounds for k in 1:n
        v=view(scratch,1:m-k+1);copyto!(v,view(F,k:m,k));beta,tau=_svd_dense_householder!(v);tl[k]=tau
        _svd_dense_reflect_left!(F,v,tau,k,k+1:n;threaded);F[k,k]=beta
        for i in k+1:m
            F[i,k]=v[i-k+1]
        end
        if k<n
            v=view(scratch,1:n-k)
            for j in k+1:n
                v[j-k]=conj(F[k,j])
            end
            beta,tau=_svd_dense_householder!(v);tr[k]=tau
            _reflect_right!(F,v,tau,k+1,k+1:m,work);F[k,k+1]=conj(beta)
            for j in k+2:n
                F[k,j]=v[j-k]
            end
        end
    end
    BidiagonalFactor(F,tl,tr,scale,false)
end

# LABRD delayed two-sided Householder panel. The unreduced matrix is represented
# as A - V*Y' - X*U'. At a panel boundary its update is two native GEMMs.
# This is the established LABRD organization (Reference LAPACK, SRC/dlabrd.f),
# expressed with Hermitian, real-tau reflectors for both real and complex types.
function _svd_dense_bidiagonal_blocked(A::AbstractMatrix{T};threaded=false,blocksize=16) where {T<:NativeFloat}
    F,scale=_normalized_copy(A);m,n=size(F);R=real(T);nb=min(blocksize,n)
    tl=zeros(R,n);tr=zeros(R,max(0,n-1))
    V=zeros(T,m,nb);X=similar(V);U=zeros(T,n,nb);Y=similar(U)
    a=Vector{T}(undef,nb);b=similar(a);scratch=Vector{T}(undef,m)
    @inbounds for first in 1:nb:n
        width=min(nb,n-first+1)
        fill!(V,zero(T));fill!(U,zero(T));fill!(X,zero(T));fill!(Y,zero(T))
        for t in 1:width
            k=first+t-1
            for h in 1:t-1
                cy=conj(Y[k,h]);cu=conj(U[k,h])
                @simd for i in k:m
                    F[i,k]-=V[i,h]*cy+X[i,h]*cu
                end
            end
            v=view(scratch,1:m-k+1);copyto!(v,view(F,k:m,k))
            beta,tau=_svd_dense_householder!(v);tl[k]=tau
            copyto!(view(V,k:m,t),v)
            F[k,k]=beta
            for i in k+1:m
                F[i,k]=v[i-k+1]
            end
            k==n && continue
            for h in 1:t-1
                a[h]=dotc(view(V,k:m,h),v)
                b[h]=dotc(view(X,k:m,h),v)
            end
            for j in k+1:n
                Y[j,t]=dotc(view(F,k:m,j),v)
            end
            for h in 1:t-1
                ah,bh=a[h],b[h]
                @simd for j in k+1:n
                    Y[j,t]-=Y[j,h]*ah+U[j,h]*bh
                end
            end
            @simd for j in k+1:n
                Y[j,t]*=tau
            end
            # Materialize the next row after applying all pending left/right
            # updates, then construct its conjugated right reflector.
            vr=view(scratch,1:n-k)
            for j in k+1:n
                vr[j-k]=conj(F[k,j])
            end
            for h in 1:t
                z=conj(V[k,h])
                @simd for j in k+1:n
                    vr[j-k]-=z*Y[j,h]
                end
            end
            for h in 1:t-1
                z=conj(X[k,h])
                @simd for j in k+1:n
                    vr[j-k]-=z*U[j,h]
                end
            end
            betar,taur=_svd_dense_householder!(vr);tr[k]=taur
            copyto!(view(U,k+1:n,t),vr)
            F[k,k+1]=conj(betar)
            for j in k+2:n
                F[k,j]=vr[j-k]
            end
            for h in 1:t
                a[h]=dotc(view(Y,k+1:n,h),vr)
            end
            for h in 1:t-1
                b[h]=dotc(view(U,k+1:n,h),vr)
            end
            xt=view(X,k+1:m,t)
            gemv!(xt,view(F,k+1:m,k+1:n),vr;threaded=false)
            for h in 1:t
                z=a[h]
                @simd for i in k+1:m
                    X[i,t]-=V[i,h]*z
                end
            end
            for h in 1:t-1
                z=b[h]
                @simd for i in k+1:m
                    X[i,t]-=X[i,h]*z
                end
            end
            @simd for i in k+1:m
                X[i,t]*=taur
            end
        end
        last=first+width-1
        if last<n
            target=view(F,last+1:m,last+1:n)
            gemm!(target,view(V,last+1:m,1:width),adjoint(view(Y,last+1:n,1:width)),-one(T),one(T);threaded)
            gemm!(target,view(X,last+1:m,1:width),adjoint(view(U,last+1:n,1:width)),-one(T),one(T);threaded)
        end
    end
    BidiagonalFactor(F,tl,tr,scale,false)
end

# Q = H1*...*Hn: reverse application on the left obtains Q*X without
# materializing the square Q, including for tall matrices.
function _svd_dense_backtransform(F::BidiagonalFactor{T},X,left;threaded=false) where T
    A=F.factors;m,n=size(A);q=left ? m : n
    result=zeros(T,q,size(X,2));copyto!(view(result,1:size(X,1),:),X)
    if n>=96 && size(X,2)>=32
        # The bidiagonal left reflectors have exactly QR's storage layout.
        # Right reflectors use its transposed row storage, offset by one.
        # Reuse the existing native compact-WY application without changing
        # any shared kernel. It applies blocks backwards and owns task buffers.
        stored=left ? (factors=A,tau=F.tauleft) :
                      (factors=transpose(view(A,1:n-1,2:n)),tau=F.tauright)
        target=left ? result : view(result,2:n,:)
        _apply_q_blocked!(target,stored,true,false,false,32,threaded,
                          1:(left ? n : n-1),axes(result,2))
        return result
    end
    scratch=Vector{T}(undef,q)
    @inbounds for k in (left ? n : max(0,n-1)):-1:1
        first=left ? k : k+1;v=view(scratch,1:q-first+1);v[1]=one(T)
        for i in 2:length(v)
            v[i]=left ? A[first+i-1,k] : A[k,first+i-1]
        end
        tau=left ? F.tauleft[k] : F.tauright[k]
        _svd_dense_reflect_left!(result,v,tau,first,axes(result,2);threaded)
    end
    result
end
_svd_dense_pair_stats(B,i,j)=_svd_dense_pair_stats_fallback(B,i,j)
function _svd_dense_pair_stats(B::Matrix{T},i,j) where {T<:NativeFloat}
    R=real(T);aa=zero(R);bb=zero(R);g=zero(T)
    @inbounds @simd for row in axes(B,1)
        x,y=B[row,i],B[row,j]
        aa+=abs2(x);bb+=abs2(y);g+=conj(x)*y
    end
    # B was globally normalized. Its Frobenius energy bounds every local
    # product. Tiny energies use the original separately normalized products:
    # the guard bounds lost underflow contributions well below working epsilon.
    floor=(floatmin(R)/eps(R))*max(1,size(B,1))
    if aa>floor && bb>floor && isfinite(aa) && isfinite(bb) && isfinite(g)
        nx,ny=sqrt(aa),sqrt(bb)
        return nx,ny,_realdiv(_realdiv(g,nx),ny)
    end
    _svd_dense_pair_stats_fallback(B,i,j)
end
