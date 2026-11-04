# SPDX-License-Identifier: Apache-2.0
# Small-bulge train with windowed accumulation and GEMM far updates.
# Upstream at the pinned revision has no train/AED/blocking implementation;
# these extend its implicit Francis/Wilkinson kernels (see QZ_SCHUR_PORT.md).

struct SchurWorkspace{T}
    localH::Matrix{T}
    localQ::Matrix{T}
    right::Matrix{T}
    left::Matrix{T}
    vector::Vector{T}
    work::Vector{T}
    guard::ReentrantLock
    busy::Base.RefValue{Bool}
end
function SchurWorkspace(::Type{T},n::Integer;blocksize=64,nshifts=6) where {T<:GenericFloat}
    isconcretetype(T) && n>=0 && blocksize>=1 && nshifts>=1 || throw(ArgumentError("invalid Schur workspace"))
    width=min(n,max(blocksize+3cld(nshifts,2)+6,blocksize+2nshifts+2))
    SchurWorkspace(zeros(T,width,width),zeros(T,width,width),zeros(T,n,width),zeros(T,width,n),zeros(T,3),zeros(T,width),ReentrantLock(),Ref(false))
end
function _schur_workspace_check(w::SchurWorkspace{T},A,n,needed) where T
    eltype(A)===T || throw(ArgumentError("Schur workspace scalar type differs"))
    size(w.localH,1)>=needed && size(w.localH,2)>=needed && size(w.localQ,1)>=needed && size(w.localQ,2)>=needed &&
        size(w.right,1)>=n && size(w.right,2)>=needed && size(w.left,1)>=needed && size(w.left,2)>=n &&
        length(w.vector)>=3 && length(w.work)>=needed || throw(DimensionMismatch("Schur workspace too small"))
    buffers=(w.localH,w.localQ,w.right,w.left,w.vector,w.work)
    for i in eachindex(buffers)
        Base.mightalias(A,buffers[i]) && throw(ArgumentError("Schur input aliases workspace"))
        for j in 1:i-1
            Base.mightalias(buffers[i],buffers[j]) && throw(ArgumentError("Schur workspace buffers alias"))
        end
    end
end

function _schur_swap!(A,Q,k)
    T=eltype(A); a=A[k,k]; b=A[k,k+1]; d=A[k+1,k+1]; scale=max(abs(b),abs(d-a))
    if iszero(scale)
        u=zero(T); v=one(T)
    else
        x=b/scale; y=(d-a)/scale; norm=hypot(abs(x),abs(y))
        iszero(y) && !iszero(x) && throw(ArgumentError("inseparable repeated Schur eigenvalues"))
        u=x/norm; v=y/norm
    end
    _unitary2_similarity!(A,Q,k,u,v); A[k+1,k]=zero(T)
end
function _restore_hessenberg!(H,Q,firstcol,lastrow)
    T=eltype(H); n=size(H,1); scratch=zeros(T,n); work=similar(scratch)
    for j in firstcol:lastrow-2
        v=view(scratch,1:lastrow-j); copyto!(v,view(H,j+1:lastrow,j)); beta,tau=_householder!(v)
        _reflect_left!(H,v,tau,j+1,1:n); _reflect_right!(H,v,tau,j+1,1:n,work); _reflect_right!(Q,v,tau,j+1,1:n,work)
        H[j+1,j]=beta
        for i in j+2:lastrow
            H[i,j]=zero(T)
        end
    end
end

# AED tests whole conjugate blocks against the transformed spike. Undeflatable
# blocks move to the front; every tentative swap is stability checked. Failure
# to swap an inseparable block only stops that AED attempt, never corrupts H.
function _aed_window!(H,Q,lo,hi,window,budget,threaded)
    T=eltype(H); R=real(T); start=max(lo,hi-window+1); range=start:hi
    F=schur_factor(view(H,range,range);maxiter=budget)
    S=copy(F.T); Z=copy(F.Z); spike=start>lo ? H[start,start-1] : zero(T)
    blocks=_schur_blocks(S); sizes=length.(blocks); remaining=length(sizes); index=remaining
    while index>=1
        k=1+sum(view(sizes,1:index-1)); block=k:k+sizes[index]-1
        threshold=max(floatmin(R)/eps(R),8eps(R)*maximum(abs,view(S,block,block)))
        if abs(spike)*nrm2(view(Z,1,block))<=threshold
            # Move a deflatable block to the bottom of the live window.
            moved=true
            for b in index:remaining-1
                pos=1+sum(view(sizes,1:b-1))
                try
                    if T<:Real
                        _real_block_swap!(S,Z,pos,sizes[b],sizes[b+1])
                    else
                        _schur_swap!(S,Z,pos)
                    end
                catch error
                    error isa ArgumentError || rethrow()
                    moved=false; break
                end
                sizes[b],sizes[b+1]=sizes[b+1],sizes[b]
            end
            if moved
                k=1+sum(view(sizes,1:remaining-1)); block=k:k+sizes[remaining]-1
                threshold=max(floatmin(R)/eps(R),8eps(R)*maximum(abs,view(S,block,block)))
                abs(spike)*nrm2(view(Z,1,block))<=threshold && (remaining-=1)
            end
        end
        index-=1
    end
    active=sum(view(sizes,1:remaining)); shifts=copy(view(S,1:active,1:active))
    # All transforms also act on the spike column and on both far slabs.
    gemm!(view(H,:,range),view(H,:,range),Z;threaded)
    gemm!(view(H,range,:),adjoint(Z),view(H,range,:);threaded)
    gemm!(view(Q,:,range),view(Q,:,range),Z;threaded); H[range,range]=S
    if start>lo
        for i in active+1:length(range)
            H[start+i-1,start-1]=zero(T)
        end
    end
    newhi=start+active-1
    active>1 && start>lo && _restore_hessenberg!(H,Q,start-1,newhi)
    newhi,shifts,F.iterations
end

# Pack real shifts without cutting a conjugate pair. Every packet is a real
# quadratic polynomial, including a pair of real shifts or a repeated singleton.
function _shift_packets(S,nshifts)
    T=eltype(S); R=real(T)
    if T<:Complex
        return [(S[j,j],) for j in max(1,size(S,1)-nshifts+1):size(S,1)]
    end
    packets=Tuple{R,R,R}[]; pending=nothing; count=0
    for block in reverse(_schur_blocks(S))
        count>=nshifts && break
        if length(block)==2
            a,b,c,d=S[first(block),first(block)],S[first(block),last(block)],S[last(block),first(block)],S[last(block),last(block)]
            scale=max(abs(a),abs(b),abs(c),abs(d)); scale=iszero(scale) ? one(R) : scale
            push!(packets,((a+d)/scale,(a/scale)*(d/scale)-(b/scale)*(c/scale),scale)); count+=2
        else
            mu=S[first(block),first(block)]; count+=1
            if pending===nothing
                pending=mu
            else
                scale=max(abs(mu),abs(pending)); scale=iszero(scale) ? one(R) : scale
                push!(packets,(mu/scale+pending/scale,(mu/scale)*(pending/scale),scale)); pending=nothing
            end
        end
    end
    if pending!==nothing
        scale=iszero(pending) ? one(R) : abs(pending)
        push!(packets,(2pending/scale,(pending/scale)^2,scale))
    end
    packets
end
function _qr_seed!(v,H,k,packet,::Type{T}) where {T<:Real}
    trace,det,scale=packet
    sc=max(scale,abs(H[k,k]),abs(H[k,k+1]),abs(H[k+1,k]),abs(H[k+1,k+1]),abs(H[k+2,k+1]))
    sc=iszero(sc) ? one(T) : sc; ratio=scale/sc
    h=H[k,k]/sc; c=H[k+1,k]/sc
    v[1]=h*h+(H[k,k+1]/sc)*c-trace*ratio*h+det*ratio*ratio
    v[2]=c*(h+H[k+1,k+1]/sc-trace*ratio)
    v[3]=c*(H[k+2,k+1]/sc)
end
function _qr_seed!(v,H,k,packet,::Type{T}) where {T<:Complex}
    v[1]=H[k,k]-packet[1]; v[2]=H[k+1,k]
end

function _schur_far_update!(H,Q,range,U,w,threaded)
    a,b=first(range),last(range); p=length(range); n=size(H,1)
    for rows in (1:a-1,b+1:n)
        isempty(rows) && continue
        tmp=view(w.right,1:length(rows),1:p)
        gemm!(tmp,view(H,rows,range),U;threaded); copyto!(view(H,rows,range),tmp)
    end
    for cols in (1:a-1,b+1:n)
        isempty(cols) && continue
        tmp=view(w.left,1:p,1:length(cols))
        gemm!(tmp,adjoint(U),view(H,range,cols);threaded); copyto!(view(H,range,cols),tmp)
    end
    tmp=view(w.right,1:n,1:p); gemm!(tmp,view(Q,:,range),U;threaded); copyto!(view(Q,:,range),tmp)
end

# Multiple *simultaneous* bulges, spaced by three rows (real double shifts) or
# two rows (complex single shifts). Advance the leading bulge first, then its
# followers, so an older bulge has cleared each column before a follower uses it.
# Accumulate each slab's reflectors in U; only the near square is touched during
# the chase. Far rows/columns and Q are updated once per slab using native GEMM.
function _small_bulge_train!(H,Q,lo,hi,packets,w,blocksize,threaded)
    T=eltype(H); spacing=T<:Real ? 3 : 2; width=T<:Real ? 3 : 2; nb=length(packets)
    lasttime=hi-1+spacing*(nb-1)
    for firsttime in lo:blocksize:lasttime
        laststep=min(lasttime,firsttime+blocksize-1)
        a=max(lo,firsttime-spacing*(nb-1)-1); b=min(hi,laststep+width)
        range=a:b; p=length(range); D=view(w.localH,1:p,1:p); U=view(w.localQ,1:p,1:p)
        copyto!(D,view(H,range,range)); fill!(U,zero(T))
        for i in 1:p
            U[i,i]=one(T)
        end
        for time in firsttime:laststep, bulge in 1:nb
            pos=time-spacing*(bulge-1)
            lo<=pos<hi || continue
            r=min(width,hi-pos+1); k=pos-a+1; v=view(w.vector,1:r)
            if pos==lo
                _qr_seed!(w.vector,D,k,packets[bulge],T)
            else
                copyto!(v,view(D,k:k+r-1,k-1))
            end
            beta,tau=_householder!(v)
            _reflect_left!(D,v,tau,k,1:p); _reflect_right!(D,v,tau,k,1:p,w.work)
            _reflect_right!(U,v,tau,k,1:p,w.work)
            if pos>lo
                D[k,k-1]=beta
                for row in k+1:k+r-1
                    D[row,k-1]=zero(T)
                end
            end
        end
        copyto!(view(H,range,range),D); _schur_far_update!(H,Q,range,U,w,threaded)
    end
    nothing
end

"""Native real/complex multishift Schur: aggressive early deflation, concurrent
small-bulge train, and accumulated blocked far updates. Real input preserves
real 1x1/2x2 blocks. `workspace` reuses panel and update buffers; a shared context
is serialized, with reentrant use rejected. No external eigensolver is called.
"""
function schur_multishift(A::AbstractMatrix{T};window=32,nshifts=6,blocksize=64,
        maxiter=200max(1,size(A,1))^2,threaded=true,workspace=nothing) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T); n=_square(A)
    min(window,nshifts,blocksize)>=1 && maxiter isa Integer && maxiter>=0 || throw(ArgumentError("invalid multishift Schur options"))
    needed=min(n,blocksize+3cld(nshifts,2)+6)
    # Complex trains need one slot per shift, rather than one per pair.
    needed=max(needed,min(n,blocksize+2nshifts+2))
    w=workspace===nothing ? SchurWorkspace(T,n;blocksize,nshifts) : workspace
    w isa SchurWorkspace || throw(ArgumentError("invalid Schur workspace"))
    lock(w.guard)
    try
        w.busy[] && throw(ArgumentError("reentrant Schur workspace use")); _schur_workspace_check(w,A,n,needed); w.busy[]=true
        try
            _schur_multishift(A,w;window,nshifts,blocksize,maxiter,threaded)
        finally
            w.busy[]=false
        end
    finally
        unlock(w.guard)
    end
end
function _schur_multishift(A::AbstractMatrix{T},w;window,nshifts,blocksize,maxiter,threaded) where T
    n=size(A,1); F=hessenberg_factor(A); H=_hessenberg_matrix(F;rescale=false); Q=hessenberg_q(F)
    hi=n; iterations=0; stagnation=0
    while hi>1
        lo=hi
        while lo>1
            if _deflatable(H,lo,1,hi)
                H[lo,lo-1]=zero(T); break
            end
            lo-=1
        end
        if lo==hi
            hi-=1; stagnation=0; continue
        elseif T<:Real && lo==hi-1
            _finish_real_pair!(H,Q,lo); hi-=2; stagnation=0; continue
        elseif T<:Complex && lo==hi-1
            _finish_complex_pair!(H,Q,lo); hi-=2; stagnation=0; continue
        end
        iterations>=maxiter && throw(ConvergenceError(:multishift_schur,iterations,lo:hi))
        oldhi=hi; hi,shifts,used=_aed_window!(H,Q,lo,hi,window,maxiter-iterations,threaded); iterations+=used
        hi<oldhi && (stagnation=0)
        hi<=lo && continue
        # An AED window can leave a real 2x2 active block.
        if T<:Real && hi==lo+1
            _finish_real_pair!(H,Q,lo); hi-=2; continue
        end
        packets=_shift_packets(shifts,nshifts); stagnation+=1
        if stagnation%20==0 || isempty(packets)
            mu=H[hi,hi]+real(T)(3)/4*abs(H[hi,hi-1])
            if T<:Real
                s=max(abs(mu),one(T)); packets=[(2mu/s,(mu/s)^2,s)]
            else
                packets=[(mu,)]
            end
        end
        iterations>=maxiter && throw(ConvergenceError(:multishift_schur,iterations,lo:hi))
        _small_bulge_train!(H,Q,lo,hi,packets,w,blocksize,threaded); iterations+=1
    end
    values=_schur_values(H,F.scale); H .*= F.scale
    SchurFactor(H,Q,values,iterations)
end
function schur_reorder(F::SchurFactor{T},select::Function) where {T<:AbstractFloat}
    _check_schur_form(F.T); _real_schur_reorder(F,select)
end

struct BalanceFactor{T,R}
    matrix::Matrix{T}
    permutation::Vector{Int}
    scales::Vector{R}
    ilo::Int
    ihi::Int
end
"""Permutation isolation and radix-two Osborne balancing.
B=Diagonal(1/scales)*A[p,p]*Diagonal(scales). Input is not changed.
"""
function balance_matrix(A::AbstractMatrix{T};permute=true,scale=true,maxiter=100) where {T<:GenericFloat}
    n=_square(A); _finite_scale(A); maxiter>=0 || throw(ArgumentError("negative balancing iteration budget"))
    B=Matrix(A); p=collect(1:n); scales=ones(real(T),n); lo=1; hi=n
    function swap(i,j)
        i==j && return
        for col in 1:n
            B[i,col],B[j,col]=B[j,col],B[i,col]
        end
        for row in 1:n
            B[row,i],B[row,j]=B[row,j],B[row,i]
        end
        p[i],p[j]=p[j],p[i]; scales[i],scales[j]=scales[j],scales[i]
    end
    if permute
        changed=true
        while changed && lo<=hi
            changed=false
            for i in hi:-1:lo
                if all(j==i || iszero(B[i,j]) for j in lo:hi)
                    swap(i,hi); hi-=1; changed=true; break
                end
            end
            changed && continue
            for j in lo:hi
                if all(i==j || iszero(B[i,j]) for i in lo:hi)
                    swap(j,lo); lo+=1; changed=true; break
                end
            end
        end
    end
    if scale
        for _ in 1:maxiter
            changed=false
            for i in lo:hi
                row=sum((abs(B[i,j]) for j in lo:hi if j!=i);init=zero(real(T)))
                column=sum((abs(B[j,i]) for j in lo:hi if j!=i);init=zero(real(T)))
                (iszero(row) || iszero(column) || !isfinite(row) || !isfinite(column)) && continue
                exponent=clamp(round(Int,(log2(row)-log2(column))/2),-32,32)
                factor=ldexp(one(real(T)),exponent); next=scales[i]*factor
                isfinite(next) && next>0 || continue
                (row/factor+column*factor)<real(T)(.95)*(row+column) || continue
                all(j==i || (isfinite(B[j,i]*factor) && isfinite(_realdiv(B[i,j],factor))) for j in 1:n) || continue
                for j in 1:n
                    j==i && continue
                    B[j,i]*=factor; B[i,j]=_realdiv(B[i,j],factor)
                end
                scales[i]=next; changed=true
            end
            !changed && break
        end
    end
    BalanceFactor(B,p,scales,lo,hi)
end
function balance_backtransform(B::BalanceFactor,V::AbstractMatrix;left=false)
    size(V,1)==length(B.permutation) || throw(DimensionMismatch("balanced vector dimensions differ"))
    T=eltype(V); R=real(T); result=similar(V)
    for col in axes(V,2)
        largest=-R(Inf)
        for i in axes(V,1)
            if !iszero(V[i,col])
                largest=max(largest,log(abs(V[i,col]))+(left ? -log(B.scales[i]) : log(B.scales[i])))
            end
        end
        for i in axes(V,1)
            value=V[i,col]
            magnitude=iszero(value) ? zero(R) : exp(log(abs(value))+(left ? -log(B.scales[i]) : log(B.scales[i]))-largest)
            result[B.permutation[i],col]=iszero(value) ? zero(T) : magnitude*(value/abs(value))
        end
        norm=nrm2(view(result,:,col)); iszero(norm) || (view(result,:,col)./=norm)
    end
    result
end
