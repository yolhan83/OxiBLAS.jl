# SPDX-License-Identifier: Apache-2.0
# Supernodal multifrontal Cholesky (P·A·Pᵀ = L·Lᴴ):
#   1. the selected triangle of A, permuted, as lower and upper compressed
#      patterns built by counting (no COO round trip, no sort);
#   2. fill-reducing ordering (AMD by default), elimination tree, column
#      counts of L by row subtrees (without the full structure), then a
#      postorder weighted by the counts composed into the permutation;
#   3. fundamental supernodes (uncapped width by default), then CHOLMOD's
#      relaxed amalgamation (widths 4/16/48, zero fractions 0.8/0.1/0.05);
#   4. numeric phase in postorder with a stack of contribution blocks: each
#      front is assembled through a relative index map into a reused dense
#      buffer, partially factored (in-place kernel for small fronts; POTRF /
#      TRSM / HERK of this package otherwise), and its columns are written
#      straight into the CSC storage of L;
#   or, when flops/nnz(L) < 40 (CHOLMOD's switch), an up-looking simplicial
#   factorization instead of steps 3–4.

# Selected triangle of A with permutation pinv applied: lower part as CSC
# (column b, row a ≥ b) and its transpose (upper part by columns, for the
# elimination tree and the up-looking kernel; skipped, left empty, when
# `withupper` is false). With `hermitian` (default) the diagonal is made
# real and the mirrored entries are conjugated; otherwise (complex symmetric)
# they are kept as stored.
function _chol_permuted(A::NativeSparse,uplo,pinv,withupper=true,hermitian=true)
    C=_as_csc(A)
    n=size(C,1); upper=_uplo(uplo); T=eltype(C)
    lcnt=zeros(Int,n+1); ucnt=zeros(Int,n+1)
    @inbounds for j in 1:n, q in C.colptr[j]:C.colptr[j+1]-1
        i=Int(C.rowind[q]); (upper ? i<=j : i>=j) || continue
        a,b=pinv[i],pinv[j]; a<b && ((a,b)=(b,a))
        lcnt[b+1]+=1; withupper && a!=b && (ucnt[a+1]+=1)
    end
    lp=Vector{Int}(undef,n+1); up=Vector{Int}(undef,n+1); lp[1]=up[1]=1
    @inbounds for j in 1:n
        lp[j+1]=lp[j]+lcnt[j+1]; up[j+1]=up[j]+ucnt[j+1]
    end
    li=Vector{Int}(undef,lp[n+1]-1); lv=Vector{T}(undef,lp[n+1]-1); ui=Vector{Int}(undef,up[n+1]-1); uv=Vector{T}(undef,up[n+1]-1)
    ln=copy(lp); un=copy(up)
    @inbounds for j in 1:n, q in C.colptr[j]:C.colptr[j+1]-1
        i=Int(C.rowind[q]); (upper ? i<=j : i>=j) || continue
        v=C.values[q]; a,b=pinv[i],pinv[j]
        if a==b
            hermitian && (v=T(real(v)))
        elseif a<b
            a,b=b,a; hermitian && (v=conj(v))
        end
        li[ln[b]]=a; lv[ln[b]]=v; ln[b]+=1
        withupper && a!=b && (ui[un[a]]=b; uv[un[a]]=hermitian ? conj(v) : v; un[a]+=1)
    end
    lp,li,lv,up,ui,uv
end
# Column counts of L (diagonal included) in O(nnz(A)·α) (Gilbert, Ng and
# Peyton; CSparse cs_counts): skeleton leaves of the row subtrees, least
# common ancestors by path compression. `lp/li`: lower pattern by columns,
# `post`: a postorder of the elimination tree `parent`.
function _chol_counts(lp,li,parent,post,n)
    delta=zeros(Int,n); first=fill(0,n); maxfirst=fill(0,n); prevleaf=fill(0,n); ancestor=collect(1:n)
    @inbounds for k in 1:n
        j=post[k]; delta[j]=first[j]==0 ? 1 : 0
        while j!=0 && first[j]==0
            first[j]=k; j=parent[j]
        end
    end
    @inbounds for k in 1:n
        j=post[k]
        parent[j]!=0 && (delta[parent[j]]-=1)
        for q in lp[j]:lp[j+1]-1
            i=li[q]
            # cs_leaf: is j a leaf of the i-th row subtree?
            (i<=j || first[j]<=maxfirst[i]) && continue
            maxfirst[i]=first[j]; jprev=prevleaf[i]; prevleaf[i]=j
            delta[j]+=1
            if jprev!=0
                a=jprev
                while a!=ancestor[a]
                    a=ancestor[a]
                end
                t=jprev
                while t!=a
                    nxt=ancestor[t]; ancestor[t]=a; t=nxt
                end
                delta[a]-=1
            end
        end
        parent[j]!=0 && (ancestor[j]=parent[j])
    end
    @inbounds for k in 1:n
        j=post[k]
        parent[j]!=0 && (delta[parent[j]]+=delta[j])
    end
    delta
end
# Fundamental supernodes, then CHOLMOD's relaxed amalgamation
# (cholmod_super_symbolic): from the root down, supernode s merges with s+1
# when the current (merged, path-compressed) parent of s is s+1 and the
# merged supernode stays within widths 4/16/48 and zero fractions
# 0.8/0.1/0.05, the zeros being tracked from the leading column counts.
function _chol_supernodes(parent,counts,n,maxwidth,zrelax=(0.8,0.1,0.05))
    nchild=zeros(Int,n)
    for j in 1:n
        parent[j]!=0 && (nchild[parent[j]]+=1)
    end
    starts=Int[]
    for j in 1:n
        (j>1 && parent[j-1]==j && counts[j-1]==counts[j]+1 && nchild[j]==1 && j-starts[end]<maxwidth) || push!(starts,j)
    end
    nf=length(starts); push!(starts,n+1)
    supermap=Vector{Int}(undef,n)
    @inbounds for s in 1:nf, j in starts[s]:starts[s+1]-1
        supermap[j]=s
    end
    sparent=[parent[starts[s+1]-1]==0 ? 0 : supermap[parent[starts[s+1]-1]] for s in 1:nf]
    merged=zeros(Int,nf); nscol=[starts[s+1]-starts[s] for s in 1:nf]
    zeros_=zeros(Int,nf); snz=[counts[starts[s]] for s in 1:nf]
    @inbounds for s in nf-1:-1:1
        sparent[s]==0 && continue
        # Current parent of s, with path compression.
        ss=sparent[s]
        while merged[ss]!=0
            ss=merged[ss]
        end
        sp=ss; ss=sparent[s]
        while merged[ss]!=0
            nxt=merged[ss]; merged[ss]=sp; ss=nxt
        end
        sp==s+1 || continue
        n0,n1=nscol[s],nscol[s+1]; ns=n0+n1
        ns<=maxwidth || continue
        tot=zeros_[s+1]
        merge=if ns<=4
            true
        else
            newz=n0*(snz[s+1]+n0-snz[s])
            if newz==0
                true
            else
                tot+=newz
                z=tot/(ns*(ns+1)/2+ns*(snz[s+1]-n1))
                (ns<=16 && z<zrelax[1]) || (ns<=48 && z<zrelax[2]) || z<zrelax[3]
            end
        end
        if merge
            zeros_[s]=tot; merged[s+1]=s; snz[s]=n0+snz[s+1]; nscol[s]=ns
        end
    end
    nodes=UnitRange{Int}[]
    @inbounds for s in 1:nf
        merged[s]==0 && push!(nodes,starts[s]:starts[s]+nscol[s]-1)
    end
    nodes
end
# Rows of each supernode below its last column: its own entries of A, and
# the rows of its child supernodes, sorted.
function _chol_supernode_rows(nodes,lp,li,parent,n)
    owner=zeros(Int,n)
    for (s,node) in enumerate(nodes)
        owner[node].=s
    end
    sparent=[parent[last(node)]==0 ? 0 : owner[parent[last(node)]] for node in nodes]
    schildren=[Int[] for _ in nodes]
    for s in eachindex(nodes)
        sparent[s]!=0 && push!(schildren[sparent[s]],s)
    end
    rows=Vector{Vector{Int}}(undef,length(nodes)); mark=zeros(Int,n)
    @inbounds for (s,node) in enumerate(nodes)
        b=last(node); r=Int[]
        for j in node, q in lp[j]:lp[j+1]-1
            i=li[q]; i>b && mark[i]!=s && (mark[i]=s; push!(r,i))
        end
        for c in schildren[s], i in rows[c]
            i>b && mark[i]!=s && (mark[i]=s; push!(r,i))
        end
        rows[s]=sort!(r)
    end
    rows,sparent,schildren
end
# Partial Cholesky of the first w columns of an f×f front (lower triangle),
# Schur complement left in the trailing block, blocked right-looking: each
# block of nb panel columns is factored within itself (rank 1, rows to f),
# then every column to its right takes the rank-nb update, two columns at a
# time with four block columns per pass (6 loads for 8 FMAs). Rank-1 updates
# of the whole trailing block ran out of L1 from f ≈ 48 on, and a panel
# factored column by column was slow for wide supernodes (natural order).
function _front_partial_small!(F,f,w,offset;nb=16)
    @inbounds for k0 in 1:nb:w
        k1=min(k0+nb-1,w)
        for k in k0:k1
            d=real(F[k,k]); isfinite(d) && d>0 || throw(PosDefException(offset+k-1))
            r=sqrt(d); F[k,k]=r; inv=1/r
            for i in k+1:f
                F[i,k]*=inv
            end
            for j in k+1:k1
                c=conj(F[j,k])
                @simd for i in j:f
                    F[i,j]=muladd(-F[i,k],c,F[i,j])
                end
            end
        end
        _front_update!(F,f,k1+1,k0,k1)
    end
    F
end
# F[i,j] -= Σ_{k0≤k≤k1} F[i,k]·conj(F[j,k]) for j0 ≤ j ≤ i ≤ f.
function _front_update!(F,f,j0,k0,k1)
    j=j0
    @inbounds while j<=f
        if j<f
            k=k0
            while k<=k1
                if k+3<=k1
                    a1,a2,a3,a4=conj(F[j,k]),conj(F[j,k+1]),conj(F[j,k+2]),conj(F[j,k+3])
                    b1,b2,b3,b4=conj(F[j+1,k]),conj(F[j+1,k+1]),conj(F[j+1,k+2]),conj(F[j+1,k+3])
                    F[j,j]=muladd(-F[j,k+3],a4,muladd(-F[j,k+2],a3,muladd(-F[j,k+1],a2,muladd(-F[j,k],a1,F[j,j]))))
                    @simd for i in j+1:f
                        f1,f2,f3,f4=F[i,k],F[i,k+1],F[i,k+2],F[i,k+3]
                        F[i,j]=muladd(-f4,a4,muladd(-f3,a3,muladd(-f2,a2,muladd(-f1,a1,F[i,j]))))
                        F[i,j+1]=muladd(-f4,b4,muladd(-f3,b3,muladd(-f2,b2,muladd(-f1,b1,F[i,j+1]))))
                    end
                    k+=4
                else
                    a,b=conj(F[j,k]),conj(F[j+1,k])
                    F[j,j]=muladd(-F[j,k],a,F[j,j])
                    @simd for i in j+1:f
                        fk=F[i,k]
                        F[i,j]=muladd(-fk,a,F[i,j]); F[i,j+1]=muladd(-fk,b,F[i,j+1])
                    end
                    k+=1
                end
            end
            j+=2
        else
            for k in k0:k1
                F[j,j]=muladd(-F[j,k],conj(F[j,k]),F[j,j])
            end
            j+=1
        end
    end
    F
end

# Up-looking simplicial Cholesky (CSparse cs_chol): row k of L from the
# triangular solve with the pattern given by the elimination-tree reach of
# the upper column k; L is filled into the CSC layout of the column counts.
function _chol_simplicial(lp,li,lv,up,ui,uv,parent,counts,n,::Type{T}) where T
    colptr=Vector{Int}(undef,n+1); colptr[1]=1
    @inbounds for j in 1:n
        colptr[j+1]=colptr[j]+counts[j]
    end
    rowind=Vector{Int}(undef,colptr[n+1]-1); values=Vector{T}(undef,colptr[n+1]-1)
    c=colptr[1:n]; x=zeros(T,n); flag=zeros(Int,n); stack=Vector{Int}(undef,n)
    @inbounds for k in 1:n
        top=n+1; flag[k]=k
        for q in up[k]:up[k+1]-1
            i=ui[q]; x[i]=uv[q]; len=0
            while flag[i]!=k
                len+=1; stack[len]=i; flag[i]=k; i=parent[i]
            end
            while len>0
                top-=1; stack[top]=stack[len]; len-=1
            end
        end
        d=zero(real(T))
        for q in lp[k]:lp[k+1]-1
            li[q]==k && (d+=real(lv[q]))
        end
        for t in top:n
            i=stack[t]; lki=x[i]/real(values[colptr[i]]); x[i]=zero(T)
            for q in colptr[i]+1:c[i]-1
                x[rowind[q]]-=values[q]*lki
            end
            d-=abs2(lki); q=c[i]; c[i]+=1
            rowind[q]=k; values[q]=conj(lki)
        end
        isfinite(d) && d>0 || throw(PosDefException(k))
        q=c[k]; c[k]+=1; rowind[q]=k; values[q]=T(sqrt(d))
    end
    CSCMatrix(_Shared(),n,n,colptr,rowind,values)
end

# Fronts up to this size use the in-place front kernel: below it, the
# per-call cost of POTRF/TRSM/HERK exceeded their gain (100², 300², 22³ and
# 40³ Laplacians, AMD and natural orders, thresholds 96–768).
const _MF_SMALL_FRONT=192
"""Supernodal multifrontal Hermitian Cholesky of the `uplo` triangle of A,
P·A·Pᵀ = L·Lᴴ, with a fill-reducing `ordering` (AMD by default). The dense
fronts use this package's POTRF/TRSM/HERK (`threaded` lets them use threads).
`supernodal=:auto` follows CHOLMOD: an up-looking simplicial factorization
when flops/nnz(L) < 40 (chain-like factors), supernodal otherwise; `true`
or `false` force one of them.
`maxwidth` caps the supernode width; uncapped by default, as in CHOLMOD (a
cap of 64 split the 3D separators and ran HERK at rank 64: ×1.35 against
CHOLMOD on a 40³ Laplacian instead of ×1.10).
"""
function multifrontal_cholesky(A::NativeSparse;uplo=:L,ordering=:amd,supernodal=:auto,maxwidth=typemax(Int),threaded=true)
    n=_square(A); maxwidth>=1 || throw(ArgumentError("positive supernode width required"))
    supernodal in (:auto,true,false) || throw(ArgumentError("supernodal must be :auto, true or false"))
    T=eltype(A); ident=collect(1:n)
    lp,li,lv,_=_chol_permuted(A,uplo,ident,false)
    # Ordering on the selected triangle (AMD symmetrizes it; the other
    # orderings receive the full Hermitian pattern).
    p=if ordering isa AbstractVector
        _checked_sparse_permutation(ordering,n)
    elseif ordering===:natural
        ident
    elseif ordering===:amd
        amd_ordering(CSCMatrix(_Shared(),n,n,lp,li,lv))
    else
        sparse_ordering(_sparse_hermitian(A,uplo),ordering)
    end
    pinv=invperm(p)
    lp,li,lv,up,ui,uv=_chol_permuted(A,uplo,pinv)
    parent=_etree(up,ui,n)
    counts=_chol_counts(lp,li,parent,_postorder(parent),n)
    if supernodal===false || (supernodal===:auto && sum(c->Float64(c)^2,counts;init=0.0)<40*sum(counts;init=0))
        L=_chol_rethrow_original(p,_chol_simplicial,lp,li,lv,up,ui,uv,parent,counts,n,T)
        return MultifrontalCholeskyFactor(L,p,parent,[j:j for j in 1:n],counts,zeros(T,2n),ReentrantLock())
    end
    # Weighted postorder (heaviest child last), composed into the permutation;
    # counts and tree are relabeled, only the pattern is rebuilt.
    post=_postorder(parent,counts)
    if post!=ident
        p=p[post]; pinv=invperm(p); ipost=invperm(post)
        counts=counts[post]
        relabeled=Vector{Int}(undef,n)
        @inbounds for k in 1:n
            q=parent[post[k]]; relabeled[k]=q==0 ? 0 : ipost[q]
        end
        parent=relabeled
        lp,li,lv,up,ui,uv=_chol_permuted(A,uplo,pinv)
    end
    nodes=_chol_supernodes(parent,counts,n,maxwidth)
    rows,sparent,schildren=_chol_supernode_rows(nodes,lp,li,parent,n)
    L,fronts=_chol_rethrow_original(p,_chol_supernodal_numeric,lp,li,lv,nodes,rows,schildren,sparent,n,threaded,T)
    MultifrontalCholeskyFactor(L,p,sparent,nodes,fronts,zeros(T,2n),ReentrantLock())
end
# A failing pivot is reported at its column of A (the kernels know only the
# permuted column). Arguments are passed through, not captured: the driver
# reassigns them, and a closure would box them.
function _chol_rethrow_original(p,f::F,args...) where F
    try
        f(args...)
    catch err
        err isa PosDefException && 1<=err.info<=length(p) && throw(PosDefException(p[err.info]))
        rethrow()
    end
end
# Front buffers of one worker: dense front, relative positions, and the
# stack of contribution blocks (lower triangles packed by columns, LIFO;
# `stack` keeps (supernode, offset) pairs).
mutable struct _CholBuffers{T}
    W::Matrix{T}
    relpos::Vector{Int}
    arena::Vector{T}
    top::Int
    stack::Vector{Tuple{Int,Int}}
end
_CholBuffers(::Type{T},f,n) where T=_CholBuffers{T}(Matrix{T}(undef,f,f),zeros(Int,n),T[],0,Tuple{Int,Int}[])

# Supernodes `range` (a postorder segment) through the buffers `B`. The
# contribution block of a supernode marked in `subroot` goes to `cbstore`
# instead of the stack, and is read from there by its parent.
function _chol_fronts!(range,B::_CholBuffers{T},lp,li,lv,nodes,rows,schildren,colptr,rowind,values,
                       subroot,cbstore,threaded,ws) where T
    W,relpos=B.W,B.relpos; leaf=T<:Complex ? 32 : 64
    @inbounds for s in range
        node=nodes[s]; a,b=first(node),last(node); w=length(node); R=rows[s]; f=w+length(R)
        for l in 1:w
            relpos[a+l-1]=l
        end
        for (l,i) in enumerate(R)
            relpos[i]=w+l
        end
        F=view(W,1:f,1:f)
        for j in 1:f, i in j:f
            F[i,j]=zero(T)
        end
        for j in node, q in lp[j]:lp[j+1]-1
            F[relpos[li[q]],j-a+1]+=lv[q]
        end
        # Children's contribution blocks: stored ones for subtree roots, the
        # others on top of the stack.
        for c in Iterators.reverse(schildren[s])
            if subroot[c]
                cc=c; src=cbstore[c]; t=1
            else
                cc,offset=pop!(B.stack); src=B.arena; t=offset; B.top=offset-1
            end
            cidx=rows[cc]; m=length(cidx)
            for jj in 1:m
                lj=relpos[cidx[jj]]
                for ii in jj:m
                    F[relpos[cidx[ii]],lj]+=src[t]; t+=1
                end
            end
            subroot[c] && (cbstore[c]=T[])
        end
        if f<=_MF_SMALL_FRONT
            _front_partial_small!(F,f,w,a)
        else
            _cholesky_recursive!(view(F,1:w,1:w),a,leaf,threaded)
            if f>w
                panel=view(F,w+1:f,1:w)
                trsm!(panel,view(F,1:w,1:w);side=:R,uplo=:L,trans=:C,threaded,workspace=ws)
                herk!(view(F,w+1:f,w+1:f),panel,-one(real(T)),one(real(T));uplo=:L,threaded,workspace=ws)
            end
        end
        for jj in 1:w
            j=a+jj-1; q=colptr[j]
            for ii in jj:w
                rowind[q]=a+ii-1; values[q]=F[ii,jj]; q+=1
            end
            for (l,i) in enumerate(R)
                rowind[q]=i; values[q]=F[w+l,jj]; q+=1
            end
        end
        if f>w
            m=f-w
            if subroot[s]
                cb=Vector{T}(undef,m*(m+1)÷2); t=1
                for jj in w+1:f, ii in jj:f
                    cb[t]=F[ii,jj]; t+=1
                end
                cbstore[s]=cb
            else
                need=B.top+m*(m+1)÷2
                need>length(B.arena) && resize!(B.arena,max(need,2*length(B.arena)))
                offset=B.top+1; t=offset
                for jj in w+1:f, ii in jj:f
                    B.arena[t]=F[ii,jj]; t+=1
                end
                B.top=need; push!(B.stack,(s,offset))
            end
        end
    end
    nothing
end

# Independent subtrees for the workers (each a postorder segment first:s):
# the heaviest candidate is split into its children while it holds more
# than 1/(4·threads) of the work (w·f² per supernode).
function _chol_subtrees(nodes,rows,sparent,schildren,nt)
    ns=length(nodes); work=zeros(Float64,ns); count=ones(Int,ns)
    for s in 1:ns
        w=length(nodes[s]); f=w+length(rows[s]); work[s]+=Float64(w)*f*f
        p=sparent[s]; p!=0 && (work[p]+=work[s]; count[p]+=count[s])
    end
    cands=[s for s in 1:ns if sparent[s]==0]
    target=sum(work[c] for c in cands;init=0.0)/(4nt)
    while true
        k=argmax(c->work[c],cands)
        (work[k]<=target || isempty(schildren[k])) && break
        filter!(!=(k),cands); append!(cands,schildren[k])
    end
    sort!(cands;by=c->-work[c]),count
end

# Numeric phase of the supernodal factorization (see the header): returns L
# in CSC and the front sizes. With threads, independent subtrees are
# factored concurrently (each worker with its own buffers; the columns of L
# they write are disjoint), then the supernodes above them in postorder with
# the threaded dense kernels.
function _chol_supernodal_numeric(lp,li,lv,nodes,rows,schildren,sparent,n,threaded,::Type{T}) where T
    # CSC layout of L: column j of supernode a:b holds rows j:b, then rows[s].
    colptr=Vector{Int}(undef,n+1); colptr[1]=1
    for (s,node) in enumerate(nodes), j in node
        colptr[j+1]=colptr[j]+(last(node)-j+1)+length(rows[s])
    end
    nzl=colptr[n+1]-1; rowind=Vector{Int}(undef,nzl); values=Vector{T}(undef,nzl)
    fronts=[length(node)+length(rows[s]) for (s,node) in enumerate(nodes)]
    maxf=maximum(fronts;init=0); ns=length(nodes)
    # One packing workspace for the dense front kernels of the sequential
    # part (the shared pool keeps only a few entries and reallocated them
    # across mixed front sizes).
    maxw=maximum(length,nodes;init=0)
    ws=T<:NativeFloat && maxf>_MF_SMALL_FRONT ? GemmWorkspace(T,maxf,maxf,maxw;workers=threaded ? Threads.nthreads() : 1) : nothing
    subroot=falses(ns); cbstore=Vector{Vector{T}}(undef,ns); inside=falses(ns)
    nt=threaded ? Threads.nthreads() : 1
    if nt>1 && sum(f->Float64(f)^3,fronts;init=0.0)>=_CHOL_TREE_PARALLEL
        subs,count=_chol_subtrees(nodes,rows,sparent,schildren,nt)
        if length(subs)>1
            for s in subs
                subroot[s]=true; inside[s-count[s]+1:s].=true
            end
            next=Threads.Atomic{Int}(1)
            tasks=map(1:min(nt,length(subs))) do _
                Threads.@spawn begin
                    # `local`: the enclosing function also assigns these names,
                    # and a closure would otherwise share them across tasks.
                    local Bw=_CholBuffers(T,maxf,n)
                    while true
                        local k=Threads.atomic_add!(next,1); k>length(subs) && break
                        local r=subs[k]
                        _chol_fronts!(r-count[r]+1:r,Bw,lp,li,lv,nodes,rows,schildren,colptr,rowind,values,
                                      subroot,cbstore,false,nothing)
                    end
                end
            end
            failure=nothing
            for t in tasks
                try
                    wait(t)
                catch err
                    failure===nothing && (failure=err isa TaskFailedException ? err.task.result : err)
                end
            end
            failure===nothing || throw(failure)
        end
    end
    B=_CholBuffers(T,maxf,n)
    _chol_fronts!((s for s in 1:ns if !inside[s]),B,lp,li,lv,nodes,rows,schildren,colptr,rowind,values,
                  subroot,cbstore,threaded,ws)
    CSCMatrix(_Shared(),n,n,colptr,rowind,values),fronts
end
# Σ f³ of the fronts from which the subtrees are factored concurrently.
const _CHOL_TREE_PARALLEL=2.0e8

# Solve L·Lᴴ·x = x by supernodes (columns a:b of a supernode hold rows j:b,
# then the same rows R below b). Forward: the dense diagonal block, then the
# updates of all w columns are accumulated in y and scattered to x[R] once;
# backward: x[R] is gathered into y once. Width-1 nodes (simplicial factor,
# banded natural factor) take the direct column algorithm. The diagonal is
# real and positive: multiplying by the inverse of its real part keeps
# divisions (complex or real) off the dependency chain of the solve.
function _chol_solve!(x,y,L::CSCMatrix,nodes)
    colptr,rowind,values=L.colptr,L.rowind,L.values; T=eltype(x)
    @inbounds for node in nodes
        a,b=first(node),last(node); w=b-a+1; qa=colptr[a]; m=colptr[a+1]-qa-w
        if w==1
            # Single column: direct update, no gather/scatter pass; rows
            # forming one range (banded factors) are addressed directly.
            xj=x[a]*Base.inv(real(values[qa])); x[a]=xj; qe=colptr[a+1]-1
            if qe>qa && rowind[qe]-rowind[qa+1]==qe-qa-1
                d=rowind[qa+1]-qa-1
                @simd for q in qa+1:qe
                    x[q+d]=muladd(-values[q],xj,x[q+d])
                end
            else
                for q in qa+1:qe
                    x[rowind[q]]-=values[q]*xj
                end
            end
            continue
        end
        for j in a:b
            q=colptr[j]; xj=x[j]*Base.inv(real(values[q])); x[j]=xj
            for ii in 1:b-j
                x[j+ii]-=values[q+ii]*xj
            end
        end
        m>0 || continue
        for l in 1:m
            y[l]=zero(T)
        end
        for j in a:b
            o=colptr[j]+b-j; xj=x[j]
            @simd for l in 1:m
                y[l]=muladd(values[o+l],xj,y[l])
            end
        end
        for l in 1:m
            x[rowind[qa+w-1+l]]-=y[l]
        end
    end
    @inbounds for s in length(nodes):-1:1
        node=nodes[s]; a,b=first(node),last(node); w=b-a+1; qa=colptr[a]; m=colptr[a+1]-qa-w
        if w==1
            qe=colptr[a+1]-1
            acc=qe>qa && rowind[qe]-rowind[qa+1]==qe-qa-1 ? _cdot(values,qa,x,rowind[qa+1]-1,qe-qa) : _cdot_idx(values,x,rowind,qa+1,qe)
            x[a]=(x[a]-acc)*Base.inv(real(values[qa]))
            continue
        end
        for l in 1:m
            y[l]=x[rowind[qa+w-1+l]]
        end
        for j in b:-1:a
            q=colptr[j]; o=q+b-j; t=x[j]
            for ii in 1:b-j
                t-=conj(values[q+ii])*x[j+ii]
            end
            x[j]=(t-_cdot(values,o,y,0,m))*Base.inv(real(values[q]))
        end
    end
    x
end

# Factor of A from the factor L of P·A·Pᵀ (`p[new] = old`), when P only
# relabels the elimination tree (natural ordering with a postorder): P·L·Pᵀ is
# lower triangular, hence the Cholesky factor of A. Exact zeros (relaxed
# supernode entries) are dropped; two counting passes keep the rows sorted.
function _chol_unpermuted(L::CSCMatrix{T},p) where T
    n=size(L,1); colptr,rowind,values=L.colptr,L.rowind,L.values
    if p==1:n
        # Identity (the usual case: a natural-order tree is already postordered).
        any(iszero,values) || return L
        cp=Vector{Int}(undef,n+1); cp[1]=1; nz=count(!iszero,values)
        ri=Vector{Int}(undef,nz); vv=Vector{T}(undef,nz); t=0
        @inbounds for c in 1:n
            for q in colptr[c]:colptr[c+1]-1
                iszero(values[q]) && continue
                t+=1; ri[t]=rowind[q]; vv[t]=values[q]
            end
            cp[c+1]=t+1
        end
        return CSCMatrix(_Shared(),n,n,cp,ri,vv)
    end
    rp=zeros(Int,n+1); cp=zeros(Int,n+1)
    @inbounds for c in 1:n, q in colptr[c]:colptr[c+1]-1
        iszero(values[q]) && continue
        rp[p[rowind[q]]+1]+=1; cp[p[c]+1]+=1
    end
    rp[1]=cp[1]=1
    @inbounds for k in 1:n
        rp[k+1]+=rp[k]; cp[k+1]+=cp[k]
    end
    nz=rp[n+1]-1; rc=Vector{Int}(undef,nz); rv=Vector{T}(undef,nz); next=rp[1:n]
    @inbounds for c in 1:n, q in colptr[c]:colptr[c+1]-1
        iszero(values[q]) && continue
        i=p[rowind[q]]; t=next[i]; next[i]+=1; rc[t]=p[c]; rv[t]=values[q]
    end
    ri=Vector{Int}(undef,nz); vv=Vector{T}(undef,nz); next=cp[1:n]
    @inbounds for i in 1:n, t in rp[i]:rp[i+1]-1
        j=rc[t]; q=next[j]; next[j]+=1; ri[q]=i; vv[q]=rv[t]
    end
    CSCMatrix(_Shared(),n,n,cp,ri,vv)
end

# Fundamental supernodes of an exact Cholesky factor with sorted columns:
# j+1 joins the supernode of j when it is the parent of j (second row of
# column j) and column j has one more entry, so that column j+1 is column j
# without j (the pattern of j below j is contained in that of its parent).
function _chol_fundamental_nodes(L::CSCMatrix)
    n=size(L,1); colptr,rowind=L.colptr,L.rowind; nodes=UnitRange{Int}[]; a=1
    @inbounds for j in 1:n-1
        c=colptr[j+1]-colptr[j]
        (c>=2 && rowind[colptr[j]+1]==j+1 && c==colptr[j+2]-colptr[j+1]+1) && continue
        push!(nodes,a:j); a=j+1
    end
    n>0 && push!(nodes,a:n)
    nodes
end

# Σ conj(v[vo+l])·x[xo+l], l = 1:len, and Σ conj(v[q])·x[rowind[q]], q = q0:q1:
# four accumulators break the latency chain of the short dot products of the
# solve (one accumulator ran a 30² Laplacian solve at ×1.6 of CHOLMOD); real
# contiguous ones vectorize.
@inline function _cdot(v,vo,x,xo,len)
    T=promote_type(eltype(v),eltype(x))
    if T<:Real
        acc=zero(T)
        @inbounds @simd for l in 1:len
            acc=muladd(v[vo+l],x[xo+l],acc)
        end
        return acc
    end
    a1=a2=a3=a4=zero(T); l=1
    @inbounds while l+3<=len
        a1=muladd(conj(v[vo+l]),x[xo+l],a1); a2=muladd(conj(v[vo+l+1]),x[xo+l+1],a2)
        a3=muladd(conj(v[vo+l+2]),x[xo+l+2],a3); a4=muladd(conj(v[vo+l+3]),x[xo+l+3],a4)
        l+=4
    end
    @inbounds while l<=len
        a1=muladd(conj(v[vo+l]),x[xo+l],a1); l+=1
    end
    (a1+a2)+(a3+a4)
end
@inline function _cdot_idx(v,x,rowind,q0,q1)
    T=promote_type(eltype(v),eltype(x))
    a1=a2=a3=a4=zero(T); q=q0
    @inbounds while q+3<=q1
        a1=muladd(conj(v[q]),x[rowind[q]],a1); a2=muladd(conj(v[q+1]),x[rowind[q+1]],a2)
        a3=muladd(conj(v[q+2]),x[rowind[q+2]],a3); a4=muladd(conj(v[q+3]),x[rowind[q+3]],a4)
        q+=4
    end
    @inbounds while q<=q1
        a1=muladd(conj(v[q]),x[rowind[q]],a1); q+=1
    end
    (a1+a2)+(a3+a4)
end
