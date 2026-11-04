# SPDX-License-Identifier: Apache-2.0
# Experimental, not wired in (kept on purpose for a later round): exact on
# the tests, less fill than the column-front engine, but 3× slower on random
# unsymmetric matrices (reports/SPARSE_LU_LDLT_ETAPES4_5_20261104.md).
# `_UMF_RESTART` selects in-place front restarts (true) or elements (false).
# Unsymmetric-pattern multifrontal LU (after UMFPACK's method, Davis and
# Duff; simplified): A[p,q] = L·U for a matrix with an unsymmetric pattern.
#
# Contribution blocks are elements: dense blocks over their own row and
# column labels, found through per-row and per-column tuple lists. A front
# starts with the next column of the column preorder: every row with an entry
# in that column enters the front, assembled whole (its original entries and
# its rows of the elements), so a front row is always a complete row of the
# active matrix and its columns join the front. The pivot row is chosen by
# threshold partial pivoting, the shortest row among those within τ of the
# largest entry. The front then grows along the chain while the next column of
# the preorder is already one of its columns and its size stays within a
# growth bound: pivots are swapped to the leading rows and columns, so rows
# and columns added later never need earlier updates (their entries in the
# earlier pivot columns and rows are zero). Updates are delayed by blocks: the
# pivot column and the pivot row are brought up to date on demand, the rest of
# the front by one GEMM per block. At the end of a front, the rows and columns
# left (zero ones dropped) become a new element.

mutable struct _UmfElement{T}
    rows::Vector{Int}
    cols::Vector{Int}
    V::Matrix{T}
    rowlive::BitVector
    collive::BitVector
    nlive::Int                 # live rows
end

const _UMF_BLOCK=32            # pivots per delayed update
const _UMF_RESTART=Ref(true)
const _UMF_GROWTH=4.0          # front growth allowed along a chain (2 to 8 measured)

# W is passed as an argument: the frontal matrix is regrown (rebound) in
# _mf_lu_umf, so closures over it were Core.Box.
function _umf_any_nonzero(W,rows,cols)
    @inbounds for j in cols, i in rows
        iszero(W[i,j]) || return true
    end
    false
end
function _mf_lu_umf(C::CSCMatrix{T},q,pivot_threshold) where T
    n=size(C,1); R=real(T); initial=maximum(abs,C.values;init=zero(R))
    # Rows of A (CSR), stored zeros skipped.
    rc=zeros(Int,n+1)
    @inbounds for j in 1:n, t in C.colptr[j]:C.colptr[j+1]-1
        iszero(C.values[t]) || (rc[Int(C.rowind[t])+1]+=1)
    end
    rc[1]=1
    for i in 1:n
        rc[i+1]+=rc[i]
    end
    rj=Vector{Int}(undef,rc[n+1]-1); rv=Vector{T}(undef,length(rj)); nxt=rc[1:n]
    @inbounds for j in 1:n, t in C.colptr[j]:C.colptr[j+1]-1
        v=C.values[t]; iszero(v) && continue
        i=Int(C.rowind[t]); x=nxt[i]; nxt[i]+=1; rj[x]=j; rv[x]=v
    end
    any(i->rc[i+1]==rc[i],1:n) && throw(SingularException(1))   # an empty row
    elements=_UmfElement{T}[]
    rowtup=[Tuple{Int,Int}[] for _ in 1:n]     # (element, local row)
    coltup=[Tuple{Int,Int}[] for _ in 1:n]     # (element, local column)
    origused=falses(n)                         # original row assembled
    rowdone=falses(n); coldone=falses(n)       # pivoted
    rpos=zeros(Int,n); cpos=zeros(Int,n)       # positions in the current front
    FR=Int[]; FC=Int[]; rlen=Int[]
    W=Matrix{T}(undef,0,0)
    prow=Vector{Int}(undef,n); pcol=Vector{Int}(undef,n); t=0
    lptr=Vector{Int}(undef,n+1); lptr[1]=1; lidx=Int[]; lval=T[]
    uptr=Vector{Int}(undef,n+1); uptr[1]=1; uidx=Int[]; uval=T[]
    pieces=UnitRange{Int}[]; shapes=Tuple{Int,Int}[]; growth=initial
    qk=1                                       # next column of the preorder
    qrank=invperm(q)
    @inbounds while t<n
        while qk<=n && coldone[q[qk]]
            qk+=1
        end
        qk>n && throw(SingularException(1))
        # New front.
        empty!(FR); empty!(FC); empty!(rlen); nr=0; nc=0; k=0; k0=1
        caprows=0; capcols=0; first=t+1
        c=q[qk]
        while true
            # Bring column c in: every active row with an entry in it.
            if cpos[c]==0
                nc+=1; push!(FC,c); cpos[c]=nc
                nc>size(W,2) && (W=_umf_grow(W,size(W,1),nc))
                for i in 1:nr
                    W[i,nc]=zero(T)
                end
            end
            newrows=Int[]
            for x in C.colptr[c]:C.colptr[c+1]-1
                i=Int(C.rowind[x]); iszero(C.values[x]) && continue
                (rowdone[i] || rpos[i]!=0) && continue
                rpos[i]=-1; push!(newrows,i)
            end
            for (e,lc) in coltup[c]
                E=elements[e]; E.collive[lc] || continue
                for (li,i) in enumerate(E.rows)
                    (E.rowlive[li] && rpos[i]==0 && !iszero(E.V[li,lc])) || continue
                    rpos[i]=-1; push!(newrows,i)
                end
            end
            # Chain test: start a new front if the rows (and their columns)
            # would outgrow the bound fixed when the front began.
            if k>0 && (nr+length(newrows)>caprows)
                for i in newrows
                    rpos[i]=0
                end
                _UMF_RESTART[] || break
                # Restart in place: emit the pivots, move the rest to the
                # leading block and go on (no element, no reassembly).
                t,nr,nc=_umf_restart!(W,nr,nc,k,k0,FR,FC,rlen,rpos,cpos,prow,pcol,lptr,lidx,lval,uptr,uidx,uval,t)
                push!(pieces,first:t); push!(shapes,(nr+k,nc+k)); first=t+1
                k=0; k0=1; caprows=max(16,ceil(Int,_UMF_GROWTH*(nr+length(newrows)))); capcols=max(16,ceil(Int,_UMF_GROWTH*nc))
                continue
            end
            for i in newrows
                nr+=1; push!(FR,i); rpos[i]=nr; push!(rlen,0)
                nr>size(W,1) && (W=_umf_grow(W,nr,size(W,2)))
                for j in 1:nc
                    W[nr,j]=zero(T)
                end
                cnt=0
                # Original entries of row i.
                if !origused[i]
                    origused[i]=true
                    for x in rc[i]:rc[i+1]-1
                        j=rj[x]; coldone[j] && continue
                        if cpos[j]==0
                            nc+=1; push!(FC,j); cpos[j]=nc
                            nc>size(W,2) && (W=_umf_grow(W,size(W,1),nc))
                            for ii in 1:nr
                                W[ii,nc]=zero(T)
                            end
                        end
                        W[nr,cpos[j]]+=rv[x]; cnt+=1
                    end
                end
                # Rows of the elements holding row i.
                for (e,li) in rowtup[i]
                    E=elements[e]; E.rowlive[li] || continue
                    for (lc,j) in enumerate(E.cols)
                        E.collive[lc] || continue
                        v=E.V[li,lc]; iszero(v) && continue
                        if cpos[j]==0
                            nc+=1; push!(FC,j); cpos[j]=nc
                            nc>size(W,2) && (W=_umf_grow(W,size(W,1),nc))
                            for ii in 1:nr
                                W[ii,nc]=zero(T)
                            end
                        end
                        W[nr,cpos[j]]+=v; cnt+=1
                    end
                    E.rowlive[li]=false; E.nlive-=1
                    E.nlive==0 && (E.V=Matrix{T}(undef,0,0))
                end
                rlen[nr]=cnt     # entries assembled (an upper bound of the length)
            end
            if k==0
                caprows=max(16,ceil(Int,_UMF_GROWTH*nr)); capcols=max(16,ceil(Int,_UMF_GROWTH*nc))
            end
            # Pivot column c to position k+1, brought up to date.
            jc=cpos[c]
            if jc!=k+1
                _umf_swapcols!(W,nr,jc,k+1); FC[jc],FC[k+1]=FC[k+1],FC[jc]; cpos[FC[jc]]=jc; cpos[FC[k+1]]=k+1
            end
            _umf_colupdate!(W,nr,k+1,k0,k)
            # Pivot row: threshold, then the shortest row.
            amax=zero(R)
            for i in k+1:nr
                amax=max(amax,abs(W[i,k+1]))
            end
            (iszero(amax) || !isfinite(amax)) && throw(SingularException(c))
            r=0; best=typemax(Int); bv=zero(R)
            for i in k+1:nr
                v=abs(W[i,k+1]); (iszero(v) || v<pivot_threshold*amax) && continue
                if rlen[i]<best || (rlen[i]==best && v>bv)
                    r=i; best=rlen[i]; bv=v
                end
            end
            if r!=k+1
                _umf_swaprows!(W,nc,r,k+1); FR[r],FR[k+1]=FR[k+1],FR[r]; rpos[FR[r]]=r; rpos[FR[k+1]]=k+1
                rlen[r],rlen[k+1]=rlen[k+1],rlen[r]
            end
            k+=1
            # Pivot row brought up to date (columns k+1:nc), then the
            # multipliers.
            _umf_rowupdate!(W,nc,k,k0,k-1)
            ip=Base.inv(W[k,k])
            @simd for i in k+1:nr
                W[i,k]*=ip
            end
            lk=rlen[k]
            for i in k+1:nr
                iszero(W[i,k]) || (rlen[i]=min(nc-k,rlen[i]+lk-2))
            end
            coldone[c]=true; rowdone[FR[k]]=true
            for (e,lc) in coltup[c]
                elements[e].collive[lc]=false
            end
            # Delayed update of the block.
            if k-k0+1>=_UMF_BLOCK
                _umf_blockupdate!(W,nr,nc,k0,k); k0=k+1
            end
            # Next pivot column: the column of the front that comes first in
            # the preorder (UMFPACK picks it among the front's columns), while
            # the front may still grow.
            (k>=nr || k>=nc) && break
            if nc>capcols
                _UMF_RESTART[] || break
                t,nr,nc=_umf_restart!(W,nr,nc,k,k0,FR,FC,rlen,rpos,cpos,prow,pcol,lptr,lidx,lval,uptr,uidx,uval,t)
                push!(pieces,first:t); push!(shapes,(nr+k,nc+k)); first=t+1
                k=0; k0=1; caprows=max(16,ceil(Int,_UMF_GROWTH*nr)); capcols=max(16,ceil(Int,_UMF_GROWTH*nc))
                (nr==0 || nc==0) && break
            end
            c=0; best=typemax(Int)
            for j in k+1:nc
                r2=qrank[FC[j]]; r2<best && (best=r2; c=FC[j])
            end
            c==0 && break
        end
        # Close the front: pending updates, factor entries, new element.
        k0<=k && _umf_blockupdate!(W,nr,nc,k0,k)
        t=_umf_emit!(W,nr,nc,k,FR,FC,prow,pcol,lptr,lidx,lval,uptr,uidx,uval,t)
        t>=first && (push!(pieces,first:t); push!(shapes,(nr,nc)))
        # Contribution block without its zero rows and columns.
        keepr=Int[]; keepc=Int[]
        for i in k+1:nr
            _umf_any_nonzero(W,i:i,k+1:nc) && push!(keepr,i)
        end
        for j in k+1:nc
            _umf_any_nonzero(W,keepr,j:j) && push!(keepc,j)
        end
        if !isempty(keepr) && !isempty(keepc)
            e=length(elements)+1
            E=_UmfElement{T}(FR[keepr],FC[keepc],W[keepr,keepc],trues(length(keepr)),trues(length(keepc)),length(keepr))
            push!(elements,E)
            for (li,i) in enumerate(E.rows)
                push!(rowtup[i],(e,li))
            end
            for (lc,j) in enumerate(E.cols)
                push!(coltup[j],(e,lc))
            end
        end
        # Rows left with nothing are singular.
        for i in k+1:nr
            _umf_any_nonzero(W,i:i,k+1:nc) || throw(SingularException(FR[i]))
        end
        for i in FR
            rpos[i]=0
        end
        for j in FC
            cpos[j]=0
        end
    end
    # Growth from the factors (U holds the largest entries of the fronts
    # but the contribution blocks, whose entries end in later factors).
    growth=max(growth,maximum(abs,uval;init=zero(R)),maximum(abs,lval;init=zero(R)))
    isfinite(growth) || throw(ArgumentError("multifrontal LU arithmetic overflow"))
    L,U=_lu_assemble(n,prow,pcol,lptr,lidx,lval,length(lidx),uptr,uidx,uval,length(uidx),T)
    parents=[g+1<=length(pieces) ? g+1 : 0 for g in eachindex(pieces)]   # a chain of fronts
    MultifrontalLUFactor(L,U,prow,pcol,parents,pieces,shapes,R(pivot_threshold),
                         iszero(initial) ? one(R) : growth/initial)
end

function _umf_grow(W::Matrix{T},nr,nc) where T
    W2=Matrix{T}(undef,max(nr,size(W,1)+size(W,1)÷2,16),max(nc,size(W,2)+size(W,2)÷2,16))
    copyto!(view(W2,1:size(W,1),1:size(W,2)),W)
    W2
end
@inline function _umf_swapcols!(W,nr,a,b)
    @inbounds for i in 1:nr
        W[i,a],W[i,b]=W[i,b],W[i,a]
    end
end
@inline function _umf_swaprows!(W,nc,a,b)
    @inbounds for j in 1:nc
        W[a,j],W[b,j]=W[b,j],W[a,j]
    end
end
# Rows k1+1:nr of column j receive the pending pivots k0..k1 (the entries of
# the pivot rows are final: each pivot row was brought up to date when chosen).
function _umf_colupdate!(W,nr,j,k0,k1)
    @inbounds for p in k0:k1
        u=W[p,j]; iszero(u) && continue
        @simd for i in k1+1:nr
            W[i,j]=muladd(-W[i,p],u,W[i,j])
        end
    end
end
# Row r receives the pending pivots k0..k1 over its columns r+1:nc (the pivot
# rows k0..k1 are final there).
function _umf_rowupdate!(W,nc,r,k0,k1)
    @inbounds for p in k0:k1
        l=W[r,p]; iszero(l) && continue
        for j in r+1:nc
            W[r,j]=muladd(-l,W[p,j],W[r,j])
        end
    end
end
# Rows k1+1:nr, columns k1+1:nc receive the pivots k0..k1: one GEMM
# (columns k0..k1 of L, rows k0..k1 of U).
function _umf_blockupdate!(W::Matrix{T},nr,nc,k0,k1) where T
    (nr>k1 && nc>k1) || return
    if T<:NativeFloat && (nr-k1)*(nc-k1)*(k1-k0+1)>=32768
        gemm!(view(W,k1+1:nr,k1+1:nc),view(W,k1+1:nr,k0:k1),view(W,k0:k1,k1+1:nc),-one(T),one(T);threaded=false)
    else
        @inbounds for j in k1+1:nc, p in k0:k1
            u=W[p,j]; iszero(u) && continue
            @simd for i in k1+1:nr
                W[i,j]=muladd(-W[i,p],u,W[i,j])
            end
        end
    end
end

# L columns and U rows of the front pivots 1:k (labels); returns the new t.
function _umf_emit!(W,nr,nc,k,FR,FC,prow,pcol,lptr,lidx,lval,uptr,uidx,uval,t)
    @inbounds for kk in 1:k
        t+=1; prow[t]=FR[kk]; pcol[t]=FC[kk]
        for i in kk+1:nr
            v=W[i,kk]; iszero(v) && continue
            push!(lidx,FR[i]); push!(lval,v)
        end
        lptr[t+1]=length(lidx)+1
        for j in kk:nc
            v=W[kk,j]; (iszero(v) && j!=kk) && continue
            push!(uidx,FC[j]); push!(uval,v)
        end
        uptr[t+1]=length(uidx)+1
    end
    t
end
# Emit the pivots 1:k of the front, then move rows k+1:nr × columns k+1:nc to
# the leading block (columns without a nonzero are dropped: no front row has
# them). Returns (t, nr, nc).
function _umf_restart!(W::Matrix{T},nr,nc,k,k0,FR,FC,rlen,rpos,cpos,prow,pcol,lptr,lidx,lval,uptr,uidx,uval,t) where T
    k0<=k && _umf_blockupdate!(W,nr,nc,k0,k)
    t=_umf_emit!(W,nr,nc,k,FR,FC,prow,pcol,lptr,lidx,lval,uptr,uidx,uval,t)
    for kk in 1:k
        rpos[FR[kk]]=0; cpos[FC[kk]]=0
    end
    mr=nr-k; mc=0
    @inbounds for j in k+1:nc
        nz=false
        for i in k+1:nr
            iszero(W[i,j]) || (nz=true; break)
        end
        if !nz
            cpos[FC[j]]=0; continue
        end
        mc+=1
        for i in 1:mr
            W[i,mc]=W[k+i,j]
        end
        FC[mc]=FC[j]; cpos[FC[mc]]=mc
    end
    @inbounds for i in 1:mr
        FR[i]=FR[k+i]; rpos[FR[i]]=i; rlen[i]=rlen[k+i]
    end
    resize!(FR,mr); resize!(FC,mc); resize!(rlen,mr)
    t,mr,mc
end
