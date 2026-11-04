# SPDX-License-Identifier: Apache-2.0
# Multifrontal sparse Householder QR (as in SPQR), A[:,p] = Q·[R; 0]:
#   1. column order q (natural, or AMD of AᵀA), column elimination tree (the
#      elimination tree of AᵀA, without forming it), chains of columns with a
#      single child grouped into fronts (fundamental supernodes);
#   2. fronts in column order (children before parents): the rows whose
#      leftmost column falls in the front, and the children's contribution
#      blocks, form a dense matrix over the union of their columns; Householder
#      QR of its pivot columns gives those rows of R and the reflectors, the
#      rows left below form the contribution block of the parent.
# Rows keep the label of an original row through the eliminations, so Q is a
# product of sparse reflectors on the rows of A; a last permutation (as
# transpositions, reflectors with τ = 1) brings the rows of R to the top.

# Column elimination tree of A[:,q] (CSparse cs_etree with ata).
function _qr_column_etree(C::CSCMatrix,q)
    m,n=size(C); parent=zeros(Int,n); ancestor=zeros(Int,n); prev=zeros(Int,m)
    @inbounds for k in 1:n, t in C.colptr[q[k]]:C.colptr[q[k]+1]-1
        iszero(C.values[t]) && continue
        i=Int(C.rowind[t]); r=prev[i]
        while r!=0 && r<k
            nxt=ancestor[r]; ancestor[r]=k
            nxt==0 && (parent[r]=k)
            r=nxt
        end
        prev[i]=k
    end
    parent
end

# Panel width of the front QR: 16 (8, 32 and 64 were not faster on a
# 4000×1000 random matrix and 2400×1600 grid least squares).
const _QR_NB=16
const _QR_BLOCKED_FRONT=2048   # nr·w from which the fronts use blocked QR
# Row counts of R (Gilbert, Ng and Peyton for AᵀA, CSparse cs_counts with
# ata): the "lower pattern" of column j is the union of the rows of A whose
# leftmost column is j.
function _qr_row_counts(C::CSCMatrix,q,qinv,parent)
    m,n=size(C); leftmost=fill(n+1,m)
    @inbounds for j in 1:n, t in C.colptr[j]:C.colptr[j+1]-1
        iszero(C.values[t]) && continue
        i=Int(C.rowind[t]); leftmost[i]=min(leftmost[i],qinv[j])
    end
    rc=zeros(Int,m+1)
    @inbounds for j in 1:n, t in C.colptr[j]:C.colptr[j+1]-1
        iszero(C.values[t]) || (rc[Int(C.rowind[t])+1]+=1)
    end
    lp=zeros(Int,n+1)
    @inbounds for i in 1:m
        leftmost[i]<=n && (lp[leftmost[i]+1]+=rc[i+1])
    end
    lp[1]=1
    for j in 1:n
        lp[j+1]+=lp[j]
    end
    li=Vector{Int}(undef,lp[n+1]-1); nxt=lp[1:n]
    @inbounds for j in 1:n, t in C.colptr[j]:C.colptr[j+1]-1
        iszero(C.values[t]) && continue
        i=Int(C.rowind[t]); l=leftmost[i]; x=nxt[l]; nxt[l]+=1; li[x]=qinv[j]
    end
    _chol_counts(lp,li,parent,_postorder(parent),n)
end
# Reflector of front column k (rows k:r, v[1] = 1), in row labels.
function _qr_record!(indices,vectors,tau,F,FR,k,r,t)
    T=eltype(F); c=1
    @inbounds for i in k+1:r
        iszero(F[i,k]) || (c+=1)
    end
    idx=Vector{Int}(undef,c); val=Vector{T}(undef,c); idx[1]=FR[k]; val[1]=one(T); l=1
    @inbounds for i in k+1:r
        x=F[i,k]; iszero(x) && continue
        l+=1; idx[l]=FR[i]; val[l]=x
    end
    push!(indices,idx); push!(vectors,val); push!(tau,t)
end
# Column-front structure shared by the multifrontal QR and the unsymmetric
# LU: column etree of A[:,q], CHOLMOD's relaxed supernodes from the row
# counts of R (column counts of the Cholesky factor of AᵀA), the supernodal
# tree, and the rows of A (as CSR in column positions) assigned to the front
# of their leftmost column.
function _colfront_symbolic(C::CSCMatrix{T},q,zrelax=(0.8,0.1,0.05),maxwidth=typemax(Int)) where T
    m,n=size(C)
    qinv=invperm(q)
    parent=_qr_column_etree(C,q)
    # Fronts: CHOLMOD's relaxed supernodes of the column etree, from the row
    # counts of R (column counts of the Cholesky factor of AᵀA).
    counts=_qr_row_counts(C,q,qinv,parent)
    nodes=_chol_supernodes(parent,counts,n,maxwidth,zrelax)
    starts=[first(node) for node in nodes]
    nf=length(starts); push!(starts,n+1)
    owner=Vector{Int}(undef,n)
    for s in 1:nf, j in starts[s]:starts[s+1]-1
        owner[j]=s
    end
    children=[Int[] for _ in 1:nf]
    for s in 1:nf
        b=starts[s+1]-1; parent[b]!=0 && push!(children[owner[parent[b]]],s)
    end
    # Rows of A by their leftmost column (in the order q), as CSR in
    # positions.
    rc=zeros(Int,m+1); leftmost=fill(n+1,m)
    @inbounds for j in 1:n, t in C.colptr[j]:C.colptr[j+1]-1
        iszero(C.values[t]) && continue
        i=Int(C.rowind[t]); rc[i+1]+=1; leftmost[i]=min(leftmost[i],qinv[j])
    end
    rc[1]=1
    for i in 1:m
        rc[i+1]+=rc[i]
    end
    rj=Vector{Int}(undef,rc[m+1]-1); rv=Vector{T}(undef,length(rj)); nxt=rc[1:m]
    @inbounds for j in 1:n, t in C.colptr[j]:C.colptr[j+1]-1
        v=C.values[t]; iszero(v) && continue
        i=Int(C.rowind[t]); x=nxt[i]; nxt[i]+=1; rj[x]=qinv[j]; rv[x]=v
    end
    frows=[Int[] for _ in 1:nf]
    for i in 1:m
        leftmost[i]<=n && push!(frows[owner[leftmost[i]]],i)
    end
    qinv,parent,starts,nf,children,rc,rj,rv,leftmost,frows
end
function _qr_multifrontal(A::NativeSparse,q)
    C=_as_csc(A); m,n=size(C); T=eltype(C); R=real(T)
    qinv,parent,starts,nf,children,rc,rj,rv,leftmost,frows=_colfront_symbolic(C,q)
    # Contribution blocks kept per front until the parent takes them.
    cbrows=Vector{Vector{Int}}(undef,nf); cbcols=Vector{Vector{Int}}(undef,nf); cbvals=Vector{Matrix{T}}(undef,nf)
    cpos=zeros(Int,n); mark=zeros(Int,n)
    indices=Vector{Vector{Int}}(); vectors=Vector{Vector{T}}(); tau=R[]
    Rrow=Vector{Int}(undef,n)                  # label of the row holding R[k,:]
    Rcols=Vector{Vector{Int}}(undef,n); Rvals=Vector{Vector{T}}(undef,n)
    # Workspace of the blocked panels (V, T, Gram, W), grown as needed.
    Fb=Matrix{T}(undef,0,0); ftau=T[]; Vb=Matrix{T}(undef,0,_QR_NB); Tb=Matrix{T}(undef,_QR_NB,_QR_NB); Gb=similar(Tb); Wb=Matrix{T}(undef,_QR_NB,0)
    @inbounds for s in 1:nf
        a,b=starts[s],starts[s+1]-1; w=b-a+1
        # Front columns: the pivots, then the others in increasing order.
        FC=collect(a:b)
        for j in a:b
            mark[j]=s
        end
        for i in frows[s], x in rc[i]:rc[i+1]-1
            j=rj[x]; mark[j]==s || (mark[j]=s; push!(FC,j))
        end
        for c in children[s], j in cbcols[c]
            mark[j]==s || (mark[j]=s; push!(FC,j))
        end
        sort!(view(FC,w+1:length(FC))); nc=length(FC)
        for (l,j) in enumerate(FC)
            cpos[j]=l
        end
        # Rows in staircase order (SPQR): by leftmost column in the front, so
        # that the reflector of column k only touches the rows already
        # started.
        nown=length(frows[s]); nr=nown
        for c in children[s]
            nr+=length(cbrows[c])
        end
        key=Vector{Int}(undef,nr); FR0=Vector{Int}(undef,nr)
        for (l,i) in enumerate(frows[s])
            key[l]=cpos[leftmost[i]]; FR0[l]=i
        end
        off=nown
        for c in children[s]
            M=cbvals[c]; cols=cbcols[c]
            for (l,i) in enumerate(cbrows[c])
                # First nonzero column of the row (rows the child's reflectors
                # did not reach keep their sparsity).
                jj=1
                while jj<length(cols) && iszero(M[l,jj])
                    jj+=1
                end
                key[off+l]=cpos[cols[jj]]; FR0[off+l]=i
            end
            off+=length(cbrows[c])
        end
        order=sortperm(key); rowat=invperm(order); FR=FR0[order]; lm=key[order]
        # Front in a reused buffer (grown as needed), cleared over nr×nc.
        if nr>size(Fb,1) || nc>size(Fb,2)
            Fb=Matrix{T}(undef,max(nr,size(Fb,1)+size(Fb,1)÷2),max(nc,size(Fb,2)+size(Fb,2)÷2))
        end
        F=view(Fb,1:nr,1:nc)
        for j in 1:nc
            @simd for i in 1:nr
                F[i,j]=zero(T)
            end
        end
        for (l,i) in enumerate(frows[s]), x in rc[i]:rc[i+1]-1
            F[rowat[l],cpos[rj[x]]]+=rv[x]
        end
        off=nown
        for c in children[s]
            M=cbvals[c]; cols=cbcols[c]
            for jj in eachindex(cols)
                lj=cpos[cols[jj]]
                for ii in axes(M,1)
                    F[rowat[off+ii],lj]+=M[ii,jj]
                end
            end
            off+=size(M,1)
            cbrows[c]=Int[]; cbcols[c]=Int[]; cbvals[c]=Matrix{T}(undef,0,0)
        end
        # Householder QR of the whole front (as SPQR), in panels of 16 columns
        # over the staircase rows: blocked (WY, this package's geqrf/ormqr)
        # for a large panel, by columns otherwise. Its first w rows are rows
        # of R, the next ones the contribution block (upper trapezoid), the
        # rest vanish. Without the staircase this cost ×2.4; with it, the
        # shorter reflectors of the parents make it 1.2–1.8× faster than
        # factoring only the pivot columns.
        kmax=min(nr,w); stair=0
        kfac=min(nr,nc)
        nbq=_QR_NB
        for k0 in 1:nbq:kfac
            k1=min(k0+nbq-1,kfac)
            while stair<nr && lm[stair+1]<=k1
                stair+=1
            end
            r1=max(stair,k1)
            if T<:NativeFloat && (r1-k0+1)*(nc-k0+1)>=_QR_BLOCKED_FRONT
                # In place (the dense QR's own steps): reflectors of the panel
                # on rows 1:r1, then V, T and one blocked update of the other
                # columns (larft/larfb); qr_factor + apply_q! copied the panel
                # and rebuilt V and T.
                Fv=view(F,1:r1,:); p=k1-k0+1
                length(ftau)<nc && resize!(ftau,nc)
                for j in k0:k1
                    ftau[j]=zero(T); τ=_qr_reflector!(Fv,ftau,j); _qr_update_columns!(Fv,j,τ,j+1:k1)
                end
                if nc>k1
                    size(Vb,1)<r1 && (Vb=Matrix{T}(undef,max(r1,2size(Vb,1)),nbq))
                    size(Wb,2)<nc && (Wb=Matrix{T}(undef,nbq,max(nc,2size(Wb,2))))
                    Vv=_qr_block_v!(Vb,Fv,k0,p)
                    Tv=_qr_tfactor!(view(Tb,1:p,1:p),Vv,ftau,k0,view(Gb,1:p,1:p);threaded=false)
                    _qr_block_apply!(view(Fv,k0:r1,k1+1:nc),Vv,Tv,Wb,true,true;threaded=false)
                end
                for k in k0:k1
                    _qr_record!(indices,vectors,tau,F,FR,k,r1,R(real(ftau[k])))
                end
            else
                for k in k0:k1
                    rk=k
                    for i in k+1:r1
                        lm[i]<=k && (rk=i)
                    end
                    v=view(F,k:rk,k); beta,t=_householder!(v)
                    if !iszero(t)
                        for j in k+1:nc
                            s2=zero(T)
                            @simd for i in k:rk
                                s2=muladd(conj(F[i,k]),F[i,j],s2)
                            end
                            s2*=t
                            iszero(s2) && continue
                            @simd for i in k:rk
                                F[i,j]=muladd(-F[i,k],s2,F[i,j])
                            end
                        end
                    end
                    F[k,k]=beta
                    _qr_record!(indices,vectors,tau,F,FR,k,rk,t)
                end
            end
        end
        for k in 1:kmax
            # Row k of the front is row a+k-1 of R.
            piv=a+k-1; Rrow[piv]=FR[k]
            cs=Int[piv]; vs=T[F[k,k]]
            for j in k+1:nc
                x=F[k,j]; iszero(x) && continue
                push!(cs,FC[j]); push!(vs,x)
            end
            Rcols[piv]=cs; Rvals[piv]=vs
        end
        # Pivots without a row left: zero rows of R (rank deficiency).
        for k in kmax+1:w
            piv=a+k-1; Rrow[piv]=0; Rcols[piv]=Int[]; Rvals[piv]=T[]
        end
        if kfac>w
            # Whole front factored: the contribution block is the upper
            # trapezoid of rows w+1:kfac (the rows below vanished).
            cb=F[w+1:kfac,w+1:nc]
            for jj in axes(cb,2), ii in jj+1:size(cb,1)
                cb[ii,jj]=zero(T)
            end
            cbrows[s]=FR[w+1:kfac]; cbcols[s]=FC[w+1:nc]; cbvals[s]=cb
        else
            cbrows[s]=Int[]; cbcols[s]=Int[]; cbvals[s]=Matrix{T}(undef,0,0)
        end
    end
    # Row permutation that brings the R rows to positions 1:n: unused labels
    # take the pivots without a row, then the positions n+1:m.
    used=falses(m)
    for k in 1:n
        Rrow[k]!=0 && (used[Rrow[k]]=true)
    end
    spare=[i for i in 1:m if !used[i]]; si=1
    target=Vector{Int}(undef,m)        # label placed at position k
    for k in 1:n
        if Rrow[k]==0
            target[k]=spare[si]; si+=1
        else
            target[k]=Rrow[k]
        end
    end
    for k in n+1:m
        target[k]=spare[si]; si+=1
    end
    # Transpositions: after them, position k holds the row labelled target[k].
    where=collect(1:m); at=collect(1:m)          # where[label] = position, at[position] = label
    for k in 1:m
        j=where[target[k]]; j==k && continue
        push!(indices,[k,j]); push!(vectors,T[one(T),-one(T)]); push!(tau,one(R))
        lk=at[k]; at[k],at[j]=target[k],lk; where[target[k]]=k; where[lk]=j
    end
    # R as an m×n CSR, rows 1:n in pivot order (each row is already sorted:
    # its pivot, the later pivots of its front, then the other columns).
    rp=Vector{Int}(undef,m+1); rp[1]=1
    for k in 1:m
        rp[k+1]=rp[k]+(k<=n ? length(Rcols[k]) : 0)
    end
    ci=Vector{Int}(undef,rp[m+1]-1); cv=Vector{T}(undef,length(ci))
    for k in 1:n
        copyto!(ci,rp[k],Rcols[k],1,length(Rcols[k])); copyto!(cv,rp[k],Rvals[k],1,length(Rvals[k]))
    end
    SparseQRFactor(CSRMatrix(_Shared(),m,n,rp,ci,cv),indices,vectors,tau,collect(q),(m,n))
end
