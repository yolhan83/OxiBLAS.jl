# SPDX-License-Identifier: Apache-2.0
# Fill-reducing ordering and symbolic analysis for the sparse direct solvers:
# approximate minimum degree (AMD, Amestoy–Davis–Duff, as in CSparse's
# cs_amd: quotient graph, element absorption, approximate external degrees,
# mass elimination by supernode detection, dense-row handling), elimination
# tree and postorder (cs_etree, cs_post), and the column structure of the
# Cholesky factor in O(nnz(L)).
#
# The AMD kernel keeps CSparse's 0-based conventions internally (node ids,
# pointers, FLIP(i) = -i-2, -1 as "none"); arrays are accessed at index+1.

# Compressed-column view of a sparse operand without copying when it already
# is one (CSCMatrix(::CSCMatrix) is a validating copy).
_as_csc(A::CSCMatrix)=A
_as_csc(A::SparseMatrixCSC)=CSCMatrix(A)
_as_csc(A)=CSCMatrix(A)
# Symmetric pattern without the diagonal, 1-based CSC (colptr, rowind).
function _amd_pattern(A::NativeSparse,kind::Symbol)
    C=_as_csc(A); m,n=size(C)
    if kind===:symmetric
        m==n || throw(DimensionMismatch("AMD of A+Aᵀ needs a square matrix"))
        colptr,rowind=_symmetric_pattern(C,nothing)
        return colptr,rowind,n
    else
        # AᵀA pattern; with :ata_sparse, dense rows of A are dropped first.
        R=[Int[] for _ in 1:m]
        for j in 1:n, p in C.colptr[j]:C.colptr[j+1]-1
            iszero(C.values[p]) || push!(R[Int(C.rowind[p])],j)
        end
        dense=min(n-2,max(16,round(Int,10*sqrt(n))))
        cols=[Int[] for _ in 1:n]
        for row in R
            kind===:ata_sparse && length(row)>dense && continue
            for a in row, b in row
                a!=b && push!(cols[a],b)
            end
        end
    end
    colptr=ones(Int,n+1); rowind=Int[]
    for j in 1:n
        u=sort!(unique!(cols[j])); append!(rowind,u); colptr[j+1]=length(rowind)+1
    end
    colptr,rowind,n
end
_amd_flip(i)=-i-2
function _amd_wclear(mark,lemax,w,n)
    if mark<2 || mark+lemax<0
        @inbounds for k in 0:n-1
            w[k+1]!=0 && (w[k+1]=1)
        end
        mark=2
    end
    mark
end
function _amd_tdfs(j,k,head,next,post,stack)
    top=0; stack[1]=j
    @inbounds while top>=0
        p=stack[top+1]; i=head[p+1]
        if i==-1
            top-=1; post[k+1]=p; k+=1
        else
            head[p+1]=next[i+1]; top+=1; stack[top+1]=i
        end
    end
    k
end
"""    amd_ordering(A; kind=:symmetric)

Approximate minimum degree ordering of A+Aᵀ (`kind=:symmetric`, for Cholesky
and LDLᵀ), of AᵀA (`:ata`, for QR) or of AᵀA without dense rows (`:ata_sparse`,
for LU). Returns `p` with `p[new] = old`."""
function amd_ordering(A::NativeSparse;kind::Symbol=:symmetric)
    kind in (:symmetric,:ata,:ata_sparse) || throw(ArgumentError("unknown AMD kind $kind"))
    Cp1,Ci1,n=_amd_pattern(A,kind)      # 1-based
    _amd_core(Cp1,Ci1,n)
end
# AMD of a symmetric pattern without diagonal, 1-based CSC (Cp1, Ci1);
# both are copied, not modified.
function _amd_core(Cp1,Ci1,n)
    n==0 && return Int[]
    cnz=Cp1[n+1]-1
    dense=max(16,10*sqrt(n)); dense=Int(floor(min(n-2,dense)))
    nzmax=cnz+cnz÷5+2n
    Cp=Vector{Int}(undef,n+1); Ci=Vector{Int}(undef,max(nzmax,1))
    @inbounds for j in 1:n+1
        Cp[j]=Cp1[j]-1
    end
    @inbounds for q in 1:cnz
        Ci[q]=Ci1[q]-1
    end
    # Every entry is set by the initialization loops below.
    len=Vector{Int}(undef,n+1); nv=Vector{Int}(undef,n+1); next=Vector{Int}(undef,n+1); head=Vector{Int}(undef,n+1)
    elen=Vector{Int}(undef,n+1); degree=Vector{Int}(undef,n+1); w=Vector{Int}(undef,n+1); hhead=Vector{Int}(undef,n+1)
    last=Vector{Int}(undef,n+1)   # also the output permutation
    @inbounds begin
    for k in 0:n-1
        len[k+1]=Cp[k+2]-Cp[k+1]
    end
    len[n+1]=0
    for i in 0:n
        head[i+1]=-1; last[i+1]=-1; next[i+1]=-1; hhead[i+1]=-1
        nv[i+1]=1; w[i+1]=1; elen[i+1]=0; degree[i+1]=len[i+1]
    end
    mark=_amd_wclear(0,0,w,n)
    elen[n+1]=-2; Cp[n+1]=-1; w[n+1]=0
    nel=0; mindeg=0; lemax=0
    for i in 0:n-1
        d=degree[i+1]
        if d==0
            elen[i+1]=-2; nel+=1; Cp[i+1]=-1; w[i+1]=0
        elseif d>dense
            nv[i+1]=0; elen[i+1]=-1; nel+=1; Cp[i+1]=_amd_flip(n); nv[n+1]+=1
        else
            head[d+1]!=-1 && (last[head[d+1]+1]=i)
            next[i+1]=head[d+1]; head[d+1]=i
        end
    end
    while nel<n
        # Select a pivot of minimum approximate degree.
        k=-1
        while mindeg<n
            k=head[mindeg+1]; k!=-1 && break
            mindeg+=1
        end
        next[k+1]!=-1 && (last[next[k+1]+1]=-1)
        head[mindeg+1]=next[k+1]
        elenk=elen[k+1]; nvk=nv[k+1]; nel+=nvk
        # Garbage collection of the quotient graph storage.
        if elenk>0 && cnz+mindeg>=nzmax
            for j in 0:n-1
                p=Cp[j+1]
                if p>=0
                    Cp[j+1]=Ci[p+1]; Ci[p+1]=_amd_flip(j)
                end
            end
            q=0; p=0
            while p<cnz
                j=_amd_flip(Ci[p+1]); p+=1
                if j>=0
                    Ci[q+1]=Cp[j+1]; Cp[j+1]=q; q+=1
                    for _ in 0:len[j+1]-2
                        Ci[q+1]=Ci[p+1]; q+=1; p+=1
                    end
                end
            end
            cnz=q
        end
        # Construct the new element.
        dk=0; nv[k+1]=-nvk; p=Cp[k+1]
        pk1=elenk==0 ? p : cnz; pk2=pk1
        for k1 in 1:elenk+1
            if k1>elenk
                e=k; pj=p; ln=len[k+1]-elenk
            else
                e=Ci[p+1]; p+=1; pj=Cp[e+1]; ln=len[e+1]
            end
            for _ in 1:ln
                i=Ci[pj+1]; pj+=1
                nvi=nv[i+1]; nvi<=0 && continue
                dk+=nvi; nv[i+1]=-nvi
                if pk2+1>length(Ci)
                    resize!(Ci,2*length(Ci)+1); nzmax=length(Ci)
                end
                Ci[pk2+1]=i; pk2+=1
                next[i+1]!=-1 && (last[next[i+1]+1]=last[i+1])
                if last[i+1]!=-1
                    next[last[i+1]+1]=next[i+1]
                else
                    head[degree[i+1]+1]=next[i+1]
                end
            end
            if e!=k
                Cp[e+1]=_amd_flip(k); w[e+1]=0
            end
        end
        elenk!=0 && (cnz=pk2)
        degree[k+1]=dk; Cp[k+1]=pk1; len[k+1]=pk2-pk1; elen[k+1]=-2
        # Set differences |Le \ Lk| for every element e adjacent to Lk.
        mark=_amd_wclear(mark,lemax,w,n)
        for pk in pk1:pk2-1
            i=Ci[pk+1]
            eln=elen[i+1]; eln<=0 && continue
            nvi=-nv[i+1]; wnvi=mark-nvi
            for p in Cp[i+1]:Cp[i+1]+eln-1
                e=Ci[p+1]
                if w[e+1]>=mark
                    w[e+1]-=nvi
                elseif w[e+1]!=0
                    w[e+1]=degree[e+1]+wnvi
                end
            end
        end
        # Approximate degree update.
        for pk in pk1:pk2-1
            i=Ci[pk+1]; p1=Cp[i+1]; p2=p1+elen[i+1]-1; pn=p1
            h=0; d=0
            for p in p1:p2
                e=Ci[p+1]
                if w[e+1]!=0
                    dext=w[e+1]-mark
                    if dext>0
                        d+=dext; Ci[pn+1]=e; pn+=1; h+=e
                    else
                        Cp[e+1]=_amd_flip(k); w[e+1]=0
                    end
                end
            end
            elen[i+1]=pn-p1+1
            p3=pn; p4=p1+len[i+1]
            for p in p2+1:p4-1
                j=Ci[p+1]
                nvj=nv[j+1]; nvj<=0 && continue
                d+=nvj; Ci[pn+1]=j; pn+=1; h+=j
            end
            if d==0
                Cp[i+1]=_amd_flip(k); nvi=-nv[i+1]
                dk-=nvi; nvk+=nvi; nel+=nvi; nv[i+1]=0; elen[i+1]=-1
            else
                degree[i+1]=min(degree[i+1],d)
                Ci[pn+1]=Ci[p3+1]; Ci[p3+1]=Ci[p1+1]; Ci[p1+1]=k
                len[i+1]=pn-p1+1
                h=mod(h,n)
                next[i+1]=hhead[h+1]; hhead[h+1]=i; last[i+1]=h
            end
        end
        degree[k+1]=dk
        lemax=max(lemax,dk)
        mark=_amd_wclear(mark+lemax,lemax,w,n)
        # Supernode detection: indistinguishable variables are merged.
        for pk in pk1:pk2-1
            i=Ci[pk+1]
            nv[i+1]>=0 && continue
            h=last[i+1]; i=hhead[h+1]; hhead[h+1]=-1
            while i!=-1 && next[i+1]!=-1
                ln=len[i+1]; eln=elen[i+1]
                for p in Cp[i+1]+1:Cp[i+1]+ln-1
                    w[Ci[p+1]+1]=mark
                end
                jlast=i; j=next[i+1]
                while j!=-1
                    ok=len[j+1]==ln && elen[j+1]==eln
                    p=Cp[j+1]+1
                    while ok && p<=Cp[j+1]+ln-1
                        w[Ci[p+1]+1]!=mark && (ok=false); p+=1
                    end
                    if ok
                        Cp[j+1]=_amd_flip(i); nv[i+1]+=nv[j+1]; nv[j+1]=0; elen[j+1]=-1
                        j=next[j+1]; next[jlast+1]=j
                    else
                        jlast=j; j=next[j+1]
                    end
                end
                i=next[i+1]; mark+=1
            end
        end
        # Finalize the new element.
        p=pk1
        for pk in pk1:pk2-1
            i=Ci[pk+1]
            nvi=-nv[i+1]; nvi<=0 && continue
            nv[i+1]=nvi
            d=degree[i+1]+dk-nvi
            d=min(d,n-nel-nvi)
            head[d+1]!=-1 && (last[head[d+1]+1]=i)
            next[i+1]=head[d+1]; last[i+1]=-1; head[d+1]=i
            mindeg=min(mindeg,d); degree[i+1]=d
            Ci[p+1]=i; p+=1
        end
        nv[k+1]=nvk
        len[k+1]=p-pk1
        if len[k+1]==0
            Cp[k+1]=-1; w[k+1]=0
        end
        elenk!=0 && (cnz=p)
    end
    # Postorder the assembly tree.
    for i in 0:n-1
        Cp[i+1]=_amd_flip(Cp[i+1])
    end
    for j in 0:n
        head[j+1]=-1
    end
    for j in n:-1:0
        nv[j+1]>0 && continue
        next[j+1]=head[Cp[j+1]+1]; head[Cp[j+1]+1]=j
    end
    for e in n:-1:0
        nv[e+1]<=0 && continue
        if Cp[e+1]!=-1
            next[e+1]=head[Cp[e+1]+1]; head[Cp[e+1]+1]=e
        end
    end
    k=0
    for i in 0:n
        Cp[i+1]==-1 && (k=_amd_tdfs(i,k,head,next,last,w))
    end
    end # @inbounds
    # last[1:n+1] holds the 0-based postordered pivots, the dummy node n last.
    p=Vector{Int}(undef,n); seen=falses(n); k=0
    @inbounds for q in 1:n+1
        v=last[q]; v==n && continue
        (k<n && 0<=v<n && !seen[v+1]) || error("AMD internal error: invalid permutation")
        k+=1; p[k]=v+1; seen[v+1]=true
    end
    k==n || error("AMD internal error: invalid permutation")
    p
end

# ---- elimination tree, postorder, factor structure ------------------------------

"""Elimination tree of the symmetric pattern given by its upper part in
1-based CSC (parent[j] = 0 for a root), path compression (cs_etree)."""
function _etree(colptr,rowind,n)
    parent=zeros(Int,n); ancestor=zeros(Int,n)
    @inbounds for k in 1:n, p in colptr[k]:colptr[k+1]-1
        i=Int(rowind[p])
        while i!=0 && i<k
            inext=ancestor[i]; ancestor[i]=k
            inext==0 && (parent[i]=k)
            i=inext
        end
    end
    parent
end
"""Postorder of a forest given by `parent`, children visited in increasing order."""
# Postorder of the elimination forest. With `weight`, the children of each
# node are visited by increasing weight (CHOLMOD's weighted postorder with the
# column counts: the heaviest child ends next to its parent, where relaxed
# amalgamation can merge it).
function _postorder(parent,weight=nothing)
    n=length(parent); head=zeros(Int,n); next=zeros(Int,n)
    if weight===nothing
        for j in n:-1:1
            parent[j]==0 && continue
            next[j]=head[parent[j]]; head[parent[j]]=j
        end
    else
        # Bucket sort of the nodes by weight, inserted heaviest first.
        maxw=maximum(weight;init=0); whead=zeros(Int,maxw+1); wnext=zeros(Int,n)
        @inbounds for j in n:-1:1
            k=min(max(weight[j],0),maxw)+1; wnext[j]=whead[k]; whead[k]=j
        end
        @inbounds for k in maxw+1:-1:1
            j=whead[k]
            while j!=0
                parent[j]!=0 && (next[j]=head[parent[j]]; head[parent[j]]=j)
                j=wnext[j]
            end
        end
    end
    post=Vector{Int}(undef,n); stack=Vector{Int}(undef,n); np=0
    @inbounds for j in 1:n
        parent[j]==0 || continue
        top=1; stack[1]=j
        while top>0
            p=stack[top]; i=head[p]
            if i==0
                top-=1; np+=1; post[np]=p
            else
                head[p]=next[i]; top+=1; stack[top]=i
            end
        end
    end
    post
end
"""Row structure of each column of the Cholesky factor of the symmetric
pattern (1-based CSC, both triangles): row k of L is the union of the paths
of the elimination tree from each i < k with a_ik ≠ 0 up to k (row subtree,
cs_ereach). Counting pass, then fill pass: rows arrive in increasing order,
so each column [j; rows > j] is sorted without a sort."""
function _factor_columns(colptr,rowind,parent,n)
    counts=ones(Int,n); flag=zeros(Int,n)
    @inbounds for k in 1:n
        flag[k]=k
        for p in colptr[k]:colptr[k+1]-1
            i=Int(rowind[p]); i<k || continue
            while flag[i]!=k
                counts[i]+=1; flag[i]=k; i=parent[i]
            end
        end
    end
    columns=[Vector{Int}(undef,c) for c in counts]; pos=fill(2,n)
    @inbounds for j in 1:n
        columns[j][1]=j
    end
    fill!(flag,0)
    @inbounds for k in 1:n
        flag[k]=k
        for p in colptr[k]:colptr[k+1]-1
            i=Int(rowind[p]); i<k || continue
            while flag[i]!=k
                columns[i][pos[i]]=k; pos[i]+=1; flag[i]=k; i=parent[i]
            end
        end
    end
    children=[Int[] for _ in 1:n]
    for j in 1:n
        parent[j]!=0 && push!(children[parent[j]],j)
    end
    columns,children
end
# Pattern of P·A·Pᵀ (or A when `pinv` is nothing): both triangles, no
# diagonal, stored zeros skipped, each index once per column (unsorted),
# 1-based CSC. Counting pass, fill pass, then marker-based deduplication.
function _symmetric_pattern(C::CSCMatrix,pinv)
    n=size(C,1); cnt=zeros(Int,n+1)
    @inbounds for j in 1:n, q in C.colptr[j]:C.colptr[j+1]-1
        i=Int(C.rowind[q]); (i==j || iszero(C.values[q])) && continue
        a,b=pinv===nothing ? (i,j) : (pinv[i],pinv[j]); cnt[a+1]+=1; cnt[b+1]+=1
    end
    colptr=Vector{Int}(undef,n+1); colptr[1]=1
    @inbounds for j in 1:n
        colptr[j+1]=colptr[j]+cnt[j+1]
    end
    rowind=Vector{Int}(undef,colptr[n+1]-1); next=copy(colptr)
    lower=upper=false
    @inbounds for j in 1:n, q in C.colptr[j]:C.colptr[j+1]-1
        i=Int(C.rowind[q]); (i==j || iszero(C.values[q])) && continue
        i>j ? (lower=true) : (upper=true)
        a,b=pinv===nothing ? (i,j) : (pinv[i],pinv[j])
        rowind[next[b]]=a; next[b]+=1; rowind[next[a]]=b; next[a]+=1
    end
    # A single stored triangle cannot repeat a pair.
    (lower && upper) || return colptr,rowind
    # Deduplicate in place (a symmetric input lists every pair twice).
    mark=zeros(Int,n); w=1
    @inbounds for j in 1:n
        lo,hi=colptr[j],colptr[j+1]-1; colptr[j]=w
        for q in lo:hi
            i=rowind[q]
            mark[i]==j && continue
            mark[i]=j; rowind[w]=i; w+=1
        end
    end
    colptr[n+1]=w; resize!(rowind,w-1)
    colptr,rowind
end
_permuted_symmetric_pattern(A::NativeSparse,p)=(C=_as_csc(A); (_symmetric_pattern(C,invperm(p))...,size(C,1)))
