# SPDX-License-Identifier: Apache-2.0
# Supernodal multifrontal LU, symmetric strategy (UMFPACK's choice for a
# mostly symmetric pattern with a nonzero diagonal): A[p,q] = L·U.
#   1. symbolic analysis of the pattern of A+Aᵀ, as for the Cholesky
#      factorization (src/sparse/supernodal_cholesky.jl): AMD, elimination tree,
#      column counts, weighted postorder, relaxed supernodes;
#   2. numeric phase in postorder with a stack of contribution blocks: each
#      front is square and dense, its fully summed rows and columns are the
#      supernode's and those delayed by its children; pivots are chosen in
#      the fully summed rows with threshold partial pivoting (the diagonal is
#      kept when it passes the threshold); a column with no acceptable pivot
#      is delayed, with an unused row, to the parent front, where all of its
#      entries arrive through the contribution block. At a root every row is
#      fully summed, so only an exactly zero column fails (singular matrix).

# Fraction of off-diagonal entries of A whose transpose is also stored, and
# the number of nonzero diagonal entries; `both`: off-diagonal entries of the
# pattern of A+Aᵀ.
function _lu_pattern_symmetry(C::CSCMatrix,both)
    n=size(C,1); off=0; diag=0
    @inbounds for j in 1:n
        hasdiag=false
        for q in C.colptr[j]:C.colptr[j+1]-1
            iszero(C.values[q]) && continue
            Int(C.rowind[q])==j ? (hasdiag=true) : (off+=1)
        end
        diag+=hasdiag
    end
    off==0 ? 1.0 : (2off-both)/off,diag
end
# UMFPACK's test for the symmetric strategy.
_lu_symmetric_strategy(C::CSCMatrix)=_lu_symmetric_strategy(C,_symmetric_pattern(C,nothing)[1])
function _lu_symmetric_strategy(C::CSCMatrix,colptr)
    n=size(C,1); n==0 && return true
    sym,diag=_lu_pattern_symmetry(C,colptr[n+1]-1)
    sym>=0.5 && diag>=0.9n
end

# Lower pattern with the diagonal (by columns) and strict upper pattern (by
# columns) of P·(A+Aᵀ)·Pᵀ, from the pattern (colptr, rowind) of A+Aᵀ without
# diagonal (each pair once per column): a relabeling by counting.
function _lu_sym_pattern(colptr,rowind,n,pinv)
    lp=zeros(Int,n+1); up=zeros(Int,n+1)
    @inbounds for j in 1:n
        b=pinv[j]; lp[b+1]+=1
        for q in colptr[j]:colptr[j+1]-1
            pinv[rowind[q]]>b ? (lp[b+1]+=1) : (up[b+1]+=1)
        end
    end
    lp[1]=up[1]=1
    @inbounds for k in 1:n
        lp[k+1]+=lp[k]; up[k+1]+=up[k]
    end
    li=Vector{Int}(undef,lp[n+1]-1); ui=Vector{Int}(undef,up[n+1]-1); ln=lp[1:n]; un=up[1:n]
    @inbounds for j in 1:n
        b=pinv[j]; li[ln[b]]=b; ln[b]+=1
        for q in colptr[j]:colptr[j+1]-1
            a=pinv[rowind[q]]
            a>b ? (li[ln[b]]=a; ln[b]+=1) : (ui[un[b]]=a; un[b]+=1)
        end
    end
    lp,li,up,ui
end
# P·A·Pᵀ by columns and by rows (stored zeros skipped, rows/columns unsorted).
function _lu_permuted(C::CSCMatrix{T},pinv) where T
    n=size(C,1); cc=zeros(Int,n+1); rc=zeros(Int,n+1)
    @inbounds for j in 1:n, q in C.colptr[j]:C.colptr[j+1]-1
        iszero(C.values[q]) && continue
        cc[pinv[j]+1]+=1; rc[pinv[Int(C.rowind[q])]+1]+=1
    end
    cc[1]=rc[1]=1
    @inbounds for k in 1:n
        cc[k+1]+=cc[k]; rc[k+1]+=rc[k]
    end
    nz=cc[n+1]-1; ci=Vector{Int}(undef,nz); cv=Vector{T}(undef,nz); ri=Vector{Int}(undef,nz); rv=Vector{T}(undef,nz)
    cn=cc[1:n]; rn=rc[1:n]
    @inbounds for j in 1:n, q in C.colptr[j]:C.colptr[j+1]-1
        v=C.values[q]; iszero(v) && continue
        a,b=pinv[Int(C.rowind[q])],pinv[j]
        t=cn[b]; cn[b]+=1; ci[t]=a; cv[t]=v
        t=rn[a]; rn[a]+=1; ri[t]=b; rv[t]=v
    end
    cc,ci,cv,rc,ri,rv
end

# F[i,j] -= Σ_{k≤e} F[i,k]·F[k,j] for e < i ≤ f, j0 ≤ j ≤ nc (Schur update
# of the columns that the panel did not update), four k per pass.
function _lu_schur!(F,f,e,j0,nc=f)
    @inbounds for j in j0:nc
        k=1
        while k+3<=e
            u1,u2,u3,u4=F[k,j],F[k+1,j],F[k+2,j],F[k+3,j]
            @simd for i in e+1:f
                F[i,j]=muladd(-F[i,k+3],u4,muladd(-F[i,k+2],u3,muladd(-F[i,k+1],u2,muladd(-F[i,k],u1,F[i,j]))))
            end
            k+=4
        end
        while k<=e
            u=F[k,j]
            @simd for i in e+1:f
                F[i,j]=muladd(-F[i,k],u,F[i,j])
            end
            k+=1
        end
    end
    F
end

# Column k receives pivots j0..j1 in order: F[j+1:f,k] -= F[j+1:f,j]·F[j,k],
# four pivots per pass over the column (their small triangle first).
function _lu_col_update!(F,f,k,j0,j1)
    j=j0
    @inbounds while j+3<=j1
        u1=F[j,k]
        F[j+1,k]=muladd(-F[j+1,j],u1,F[j+1,k]); u2=F[j+1,k]
        F[j+2,k]=muladd(-F[j+2,j+1],u2,muladd(-F[j+2,j],u1,F[j+2,k])); u3=F[j+2,k]
        F[j+3,k]=muladd(-F[j+3,j+2],u3,muladd(-F[j+3,j+1],u2,muladd(-F[j+3,j],u1,F[j+3,k]))); u4=F[j+3,k]
        @simd for i in j+4:f
            F[i,k]=muladd(-F[i,j+3],u4,muladd(-F[i,j+2],u3,muladd(-F[i,j+1],u2,muladd(-F[i,j],u1,F[i,k]))))
        end
        j+=4
    end
    @inbounds while j<=j1
        u=F[j,k]
        @simd for i in j+1:f
            F[i,k]=muladd(-F[i,j],u,F[i,k])
        end
        j+=1
    end
    F
end

function _mf_lu_symmetric(A::NativeSparse,pivot_threshold,maxwidth,threaded,symmetric=_symmetric_pattern(_as_csc(A),nothing))
    C=_as_csc(A); n=size(C,1); T=eltype(C); R=real(T); ident=collect(1:n)
    initial=maximum(abs,C.values;init=zero(R))
    # Symbolic analysis of A+Aᵀ (see src/sparse/supernodal_cholesky.jl).
    sp,si=symmetric   # pattern of A+Aᵀ without diagonal, shared with the strategy test
    pamd=_amd_core(sp,si,n); pinv=invperm(pamd)
    lp,li,up,ui=_lu_sym_pattern(sp,si,n,pinv)
    parent=_etree(up,ui,n)
    counts=_chol_counts(lp,li,parent,_postorder(parent),n)
    post=_postorder(parent,counts)
    # One assignment: the final order p is captured by closures further down.
    p=post!=ident ? pamd[post] : pamd
    if post!=ident
        pinv=invperm(p); ipost=invperm(post); counts=counts[post]
        relabeled=Vector{Int}(undef,n)
        @inbounds for k in 1:n
            q=parent[post[k]]; relabeled[k]=q==0 ? 0 : ipost[q]
        end
        parent=relabeled
        lp,li,up,ui=_lu_sym_pattern(sp,si,n,pinv)
    end
    nodes=_chol_supernodes(parent,counts,n,maxwidth)
    rows,sparent,schildren=_chol_supernode_rows(nodes,lp,li,parent,n)
    cp,ci,cv,rp,ri,rv=_lu_permuted(C,pinv)
    # Output: pivot sequence (permuted labels) and the L columns / U rows.
    prow=Vector{Int}(undef,n); pcol=Vector{Int}(undef,n); t=0
    guess=sum(counts;init=0)+n   # nnz of L without delayed pivots
    lptr=Vector{Int}(undef,n+1); lptr[1]=1; lidx=Vector{Int}(undef,guess); lval=Vector{T}(undef,guess); lnz=0
    uptr=Vector{Int}(undef,n+1); uptr[1]=1; uidx=Vector{Int}(undef,guess); uval=Vector{T}(undef,guess); unz=0
    # Contribution blocks: values (m×m, by columns) in `arena`, their row
    # then column labels in `iarena`; `stack` keeps (value offset, index
    # offset, m, delayed count).
    arena=Vector{T}(undef,0); vtop=0; iarena=Vector{Int}(undef,0); itop=0; stack=NTuple{4,Int}[]
    rpos=zeros(Int,n); cpos=zeros(Int,n); FR=Int[]; FC=Int[]; done=Int[]
    W=Matrix{T}(undef,0,0)
    pieces=UnitRange{Int}[]; piece_parent=Int[]; shapes=Tuple{Int,Int}[]; growth=initial
    pending=Vector{Vector{Int}}(undef,length(nodes))   # pieces waiting for their parent
    @inbounds for (s,node) in enumerate(nodes)
        a,b=first(node),last(node); w=length(node); Rs=rows[s]; nch=length(schildren[s])
        # Front labels: delayed rows/columns of the children, the supernode,
        # then the rows below it.
        empty!(FR); empty!(FC)
        for c in length(stack)-nch+1:length(stack)
            _,io,m,nd=stack[c]
            for l in 1:nd
                push!(FR,iarena[io+l-1]); push!(FC,iarena[io+m+l-1])
            end
        end
        append!(FR,node); append!(FC,node); nfs=length(FR)
        append!(FR,Rs); append!(FC,Rs); f=length(FR)
        for l in 1:f
            rpos[FR[l]]=l; cpos[FC[l]]=l
        end
        if f>size(W,1)
            W=Matrix{T}(undef,max(f,2size(W,1)),max(f,2size(W,1)))
        end
        F=view(W,1:f,1:f)
        for j in 1:f, i in 1:f
            F[i,j]=zero(T)
        end
        # Original entries: columns of the supernode (rows ≥ a), rows of the
        # supernode (columns > b).
        for j in node, q in cp[j]:cp[j+1]-1
            i=ci[q]; i>=a && (F[rpos[i],cpos[j]]+=cv[q])
        end
        for i in node, q in rp[i]:rp[i+1]-1
            j=ri[q]; j>b && (F[rpos[i],cpos[j]]+=rv[q])
        end
        # Children's contribution blocks (top of the stack).
        for _ in 1:nch
            vo,io,m,_=pop!(stack); tt=vo
            for jj in 1:m
                lj=cpos[iarena[io+m+jj-1]]
                for ii in 1:m
                    F[rpos[iarena[io+ii-1]],lj]+=arena[tt]; tt+=1
                end
            end
            vtop=vo-1; itop=io-1
        end
        # Threshold partial pivoting among the fully summed rows, left-looking:
        # column k receives the pivots it misses (done[k] counts those already
        # applied), then its pivot is chosen; a column without an acceptable
        # pivot moves behind the candidates and is brought up to date at the
        # end. Rows are swapped whole, so the lazy updates stay consistent.
        isroot=sparent[s]==0; k=1; ncand=nfs
        length(done)<nfs && resize!(done,nfs)
        for l in 1:nfs
            done[l]=0
        end
        while k<=ncand
            _lu_col_update!(F,f,k,done[k]+1,k-1); done[k]=k-1
            colmax=zero(R)
            for i in k:f
                colmax=max(colmax,abs(F[i,k]))
            end
            iszero(colmax) && throw(SingularException(p[FC[k]]))
            r=0; d=rpos[FC[k]]
            if k<=d<=nfs && FR[d]==FC[k] && !iszero(F[d,k]) && abs(F[d,k])>=pivot_threshold*colmax
                r=d
            else
                best=k; bv=abs(F[k,k])
                for i in k+1:nfs
                    v=abs(F[i,k]); v>bv && (best=i; bv=v)
                end
                !iszero(bv) && (isroot || bv>=pivot_threshold*colmax) && (r=best)
            end
            if r==0
                isroot && throw(SingularException(p[FC[k]]))
                # Delay column k: swap it with the last candidate.
                if k!=ncand
                    for i in 1:f
                        F[i,k],F[i,ncand]=F[i,ncand],F[i,k]
                    end
                    FC[k],FC[ncand]=FC[ncand],FC[k]; cpos[FC[k]]=k; cpos[FC[ncand]]=ncand
                    done[k],done[ncand]=done[ncand],done[k]
                end
                ncand-=1; continue
            end
            if r!=k
                for j in 1:f
                    F[k,j],F[r,j]=F[r,j],F[k,j]
                end
                FR[k],FR[r]=FR[r],FR[k]; rpos[FR[k]]=k; rpos[FR[r]]=r
            end
            inv=one(T)/F[k,k]
            @simd for i in k+1:f
                F[i,k]*=inv
            end
            k+=1
        end
        e=ncand
        for k2 in e+1:nfs
            _lu_col_update!(F,f,k2,done[k2]+1,e)
        end
        if f>nfs && e>0
            # U12 = L11⁻¹·A12, then the Schur complement of the other columns.
            if f>_MF_SMALL_FRONT && T<:NativeFloat
                U12=view(F,1:e,nfs+1:f)
                trsm!(U12,view(F,1:e,1:e);uplo=:L,unit=true,threaded)
                gemm!(view(F,e+1:f,nfs+1:f),view(F,e+1:f,1:e),U12,-one(T),one(T);threaded)
            else
                for j in nfs+1:f, k2 in 1:e
                    u=F[k2,j]
                    iszero(u) && continue
                    @simd for i in k2+1:e
                        F[i,j]=muladd(-F[i,k2],u,F[i,j])
                    end
                end
                _lu_schur!(F,f,e,nfs+1)
            end
        end
        g2=zero(R)
        for j in 1:f
            @simd for i in 1:f
                g2=max(g2,abs2(F[i,j]))
            end
        end
        # abs2 overflows before the entries do: fall back to abs then.
        g=isfinite(g2) ? sqrt(g2) : maximum(abs,F)
        isfinite(g) || throw(ArgumentError("multifrontal LU arithmetic overflow"))
        growth=max(growth,g)
        # Pivots 1:e: L columns below the pivot, U rows from the pivot
        # (exact zeros dropped, the U diagonal kept).
        lneed=lnz+e*f-e*(e+1)÷2; uneed=unz+e*f-e*(e-1)÷2
        lneed>length(lidx) && (resize!(lidx,max(lneed,2length(lidx))); resize!(lval,length(lidx)))
        uneed>length(uidx) && (resize!(uidx,max(uneed,2length(uidx))); resize!(uval,length(uidx)))
        for k2 in 1:e
            t+=1; prow[t]=FR[k2]; pcol[t]=FC[k2]
            for i in k2+1:f
                v=F[i,k2]; iszero(v) && continue
                lnz+=1; lidx[lnz]=FR[i]; lval[lnz]=v
            end
            lptr[t+1]=lnz+1
            for j in k2:f
                v=F[k2,j]; (iszero(v) && j!=k2) && continue
                unz+=1; uidx[unz]=FC[j]; uval[unz]=v
            end
            uptr[t+1]=unz+1
        end
        # Output pieces of at most maxwidth pivots, chained; the first piece
        # of this front is the parent of its children's last pieces.
        first_piece=length(pieces)+1
        for k0 in 1:max(maxwidth,1):max(e,1)
            e==0 && break
            push!(pieces,(t-e+k0):(t-e+min(k0+maxwidth-1,e))); push!(shapes,(f,f)); push!(piece_parent,0)
            length(pieces)>first_piece && (piece_parent[end-1]=length(pieces))
        end
        last_piece=length(pieces)>=first_piece ? length(pieces) : 0
        # Children's last pieces point to this front's first piece (or, if
        # this front eliminated nothing, they wait for the parent).
        waiting=Int[]
        for c in schildren[s]
            append!(waiting,pending[c])
        end
        if last_piece!=0
            for g in waiting
                piece_parent[g]=first_piece
            end
            pending[s]=[last_piece]
        else
            pending[s]=waiting
        end
        # Contribution block: rows/columns e+1:f (delayed ones first).
        if f>e
            m=f-e; nd=nfs-e
            need=vtop+m*m; need>length(arena) && resize!(arena,max(need,2length(arena)))
            ineed=itop+2m; ineed>length(iarena) && resize!(iarena,max(ineed,2length(iarena)))
            vo=vtop+1; io=itop+1; tt=vo
            for jj in e+1:f, ii in e+1:f
                arena[tt]=F[ii,jj]; tt+=1
            end
            for l in 1:m
                iarena[io+l-1]=FR[e+l]; iarena[io+m+l-1]=FC[e+l]
            end
            vtop=need; itop=ineed; push!(stack,(vo,io,m,nd))
        end
    end
    t==n || error("multifrontal LU did not eliminate every pivot")
    L,U=_lu_assemble(n,prow,pcol,lptr,lidx,lval,lnz,uptr,uidx,uval,unz,T)
    rowperm=[p[prow[k]] for k in 1:n]; colperm=[p[pcol[k]] for k in 1:n]
    MultifrontalLUFactor(L,U,rowperm,colperm,piece_parent,pieces,shapes,R(pivot_threshold),
                         iszero(initial) ? one(R) : growth/initial)
end

# L (unit diagonal stored last in each row) and U (pivot first in each row)
# as CSR in the pivot order, from the L columns and U rows recorded per pivot
# in labels (prow[k], pcol[k]: row and column of pivot k).
function _lu_assemble(n,prow,pcol,lptr,lidx,lval,lnz,uptr,uidx,uval,unz,::Type{T}) where T
    # Relabel to the pivot order: L by columns → CSR rows, U rows by pivot.
    rowpos=Vector{Int}(undef,n); colpos=Vector{Int}(undef,n)
    @inbounds for k in 1:n
        rowpos[prow[k]]=k; colpos[pcol[k]]=k
    end
    # L (unit diagonal stored) as CSR: rows = pivot positions.
    lcnt=ones(Int,n+1); lcnt[1]=1
    @inbounds for q in 1:lnz
        lcnt[rowpos[lidx[q]]+1]+=1
    end
    @inbounds for k in 1:n
        lcnt[k+1]+=lcnt[k]
    end
    Lc=Vector{Int}(undef,lcnt[n+1]-1); Lv=Vector{T}(undef,lcnt[n+1]-1); nxt=lcnt[1:n]
    @inbounds for k in 1:n
        for q in lptr[k]:lptr[k+1]-1
            i=rowpos[lidx[q]]; x=nxt[i]; nxt[i]+=1; Lc[x]=k; Lv[x]=lval[q]
        end
    end
    @inbounds for i in 1:n
        x=nxt[i]; Lc[x]=i; Lv[x]=one(T)   # columns k < i were written in order
    end
    # U: rows are pivots, columns relabeled; by columns (rows ascending),
    # then back to rows (columns ascending), two counting passes.
    ccnt=zeros(Int,n+1); ccnt[1]=1
    @inbounds for q in 1:unz
        ccnt[colpos[uidx[q]]+1]+=1
    end
    @inbounds for k in 1:n
        ccnt[k+1]+=ccnt[k]
    end
    Ur=Vector{Int}(undef,unz); Uw=Vector{T}(undef,unz); nxt=ccnt[1:n]
    @inbounds for k in 1:n, q in uptr[k]:uptr[k+1]-1
        c=colpos[uidx[q]]; x=nxt[c]; nxt[c]+=1; Ur[x]=k; Uw[x]=uval[q]
    end
    Uc=Vector{Int}(undef,unz); Uv=Vector{T}(undef,unz); nxt=uptr[1:n]
    @inbounds for c in 1:n, x in ccnt[c]:ccnt[c+1]-1
        k=Ur[x]; y=nxt[k]; nxt[k]+=1; Uc[y]=c; Uv[y]=Uw[x]
    end
    CSRMatrix(_Shared(),n,n,lcnt,Lc,Lv),CSRMatrix(_Shared(),n,n,uptr,Uc,Uv)
end

# L·U·x = x with CSR factors: L unit lower (diagonal stored last in each
# row), U upper (diagonal first). Row dot products with four accumulators;
# a row whose columns form one range is addressed directly. The inverse of
# the U diagonal is taken off the dependency chain.
function _lu_csr_solve!(x,L::CSRMatrix,U::CSRMatrix)
    n=length(x); lp,lc,lv=L.rowptr,L.colind,L.values; up,uc,uv=U.rowptr,U.colind,U.values
    @inbounds for i in 1:n
        lo,hi=lp[i],lp[i+1]-2
        hi>=lo && (x[i]-=_lu_rowdot(lv,lc,x,lo,hi))
    end
    @inbounds for i in n:-1:1
        lo,hi=up[i]+1,up[i+1]-1
        d=Base.inv(uv[up[i]])
        x[i]=(hi>=lo ? x[i]-_lu_rowdot(uv,uc,x,lo,hi) : x[i])*d
    end
    x
end
@inline function _lu_rowdot(v,idx,x,lo,hi)
    T=promote_type(eltype(v),eltype(x))
    @inbounds if idx[hi]-idx[lo]==hi-lo
        d=idx[lo]-lo
        if T<:Real
            acc=zero(T)
            @simd for q in lo:hi
                acc=muladd(v[q],x[q+d],acc)
            end
            return acc
        end
        a1=a2=a3=a4=zero(T); q=lo
        while q+3<=hi
            a1=muladd(v[q],x[q+d],a1); a2=muladd(v[q+1],x[q+1+d],a2)
            a3=muladd(v[q+2],x[q+2+d],a3); a4=muladd(v[q+3],x[q+3+d],a4); q+=4
        end
        while q<=hi
            a1=muladd(v[q],x[q+d],a1); q+=1
        end
        return (a1+a2)+(a3+a4)
    end
    a1=a2=a3=a4=zero(T); q=lo
    @inbounds while q+3<=hi
        a1=muladd(v[q],x[idx[q]],a1); a2=muladd(v[q+1],x[idx[q+1]],a2)
        a3=muladd(v[q+2],x[idx[q+2]],a3); a4=muladd(v[q+3],x[idx[q+3]],a4); q+=4
    end
    @inbounds while q<=hi
        a1=muladd(v[q],x[idx[q]],a1); q+=1
    end
    (a1+a2)+(a3+a4)
end

# Left-looking sparse LU with threshold partial row pivoting (Gilbert–
# Peierls; CSparse cs_lu), columns in the order q (natural by default):
# column j of L and U from the sparse triangular solve L·x = A[:,q[j]], whose
# pattern is the reach of that column in the graph of L (depth-first search).
# The preferred row (pref[j], by default the diagonal row q[j]) is kept when
# |x[row]| ≥ τ·max|candidates|. Returns
# L (unit diagonal, CSR, rows in pivot order) and U (CSR) with
# A[p,q] = L·U.
function _gp_lu(C::CSCMatrix{T},τ,q=nothing,pref=nothing) where T
    n=size(C,1); R=real(T)
    Lp=Vector{Int}(undef,n+1); Lp[1]=1; Li=Int[]; Lx=T[]   # by columns, original rows
    Up=Vector{Int}(undef,n+1); Up[1]=1; Ui=Int[]; Ux=T[]   # by columns, pivot rows
    nz=C.colptr[n+1]-1; sizehint!(Li,4nz); sizehint!(Lx,4nz); sizehint!(Ui,4nz); sizehint!(Ux,4nz)
    pinv=zeros(Int,n); x=zeros(T,n); xi=Vector{Int}(undef,n); stack=Vector{Int}(undef,n); pstack=Vector{Int}(undef,n)
    mark=zeros(Int,n)
    @inbounds for j in 1:n
        col=q===nothing ? j : q[j]
        # Reach of the pattern of A[:,col] in the graph of L, in topological
        # order in xi[top:n].
        top=n+1
        for q in C.colptr[col]:C.colptr[col+1]-1
            i0=Int(C.rowind[q]); (mark[i0]==j || iszero(C.values[q])) && continue
            head=1; stack[1]=i0
            while head>0
                i=stack[head]
                if mark[i]!=j
                    mark[i]=j; k=pinv[i]; pstack[head]=k==0 ? 0 : Lp[k]+1
                end
                k=pinv[i]; done=true
                if k!=0
                    pe=Lp[k+1]-1; pp=pstack[head]
                    while pp<=pe
                        r=Li[pp]; pp+=1
                        if mark[r]!=j
                            pstack[head]=pp; head+=1; stack[head]=r; done=false; break
                        end
                    end
                    done && (pstack[head]=pp)
                end
                if done
                    head-=1; top-=1; xi[top]=i
                end
            end
        end
        # x = L \ A[:,j] on that pattern.
        for q in C.colptr[col]:C.colptr[col+1]-1
            v=C.values[q]; iszero(v) || (x[Int(C.rowind[q])]+=v)
        end
        for t in top:n
            i=xi[t]; k=pinv[i]; k==0 && continue
            xv=x[i]
            for q in Lp[k]+1:Lp[k+1]-1
                x[Li[q]]-=Lx[q]*xv
            end
        end
        # Pivot among the rows not yet pivoted.
        ipiv=0; amax=zero(R)
        for t in top:n
            i=xi[t]
            if pinv[i]==0
                a=abs(x[i]); a>amax && (amax=a; ipiv=i)
            else
                push!(Ui,pinv[i]); push!(Ux,x[i])
            end
        end
        (ipiv==0 || iszero(amax) || !isfinite(amax)) && throw(SingularException(col))
        d=pref===nothing ? col : pref[j]
        d!=0 && pinv[d]==0 && mark[d]==j && !iszero(x[d]) && abs(x[d])>=τ*amax && (ipiv=d)
        piv=x[ipiv]; push!(Ui,j); push!(Ux,piv); Up[j+1]=length(Ui)+1
        pinv[ipiv]=j; push!(Li,ipiv); push!(Lx,one(T))
        ip=Base.inv(piv)
        for t in top:n
            i=xi[t]
            if pinv[i]==0
                push!(Li,i); push!(Lx,x[i]*ip)
            end
            x[i]=zero(T)
        end
        Lp[j+1]=length(Li)+1
    end
    # L rows relabeled to pivot order; both factors as sorted CSR by two
    # counting passes (columns ascending within each row).
    p=Vector{Int}(undef,n)
    @inbounds for i in 1:n
        p[pinv[i]]=i
    end
    _csc_to_csr(n,Lp,[pinv[i] for i in Li],Lx),_csc_to_csr(n,Up,Ui,Ux),p
end
# CSR of an n×n matrix given by columns (rows in any order within a column):
# entries of each row come out by increasing column.
function _csc_to_csr(n,colptr,rowind,values::Vector{T}) where T
    rp=zeros(Int,n+1); rp[1]=1
    @inbounds for q in 1:colptr[n+1]-1
        rp[rowind[q]+1]+=1
    end
    @inbounds for i in 1:n
        rp[i+1]+=rp[i]
    end
    ci=Vector{Int}(undef,colptr[n+1]-1); cv=Vector{T}(undef,length(ci)); nxt=rp[1:n]
    @inbounds for j in 1:n, q in colptr[j]:colptr[j+1]-1
        i=rowind[q]; x=nxt[i]; nxt[i]+=1; ci[x]=j; cv[x]=values[q]
    end
    CSRMatrix(_Shared(),n,n,rp,ci,cv)
end

# Row and column singletons of the pattern of A (UMFPACK umf_singletons):
# a column with one active entry is eliminated first (no L entries), a row
# with one active entry last (no U entries beyond its pivot); removing one
# may create others. Returns the column order and preferred pivot rows of
# the leading and trailing singletons, and the remaining (core) rows and
# columns.
function _lu_singletons(C::CSCMatrix)
    n=size(C,1); rc=zeros(Int,n); cc=zeros(Int,n)
    @inbounds for j in 1:n, q in C.colptr[j]:C.colptr[j+1]-1
        iszero(C.values[q]) && continue
        rc[Int(C.rowind[q])]+=1; cc[j]+=1
    end
    # Row structure (stored zeros skipped).
    rp=zeros(Int,n+1); rp[1]=1
    @inbounds for i in 1:n
        rp[i+1]=rp[i]+rc[i]
    end
    rj=Vector{Int}(undef,rp[n+1]-1); nxt=rp[1:n]
    @inbounds for j in 1:n, q in C.colptr[j]:C.colptr[j+1]-1
        iszero(C.values[q]) && continue
        i=Int(C.rowind[q]); rj[nxt[i]]=j; nxt[i]+=1
    end
    rowalive=trues(n); colalive=trues(n)
    headc=Int[]; headr=Int[]; tailc=Int[]; tailr=Int[]
    cq=[j for j in 1:n if cc[j]==1]; rq=[i for i in 1:n if rc[i]==1]
    @inbounds while !isempty(cq) || !isempty(rq)
        while !isempty(cq)
            c=pop!(cq); (colalive[c] && cc[c]==1) || continue
            r=0
            for q in C.colptr[c]:C.colptr[c+1]-1
                i=Int(C.rowind[q]); (rowalive[i] && !iszero(C.values[q])) && (r=i; break)
            end
            r==0 && continue
            push!(headc,c); push!(headr,r); colalive[c]=false; rowalive[r]=false
            for t in rp[r]:rp[r+1]-1
                j=rj[t]; colalive[j] || continue
                cc[j]-=1; cc[j]==1 && push!(cq,j)
            end
        end
        while !isempty(rq)
            r=pop!(rq); (rowalive[r] && rc[r]==1) || continue
            c=0
            for t in rp[r]:rp[r+1]-1
                j=rj[t]; colalive[j] && (c=j; break)
            end
            c==0 && continue
            push!(tailc,c); push!(tailr,r); colalive[c]=false; rowalive[r]=false
            for q in C.colptr[c]:C.colptr[c+1]-1
                i=Int(C.rowind[q]); (rowalive[i] && !iszero(C.values[q])) || continue
                rc[i]-=1; rc[i]==1 && push!(rq,i)
            end
            !isempty(cq) && break
        end
        # Removing a row singleton's column can create column singletons.
        if isempty(cq)
            for j in 1:n
                colalive[j] && cc[j]==1 && push!(cq,j)
            end
            isempty(cq) && isempty(rq) && break
        end
    end
    headc,headr,reverse!(tailc),reverse!(tailr),findall(rowalive),findall(colalive)
end

# Unsymmetric pattern: row/column singletons, then the core columns ordered
# by COLAMD (src/sparse/sparse_colamd.jl; AMD of AᵀA left 20–35% more fill); Gilbert–
# Peierls LU with the singleton rows as preferred pivots, returned as a
# multifrontal factor of width-1 nodes. A core larger than n/4 goes to the
# column-front multifrontal LU instead, in the same column order.
function _mf_lu_unsymmetric(C::CSCMatrix{T},pivot_threshold,maxwidth,threaded,P) where T
    n=size(C,1); R=real(T); initial=maximum(abs,C.values;init=zero(R))
    headc,headr,tailc,tailr,crows,ccols=_lu_singletons(C)
    qcore=if isempty(ccols)
        Int[]
    else
        # Core submatrix (core rows × core columns), AMD of its AᵀA.
        rmap=zeros(Int,n)
        for (k,i) in enumerate(crows)
            rmap[i]=k
        end
        I=Int[]; J=Int[]; V=T[]
        for (k,j) in enumerate(ccols), q in C.colptr[j]:C.colptr[j+1]-1
            i=rmap[Int(C.rowind[q])]; (i==0 || iszero(C.values[q])) && continue
            push!(I,i); push!(J,k); push!(V,C.values[q])
        end
        Ccore=CSCMatrix(COOMatrix(length(crows),length(ccols),I,J,V))
        ccols[colamd_ordering(Ccore)]
    end
    q=vcat(headc,qcore,tailc); pref=vcat(headr,zeros(Int,length(qcore)),tailr)
    length(q)==n || throw(SingularException(1))   # structurally rank deficient
    # A large core goes to the column-front multifrontal LU (dense fronts);
    # the left-looking kernel is kept when the singletons leave little.
    length(ccols)>n÷4 && return _mf_lu_colfront(C,q,pivot_threshold)
    L,U,p=_gp_lu(C,R(pivot_threshold),q,pref)
    growth=max(initial,maximum(abs,L.values;init=zero(R)),maximum(abs,U.values;init=zero(R)))
    isfinite(growth) || throw(ArgumentError("multifrontal LU arithmetic overflow"))
    MultifrontalLUFactor(L,U,p,q,zeros(Int,n),[j:j for j in 1:n],fill((1,1),n),R(pivot_threshold),
                         iszero(initial) ? one(R) : growth/initial)
end

# Unsymmetric multifrontal LU on the column fronts of the QR (George and Ng:
# the pattern of R bounds those of L and U under any row pivoting): column
# order q, fronts of the column etree of AᵀA (src/sparse/sparse_qr_multifrontal.jl),
# rows of A at the front of their leftmost column (the rows a front leaves go,
# sparse, to the front of their first nonzero column). In each front, threshold
# partial pivoting among all its rows on the pivot columns (the shortest row
# within τ of the largest entry, as UMFPACK), then
# U12 = L11⁻¹·A12 and the Schur complement of the rows left, which go to the
# parent. A[p,q] = L·U.
# Width cap of the column fronts: the relaxed supernodes follow the pattern of
# R (the bound of George and Ng), much denser than L and U at the top of the
# tree; uncapped, a 5000² unsymmetric matrix got a 3109² root front (×8.4 of
# UMFPACK), 256 gave ×6.4 (64 to 512 measured).
const _LU_COLFRONT_WIDTH=256
function _mf_lu_colfront(C::CSCMatrix{T},q,pivot_threshold) where T
    n=size(C,1); R=real(T); initial=maximum(abs,C.values;init=zero(R))
    qinv,parent,starts,nf,children,rc,rj,rv,leftmost,frows=_colfront_symbolic(C,q,(0.8,0.1,0.05),_LU_COLFRONT_WIDTH)
    any(>(n),leftmost) && throw(SingularException(1))    # an empty row
    # Rows left by a front travel, sparse, to the front of their first
    # nonzero column (always a later front: the fronts partition the columns
    # in order), not through the parent: a row untouched by the parent's
    # pivots would only widen its front. pend[s]: (row, columns, values).
    pend=[Tuple{Int,Vector{Int},Vector{T}}[] for _ in 1:nf]
    owner=Vector{Int}(undef,n)
    for s in 1:nf, j in starts[s]:starts[s+1]-1
        owner[j]=s
    end
    cpos=zeros(Int,n); mark=zeros(Int,n)
    prow=Vector{Int}(undef,n); pcol=Vector{Int}(undef,n); t=0
    lptr=Vector{Int}(undef,n+1); lptr[1]=1; lidx=Int[]; lval=T[]
    uptr=Vector{Int}(undef,n+1); uptr[1]=1; uidx=Int[]; uval=T[]
    fparent=zeros(Int,nf); shapes=Vector{Tuple{Int,Int}}(undef,nf); growth=initial; rlen=Int[]
    for s in 1:nf, c in children[s]
        fparent[c]=s
    end
    @inbounds for s in 1:nf
        a,b=starts[s],starts[s+1]-1; w=b-a+1
        FC=collect(a:b)
        for j in a:b
            mark[j]=s
        end
        for i in frows[s], x in rc[i]:rc[i+1]-1
            j=rj[x]; mark[j]==s || (mark[j]=s; push!(FC,j))
        end
        for (_,cols,_) in pend[s], j in cols
            mark[j]==s || (mark[j]=s; push!(FC,j))
        end
        sort!(view(FC,w+1:length(FC))); nc=length(FC)
        for (l,j) in enumerate(FC)
            cpos[j]=l
        end
        # Rows in staircase order (as the QR): by first column in the front,
        # so that a block of pivots only updates the rows already started
        # (the others have zero multipliers there).
        nown=length(frows[s]); nr=nown+length(pend[s])
        nr>=w || throw(SingularException(q[a+nr]))
        key=Vector{Int}(undef,nr); FR0=Vector{Int}(undef,nr)
        for (l,i) in enumerate(frows[s])
            key[l]=cpos[leftmost[i]]; FR0[l]=i
        end
        for (l,(i,cols,_)) in enumerate(pend[s])
            key[nown+l]=cpos[cols[1]]; FR0[nown+l]=i
        end
        order=sortperm(key); rowat=invperm(order); FR=FR0[order]; lm=key[order]
        shapes[s]=(nr,nc)
        F=zeros(T,nr,nc)
        for (l,i) in enumerate(frows[s]), x in rc[i]:rc[i+1]-1
            F[rowat[l],cpos[rj[x]]]+=rv[x]
        end
        for (l,(_,cols,vals)) in enumerate(pend[s]), x in eachindex(cols)
            F[rowat[nown+l],cpos[cols[x]]]=vals[x]
        end
        empty!(pend[s])
        # Row lengths in the front (fill-aware pivot choice, as UMFPACK).
        length(rlen)<nr && resize!(rlen,nr)
        for i in 1:nr
            rlen[i]=0
        end
        for j in 1:nc, i in 1:nr
            iszero(F[i,j]) || (rlen[i]+=1)
        end
        # Panel: threshold partial pivoting on columns 1:w over all rows, the
        # shortest row among those within τ of the largest entry (rows swapped
        # whole). Pivots in blocks of nb: right-looking within the block, then
        # TRSM/GEMM on every column to its right (a whole-panel rank-1 loop
        # ran a 2523² root front at BLAS 2 speed). Small fronts: one block,
        # then the Schur complement by loops.
        blocked=T<:NativeFloat && nr*nc>_MF_SMALL_FRONT^2
        nb=blocked ? 64 : w; stair=0
        for k0 in 1:nb:w
            k1=min(k0+nb-1,w)
            # Rows started by the block's columns (staircase); a pivot row
            # swap keeps that prefix as a set.
            while stair<nr && lm[stair+1]<=k1
                stair+=1
            end
            r1=max(stair,k1)
            for k in k0:k1
                # Column refinement (as UMFPACK): the candidate column of the
                # block with the fewest nonzeros in the remaining rows.
                jbest=0; cbest=typemax(Int)
                for j in k:k1
                    c=0
                    for i in k:r1
                        iszero(F[i,j]) || (c+=1)
                    end
                    0<c<cbest && (cbest=c; jbest=j)
                end
                jbest==0 && throw(SingularException(q[FC[k]]))
                if jbest!=k
                    for i in 1:nr
                        F[i,k],F[i,jbest]=F[i,jbest],F[i,k]
                    end
                    FC[k],FC[jbest]=FC[jbest],FC[k]
                end
                amax=zero(R)
                for i in k:r1
                    amax=max(amax,abs(F[i,k]))
                end
                (iszero(amax) || !isfinite(amax)) && throw(SingularException(q[FC[k]]))
                r=0; best=typemax(Int); bv=zero(R)
                for i in k:r1
                    v=abs(F[i,k]); (iszero(v) || v<pivot_threshold*amax) && continue
                    if rlen[i]<best || (rlen[i]==best && v>bv)
                        r=i; best=rlen[i]; bv=v
                    end
                end
                if r!=k
                    for j in 1:nc
                        F[k,j],F[r,j]=F[r,j],F[k,j]
                    end
                    FR[k],FR[r]=FR[r],FR[k]; rlen[k],rlen[r]=rlen[r],rlen[k]
                end
                ip=Base.inv(F[k,k])
                @simd for i in k+1:r1
                    F[i,k]*=ip
                end
                # A row updated by this pivot takes the pattern of the pivot
                # row: its length is bounded by |row ∪ pivot row| - 1 (the
                # stale lengths chose rows already filled in).
                lk=rlen[k]
                for i in k+1:r1
                    iszero(F[i,k]) || (rlen[i]=min(nc-k,rlen[i]+lk-2))
                end
                for j in k+1:k1
                    u=F[k,j]; iszero(u) && continue
                    @simd for i in k+1:r1
                        F[i,j]=muladd(-F[i,k],u,F[i,j])
                    end
                end
            end
            if blocked && nc>k1
                U12=view(F,k0:k1,k1+1:nc)
                trsm!(U12,view(F,k0:k1,k0:k1);uplo=:L,unit=true,threaded=false)
                r1>k1 && gemm!(view(F,k1+1:r1,k1+1:nc),view(F,k1+1:r1,k0:k1),U12,-one(T),one(T);threaded=false)
            end
        end
        if !blocked && nc>w
            for j in w+1:nc, k2 in 1:w
                u=F[k2,j]; iszero(u) && continue
                @simd for i in k2+1:w
                    F[i,j]=muladd(-F[i,k2],u,F[i,j])
                end
            end
            _lu_schur!(F,nr,w,w+1,nc)
        end
        g2=zero(R)
        for j in 1:nc
            @simd for i in 1:nr
                g2=max(g2,abs2(F[i,j]))
            end
        end
        g=isfinite(g2) ? sqrt(g2) : maximum(abs,F)
        isfinite(g) || throw(ArgumentError("multifrontal LU arithmetic overflow"))
        growth=max(growth,g)
        for k in 1:w
            t+=1; prow[t]=FR[k]; pcol[t]=FC[k]
            for i in k+1:nr
                v=F[i,k]; iszero(v) && continue
                push!(lidx,FR[i]); push!(lval,v)
            end
            lptr[t+1]=length(lidx)+1
            for j in k:nc
                v=F[k,j]; (iszero(v) && j!=k) && continue
                push!(uidx,FC[j]); push!(uval,v)
            end
            uptr[t+1]=length(uidx)+1
        end
        # Rows left: their nonzeros over the non-pivot columns (in increasing
        # position), sent to the front of the first one. A row left with no
        # nonzero means a singular matrix.
        for i in w+1:nr
            c=0
            for j in w+1:nc
                iszero(F[i,j]) || (c+=1)
            end
            c==0 && throw(SingularException(q[FC[w]]))
            cols=Vector{Int}(undef,c); vals=Vector{T}(undef,c); l=0
            for j in w+1:nc
                v=F[i,j]; iszero(v) && continue
                l+=1; cols[l]=FC[j]; vals[l]=v
            end
            push!(pend[owner[cols[1]]],(FR[i],cols,vals))
        end
    end
    t==n || error("multifrontal LU did not eliminate every pivot")
    # Column labels are positions in q; rows are rows of A.
    L,U=_lu_assemble(n,prow,pcol,lptr,lidx,lval,length(lidx),uptr,uidx,uval,length(uidx),T)
    pieces=[starts[s]:starts[s+1]-1 for s in 1:nf]
    MultifrontalLUFactor(L,U,copy(prow),[q[k] for k in pcol],fparent,pieces,shapes,R(pivot_threshold),
                         iszero(initial) ? one(R) : growth/initial)
end
