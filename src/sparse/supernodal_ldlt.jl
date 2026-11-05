# SPDX-License-Identifier: Apache-2.0
# Supernodal multifrontal LDLᵀ / LDLᴴ with Bunch–Kaufman pivoting:
# A[p,p] = L·D·Lᴴ (or L·D·Lᵀ for a complex symmetric matrix), D with 1×1 and
# 2×2 blocks.
#   1. symbolic analysis as for the Cholesky factorization (ordering,
#      elimination tree, column counts, weighted postorder, relaxed
#      supernodes; src/sparse/supernodal_cholesky.jl);
#   2. numeric phase in postorder with a stack of contribution blocks (lower
#      triangles). In each front the fully summed variables are those of the
#      supernode and those delayed by the children. Bunch–Kaufman runs on
#      them with the column and row maxima taken over the whole front, so its
#      growth bound holds; when the largest entry of a column lies outside the
#      candidates, its diagonal is accepted if it passes the threshold test of
#      the sparse codes (|a_kk| ≥ u·colmax, u = 0.01 as in MA57: multipliers
#      ≤ 100), and the variable is delayed to the parent front otherwise. With
#      the Bunch–Kaufman bound alone (α ≈ 0.64) the delays nearly doubled the
#      factor of a 150² Helmholtz matrix. At a root every variable is fully
#      summed.

const _LDL_THRESHOLD=0.01

# Swap variables p < q of the lower-stored symmetric f×f front F (LAPACK
# xSYTF2/xHETF2 pattern).
function _ldl_sym_swap!(F,f,p,q,hermitian)
    p==q && return F
    @inbounds begin
        for j in 1:p-1
            F[p,j],F[q,j]=F[q,j],F[p,j]
        end
        F[p,p],F[q,q]=F[q,q],F[p,p]
        for j in p+1:q-1
            t=F[j,p]; F[j,p]=_ldlconj(F[q,j],hermitian); F[q,j]=_ldlconj(t,hermitian)
        end
        F[q,p]=_ldlconj(F[q,p],hermitian)
        for i in q+1:f
            F[i,p],F[i,q]=F[i,q],F[i,p]
        end
    end
    F
end
# Entry (i,j) of the lower-stored symmetric front.
@inline _ldl_entry(F,i,j,hermitian)=i>=j ? F[i,j] : _ldlconj(F[j,i],hermitian)

# F[i,j] -= Σ_{c≤e} F[i,c]·conj(W[j,c]) for j0 ≤ j ≤ i ≤ f (Schur update of the
# columns the panel did not update; W holds the eliminated columns of L·D),
# four c per pass.
function _ldl_schur!(F,f,j0,e,Wb,hermitian)
    @inbounds for j in j0:f
        c=1
        while c+3<=e
            a1,a2,a3,a4=_ldlconj(Wb[j,c],hermitian),_ldlconj(Wb[j,c+1],hermitian),_ldlconj(Wb[j,c+2],hermitian),_ldlconj(Wb[j,c+3],hermitian)
            @simd for i in j:f
                F[i,j]=muladd(-F[i,c+3],a4,muladd(-F[i,c+2],a3,muladd(-F[i,c+1],a2,muladd(-F[i,c],a1,F[i,j]))))
            end
            c+=4
        end
        while c<=e
            a=_ldlconj(Wb[j,c],hermitian)
            @simd for i in j:f
                F[i,j]=muladd(-F[i,c],a,F[i,j])
            end
            c+=1
        end
    end
    F
end

function _mf_ldlt(A::NativeSparse,uplo,hermitian,pivot,ordering,maxwidth;check::Bool=true)
    n=_square(A); T=eltype(A); R=real(T); ident=collect(1:n)
    hermitian=hermitian && T<:Complex || T<:Real   # real symmetric = Hermitian
    lp,li,lv,_=_chol_permuted(A,uplo,ident,false,hermitian)
    all(isfinite,lv) || throw(ArgumentError("finite representable LDL entries required"))
    initial=maximum(abs,lv;init=zero(R))
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
    lp,li,lv,up,ui,_=_chol_permuted(A,uplo,pinv,true,hermitian)
    parent=_etree(up,ui,n)
    counts=_chol_counts(lp,li,parent,_postorder(parent),n)
    post=_postorder(parent,counts)
    if post!=ident
        p=p[post]; pinv=invperm(p); ipost=invperm(post); counts=counts[post]
        relabeled=Vector{Int}(undef,n)
        @inbounds for k in 1:n
            q=parent[post[k]]; relabeled[k]=q==0 ? 0 : ipost[q]
        end
        parent=relabeled
        lp,li,lv,up,ui,_=_chol_permuted(A,uplo,pinv,true,hermitian)
    end
    nodes=_chol_supernodes(parent,counts,n,maxwidth)
    rows,sparent,schildren=_chol_supernode_rows(nodes,lp,li,parent,n)
    α=(one(R)+sqrt(R(17)))/8
    # Output in elimination order t: L columns (labels), D blocks.
    order=Vector{Int}(undef,n); t=0
    guess=sum(counts;init=0)+n
    lptr=Vector{Int}(undef,n+1); lptr[1]=1; lidx=Vector{Int}(undef,guess); lval=Vector{T}(undef,guess); lnz=0
    diagonal=zeros(T,n); offdiagonal=zeros(T,max(n-1,0)); blocks=Int[]
    # Contribution blocks: lower triangles (m×m, packed by columns) in
    # `arena`, labels in `iarena`; `stack` keeps (value offset, label offset,
    # m, delayed count).
    arena=Vector{T}(undef,0); vtop=0; iarena=Vector{Int}(undef,0); itop=0; stack=NTuple{4,Int}[]
    # Front buffers sized from the symbolic fronts (delays may enlarge them).
    maxf=maximum(s->length(nodes[s])+length(rows[s]),eachindex(nodes);init=0)
    m0=maxf+maxf÷8
    pos=zeros(Int,n); FR=Int[]; W=Matrix{T}(undef,m0,m0); Wb=Matrix{T}(undef,m0,m0)
    l1=Vector{T}(undef,m0); l2=Vector{T}(undef,m0); growth=initial
    sizehint!(FR,m0); resize!(arena,maxf*(maxf+1)); resize!(iarena,4maxf)
    @inbounds for (s,node) in enumerate(nodes)
        a,b=first(node),last(node); Rs=rows[s]; nch=length(schildren[s])
        empty!(FR)
        for c in length(stack)-nch+1:length(stack)
            _,io,m,nd=stack[c]
            for l in 1:nd
                push!(FR,iarena[io+l-1])
            end
        end
        append!(FR,node); nfs=length(FR); append!(FR,Rs); f=length(FR)
        for l in 1:f
            pos[FR[l]]=l
        end
        if f>size(W,1)
            m2=max(f,2size(W,1)); W=Matrix{T}(undef,m2,m2); Wb=Matrix{T}(undef,m2,m2); resize!(l1,m2); resize!(l2,m2)
        end
        F=view(W,1:f,1:f)
        for j in 1:f, i in j:f
            F[i,j]=zero(T)
        end
        # Original entries of the supernode's columns (lower part).
        for j in node, q in lp[j]:lp[j+1]-1
            i,jj=pos[li[q]],pos[j]
            i>=jj ? (F[i,jj]+=lv[q]) : (F[jj,i]+=_ldlconj(lv[q],hermitian))
        end
        # Children's contribution blocks (lower triangles over their labels).
        for _ in 1:nch
            vo,io,m,_=pop!(stack); tt=vo
            for jj in 1:m
                pj=pos[iarena[io+jj-1]]
                for ii in jj:m
                    pi=pos[iarena[io+ii-1]]; v=arena[tt]; tt+=1
                    pi>=pj ? (F[pi,pj]+=v) : (F[pj,pi]+=_ldlconj(v,hermitian))
                end
            end
            vtop=vo-1; itop=io-1
        end
        # Bunch–Kaufman on the fully summed variables 1:ncand, right-looking
        # within columns 1:nfs; delayed variables move behind the candidates.
        isroot=sparent[s]==0; k=1; ncand=nfs; nb0=length(blocks)
        while k<=ncand
            hermitian && (F[k,k]=T(real(F[k,k])))
            absakk=abs(F[k,k]); colmax=zero(R); imax=k
            for i in k+1:f
                v=abs(F[i,k]); isfinite(v) || throw(ArgumentError("sparse LDL arithmetic overflow"))
                v>colmax && (colmax=v; imax=i)
            end
            isfinite(absakk) || throw(ArgumentError("sparse LDL arithmetic overflow"))
            step=1; r=k
            if pivot===:none
                iszero(absakk) && (check || !iszero(colmax)) && throw(SingularException(p[FR[k]]))
            elseif absakk>=α*colmax && !iszero(absakk)
                step=1
            elseif iszero(colmax)
                check && throw(SingularException(p[FR[k]]))
            elseif imax>ncand && absakk>=_LDL_THRESHOLD*colmax && !iszero(absakk)
                # The largest entry is outside the candidates: a 1×1 pivot
                # passing the sparse threshold test (multipliers ≤ 1/u).
                step=1
            elseif imax>ncand
                # Otherwise delay k.
                isroot && error("LDL: delayed pivot at a root front")
                _ldl_sym_swap!(F,f,k,ncand,hermitian); FR[k],FR[ncand]=FR[ncand],FR[k]
                pos[FR[k]]=k; pos[FR[ncand]]=ncand; ncand-=1; continue
            else
                rowmax=zero(R)
                for j in k:f
                    j==imax && continue
                    rowmax=max(rowmax,abs(_ldl_entry(F,imax,j,hermitian)))
                end
                if absakk*rowmax>=α*colmax^2
                    step=1
                elseif abs(F[imax,imax])>=α*rowmax
                    step=1; r=imax
                else
                    step=2; r=imax
                end
            end
            if step==1
                if r!=k
                    _ldl_sym_swap!(F,f,k,r,hermitian); FR[k],FR[r]=FR[r],FR[k]; pos[FR[k]]=k; pos[FR[r]]=r
                end
                hermitian && (F[k,k]=T(real(F[k,k])))
                d=F[k,k]; isfinite(d) || throw(ArgumentError("sparse LDL arithmetic overflow"))
                if iszero(d)
                    check && throw(SingularException(p[FR[k]]))
                else
                    id=Base.inv(d)
                    for i in k+1:f
                        Wb[i,k]=F[i,k]
                    end
                    for j in k+1:nfs
                        c=_ldlconj(Wb[j,k],hermitian)*id
                        @simd for i in j:f
                            F[i,j]=muladd(-Wb[i,k],c,F[i,j])
                        end
                    end
                    @simd for i in k+1:f
                        F[i,k]*=id
                    end
                end
                t+=1; order[t]=FR[k]; diagonal[t]=d; push!(blocks,1)
            else
                if r!=k+1
                    _ldl_sym_swap!(F,f,k+1,r,hermitian); FR[k+1],FR[r]=FR[r],FR[k+1]; pos[FR[k+1]]=k+1; pos[FR[r]]=r
                end
                hermitian && (F[k+1,k+1]=T(real(F[k+1,k+1])))
                da,db,dc=F[k,k],F[k+1,k],F[k+1,k+1]
                scale,aa,bb,cc,bh,det=_ldl_block_scaled(da,db,dc,hermitian,t+1)
                for i in k+2:f
                    Wb[i,k]=F[i,k]; Wb[i,k+1]=F[i,k+1]
                    x,y=F[i,k]/scale,F[i,k+1]/scale
                    l1[i]=(x*cc-y*bb)/det; l2[i]=(y*aa-x*bh)/det
                end
                for j in k+2:nfs
                    c1=_ldlconj(Wb[j,k],hermitian); c2=_ldlconj(Wb[j,k+1],hermitian)
                    @simd for i in j:f
                        F[i,j]=muladd(-l2[i],c2,muladd(-l1[i],c1,F[i,j]))
                    end
                end
                for i in k+2:f
                    F[i,k]=l1[i]; F[i,k+1]=l2[i]
                end
                # Wb keeps the eliminated columns of L·D for the Schur update.
                t+=1; order[t]=FR[k]; diagonal[t]=da; offdiagonal[t]=db
                t+=1; order[t]=FR[k+1]; diagonal[t]=dc; push!(blocks,2)
            end
            k+=step
        end
        # Delayed variables (e+1:nfs) were updated by every pivot of the panel.
        e=ncand
        if f>nfs && e>0
            # Columns nfs+1:f: S -= L[:,1:e]·W[:,1:e]ᴴ (W = L·D, saved).
            if f>_MF_SMALL_FRONT && T<:NativeFloat
                gemmt!(view(F,nfs+1:f,nfs+1:f),view(F,nfs+1:f,1:e),view(Wb,nfs+1:f,1:e),-one(T),one(T);
                       uplo=:L,transB=hermitian ? :C : :T)
            else
                _ldl_schur!(F,f,nfs+1,e,Wb,hermitian)
            end
        end
        g2=zero(R)
        for j in 1:f
            @simd for i in j:f
                g2=max(g2,abs2(F[i,j]))
            end
        end
        g=isfinite(g2) ? sqrt(g2) : maximum(abs,F)
        isfinite(g) || throw(ArgumentError("sparse LDL multipliers overflow"))
        growth=max(growth,g)
        # L columns of the pivots 1:e (the second column of a 2×2 block has
        # no entry in the first's row: rows start after the block).
        need=lnz+e*f
        need>length(lidx) && (resize!(lidx,max(need,2length(lidx))); resize!(lval,length(lidx)))
        k=1; tt=t-e
        for bk in nb0+1:length(blocks)
            step=blocks[bk]
            for c in k:k+step-1
                tt+=1
                for i in k+step:f
                    v=F[i,c]; iszero(v) && continue
                    lnz+=1; lidx[lnz]=FR[i]; lval[lnz]=v
                end
                lptr[tt+1]=lnz+1
            end
            k+=step
        end
        # Contribution block: positions e+1:f (delayed variables first).
        if f>e
            m=f-e; nd=nfs-e
            need=vtop+m*(m+1)÷2; need>length(arena) && resize!(arena,max(need,2length(arena)))
            ineed=itop+m; ineed>length(iarena) && resize!(iarena,max(ineed,2length(iarena)))
            vo=vtop+1; io=itop+1; tt2=vo
            for jj in e+1:f, ii in jj:f
                arena[tt2]=F[ii,jj]; tt2+=1
            end
            for l in 1:m
                iarena[io+l-1]=FR[e+l]
            end
            vtop=need; itop=ineed; push!(stack,(vo,io,m,nd))
        end
    end
    t==n || error("sparse LDL did not eliminate every variable")
    # L in elimination order (unit diagonal stored), sorted columns.
    position=Vector{Int}(undef,n)
    @inbounds for k in 1:n
        position[order[k]]=k
    end
    colptr=Vector{Int}(undef,n+1); colptr[1]=1
    @inbounds for k in 1:n
        colptr[k+1]=colptr[k]+1+(lptr[k+1]-lptr[k])
    end
    ri=Vector{Int}(undef,colptr[n+1]-1); rv=Vector{T}(undef,length(ri))
    @inbounds for k in 1:n
        x=colptr[k]; ri[x]=k; rv[x]=one(T)
        for q in lptr[k]:lptr[k+1]-1
            x+=1; ri[x]=position[lidx[q]]; rv[x]=lval[q]
        end
    end
    # Rows are already ascending unless pivots were delayed or swapped:
    # insertion sort of the columns that need it.
    @inbounds for k in 1:n
        lo,hi=colptr[k]+1,colptr[k+1]-1
        for x in lo+1:hi
            r=ri[x]; ri[x-1]<=r && continue
            v=rv[x]; y=x-1
            while y>=lo && ri[y]>r
                ri[y+1]=ri[y]; rv[y+1]=rv[y]; y-=1
            end
            ri[y+1]=r; rv[y+1]=v
        end
    end
    L=CSCMatrix(_Shared(),n,n,colptr,ri,rv)
    # D as CSR: a 1×1 block is one row entry, a 2×2 block two per row.
    dp=Vector{Int}(undef,n+1); dp[1]=1; dc=Int[]; dv=T[]; sizehint!(dc,n+2length(blocks)); sizehint!(dv,n+2length(blocks)); k=1
    for step in blocks
        if step==1
            push!(dc,k); push!(dv,diagonal[k]); dp[k+1]=length(dc)+1
        else
            push!(dc,k); push!(dv,diagonal[k]); push!(dc,k+1); push!(dv,_ldlconj(offdiagonal[k],hermitian)); dp[k+1]=length(dc)+1
            push!(dc,k); push!(dv,offdiagonal[k]); push!(dc,k+1); push!(dv,diagonal[k+1]); dp[k+2]=length(dc)+1
        end
        k+=step
    end
    D=CSRMatrix(_Shared(),n,n,dp,dc,dv)
    SparseLDLFactor(L,D,diagonal,offdiagonal,p[order],blocks,Bool(hermitian),iszero(initial) ? one(R) : growth/initial)
end
