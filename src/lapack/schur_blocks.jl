# SPDX-License-Identifier: Apache-2.0
# Block back/forward substitution ports TREVC/TRSNA from evd/schur.rs.
# Unlike upstream's scalar recurrence, substitution retains every intervening
# real 2x2 block. Complex arithmetic represents the paired real/imag columns.

function _schur_blocks(S)
    n=_square(S); blocks=UnitRange{Int}[]; j=1
    while j<=n
        r=(eltype(S)<:Real && j<n && !iszero(S[j+1,j])) ? (j:j+1) : (j:j)
        push!(blocks,r); j=last(r)+1
    end
    blocks
end

function _check_schur_form(S,B=nothing)
    n=_square(S); _finite_scale(S)
    B===nothing || (size(B)==(n,n) || throw(DimensionMismatch("pencil dimensions differ")); _finite_scale(B))
    for j in 1:n, i in j+1:n
        (i==j+1 && eltype(S)<:Real) || iszero(S[i,j]) || throw(ArgumentError("matrix is not in Schur form"))
        B===nothing || iszero(B[i,j]) || throw(ArgumentError("second Schur matrix must be triangular"))
    end
    for i in 2:n-1
        !iszero(S[i,i-1]) && !iszero(S[i+1,i]) && throw(ArgumentError("overlapping real Schur blocks"))
    end
    nothing
end

function _small_solve(A,b;perturb=false)
    T=eltype(A); R=real(T); s=maximum(abs,A;init=zero(R))
    !perturb && all(iszero,b) && return zeros(T,size(b))
    iszero(s) && (s=one(R))
    perturb && (s=max(s,one(R)))
    F=lu_complete(A./s;check=false,threaded=false)
    if perturb
        # TREVC perturbation applies to the tiny complete-pivot LU pivots,
        # relative to the normalized pencil, before division of the RHS.
        for i in axes(F.factors,1)
            z=F.factors[i,i]; floor=max(floatmin(R)/eps(R),eps(R))
            if abs(z)<floor
                F.factors[i,i]=iszero(z) ? T(floor) : (z/abs(z))*floor
            end
        end
        F=CompleteLUFactor(F.factors,F.p,F.q,0,F.pivot)
    elseif F.info!=0
        throw(ArgumentError("inseparable Schur blocks / singular Sylvester operator"))
    end
    solve(F,b./s)
end

# _small_solve(D,r;perturb=true) for a 1×1 block D=[d], without arrays:
# same normalization s, same pivot perturbation, x = (r/s)/(d/s).
@inline function _small_solve1(d::C,r::C) where {C}
    R=real(C); s=max(abs(d),one(R)); z=d/s
    floor=max(floatmin(R)/eps(R),eps(R))
    abs(z)<floor && (z=iszero(z) ? C(floor) : (z/abs(z))*floor)
    (r/s)/z
end

# _small_solve(D,r;perturb=true) for a 2×2 block D=[d11 d12; d21 d22], in
# scalars: same normalization s, complete pivoting (first largest entry in
# column-major order, as _complete_pivot), the same perturbation of tiny
# pivots, then L and U substitutions. Returns (x1, x2).
@inline function _small_solve2(d11::C,d21::C,d12::C,d22::C,r1::C,r2::C) where {C}
    R=real(C); s=max(abs(d11),abs(d21),abs(d12),abs(d22),one(R))
    a11,a21,a12,a22=d11/s,d21/s,d12/s,d22/s; b1,b2=r1/s,r2/s
    # Pivot position (p,q) and swaps bringing it to (1,1).
    p=q=1; best=abs(a11)
    abs(a21)>best && (p=2; q=1; best=abs(a21))
    abs(a12)>best && (p=1; q=2; best=abs(a12))
    abs(a22)>best && (p=2; q=2)
    if p==2
        a11,a21=a21,a11; a12,a22=a22,a12; b1,b2=b2,b1
    end
    if q==2
        a11,a12=a12,a11; a21,a22=a22,a21
    end
    # Elimination (skipped on an exactly zero pivot, as lu_complete with
    # check=false), then the perturbation of the factored diagonal.
    l=a21; u22=a22
    if !iszero(a11)
        l=a21/a11; u22=a22-l*a12
    end
    floor=max(floatmin(R)/eps(R),eps(R))
    u11=a11
    abs(u11)<floor && (u11=iszero(u11) ? C(floor) : (u11/abs(u11))*floor)
    abs(u22)<floor && (u22=iszero(u22) ? C(floor) : (u22/abs(u22))*floor)
    y2=b2-l*b1
    x2=y2/u22; x1=(b1-a12*x2)/u11
    q==2 ? (x2,x1) : (x1,x2)
end

function _pencil_weights(a,b,sa,sb)
    R=real(typeof(a)); C=Complex{R}
    aa=a*sb; bb=b*sa; t=max(abs(aa),abs(bb))
    if isfinite(t) && t>0 && (iszero(a) || !iszero(aa)) && (iszero(b) || !iszero(bb))
        return _realdiv(aa,t),_realdiv(bb,t)
    end
    function phase_log(z)
        s=max(abs(real(z)),abs(imag(z)))
        iszero(s) && return (zero(C),-R(Inf))
        # wt, not t: assigning t here rebound the enclosing function's t.
        w=_realdiv(C(z),s); wt=abs(w)
        w/wt,log(s)+log(wt)
    end
    pa,la=phase_log(a); pb,lb=phase_log(b)
    la+=log(sb); lb+=log(sa); largest=max(la,lb)
    pa*exp(la-largest),pb*exp(lb-largest)
end

# w[1:hi] -= (ks·S[1:hi,j] - kb·B[1:hi,j])·xj: one contiguous column of S
# (and B), as in LAPACK trevc.
@inline function _trevc_update!(w,S,B,ordinary,ks,kb,j,hi,xj)
    c=ks*xj
    if ordinary
        @inbounds @simd for i in 1:hi
            w[i]=muladd(-S[i,j],c,w[i])
        end
    else
        d=kb*xj
        @inbounds @simd for i in 1:hi
            w[i]=muladd(B[i,j],d,muladd(-S[i,j],c,w[i]))
        end
    end
    w
end

# B === nothing: ordinary eigenvectors (B is the identity, never formed).
function _pencil_vectors(S,B,alpha,beta;left=false)
    ordinary=B===nothing
    ordinary ? _check_schur_form(S) : _check_schur_form(S,B)
    n=size(S,1); R=real(eltype(S)); C=Complex{R}
    length(alpha)==length(beta)==n || throw(DimensionMismatch("homogeneous pair dimensions differ"))
    blocks=_schur_blocks(S); X=zeros(C,n,n)
    sa=_finite_scale(S); sb=ordinary ? one(R) : _finite_scale(B)
    Bij(i,j)=ordinary ? (i==j ? one(R) : zero(R)) : B[i,j]
    # Right eigenvectors are computed column-oriented (w: pending right-hand
    # sides); Schur blocks are 1×1 or 2×2, the general path stays for others.
    columnwise=!left && maximum(length,blocks;init=0)<=2
    w=columnwise ? Vector{C}(undef,n) : C[]
    for (index,block) in enumerate(blocks), col in block
        a,b=alpha[col],beta[col]
        iszero(a) && iszero(b) && throw(ArgumentError("indeterminate singular pencil has no unique eigenvectors"))
        # Normalize S/B separately before homogeneous combinations.
        aa,bb=_pencil_weights(a,b,sa,sb)
        entry(i,j)=bb*_realdiv(C(S[i,j]),sa)-aa*_realdiv(C(Bij(i,j)),sb)
        m(i,j)=left ? conj(entry(j,i)) : entry(i,j)
        # Off-diagonal entries for the O(n^3) substitution sums: the scale
        # factors are formed once per column, and the B term vanishes for
        # ordinary eigenvectors (two complex divisions per term before).
        ks=bb/sa; kb=aa/sb
        offentry(i,j)=ordinary ? ks*S[i,j] : ks*S[i,j]-kb*B[i,j]
        off(i,j)=left ? conj(offentry(j,i)) : offentry(i,j)
        if length(block)==1
            X[first(block),col]=one(C)
        else
            i,j=first(block),last(block)
            if hypot(abs(m(i,i)),abs(m(i,j)))>=hypot(abs(m(j,i)),abs(m(j,j)))
                u,v=-m(i,j),m(i,i)
            else
                u,v=-m(j,j),m(j,i)
            end
            r=hypot(abs(u),abs(v))
            iszero(r) && (u=one(C); r=one(R))
            X[i,col]=u/r; X[j,col]=v/r
        end
        if columnwise
            # Same equations as the row-oriented loop below, summed in
            # another order: after each solved block, w is updated by the
            # columns of S above it (contiguous), and the column maximum is
            # tracked instead of rescanned.
            fill!(w,zero(C)); xmax=zero(R)
            for j in block
                xmax=max(xmax,abs(X[j,col]))
                _trevc_update!(w,S,B,ordinary,ks,kb,j,first(block)-1,X[j,col])
            end
            for k in index-1:-1:1
                rows=blocks[k]
                if length(rows)==1
                    row=first(rows); acc=w[row]
                    if abs(acc)>0
                        x1=_small_solve1(C(m(row,row)),acc)
                        isfinite(x1) || throw(OverflowError("Schur eigenvector substitution overflow"))
                        X[row,col]=x1; xmax=max(xmax,abs(x1))
                    end
                else
                    r1,r2=first(rows),last(rows); acc1,acc2=w[r1],w[r2]
                    if max(abs(acc1),abs(acc2))>0
                        x1,x2=_small_solve2(C(m(r1,r1)),C(m(r2,r1)),C(m(r1,r2)),C(m(r2,r2)),acc1,acc2)
                        isfinite(x1) && isfinite(x2) || throw(OverflowError("Schur eigenvector substitution overflow"))
                        X[r1,col]=x1; X[r2,col]=x2; xmax=max(xmax,abs(x1),abs(x2))
                    end
                end
                # Rescale the computed part and the pending right-hand sides
                # (w is linear in x), preserving the equations.
                if xmax>1
                    for q in first(rows):last(block); X[q,col]/=xmax; end
                    for q in 1:first(rows)-1; w[q]/=xmax; end
                    xmax=one(R)
                end
                for j in rows
                    xj=X[j,col]; iszero(xj) || _trevc_update!(w,S,B,ordinary,ks,kb,j,first(rows)-1,xj)
                end
            end
            t=nrm2(view(X,:,col)); view(X,:,col)./=t
            continue
        end
        order=left ? (index+1:length(blocks)) : (index-1:-1:1)
        for k in order
            rows=blocks[k]
            known=left ? (first(block):first(rows)-1) : (last(rows)+1:last(block))
            if length(rows)==1
                # 1×1 block (every block of a complex Schur form): the scalar
                # form of the 2×2 path below, without its small arrays.
                # (New names: r and i of the enclosing loop hold other values.)
                row=first(rows); acc=zero(C)
                for j in known
                    acc-=off(row,j)*X[j,col]
                end
                if abs(acc)>0
                    x1=_small_solve1(C(m(row,row)),acc)
                    isfinite(x1) || throw(OverflowError("Schur eigenvector substitution overflow"))
                    X[row,col]=x1
                    t=maximum(abs,view(X,:,col)); t>1 && (view(X,:,col)./=t)
                end
                continue
            end
            if length(rows)==2
                # 2×2 block (a complex pair of a real Schur form): scalar
                # form of the general path below, without its small arrays.
                r1,r2=first(rows),last(rows); acc1=zero(C); acc2=zero(C)
                for j in known
                    xj=X[j,col]; acc1-=off(r1,j)*xj; acc2-=off(r2,j)*xj
                end
                if max(abs(acc1),abs(acc2))>0
                    x1,x2=_small_solve2(C(m(r1,r1)),C(m(r2,r1)),C(m(r1,r2)),C(m(r2,r2)),acc1,acc2)
                    isfinite(x1) && isfinite(x2) || throw(OverflowError("Schur eigenvector substitution overflow"))
                    X[r1,col]=x1; X[r2,col]=x2
                    t=maximum(abs,view(X,:,col)); t>1 && (view(X,:,col)./=t)
                end
                continue
            end
            D=C[m(i,j) for i in rows,j in rows]; rhs=zeros(C,length(rows))
            for (ii,i) in enumerate(rows), j in known
                rhs[ii]-=off(i,j)*X[j,col]
            end
            if maximum(abs,rhs;init=zero(R))>0
                y=_small_solve(D,rhs;perturb=true)
                all(isfinite,y) || throw(OverflowError("Schur eigenvector substitution overflow"))
                X[rows,col]=y
                # Rescale the entire partial vector, preserving its equation.
                t=maximum(abs,view(X,:,col)); t>1 && (view(X,:,col)./=t)
            end
        end
        t=nrm2(view(X,:,col)); view(X,:,col)./=t
    end
    X
end

function _ordinary_vectors(S;left=false)
    _check_schur_form(S); C=Complex{real(eltype(S))}; n=size(S,1)
    values=_schur_values(S,one(real(eltype(S))))
    _pencil_vectors(S,nothing,C.(values),ones(C,n);left)
end

"""
    trevc_right(S::AbstractMatrix{T}; packed=T<:Real) -> AbstractMatrix

Native block TREVC. Real input stores a conjugate pair as real/imaginary
columns, as in OxiBLAS Rust; complex input returns complex eigenvectors.
`packed=false` returns one complex vector per eigenvalue for real input too.
"""
function trevc_right(S::AbstractMatrix{T};packed=T<:Real) where {T<:GenericFloat}
    _trevc(S,false,packed)
end
function trevc_left(S::AbstractMatrix{T};packed=T<:Real) where {T<:GenericFloat}
    _trevc(S,true,packed)
end
function _trevc(S,left,packed)
    X=_ordinary_vectors(S;left)
    (!packed || eltype(S)<:Complex) && return X
    V=zeros(eltype(S),size(S)); values=_schur_values(S,one(eltype(S)))
    for block in _schur_blocks(S)
        j=first(block)
        if length(block)==2 && !iszero(imag(values[j]))
            V[:,j]=real.(view(X,:,j)); V[:,j+1]=imag.(view(X,:,j))
        else
            for col in block
                V[:,col]=real.(view(X,:,col))
            end
        end
    end
    V
end

"""
    trsna_s(S::AbstractMatrix{T}) -> Vector{real(T)}

Reciprocal eigenvalue condition numbers |y'*x|/(norm(y)*norm(x)).
Ports Rust TRSNA-S, including one common value for a real conjugate pair.
"""
function trsna_s(S::AbstractMatrix{T}) where {T<:GenericFloat}
    X=_ordinary_vectors(S); Y=_ordinary_vectors(S;left=true)
    [min(one(real(T)),abs(dotc(view(Y,:,j),view(X,:,j)))) for j in axes(S,1)]
end

"""
    trsna_sep(S::AbstractMatrix{T}) -> Vector{real(T)}

Rust TRSNA-SEP's spectral-distance estimate (not a singular-value sep).
Repeated eigenvalues return zero; an isolated block has no complement and
returns Inf. Use `schur_condition` for the coupled invariant-subspace estimate.
"""
function trsna_sep(S::AbstractMatrix{T}) where {T<:GenericFloat}
    _check_schur_form(S); vals=_schur_values(S,one(real(T))); R=real(T); result=fill(R(Inf),length(vals))
    for block in _schur_blocks(S), j in block, k in eachindex(vals)
        k in block && continue
        result[j]=min(result[j],abs(vals[j]-vals[k]))
    end
    result
end

schur_eigenvectors(F::SchurFactor;left=false)=_mm(Complex{real(eltype(F.Z))}.(F.Z),_ordinary_vectors(F.T;left))

# Local generalized Sylvester systems. Blocks are at most 2x2: at most eight
# unknowns. No Kronecker matrix proportional to n^4 is constructed.
function _small_scaled_solve(A,b)
    T=eltype(A); R=real(T); n=size(A,1)
    all(iszero,b) && return (zeros(T,n),one(R))
    s=_finite_scale(A); F=lu_complete(A./s;check=false,threaded=false)
    F.info==0 || throw(ArgumentError("inseparable Schur blocks / singular Sylvester operator"))
    norm=max(s,_finite_scale(b)); scale=s/norm; x=b[F.p]./norm
    limit=sqrt(floatmax(R))/max(1,4n)
    for j in 1:n, i in j+1:n
        x[i]-=F.factors[i,j]*x[j]
    end
    for j in n:-1:1
        bound=limit*abs(F.factors[j,j]); magnitude=abs(x[j])
        if magnitude>bound
            factor=bound/magnitude; x .*= factor; scale*=factor
            scale>0 || throw(OverflowError("Sylvester scale underflow; use a wider scalar type"))
        end
        x[j]/=F.factors[j,j]
        for i in 1:j-1
            x[i]-=F.factors[i,j]*x[j]
        end
    end
    result=similar(x)
    for j in 1:n
        result[F.q[j]]=x[j]
    end
    result,scale
end

function _coupled_local(A,D,B,E,C,F)
    T=eltype(A); p,q=size(C); m=p*q; K=zeros(T,2m,2m)
    for j in 1:q, i in 1:p
        row=i+(j-1)*p
        for k in 1:p
            K[row,k+(j-1)*p]=A[i,k]; K[m+row,k+(j-1)*p]=D[i,k]
        end
        for k in 1:q
            K[row,m+i+(k-1)*p]=-B[k,j]; K[m+row,m+i+(k-1)*p]=-E[k,j]
        end
    end
    v,scale=_small_scaled_solve(K,vcat(vec(C),vec(F)))
    reshape(v[1:m],p,q),reshape(v[m+1:end],p,q),scale
end

"""
    generalized_sylvester(A::AbstractMatrix{T}, D::AbstractMatrix{T}, B::AbstractMatrix{T}, E::AbstractMatrix{T}, C::AbstractMatrix{T}, F::AbstractMatrix{T}) -> Tuple{Matrix{T}, Matrix{T}, Number}

Solve A*R-L*B=C, D*R-L*E=F for two (quasi-)triangular pencils.
Block substitution with complete pivoting of each 1x1/2x2 coupled system.
The result satisfies the *scaled* right-hand sides: scale*C and scale*F.
"""
function generalized_sylvester(A::AbstractMatrix{T},D::AbstractMatrix{T},B::AbstractMatrix{T},E::AbstractMatrix{T},C::AbstractMatrix{T},F::AbstractMatrix{T}) where {T<:GenericFloat}
    Base.require_one_based_indexing(A,D,B,E,C,F)
    p,q=_square(A),_square(B)
    size(D)==(p,p) && size(E)==(q,q) && size(C)==size(F)==(p,q) || throw(DimensionMismatch("Sylvester matrix dimensions differ"))
    _check_schur_form(A,D); _check_schur_form(B,E)
    _finite_scale(C); _finite_scale(F); RR=zeros(T,p,q); L=zeros(T,p,q)
    # A single global normalization preserves both coupled equations.
    s=max(_finite_scale(A),_finite_scale(B),_finite_scale(D),_finite_scale(E))
    rhsnorm=max(s,_finite_scale(C),_finite_scale(F))
    a=A./s; d=D./s; b=B./s; e=E./s; c=C./rhsnorm; f=F./rhsnorm; scale=s/rhsnorm
    rb=_schur_blocks(A); cb=_schur_blocks(B); limit=sqrt(floatmax(real(T)))/max(1,p+q)
    for J in cb, I in reverse(rb)
        rhs=copy(view(c,I,J)); rhs2=copy(view(f,I,J))
        for (jj,j) in enumerate(J), (ii,i) in enumerate(I)
            for k in last(I)+1:p
                rhs[ii,jj]-=a[i,k]*RR[k,j]; rhs2[ii,jj]-=d[i,k]*RR[k,j]
            end
            for k in 1:first(J)-1
                rhs[ii,jj]+=L[i,k]*b[k,j]; rhs2[ii,jj]+=L[i,k]*e[k,j]
            end
        end
        r,l,local_scale=_coupled_local(view(a,I,I),view(d,I,I),view(b,J,J),view(e,J,J),rhs,rhs2)
        if local_scale<1
            RR .*= local_scale; L .*= local_scale; c .*= local_scale; f .*= local_scale; scale*=local_scale
            scale>0 || throw(OverflowError("Sylvester global scale underflow; use a wider scalar type"))
        end
        t=max(maximum(abs,r;init=zero(real(T))),maximum(abs,l;init=zero(real(T))))
        isfinite(t) || throw(OverflowError("generalized Sylvester solve overflow"))
        if t>limit
            factor=limit/t; RR .*= factor; L .*= factor; c .*= factor; f .*= factor; scale*=factor
            scale>0 || throw(OverflowError("Sylvester global scale underflow; use a wider scalar type"))
            r .*= factor; l .*= factor
        end
        RR[I,J]=r; L[I,J]=l
    end
    (;R=RR,L,scale)
end

function _orthogonal_span(X)
    # Full Householder QR: first columns span X, remaining columns complete it.
    _qr_unblocked(X).Q
end

function _equivalence_block!(S,B,Q,Z,range,U,V;threaded=false)
    S[range,:]=_mm(adjoint(U),view(S,range,:);threaded)
    S[:,range]=_mm(view(S,:,range),V;threaded)
    B[range,:]=_mm(adjoint(U),view(B,range,:);threaded)
    B[:,range]=_mm(view(B,:,range),V;threaded)
    Q[:,range]=_mm(view(Q,:,range),U;threaded)
    Z[:,range]=_mm(view(Z,:,range),V;threaded)
end

# Adjacent block swap (1/1, 1/2, 2/1 and 2/2), with weak and strong stability
# tests before committing any output. Orthogonal transformations retain reality.
function _pencil_swap!(S,B,Q,Z,k,p,q)
    T=eltype(S); R=real(T); range=k:k+p+q-1; localS=copy(view(S,range,range)); localB=copy(view(B,range,range))
    I=1:p; J=p+1:p+q
    X=generalized_sylvester(view(localS,I,I),view(localB,I,I),view(localS,J,J),view(localB,J,J),-localS[I,J],-localB[I,J])
    V=_orthogonal_span(vcat(X.R,X.scale*_identity(T,q)))
    U=_orthogonal_span(vcat(X.L,X.scale*_identity(T,q)))
    SS=_mm(_mm(adjoint(U),localS),V); BB=_mm(_mm(adjoint(U),localB),V)
    norm=max(nrm2(localS),nrm2(localB)); tolerance=max(floatmin(R)/eps(R),R(128)*eps(R)*norm)
    weak=max(nrm2(view(SS,q+1:p+q,1:q)),nrm2(view(BB,q+1:p+q,1:q)))
    weak<=tolerance || throw(ArgumentError("unstable generalized Schur block swap"))
    SS[q+1:end,1:q].=zero(T); BB[q+1:end,1:q].=zero(T)
    # Restore triangular B inside both retained 2x2 diagonal blocks; independent
    # QR rotations modify U only, so the block partition remains invariant.
    for block in (1:q,q+1:p+q)
        length(block)==2 || continue
        i=first(block); co,si,_=rotg(BB[i,i],BB[i+1,i])
        dummy=_identity(T,p+q); _qz_left!(SS,BB,dummy,i,i+1,co,si)
        U=_mm(U,dummy); BB[i+1,i]=zero(T)
    end
    strong=max(nrm2(_mm(_mm(U,SS),adjoint(V))-localS),nrm2(_mm(_mm(U,BB),adjoint(V))-localB))
    strong<=4tolerance || throw(ArgumentError("generalized Schur swap failed reconstruction test"))
    _equivalence_block!(S,B,Q,Z,range,U,V)
    S[range,range]=SS; B[range,range]=BB
    nothing
end

"""
    qz_reorder(F::QZFactor, select::Function) -> QZFactor

Reorder whole real 1x1/2x2 or complex 1x1 generalized Schur blocks.
The selector receives a generalized eigenvalue (possibly Inf/NaN). Selecting
either member of a real pair selects its entire block. Input is not modified.
"""
function qz_reorder(F::QZFactor,select::Function)
    _check_schur_form(F.S,F.T); S=copy(F.S); B=copy(F.T); Q=copy(F.Q); Z=copy(F.Z)
    blocks=_schur_blocks(S); sizes=length.(blocks); chosen=[any(select(F.values[j]) for j in block) for block in blocks]; next=1
    for j in eachindex(sizes)
        chosen[j] || continue
        for b in j-1:-1:next
            k=1+sum(view(sizes,1:b-1)); _pencil_swap!(S,B,Q,Z,k,sizes[b],sizes[b+1])
            sizes[b],sizes[b+1]=sizes[b+1],sizes[b]; chosen[b],chosen[b+1]=chosen[b+1],chosen[b]
        end
        next+=1
    end
    alpha,beta=_qz_pairs(S,B); QZFactor(S,B,Q,Z,alpha,beta,F.iterations)
end

# Standard real swap: one orthogonal similarity, not a conversion to complex
# Schur and not two independently rounded generalized bases.
function _real_block_swap!(H,Z,k,p,q)
    T=eltype(H); R=real(T); range=k:k+p+q-1; S=copy(view(H,range,range)); m=p*q
    K=zeros(T,m,m); C=-S[1:p,p+1:p+q]
    for j in 1:q, i in 1:p
        row=i+(j-1)*p
        for l in 1:p
            K[row,l+(j-1)*p]+=S[i,l]
        end
        for l in 1:q
            K[row,i+(l-1)*p]-=S[p+l,p+j]
        end
    end
    x=reshape(_small_solve(K,vec(C)),p,q)
    U=_orthogonal_span(vcat(x,_identity(T,q))); SS=_mm(_mm(adjoint(U),S),U)
    tolerance=max(floatmin(R)/eps(R),128eps(R)*nrm2(S))
    nrm2(view(SS,q+1:p+q,1:q))<=tolerance || throw(ArgumentError("unstable real Schur block swap"))
    SS[q+1:end,1:q].=zero(T)
    nrm2(_mm(_mm(U,SS),adjoint(U))-S)<=4tolerance || throw(ArgumentError("real Schur swap failed reconstruction test"))
    H[range,:]=_mm(adjoint(U),view(H,range,:)); H[:,range]=_mm(view(H,:,range),U)
    Z[:,range]=_mm(view(Z,:,range),U); H[range,range]=SS
    nothing
end

function _real_schur_reorder(F,select)
    T=eltype(F.T); H=copy(F.T); Z=copy(F.Z); blocks=_schur_blocks(H)
    sizes=length.(blocks); chosen=[any(select(F.values[j]) for j in block) for block in blocks]; next=1
    for j in eachindex(sizes)
        chosen[j] || continue
        for b in j-1:-1:next
            k=1+sum(view(sizes,1:b-1)); _real_block_swap!(H,Z,k,sizes[b],sizes[b+1])
            sizes[b],sizes[b+1]=sizes[b+1],sizes[b]; chosen[b],chosen[b+1]=chosen[b+1],chosen[b]
        end
        next+=1
    end
    for block in _schur_blocks(H)
        length(block)==2 && _finish_real_pair!(H,Z,first(block))
    end
    SchurFactor(H,Z,_schur_values(H,one(T)),F.iterations)
end

# Hager/Higham inverse 1-norm estimation on the coupled Sylvester operator.
# Adjoint solves are represented by reversal+adjoint, keeping triangular form.
function _sylvester_inverse(A,D,B,E,c,f;adj=false)
    if !adj
        X=generalized_sylvester(A,D,B,E,c,f)
        return X.R./X.scale,X.L./X.scale
    end
    # K' maps (X,Y) -> (A'*X+D'*Y, -X*B'-Y*E'). Solve by swapping
    # the roles of the two pencils and transposing the unknown matrices.
    p,q=size(c); T=eltype(c); r=zeros(T,p,q); l=zeros(T,p,q)
    ab=_schur_blocks(A); bb=_schur_blocks(B)
    for J in reverse(bb), I in ab
        u=copy(view(c,I,J)); v=copy(view(f,I,J))
        for (jj,j) in enumerate(J), (ii,i) in enumerate(I)
            for k in 1:first(I)-1
                u[ii,jj]-=conj(A[k,i])*r[k,j]+conj(D[k,i])*l[k,j]
            end
            for k in last(J)+1:q
                v[ii,jj]+=r[i,k]*conj(B[j,k])+l[i,k]*conj(E[j,k])
            end
        end
        # Transpose the explicit local operator, not its equations individually.
        pi,qj=length(I),length(J); m=pi*qj; K=zeros(T,2m,2m)
        for col in 1:2m
            cr=zeros(T,pi,qj); cf=similar(cr); fill!(cf,zero(T))
            col<=m ? (cr[col]=one(T)) : (cf[col-m]=one(T))
            ar=_mm(view(A,I,I),cr)-_mm(cf,view(B,J,J))
            af=_mm(view(D,I,I),cr)-_mm(cf,view(E,J,J))
            K[:,col]=vcat(vec(ar),vec(af))
        end
        x=_small_solve(Matrix(adjoint(K)),vcat(vec(u),vec(v)))
        r[I,J]=reshape(x[1:m],pi,qj); l[I,J]=reshape(x[m+1:end],pi,qj)
    end
    r,l
end

function _coupled_sep(A,D,B,E;iterations=8)
    p,q=size(A,1),size(B,1); R=real(eltype(A)); T=eltype(A); m=2p*q
    m==0 && return R(Inf)
    c=fill(T(one(R)/m),p,q); f=copy(c); estimate=zero(R); previous=0
    for _ in 1:iterations
        x,y=try
            _sylvester_inverse(A,D,B,E,c,f)
        catch error
            (error isa ArgumentError || error isa OverflowError) || rethrow()
            return zero(R)
        end
        next=sum(abs,x)+sum(abs,y)
        isfinite(next) || return zero(R)
        next<=estimate && break
        estimate=next
        sx=map(z->iszero(z) ? one(T) : z/abs(z),x); sy=map(z->iszero(z) ? one(T) : z/abs(z),y)
        u,v=try
            _sylvester_inverse(A,D,B,E,sx,sy;adj=true)
        catch error
            (error isa ArgumentError || error isa OverflowError) || rethrow()
            return zero(R)
        end
        all(isfinite,u) && all(isfinite,v) || return zero(R)
        a=argmax(abs.(vcat(vec(u),vec(v)))); a==previous && break; previous=a
        fill!(c,zero(T)); fill!(f,zero(T)); a<=p*q ? (c[a]=one(T)) : (f[a-p*q]=one(T))
    end
    iszero(estimate) ? R(Inf) : Base.inv(estimate)
end

"""
    qz_condition(F::QZFactor; split=nothing) -> NamedTuple

Pencil diagnostics: homogeneous eigenvalue reciprocal condition numbers,
and (when `split` is given) left/right deflating-subspace projection conditions
plus a coupled Sylvester inverse-1-norm separation estimate. No dense n^4
Kronecker operator is formed. Estimates are not certified error bounds.
"""
function qz_condition(F::QZFactor;split=nothing)
    X=_pencil_vectors(F.S,F.T,F.alpha,F.beta); Y=_pencil_vectors(F.S,F.T,F.alpha,F.beta;left=true)
    R=real(eltype(F.S)); n=size(F.S,1); s=zeros(R,n)
    scale=max(_finite_scale(F.S),_finite_scale(F.T))
    A=_mm(Complex{R}.(F.S./scale),X); B=_mm(Complex{R}.(F.T./scale),X)
    for j in 1:n
        s[j]=hypot(abs(dotc(view(Y,:,j),view(A,:,j))),abs(dotc(view(Y,:,j),view(B,:,j))))
    end
    split===nothing && return (;eigenvalue=s,normalization=scale,left=nothing,right=nothing,separation=nothing)
    0<=split<=n || throw(ArgumentError("invalid QZ split"))
    split in (0,n) && return (;eigenvalue=s,normalization=scale,left=one(R),right=one(R),separation=R(Inf))
    iszero(F.S[split+1,split]) || throw(ArgumentError("split cuts a real 2x2 block"))
    I=1:split; J=split+1:n
    G=try
        generalized_sylvester(view(F.S,I,I),view(F.T,I,I),view(F.S,J,J),view(F.T,J,J),-F.S[I,J],-F.T[I,J])
    catch error
        error isa ArgumentError || rethrow()
        return (;eigenvalue=s,normalization=scale,left=zero(R),right=zero(R),separation=zero(R))
    end
    pl=G.scale/hypot(G.scale,nrm2(G.L)); pr=G.scale/hypot(G.scale,nrm2(G.R))
    sep=_coupled_sep(view(F.S,I,I),view(F.T,I,I),view(F.S,J,J),view(F.T,J,J))
    (;eigenvalue=s,normalization=scale,left=pl,right=pr,separation=sep)
end

function schur_condition(F::SchurFactor;split=nothing)
    T=eltype(F.T); n=size(F.T,1); I=_identity(T,n); a,b=_qz_pairs(F.T,I)
    G=QZFactor(F.T,I,F.Z,F.Z,a,b,F.iterations)
    (;eigenvalue=trsna_s(F.T),spectral_gap=trsna_sep(F.T),subspace=qz_condition(G;split))
end
