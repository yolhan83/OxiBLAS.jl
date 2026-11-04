# Catalogue helper only. The existing supervised run_all.sh owns execution,
# 1/2/4/8-thread processes, actual backend verification and the single report.
function full_trevc_unpack(V,S)
    C=Complex{real(eltype(S))}; X=C.(V)
    if eltype(S)<:Real
        for block in OxiBLAS._schur_blocks(S)
            if length(block)==2
                i=first(block); X[:,i]=V[:,i]+im*V[:,i+1]; X[:,i+1]=conj.(X[:,i])
            end
        end
    end
    X
end
function full_trevc_normalize!(V,S)
    for block in OxiBLAS._schur_blocks(S)
        if eltype(S)<:Real && length(block)==2
            norm=sqrt(sum(abs2,view(V,:,block))); V[:,block]./=norm
        else
            for j in block
                norm=sqrt(sum(abs2,view(V,:,j))); V[:,j]./=norm
            end
        end
    end
    V
end
function full_qz_schur_case(operation,T,n,data;reference=false)
    (;A,B,H,x)=data; R=real(T); prefix(op)=full_symbol(T,op)
    setup=()->nothing; f=nothing; symbols=String[]; flops=NaN
    note="native QZ/Schur; complete returned result; setup excluded; missing equivalent references explicitly N/A"
    expect(a,b)=full_relative(a,b)<=5000max(1,n)*eps(R)
    verify=result->result!==nothing
    if operation in (:schur_reorder,:qz_reorder)
        generalized=operation===:qz_reorder
        if generalized
            F=qz_factor(A,H); selected=real.(F.values).>0
            # Identical input factor for both backends; no external re-factorization.
            RF=reference ? LA.GeneralizedSchur(copy(F.S),copy(F.T),copy(F.alpha),T<:Real ? T.(real.(F.beta)) : T.(F.beta),copy(F.Q),copy(F.Z)) : nothing
            f=reference ? ()->LA.ordschur(RF,selected) : ()->qz_reorder(F,z->real(z)>0)
            verify=G->expect(G.Q*G.S*G.Z',A) && expect(G.Q*G.T*G.Z',H)
            symbols=[prefix("tgsen")]
        else
            F=schur_factor(A); selected=real.(F.values).>0
            RF=reference ? LA.Schur(copy(F.T),copy(F.Z),copy(F.values)) : nothing
            f=reference ? ()->LA.ordschur(RF,selected) : ()->schur_reorder(F,z->real(z)>0)
            verify=G->expect(G.Z*G.T*G.Z',A); symbols=[prefix("trsen")]
        end
        note="stable selected real blocks moved to front; identical precomputed native Schur input; factorization excluded; allocating result"
    elseif operation in (:trevc_right,:trevc_left,:trsna_s,:trsna_sep,:schur_condition)
        F=schur_factor(A); S=F.T; left=operation===:trevc_left
        if operation in (:trevc_right,:trevc_left)
            if reference
                f=()->full_trevc_normalize!(LA.LAPACK.trevc!(left ? 'L' : 'R','A',zeros(LA.BlasInt,n),S),S)
            else
                g=left ? trevc_left : trevc_right; f=()->g(S)
            end
            verify=V->begin
                X=full_trevc_unpack(V,S)
                left ? expect(S'*X,X*LA.Diagonal(conj.(F.values))) : expect(S*X,X*LA.Diagonal(F.values))
            end
            symbols=[prefix("trevc")]
            note="all block triangular eigenvectors; real pairs packed as real/imag columns; joint unit 2-norm normalization included for both backends; same precomputed S"
        elseif operation===:trsna_s
            f=reference ? nothing : ()->trsna_s(S); verify=v->all(z->isfinite(z) && 0<=z<=1,v)
        elseif operation===:trsna_sep
            f=reference ? nothing : ()->trsna_sep(S); verify=v->all(z->z>=0,v)
            note="Rust spectral-distance estimator, not LAPACK singular-value sep; no false equivalent timing"
        else
            blocks=OxiBLAS._schur_blocks(S); split=last(first(blocks))
            f=reference ? nothing : ()->schur_condition(F;split)
            verify=v->all(z->0<=z<=1,v.eigenvalue) && v.subspace.separation>=0
        end
    elseif operation===:qz_condition
        F=qz_factor(A,H); split=last(first(OxiBLAS._schur_blocks(F.S)))
        f=reference ? nothing : ()->qz_condition(F;split)
        verify=v->all(z->isfinite(z) && z>=0,v.eigenvalue) && v.separation>=0
        note="homogeneous eigenvalue conditions + left/right deflating subspaces + coupled inverse-1-norm estimate; no n^4 operator; no identical reference wrapper"
    elseif operation===:generalized_sylvester
        p=max(1,n÷2); q=n-p; q=max(1,q)
        AA=Matrix(LA.UpperTriangular(A[1:p,1:p])); DD=Matrix(LA.UpperTriangular(H[1:p,1:p]))
        BB=Matrix(LA.UpperTriangular(B[1:q,1:q])); EE=Matrix(LA.UpperTriangular(H[1:q,1:q]))
        for j in 1:p
            AA[j,j]+=T(4n); DD[j,j]=T(2)
        end
        for j in 1:q
            BB[j,j]-=T(4n); EE[j,j]=one(T)
        end
        C=copy(A[1:p,1:q]); FF=copy(B[1:p,1:q])
        f=reference ? nothing : ()->generalized_sylvester(AA,DD,BB,EE,C,FF)
        verify=G->expect(AA*G.R-G.L*BB,G.scale*C) && expect(DD*G.R-G.L*EE,G.scale*FF)
    elseif operation===:qz_complex
        C=Complex{R}; f=reference ? ()->LA.schur(C.(A),C.(H)) : ()->qz_factor(A,H;form=:complex)
        verify=F->expect(F.Q*F.S*F.Z',A) && expect(F.Q*F.T*F.Z',H)
        symbols=[full_symbol(C,"gges")]
        note="complex triangular QZ including input conversion/copy; no real-to-complex shortcut in the real QZ case"
    elseif operation===:schur_multishift_workspace
        W=SchurWorkspace(T,n;blocksize=16,nshifts=6)
        f=reference ? ()->LA.schur(A) : ()->schur_multishift(A;window=4,blocksize=16,nshifts=6,workspace=W)
        verify=F->expect(F.Z*F.T*F.Z',A); symbols=[prefix("gees")]
        note="full Schur result; reused native panel/update buffers; small AED window forces bulge trains; external algorithm may differ"
    else
        if T<:Real && n<3
            return (;f=nothing,setup,verify,symbols,flops,note="a real double-shift bulge train requires an active block of order at least three")
        end
        HF=hessenberg_factor(A); initial=OxiBLAS._hessenberg_matrix(HF;rescale=false)
        shifts=schur_factor(initial[max(1,n-7):n,max(1,n-7):n]).T
        packets=OxiBLAS._shift_packets(shifts,6); HH=copy(initial); Q=Matrix{T}(LA.I,n,n)
        W=SchurWorkspace(T,n;blocksize=16,nshifts=6)
        setup=()->(copyto!(HH,initial);copyto!(Q,Matrix{T}(LA.I,n,n)))
        f=reference ? nothing : ()->(OxiBLAS._small_bulge_train!(HH,Q,1,n,packets,W,16,true);HH)
        verify=_ ->expect(Q*HH*Q',initial)
        note="one simultaneous small-bulge train with blocked far updates; excludes Hessenberg reduction/AED; no equivalent full-solver timing substituted"
    end
    (;f,setup,verify,symbols,flops,note)
end
