# SPDX-License-Identifier: Apache-2.0
function _interval_prepare(A::AbstractMatrix{T},low,high,check_hermitian) where {T<:NativeFloat}
    _square(A); Base.require_one_based_indexing(A); R=real(T); lo,hi=R(low),R(high)
    isfinite(lo) && isfinite(hi) && lo<=hi || throw(ArgumentError("finite ordered interval endpoints required"))
    C=CSRMatrix(COOMatrix(CSRMatrix(A)))
    all(isfinite,C.values) || throw(ArgumentError("finite interval operator required"))
    check_hermitian && !_block_is_hermitian(C) && throw(ArgumentError("Hermitian interval operator required"))
    C,lo,hi
end
function _interval_options(rtol,atol,boundary_tol)
    all(isfinite,(rtol,atol,boundary_tol)) && min(rtol,atol,boundary_tol)>=0 || throw(ArgumentError("invalid interval tolerances"))
end
_interval_contains(x,low,high,tolerance)=low<=x<=high || (x<low && low-x<=tolerance) || (x>high && x-high<=tolerance)
function _interval_bands(C::CSRMatrix{T}) where T
    n=size(C,1); R=real(T); d=zeros(R,n); upper=zeros(T,max(0,n-1))
    for i in 1:n, p in C.rowptr[i]:C.rowptr[i+1]-1
        j=C.colind[p]; value=C.values[p]; iszero(value) && continue
        abs(i-j)>1 && return nothing
        if i==j
            d[i]=real(value)
        elseif j==i+1
            upper[i]=value
        end
    end
    phases=ones(T,n); e=abs.(upper)
    for i in eachindex(e)
        phase=iszero(e[i]) ? one(T) : phases[i]*conj(upper[i])/e[i]
        phases[i+1]=phase/abs(phase)
    end
    (;diagonal=d,offdiagonal=e,phases)
end
function _interval_candidates(A,values,vectors,low,high,rtol,atol,boundary_tol;threaded=true)
    selected=findall(x->_interval_contains(x,low,high,boundary_tol),values)
    vals=values[selected]; X=copy(vectors[:,selected]); R=real(eltype(A))
    for j in axes(X,2)
        norm=nrm2(view(X,:,j)); isfinite(norm) && norm>0 || throw(ArgumentError("invalid interval Ritz vector"))
        view(X,:,j)./=norm
    end
    AX=_block_images(A,X;threaded); residuals=R[]
    for j in axes(X,2)
        axpy!(view(AX,:,j),view(X,:,j),-vals[j]); push!(residuals,nrm2(view(AX,:,j)))
    end
    flags=[isfinite(residuals[j]) && residuals[j]<=max(atol,rtol*abs(vals[j])) for j in eachindex(vals)]
    uncertain=[vals[j]-low<=residuals[j] || high-vals[j]<=residuals[j] for j in eachindex(vals)]
    (;values=vals,vectors=X,residuals,flags,boundary_uncertain=uncertain)
end
function _interval_project(A,V,low,high,rtol,atol,boundary_tol;images=nothing,threaded=true)
    AV=images===nothing ? _block_images(A,V;threaded) : images
    H=Matrix{eltype(V)}(undef,size(V,2),size(V,2)); gemm!(H,V,AV;transA=:C,threaded)
    E=eigen_hermitian(H)
    selected=findall(x->_interval_contains(x,low,high,boundary_tol),E.values)
    lifted=_block_lift(V,E.vectors[:,selected];threaded)
    candidate=_interval_candidates(A,E.values[selected],lifted,low,high,rtol,atol,boundary_tol;threaded)
    candidate,E
end
function _interval_result(candidate;vectors=true,full_basis=false,requested=nothing,center=0,
                          iterations=0,basis_dimension=0,matrix_products=0,method=:krylov,reason=:subspace_limit)
    selected=findall(identity,candidate.flags)
    if requested!==nothing && length(selected)>requested
        order=sortperm(abs.(candidate.values[selected].-center)); selected=selected[order[1:requested]]
    end
    sort!(selected;by=i->candidate.values[i])
    complete=full_basis && all(candidate.flags) && length(selected)==length(candidate.values)
    converged=requested===nothing ? complete : length(selected)==requested
    stop=converged ? :converged : full_basis ? (all(candidate.flags) ? :insufficient_interval_pairs : :residual_failure) : reason
    (;values=candidate.values[selected],vectors=vectors ? candidate.vectors[:,selected] : nothing,
      residuals=candidate.residuals[selected],boundary_uncertain=candidate.boundary_uncertain[selected],
      count=length(selected),candidate_count=length(candidate.values),candidate_residuals=copy(candidate.residuals),
      converged,complete,certified_count=false,iterations,basis_dimension,matrix_products,method,
      reason=stop)
end
function _interval_empty(::Type{T},n;vectors=true,complete=false,requested=nothing,method=:krylov,reason=:iteration_limit) where T
    candidate=(;values=real(T)[],vectors=zeros(T,n,0),residuals=real(T)[],flags=Bool[],boundary_uncertain=Bool[])
    _interval_result(candidate;vectors,full_basis=complete,requested,method,reason)
end
function _diagonal_interval(C,bands,low,high,boundary_tol,vectors;exact_endpoints=true)
    T=eltype(C); R=real(T); n=size(C,1)
    selected=findall(x->_interval_contains(x,low,high,boundary_tol),bands.diagonal)
    sort!(selected;by=i->bands.diagonal[i]); count=length(selected)
    X=vectors ? zeros(T,n,count) : nothing
    if vectors
        for (j,i) in enumerate(selected)
            X[i,j]=one(T)
        end
    end
    (;values=bands.diagonal[selected],vectors=X,residuals=zeros(R,count),boundary_uncertain=fill(!iszero(boundary_tol),count),
      count,candidate_count=count,candidate_residuals=zeros(R,count),converged=true,complete=true,
      certified_count=iszero(boundary_tol) && exact_endpoints,iterations=0,basis_dimension=0,matrix_products=0,method=:diagonal,reason=:converged)
end

"""Find verified Hermitian eigenpairs in the closed interval [low,high].
:auto handles diagonal/tridiagonal structure directly, otherwise builds one
bounded block Krylov projection. General partial projections never assert a
complete interval count. boundary_tol explicitly widens numerical selection.
Current tridiagonal vector extraction computes the full native MRRR spectrum;
even vectors=false still forms vectors internally to verify true residuals.
"""
function eigenvalues_in_interval(A::AbstractMatrix{T},low::Real,high::Real;
                                  method=:auto,vectors=true,subspace::Integer=min(size(A,1),50),
                                  blocksize::Integer=min(2,subspace),maxblocks::Integer=max(1,subspace),initial=nothing,
                                  rtol=sqrt(eps(real(T))),atol=zero(real(T)),boundary_tol=zero(real(T)),
                                  deflation_tol=_block_deflation_default(T,size(A,1)),seed::Integer=0x0b1a5,
                                  check_hermitian=true,threaded=true) where {T<:NativeFloat}
    _interval_options(rtol,atol,boundary_tol); method in (:auto,:krylov,:tridiagonal) || throw(ArgumentError("unknown interval method"))
    C,lo,hi=_interval_prepare(A,low,high,check_hermitian); n=size(C,1)
    n==0 && return _interval_empty(T,0;vectors,complete=true,method=:empty)
    bands=method===:krylov ? nothing : _interval_bands(C)
    method===:tridiagonal && bands===nothing && throw(ArgumentError("operator is not tridiagonal"))
    if bands!==nothing
        all(iszero,bands.offdiagonal) && return _diagonal_interval(C,bands,lo,hi,boundary_tol,vectors;
                                                                 exact_endpoints=lo==low && hi==high)
        E=eigen_tridiagonal_mrrr(bands.diagonal,bands.offdiagonal;threaded)
        selected=findall(x->_interval_contains(x,lo,hi,boundary_tol),E.values)
        X=Matrix{T}(undef,n,length(selected))
        for j in eachindex(selected), i in 1:n
            X[i,j]=bands.phases[i]*E.vectors[i,selected[j]]
        end
        candidate=_interval_candidates(C,E.values[selected],X,lo,hi,rtol,atol,boundary_tol;threaded)
        return _interval_result(candidate;vectors,full_basis=true,basis_dimension=n,
                                matrix_products=length(candidate.values),method=:tridiagonal,reason=:residual_failure)
    end
    F=block_lanczos_factor(C;blocksize,subspace,maxblocks,initial,deflation_tol,seed,threaded,check_hermitian=false)
    candidate,_=_interval_project(C,F.basis,lo,hi,rtol,atol,boundary_tol;images=F.images,threaded)
    _interval_result(candidate;vectors,full_basis=size(F.basis,2)==n,basis_dimension=size(F.basis,2),
                     iterations=F.iterations,matrix_products=F.matvecs+length(candidate.values),method=:krylov,
                     reason=size(F.basis,2)==n ? :residual_failure : :partial_projection)
end

# Choose the sign of a zero pivot by the requested side limit at the endpoint.
function _interval_sturm(d,e,x,inclusive)
    R=eltype(d); isempty(d) && return 0
    floor=floatmin(R); replacement=inclusive ? -floor : floor
    pivot=d[1]-x; abs(pivot)<floor && (pivot=replacement); count=Int(pivot<0)
    for i in 2:length(d)
        pivot=(d[i]-x)-(e[i-1]/pivot)*e[i-1]
        abs(pivot)<floor && (pivot=replacement)
        count+=Int(pivot<0)
    end
    count
end
"""Count with provenance. Returns a NamedTuple, not an unqualified integer.
Diagonal counts with boundary_tol=0 and exact endpoint conversion are exact
comparisons. Tridiagonal counts
use scaled Sturm recurrences in working precision (not certified intervals).
General operators report only the verified pairs of a bounded projection.
"""
function count_eigenvalues_in_interval(A::AbstractMatrix{T},low::Real,high::Real;
                                       method=:auto,boundary_tol=zero(real(T)),check_hermitian=true,kwargs...) where {T<:NativeFloat}
    _interval_options(zero(real(T)),zero(real(T)),boundary_tol)
    method in (:auto,:krylov,:tridiagonal) || throw(ArgumentError("unknown interval count method"))
    C,lo,hi=_interval_prepare(A,low,high,check_hermitian); n=size(C,1)
    bands=method===:krylov ? nothing : _interval_bands(C)
    method===:tridiagonal && bands===nothing && throw(ArgumentError("operator is not tridiagonal"))
    if bands!==nothing && all(iszero,bands.offdiagonal)
        result=_diagonal_interval(C,bands,lo,hi,boundary_tol,false;exact_endpoints=lo==low && hi==high)
        return (;count=result.count,complete=true,certified=result.certified_count,method=:diagonal)
    elseif bands!==nothing && iszero(boundary_tol)
        R=real(T); scale=max(maximum(abs,bands.diagonal;init=zero(R)),maximum(abs,bands.offdiagonal;init=zero(R)))
        d=bands.diagonal./scale; e=bands.offdiagonal./scale
        left=lo/scale; right=hi/scale
        below=_interval_sturm(d,e,left,false)
        through=_interval_sturm(d,e,right,true); count=through-below
        0<=count<=n || throw(ConvergenceError(:interval_sturm,0,1:n))
        return (;count,complete=true,certified=false,method=:sturm_working_precision)
    end
    result=eigenvalues_in_interval(C,lo,hi;method,boundary_tol,check_hermitian=false,vectors=false,kwargs...)
    (;count=result.count,complete=result.complete,certified=false,method=result.method)
end

mutable struct _FilteredOperator{T,P,W} <: AbstractMatrix{T}
    filter::P
    workspace::W
    products::Int
end
_FilteredOperator(P::ChebyshevFilter{T},W::FilterWorkspace{T}) where T=_FilteredOperator{T,typeof(P),typeof(W)}(P,W,0)
Base.size(A::_FilteredOperator)=size(A.filter.matrix)
Base.getindex(::_FilteredOperator,i::Int,j::Int)=throw(ArgumentError("polynomial filter is an implicit operator"))
_block_is_hermitian(::_FilteredOperator)=true
function _block_apply!(Y,A::_FilteredOperator,X;threaded=true)
    apply_filter!(Y,A.filter,X;workspace=A.workspace,threaded)
    A.products=Base.checked_add(A.products,Base.checked_mul(length(A.filter.coefficients)-1,size(X,2)))
    Y
end

"""Filtered block Lanczos with original-operator Rayleigh–Ritz extraction.
Retains multiplicities by vector subspaces, never by deduplicating equal values.
Requested-pair convergence does not imply completeness of the interval. Constant
filters fall back to ordinary block Lanczos. Restarts retain original Ritz
vectors and residual enrichment; no partial reorthogonalization or hard locking.
"""
function polynomial_filtered_eigenvalues(A::AbstractMatrix{T},low::Real,high::Real,
                                         nev::Integer=min(6,size(A,1));degree::Integer=20,bounds=nothing,
                                         damping=:jackson,vectors=true,subspace::Integer=min(size(A,1),max(20,4nev)),
                                         blocksize::Integer=min(size(A,1),max(2,nev)),
                                         keep::Integer=max(0,min(subspace-1,max(nev,2blocksize))),
                                         maxiter::Integer=30,initial=nothing,rtol=sqrt(eps(real(T))),atol=zero(real(T)),
                                         boundary_tol=zero(real(T)),deflation_tol=_block_deflation_default(T,size(A,1)),
                                         seed::Integer=0x0b1a5,check_hermitian=true,threaded=true) where {T<:NativeFloat}
    _interval_options(rtol,atol,boundary_tol); n=_square(A)
    0<=nev<=n && maxiter>=0 && seed>=0 || throw(ArgumentError("invalid filtered eigenvalue count/budget"))
    isfinite(deflation_tol) && 0<=deflation_tol<1 || throw(ArgumentError("invalid filtered deflation tolerance"))
    P=polynomial_filter(A,low,high;degree,bounds,damping,check_hermitian)
    if nev==0 || maxiter==0
        return merge(_interval_empty(T,n;vectors,complete=n==0,requested=nev,method=:filtered),
                     (;filter=P,history=NamedTuple[],filter_products=0,projection_products=0,verification_products=0))
    end
    1<=blocksize<=subspace<=n && 0<=keep<subspace && (subspace==n || keep>=nev) || throw(ArgumentError("invalid filtered Krylov space/retention sizes"))
    W=FilterWorkspace(P;columns=blocksize); operator=_FilteredOperator(P,W)
    active=length(P.coefficients)>1; current=initial; width=blocksize
    iterations=0; ordinary_products=0; projection_products=0; verification_products=0; history=NamedTuple[]
    lo,hi=P.interval; center=lo/2+hi/2
    for cycle in 1:maxiter
        F=block_lanczos_factor(active ? operator : P.matrix;blocksize=width,maxblocks=subspace,subspace,
                               initial=current,deflation_tol,seed,threaded,check_hermitian=false)
        images=active ? nothing : F.images
        candidate,E=_interval_project(P.matrix,F.basis,lo,hi,rtol,atol,boundary_tol;images,threaded)
        active ? (projection_products+=F.matvecs) : (ordinary_products+=F.matvecs)
        verification_products+=length(candidate.values); iterations+=F.iterations
        result=_interval_result(candidate;vectors,full_basis=size(F.basis,2)==n,requested=nev,center,
                                iterations,basis_dimension=size(F.basis,2),method=:filtered,reason=:iteration_limit)
        push!(history,(;cycle,dimension=size(F.basis,2),count=result.count,candidate_count=result.candidate_count,
                        candidate_residuals=copy(candidate.residuals)))
        if result.converged || cycle==maxiter || size(F.basis,2)==n
            total=operator.products+ordinary_products+projection_products+verification_products
            return merge(result,(;matrix_products=total,filter=P,history,filter_products=operator.products,
                                  projection_products=projection_products+ordinary_products,verification_products))
        end
        retained=min(keep,size(F.basis,2)); order=sortperm(abs.(E.values.-center))[1:retained]
        X=_block_lift(F.basis,E.vectors[:,order];threaded)
        AX=_block_images(P.matrix,X;threaded); verification_products+=retained; scale=_block_column_scale(AX)
        for j in axes(X,2)
            axpy!(view(AX,:,j),view(X,:,j),-E.values[order[j]])
        end
        if active
            # Advance every retained direction through the filter. The next
            # truncated block expansion alone may select only residual-enrichment
            # images and leave the wanted Ritz directions unchanged indefinitely.
            filtered=_block_images(operator,X;threaded)
            filtered_basis=_block_basis(filtered,zeros(T,n,0);limit=retained,rtol=deflation_tol,threaded)
            if size(filtered_basis,2)<retained
                supplement=_block_basis(X,filtered_basis;limit=retained-size(filtered_basis,2),rtol=deflation_tol,threaded)
                filtered_basis=hcat(filtered_basis,supplement)
            end
            X=filtered_basis
            retained=size(X,2)
        end
        # Leave room for a fresh polynomial Krylov expansion. Filling the entire
        # space with Ritz vectors and their original-operator residuals makes
        # the next cycle a projection of that same space without filtering.
        enrichment=min(blocksize,max(0,subspace-retained-1))
        extra=_block_basis(AX,X;limit=enrichment,rtol=deflation_tol,scale,threaded)
        if size(extra,2)==0
            fresh=_random_projection(T,n,enrichment,UInt64(seed)+UInt64(cycle))
            extra=_block_basis(fresh,X;rtol=deflation_tol,threaded)
        end
        current=hcat(X,extra); width=size(current,2)
    end
    error("unreachable filtered iteration state")
end
