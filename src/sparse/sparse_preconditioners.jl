# SPDX-License-Identifier: Apache-2.0
struct RelaxationPreconditioner{T,R}
    A::CSRMatrix{T,Int}
    diagonal::Vector{T}
    omega::R
    sweeps::Int
    symmetric::Bool
end
function relaxation_preconditioner(A::NativeSparse;omega=one(real(eltype(A))),sweeps=1,symmetric=false)
    n=_square(A); C=CSRMatrix(COOMatrix(CSRMatrix(A))); T=eltype(C); R=real(T)
    0<omega<2 && sweeps>=1 || throw(ArgumentError("invalid relaxation options"))
    diagonal=T[C[i,i] for i in 1:n]
    all(x->isfinite(x) && !iszero(x),diagonal) || throw(ArgumentError("relaxation needs nonzero finite pivots"))
    RelaxationPreconditioner(C,diagonal,R(omega),sweeps,symmetric)
end
function _precondition!(z,r,P::RelaxationPreconditioner)
    length(z)==length(r)==size(P.A,1) || throw(DimensionMismatch("relaxation dimensions differ"))
    rr=Base.mightalias(z,r) ? copy(r) : r; fill!(z,zero(eltype(z))); n=length(z)
    for _ in 1:P.sweeps
        for backwards in (P.symmetric ? (false,true) : (false,))
            for i in (backwards ? (n:-1:1) : (1:n))
                s=rr[i]
                for p in P.A.rowptr[i]:P.A.rowptr[i+1]-1
                    j=P.A.colind[p]; j!=i && (s-=P.A.values[p]*z[j])
                end
                z[i]=(1-P.omega)*z[i]+P.omega*s/P.diagonal[i]
            end
        end
    end
    z
end
_check_preconditioner(P::RelaxationPreconditioner,n)=size(P.A)==(n,n) ? nothing : throw(DimensionMismatch("preconditioner dimensions differ"))

struct PolynomialPreconditioner{T,R}
    A::CSRMatrix{T,Int}
    lower::R
    upper::R
    degree::Int
    kind::Symbol
    work::NTuple{3,Vector{T}}
    guard::ReentrantLock
end
function polynomial_preconditioner(A::NativeSparse;degree=8,bounds=nothing,kind=:chebyshev)
    n=_square(A); C=CSRMatrix(COOMatrix(CSRMatrix(A))); T=eltype(C); R=real(T)
    degree>=1 && kind in (:chebyshev,:neumann) || throw(ArgumentError("invalid polynomial options"))
    if bounds===nothing
        lower=R(Inf); upper=-R(Inf)
        for i in 1:n
            center=real(C[i,i]); radius=zero(R)
            for p in C.rowptr[i]:C.rowptr[i+1]-1
                C.colind[p]!=i && (radius+=abs(C.values[p]))
            end
            lower=min(lower,center-radius); upper=max(upper,center+radius)
        end
        n==0 && (lower=upper=one(R))
    else
        lower,upper=R.(bounds)
    end
    isfinite(lower) && isfinite(upper) && 0<lower<=upper || throw(ArgumentError("positive finite spectral bounds required"))
    PolynomialPreconditioner(C,lower,upper,degree,kind,ntuple(_->zeros(T,n),3),ReentrantLock())
end
function _precondition!(z,r,P::PolynomialPreconditioner)
    _check_preconditioner(P,length(r)); length(z)==length(r) || throw(DimensionMismatch("polynomial dimensions differ"))
    lock(P.guard) do
        residual,direction,product=P.work; copyto!(residual,r); fill!(z,zero(eltype(z)))
        center=P.lower/2+P.upper/2; radius=(P.upper-P.lower)/2
        if P.kind===:neumann
            for _ in 1:P.degree
                axpy!(z,residual,Base.inv(center)); sparse_mul!(product,P.A,residual;threaded=false)
                for i in eachindex(residual)
                    residual[i]-=product[i]/center
                end
            end
        elseif iszero(radius)
            for i in eachindex(z)
                z[i]=residual[i]/center
            end
        else
            sigma=center/radius; rho=Base.inv(sigma)
            for i in eachindex(direction)
                direction[i]=residual[i]/center
            end
            for iteration in 1:P.degree
                axpy!(z,direction,one(eltype(z)))
                iteration==P.degree && break
                sparse_mul!(product,P.A,direction;threaded=false); axpy!(residual,product,-one(eltype(z)))
                next=Base.inv(2sigma-rho)
                for i in eachindex(direction)
                    direction[i]=next*rho*direction[i]+(2next/radius)*residual[i]
                end
                rho=next
            end
        end
        z
    end
end
_check_preconditioner(P::PolynomialPreconditioner,n)=size(P.A)==(n,n) ? nothing : throw(DimensionMismatch("preconditioner dimensions differ"))

struct ApproxInversePreconditioner{T}
    inverse::CSRMatrix{T,Int}
end
function _sparse_columns(A::CSRMatrix{T},selected) where T
    lookup=Dict(col=>j for (j,col) in enumerate(selected)); rows=[Dict{Int,T}() for _ in 1:A.m]
    for i in 1:A.m, p in A.rowptr[i]:A.rowptr[i+1]-1
        col=get(lookup,A.colind[p],0); col!=0 && (rows[i][col]=A.values[p])
    end
    _rows_to_csr(rows,T,length(selected))
end
"""
    spai(A::NativeSparse; pattern_levels=1, droptol=zero(real(eltype(A))), threaded=true) -> SPAIPreconditioner

Sparse approximate inverse from independent column least-squares problems.
`pattern_levels` expands the initial column graph support. Each task owns its
sparse QR and scratch; assembly happens after all column solves complete.
"""
function spai(A::NativeSparse;pattern_levels=1,droptol=zero(real(eltype(A))),threaded=true)
    n=_square(A); C=CSRMatrix(COOMatrix(CSRMatrix(A))); Z=CSCMatrix(C); T=eltype(C)
    pattern_levels>=0 && droptol>=0 || throw(ArgumentError("invalid SPAI options"))
    columns=Vector{Vector{Int}}(undef,n); coefficients=Vector{Vector{T}}(undef,n)
    workers=threaded ? Threads.nthreads() : 1
    _foreach_range(n,workers) do range,_
        rhs=zeros(T,n)
        for j in range
            support=Set{Int}([j])
            for _ in 1:pattern_levels
                next=copy(support)
                for k in support, p in Z.colptr[k]:Z.colptr[k+1]-1
                    push!(next,Z.rowind[p])
                end
                support=next
            end
            selected=sort!(collect(support)); B=_sparse_columns(C,selected); rhs[j]=one(T)
            x=solve(sparse_qr(B),rhs); rhs[j]=zero(T)
            keep=findall(v->abs(v)>droptol,x); columns[j]=selected[keep]; coefficients[j]=x[keep]
        end
    end
    rows=Int[]; cols=Int[]; values=T[]
    for j in 1:n
        append!(rows,columns[j]); append!(cols,fill(j,length(columns[j]))); append!(values,coefficients[j])
    end
    ApproxInversePreconditioner(CSRMatrix(COOMatrix(n,n,rows,cols,values)))
end
_precondition!(z,r,P::ApproxInversePreconditioner)=sparse_mul!(z,P.inverse,r;threaded=false)
_check_preconditioner(P::ApproxInversePreconditioner,n)=size(P.inverse)==(n,n) ? nothing : throw(DimensionMismatch("preconditioner dimensions differ"))

struct AdditiveSchwarzPreconditioner{T,R,F}
    factors::Vector{F}
    domains::Vector{Vector{Int}}
    weights::Vector{R}
    work::Vector{Vector{T}}
    rhs::Vector{Vector{T}}
    guard::ReentrantLock
    threaded::Bool
end
function additive_schwarz(A::NativeSparse;subdomains=max(1,Threads.nthreads()),overlap=1,solver=:cholesky,threaded=true)
    n=_square(A); C=CSRMatrix(A); T=eltype(C); R=real(T)
    subdomains>=1 && overlap>=0 && solver in (:cholesky,:lu,:ilu0) || throw(ArgumentError("invalid Schwarz options"))
    domains=Vector{Vector{Int}}(); multiplicity=zeros(Int,n); factors=Any[]
    for worker in 1:min(n,subdomains)
        range=_task_range(n,worker,min(n,subdomains)); domain=collect(max(1,first(range)-overlap):min(n,last(range)+overlap))
        push!(domains,domain); multiplicity[domain].+=1
        inverse=Dict(i=>k for (k,i) in enumerate(domain)); rows=[Dict{Int,T}() for _ in domain]
        for (row,i) in enumerate(domain), p in C.rowptr[i]:C.rowptr[i+1]-1
            j=get(inverse,C.colind[p],0); j!=0 && (rows[row][j]=C.values[p])
        end
        localmatrix=_rows_to_csr(rows,T,length(domain))
        push!(factors,solver===:cholesky ? sparse_cholesky(localmatrix) : solver===:lu ? sparse_lu(localmatrix) : ilu0(localmatrix))
    end
    typed=isempty(factors) ? SparseLUFactor{T,R}[] : Vector{typeof(first(factors))}(factors)
    weights=R[Base.inv(sqrt(R(m))) for m in multiplicity]
    AdditiveSchwarzPreconditioner(typed,domains,weights,[zeros(T,length(d)) for d in domains],[zeros(T,length(d)) for d in domains],ReentrantLock(),threaded)
end
function _precondition!(z,r,P::AdditiveSchwarzPreconditioner)
    length(z)==length(r)==length(P.weights) || throw(DimensionMismatch("Schwarz dimensions differ"))
    lock(P.guard) do
        _foreach_range(length(P.domains),P.threaded ? Threads.nthreads() : 1) do range,_
            @inbounds for k in range
                domain=P.domains[k]; rhs=P.rhs[k]
                for (j,i) in enumerate(domain)
                    rhs[j]=P.weights[i]*r[i]
                end
                _precondition!(P.work[k],rhs,P.factors[k])
            end
        end
        fill!(z,zero(eltype(z)))
        @inbounds for k in eachindex(P.domains), (j,i) in enumerate(P.domains[k])
            z[i]+=P.weights[i]*P.work[k][j]
        end
        z
    end
end
_check_preconditioner(P::AdditiveSchwarzPreconditioner,n)=length(P.weights)==n ? nothing : throw(DimensionMismatch("preconditioner dimensions differ"))

struct AMGLevel{T,R}
    A::CSRMatrix{T,Int}
    P::Union{Nothing,CSRMatrix{T,Int}}
    diagonal::Vector{R}
    omega::R
    factor::Union{Nothing,SparseCholeskyFactor{T,R}}
    work::NTuple{4,Vector{T}}
end
struct AMGPreconditioner{T,R}
    levels::Vector{AMGLevel{T,R}}
    presweeps::Int
    postsweeps::Int
    cycle::Symbol
    guard::ReentrantLock
end
"""
    amg(A::NativeSparse; smoothed=true, coarse_size=32, maxlevels=10, presweeps=2, postsweeps=2, cycle=:V) -> AMGPreconditioner

Aggregation algebraic multigrid for Hermitian positive-definite sparse A.
Piecewise-constant aggregates may be smoothed before Galerkin coarsening.
V/W cycles use weighted Jacobi and a native sparse Cholesky coarse solve.
"""
function amg(A::NativeSparse;smoothed=true,coarse_size=32,maxlevels=10,presweeps=2,postsweeps=2,cycle=:V)
    _square(A); T=eltype(A); R=real(T); C=CSRMatrix(COOMatrix(CSRMatrix(A)))
    min(coarse_size,maxlevels)>=1 && min(presweeps,postsweeps)>=0 && cycle in (:V,:W) || throw(ArgumentError("invalid AMG options"))
    levels=AMGLevel{T,R}[]
    for depth in 1:maxlevels
        # Cl: this level's matrix, bound once per iteration. C is rebound at the
        # end of the loop and the generators below captured it (Core.Box).
        Cl=C; n=size(Cl,1); diagonal=R[real(Cl[i,i]) for i in 1:n]
        all(d->isfinite(d) && d>0,diagonal) || throw(ArgumentError("AMG needs a positive finite diagonal"))
        bound=maximum((sum(abs,view(Cl.values,Cl.rowptr[i]:Cl.rowptr[i+1]-1))/diagonal[i] for i in 1:n);init=one(R))
        omega=R(4)/(3bound); aggregate=zeros(Int,n); groups=0
        for i in 1:n
            aggregate[i]!=0 && continue
            groups+=1; aggregate[i]=groups; partner=0; strength=zero(R)
            for p in C.rowptr[i]:C.rowptr[i+1]-1
                j=C.colind[p]
                if j!=i && aggregate[j]==0
                    value=abs(C.values[p])/sqrt(diagonal[i]*diagonal[j])
                    value>strength && (strength=value; partner=j)
                end
            end
            partner!=0 && (aggregate[partner]=groups)
        end
        if n<=coarse_size || depth==maxlevels || groups>=n
            push!(levels,AMGLevel(C,nothing,diagonal,omega,sparse_cholesky(C),ntuple(_->zeros(T,n),4))); break
        end
        counts=zeros(Int,groups)
        for j in aggregate
            counts[j]+=1
        end
        P=CSRMatrix(COOMatrix(n,groups,collect(1:n),aggregate,T[Base.inv(sqrt(R(counts[j]))) for j in aggregate]))
        if smoothed
            D=CSRMatrix(C)
            for i in 1:n, p in D.rowptr[i]:D.rowptr[i+1]-1
                D.values[p]/=diagonal[i]
            end
            P=sparse_add(P,sparse_matmul(D,P),one(T),T(-omega))
        end
        push!(levels,AMGLevel(C,P,diagonal,omega,nothing,ntuple(_->zeros(T,n),4)))
        C=sparse_matmul(sparse_transpose(P;conjugate=true),sparse_matmul(C,P))
    end
    AMGPreconditioner(levels,presweeps,postsweeps,cycle,ReentrantLock())
end
function _amg_cycle!(x,b,P,depth;zero=true)
    level=P.levels[depth]
    if level.factor!==nothing
        return _precondition!(x,b,level.factor)
    end
    zero && fill!(x,Base.zero(eltype(x)))
    residual,product=level.work[3:4]
    function smooth(sweeps)
        for _ in 1:sweeps
            sparse_mul!(product,level.A,x;threaded=false)
            for i in eachindex(x)
                x[i]+=level.omega*(b[i]-product[i])/level.diagonal[i]
            end
        end
    end
    smooth(P.presweeps); sparse_mul!(product,level.A,x;threaded=false)
    for i in eachindex(residual)
        residual[i]=b[i]-product[i]
    end
    next=P.levels[depth+1]; coarse_rhs=next.work[1]; coarse_x=next.work[2]
    sparse_mul!(coarse_rhs,level.P,residual;trans=:C,threaded=false)
    _amg_cycle!(coarse_x,coarse_rhs,P,depth+1)
    P.cycle===:W && _amg_cycle!(coarse_x,coarse_rhs,P,depth+1;zero=false)
    sparse_mul!(product,level.P,coarse_x;threaded=false); axpy!(x,product,one(eltype(x)))
    smooth(P.postsweeps); x
end
function _precondition!(z,r,P::AMGPreconditioner)
    _check_preconditioner(P,length(r)); length(z)==length(r) || throw(DimensionMismatch("AMG dimensions differ"))
    lock(P.guard) do
        rhs=P.levels[1].work[1]; copyto!(rhs,r); _amg_cycle!(z,rhs,P,1)
    end
end
_check_preconditioner(P::AMGPreconditioner,n)=size(P.levels[1].A)==(n,n) ? nothing : throw(DimensionMismatch("preconditioner dimensions differ"))
