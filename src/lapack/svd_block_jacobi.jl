# SPDX-License-Identifier: Apache-2.0
# One-sided block Jacobi preconditioning followed by the original robust
# scalar-pair convergence test. Only small local Gram matrices are formed.
# An unsafe local condition estimate rejects the Gram rotation before any
# column is changed; tiny/graded/rank-deficient blocks retain the robust path.
function _svd_block_workspace(::Type{T},m,n,b,vectors) where T
    (;panel=Matrix{T}(undef,m,2b),out=Matrix{T}(undef,m,2b),
      vp=Matrix{T}(undef,vectors ? n : 0,2b),vo=Matrix{T}(undef,vectors ? n : 0,2b),
      gram=Matrix{T}(undef,2b,2b),rotation=Matrix{T}(undef,2b,2b))
end

function _svd_block_rotation_cyclic!(G,W,rtol,m)
    T=eltype(G);R=real(T);q=size(G,1);dmax=zero(R);dmin=typemax(R)
    @inbounds for j in 1:q
        d=real(G[j,j]);G[j,j]=T(d);dmax=max(dmax,d);dmin=min(dmin,d)
        for i in 1:j-1
            G[j,i]=conj(G[i,j])
        end
    end
    floor=(floatmin(R)/eps(R))*max(1,m)
    (isfinite(dmax) && dmin>max(floor,sqrt(eps(R))*dmax)) || return false
    fill!(W,zero(T))
    @inbounds for i in 1:q
        W[i,i]=one(T)
    end
    changed=false;localtol=max(8eps(R),rtol/8)
    @inbounds for _ in 1:16
        rotated=false
        for j in 2:q,i in 1:j-1
            a,b=real(G[i,i]),real(G[j,j]);z=G[i,j];g=abs(z)
            (a>0 && b>0) || return false
            g<=localtol*sqrt(a)*sqrt(b) && continue
            delta=(b-a)/2;t=g/(delta+copysign(hypot(delta,g),delta))
            c=Base.inv(hypot(one(R),t));s=(c*t)*_realdiv(z,g)
            for k in 1:q
                (k==i || k==j) && continue
                x,y=G[k,i],G[k,j]
                xnew=c*x-conj(s)*y;ynew=s*x+c*y
                G[k,i]=xnew;G[i,k]=conj(xnew)
                G[k,j]=ynew;G[j,k]=conj(ynew)
            end
            G[i,i]=T(a-t*g);G[j,j]=T(b+t*g)
            G[i,j]=zero(T);G[j,i]=zero(T)
            _svd_rotate_columns!(W,i,j,c,s)
            rotated=true;changed=true
        end
        if !rotated
            dmin=minimum(i->real(G[i,i]),1:q);dmax=maximum(i->real(G[i,i]),1:q)
            return changed && dmin>sqrt(eps(R))*dmax
        end
    end
    false
end

function _svd_block_rotation!(G,W,rtol,m)
    R=real(eltype(G));q=size(G,1)
    lo=minimum(i->real(G[i,i]),1:q);hi=maximum(i->real(G[i,i]),1:q)
    lo>max((floatmin(R)/eps(R))*max(1,m),sqrt(eps(R))*hi) || return false
    changed=false
    @inbounds for j in 2:q,i in 1:j-1
        if abs(G[i,j])>rtol*sqrt(real(G[i,i]))*sqrt(real(G[j,j]))
            changed=true;break
        end
    end
    changed || return false
    F=try
        eigen_hermitian(G)
    catch err
        err isa ConvergenceError || rethrow()
        return _svd_block_rotation_cyclic!(G,W,rtol,m)
    end
    minimum(F.values)>sqrt(eps(R))*maximum(F.values) || return false
    copyto!(W,F.vectors)
    true
end

function _svd_block_pair!(B,V,p,q,b,rtol,w)
    m,n=size(B);ip=(p-1)*b+1:min(p*b,n);iq=(q-1)*b+1:min(q*b,n)
    np=length(ip);width=np+length(iq)
    P=view(w.panel,:,1:width);G=view(w.gram,1:width,1:width);W=view(w.rotation,1:width,1:width)
    @inbounds for j in 1:width
        source=j<=np ? ip[j] : iq[j-np]
        @simd for i in 1:m
            P[i,j]=B[i,source]
        end
    end
    # The output is small but the reduction dimension is long. Explicitly
    # select the existing packed SIMD GEMM instead of its tiny-matrix path.
    gemm!(G,adjoint(P),P;algorithm=:simd,threaded=false)
    _svd_block_rotation!(G,W,rtol,m) || return false
    output=view(w.out,:,1:width)
    gemm!(output,P,W;threaded=false)
    if V!==nothing
        PV=view(w.vp,:,1:width);OV=view(w.vo,:,1:width)
        @inbounds for j in 1:width
            source=j<=np ? ip[j] : iq[j-np]
            @simd for i in 1:n
                PV[i,j]=V[i,source]
            end
        end
        gemm!(OV,PV,W;threaded=false)
        @inbounds for j in 1:width
            dest=j<=np ? ip[j] : iq[j-np]
            @simd for i in 1:n
                V[i,dest]=OV[i,j]
            end
        end
    end
    @inbounds for j in 1:width
        dest=j<=np ? ip[j] : iq[j-np]
        @simd for i in 1:m
            B[i,dest]=output[i,j]
        end
    end
    true
end

function _svd_block_round!(B,V,b,rtol,schedule,flags,workspaces,blocks,pairs,workers)
    slots=length(schedule)
    _l1_foreach(pairs,workers) do range,worker
        changed=false;w=workspaces[worker]
        @inbounds for k in range
            p,q=schedule[k],schedule[slots+1-k]
            (p>blocks || q>blocks) && continue
            changed=_svd_block_pair!(B,V,min(p,q),max(p,q),b,rtol,w) | changed
        end
        @inbounds flags[worker]=UInt8(changed)
    end
    any(!iszero,flags)
end

function _svd_dense_block_jacobi!(B,V,rtol,maxsweeps,parallel)
    m,n=size(B);b=32;blocks=cld(n,b);slots=iseven(blocks) ? blocks : blocks+1
    pairs=slots÷2;schedule=collect(1:slots)
    workers=_operation_workers(pairs,Float64(m)*n*n,parallel,2_000_000)
    workspaces=[_svd_block_workspace(eltype(B),m,n,b,V!==nothing) for _ in 1:workers]
    flags=zeros(UInt8,workers);sweeps=0
    for _ in 1:min(3,maxsweeps-1)
        changed=false
        for _ in 1:slots-1
            changed=_svd_block_round!(B,V,b,rtol,schedule,flags,workspaces,blocks,pairs,workers) | changed
            tail=schedule[end]
            for k in slots:-1:3
                schedule[k]=schedule[k-1]
            end
            schedule[2]=tail
        end
        sweeps+=1
        !changed && break
    end
    # A local Gram is only a preconditioner: final convergence and tiny-column
    # handling always use the original normalized column-pair algorithm.
    try
        count=parallel ? _jacobi_parallel_svd!(B,V,rtol,maxsweeps-sweeps,true) :
                         _jacobi_svd!(B,V,rtol,maxsweeps-sweeps)
        sweeps+count
    catch err
        err isa ConvergenceError || rethrow()
        throw(ConvergenceError(err.algorithm,err.iterations+sweeps,err.active))
    end
end
