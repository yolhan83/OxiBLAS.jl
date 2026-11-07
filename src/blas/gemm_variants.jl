# SPDX-License-Identifier: Apache-2.0
function _cache_recursive!(C,A,B,alpha,beta,leafsize,options)
    m,n=size(C); k=size(A,2)
    if max(m,n,k)<=leafsize || min(m,n,k)==0
        return gemm!(C,A,B,alpha,beta;options...)
    elseif m>=max(n,k)
        h=m÷2
        _cache_recursive!(view(C,1:h,:),view(A,1:h,:),B,alpha,beta,leafsize,options)
        _cache_recursive!(view(C,h+1:m,:),view(A,h+1:m,:),B,alpha,beta,leafsize,options)
    elseif n>=k
        h=n÷2
        _cache_recursive!(view(C,:,1:h),A,view(B,:,1:h),alpha,beta,leafsize,options)
        _cache_recursive!(view(C,:,h+1:n),A,view(B,:,h+1:n),alpha,beta,leafsize,options)
    else
        h=k÷2
        _cache_recursive!(C,view(A,:,1:h),view(B,1:h,:),alpha,beta,leafsize,options)
        _cache_recursive!(C,view(A,:,h+1:k),view(B,h+1:k,:),alpha,one(beta),leafsize,options)
    end
    C
end
"""
    gemm_cache_oblivious!(C, A, B, α=1, β=0; transA=:N, transB=:N, leafsize=64, kwargs...) -> C

Cache-oblivious largest-dimension recursion with native SIMD leaf kernels.
Parallel output regions have task-owned packing panels and no nested tasks.
"""
function gemm_cache_oblivious!(C::AbstractMatrix{T},A::AbstractMatrix{T},B::AbstractMatrix{T},alpha0=one(T),beta0=zero(T);
                               transA=:N,transB=:N,leafsize=64,kernel=:auto,threaded=true,workspace=nothing,config=GemmConfig(T)) where {T<:NativeFloat}
    opa,opb=_op(A,transA),_op(B,transB); m,n,k=_gemm_dims(C,opa,opb)
    leafsize>=1 || throw(ArgumentError("positive recursive leaf size required")); _kernel_kind(kernel)
    alpha,beta=convert(T,alpha0),convert(T,beta0)
    (iszero(alpha) || k==0) && return _scale!(C,beta)
    # AA, BB and workers are assigned once: the parallel closures capture them.
    AA=Base.mightalias(C,opa) ? copy(opa) : opa; BB=Base.mightalias(C,opb) ? copy(opb) : opb
    wanted=_operation_workers(max(1,m*n),Float64(m)*n*k,threaded,config.parallel_threshold)
    w=workspace===nothing ? GemmWorkspace(T,min(m,leafsize),min(n,leafsize),min(k,leafsize);config,workers=max(1,wanted)) : workspace
    options=_l3_options(T,min(m,leafsize),min(n,leafsize),min(k,leafsize),:auto,kernel,false,w,config)
    workers=min(wanted,length(w.buffers)); jobs=[(1:m,1:n)]
    while length(jobs)<workers
        index=argmax([length(r)*length(c) for (r,c) in jobs]); rows,cols=jobs[index]
        max(length(rows),length(cols))<=1 && break
        if length(rows)>=length(cols)
            middle=first(rows)+length(rows)÷2-1
            jobs[index]=(first(rows):middle,cols); push!(jobs,(middle+1:last(rows),cols))
        else
            middle=first(cols)+length(cols)÷2-1
            jobs[index]=(rows,first(cols):middle); push!(jobs,(rows,middle+1:last(cols)))
        end
    end
    lock(w.guard) do
        children=_workspace_children!(w,max(1,workers))
        _foreach_range(length(jobs),workers) do range,worker
            opts=merge(options,(;workspace=children[worker],threaded=false))
            for job in range
                # Task-local: the splitting loop above has its own rows/cols,
                # which this closure shared (and raced on) before.
                local jr,jc=jobs[job]
                _cache_recursive!(view(C,jr,jc),view(AA,jr,:),view(BB,:,jc),alpha,beta,leafsize,opts)
            end
        end
        C
    end
end

function _seven_products(A,B,method)
    m,k=size(A); n=size(B,2); mh=m÷2; kh=k÷2; nh=n÷2
    a=view(A,1:mh,1:kh); b=view(A,1:mh,kh+1:k); c=view(A,mh+1:m,1:kh); d=view(A,mh+1:m,kh+1:k)
    e=view(B,1:kh,1:nh); f=view(B,1:kh,nh+1:n); g=view(B,kh+1:k,1:nh); h=view(B,kh+1:k,nh+1:n)
    if method===:strassen
        return ((a+d,e+h),(c+d,e),(a,f-h),(d,g-e),(a+b,h),(c-a,e+f),(b-d,g+h))
    end
    s1=c+d; s2=s1-a; s3=a-c; s4=b-s2
    t1=f-e; t2=h-t1; t3=h-f; t4=t2-g
    ((a,e),(b,g),(s4,h),(d,t4),(s1,t1),(s2,t2),(s3,t3))
end
function _seven_assemble!(C,P,method,m,n)
    mh=m÷2; nh=n÷2
    for j in 1:nh, i in 1:mh
        p1,p2,p3,p4,p5,p6,p7=(P[k][i,j] for k in 1:7)
        if method===:strassen
            C[i,j]=p1+p4-p5+p7; C[i,j+nh]=p3+p5; C[i+mh,j]=p2+p4; C[i+mh,j+nh]=p1-p2+p3+p6
        else
            u2=p1+p6; u3=u2+p7; u4=u2+p5
            C[i,j]=p1+p2; C[i,j+nh]=u4+p3; C[i+mh,j]=u3-p4; C[i+mh,j+nh]=u3+p5
        end
    end
end
function _seven_recursive(A::AbstractMatrix{T},B,method,leafsize,w,config,kernel;workers=1) where T
    m,k=size(A); n=size(B,2); C=Matrix{T}(undef,m,n)
    if min(m,n,k)<=leafsize
        gemm!(C,A,B;threaded=false,workspace=w,config,kernel); return C
    end
    mm=m-mod(m,2); nn=n-mod(n,2); kk=k-mod(k,2)
    operands=_seven_products(view(A,1:mm,1:kk),view(B,1:kk,1:nn),method)
    products=Vector{Matrix{T}}(undef,7)
    if workers>1
        children=_workspace_children!(w,workers)
        _foreach_range(7,workers) do range,worker
            for j in range
                products[j]=_seven_recursive(operands[j][1],operands[j][2],method,leafsize,children[worker],config,kernel)
            end
        end
    else
        for j in 1:7
            products[j]=_seven_recursive(operands[j][1],operands[j][2],method,leafsize,w,config,kernel)
        end
    end
    _seven_assemble!(C,products,method,mm,nn)
    if kk<k
        gemm!(view(C,1:mm,1:nn),view(A,1:mm,k:k),view(B,k:k,1:nn),one(T),one(T);threaded=false,workspace=w,config,kernel)
    end
    if mm<m
        gemm!(view(C,m:m,:),view(A,m:m,:),B;threaded=false,workspace=w,config,kernel)
    end
    if nn<n
        gemm!(view(C,1:mm,n:n),view(A,1:mm,:),view(B,:,n:n);threaded=false,workspace=w,config,kernel)
    end
    C
end
function _gemm_seven!(C::AbstractMatrix{T},A::AbstractMatrix{T},B::AbstractMatrix{T},alpha,beta,method;
                      transA=:N,transB=:N,leafsize=128,kernel=:auto,threaded=true,workspace=nothing,config=GemmConfig(T)) where {T<:NativeFloat}
    AA,BB=_op(A,transA),_op(B,transB); m,n,k=_gemm_dims(C,AA,BB)
    leafsize>=1 || throw(ArgumentError("positive seven-product leaf size required")); _kernel_kind(kernel)
    alpha,beta=convert(T,alpha),convert(T,beta)
    (iszero(alpha) || k==0) && return _scale!(C,beta)
    wanted=_operation_workers(7,Float64(m)*n*k,threaded,config.parallel_threshold)
    options=_l3_options(T,m,n,k,:auto,kernel,threaded,workspace,config); w=options.workspace
    workers=min(wanted,length(w.buffers))   # one assignment: captured below
    result=lock(w.guard) do
        _seven_recursive(AA,BB,method,leafsize,w,config,kernel;workers)
    end
    for j in 1:n, i in 1:m
        C[i,j]=iszero(beta) ? alpha*result[i,j] : alpha*result[i,j]+beta*C[i,j]
    end
    C
end
"""
    gemm_strassen!(C, A, B, α=1, β=0; kwargs...) -> C

Native Strassen seven-product recursion, including odd-dimension fringes.
This trades additional workspace and rounding error for fewer products; select
it explicitly after checking the application's accuracy and performance needs.
"""
gemm_strassen!(C,A,B,alpha=one(eltype(C)),beta=zero(eltype(C));kwargs...)=_gemm_seven!(C,A,B,alpha,beta,:strassen;kwargs...)
"""
    gemm_winograd!(C, A, B, α=1, β=0; kwargs...) -> C

Native Strassen–Winograd addition schedule with seven recursive products.
"""
gemm_winograd!(C,A,B,alpha=one(eltype(C)),beta=zero(eltype(C));kwargs...)=_gemm_seven!(C,A,B,alpha,beta,:winograd;kwargs...)
