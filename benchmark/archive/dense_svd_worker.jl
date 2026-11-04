# Dedicated dense SVD campaign. No sparse or other numerical families run here.
using OxiBLAS, Random, SHA
if get(ENV,"OXIBLAS_SVD_BASELINE","0")=="1"
    # Restore the two exact original method sets inside this worker only.
    # No package file or shared kernel is restored/overwritten on disk.
    for name in ("svd.jl.before","svd_advanced.jl.before")
        path=joinpath(@__DIR__,"..","..","reports","dense_svd_20261104",name)
        println("BASELINE_METHOD_SNAPSHOT ",name," ",bytes2hex(sha256(read(path))))
        original=read(path,String)
        # The new high-level API owns OxiBLAS.inv. The historical engines
        # meant Base's scalar reciprocal; qualify that binding in memory,
        # without changing the archived algorithm or any source on disk.
        compatible=replace(original,r"(?<![\w.])inv\("=>"Base.inv(")
        println("BASELINE_SCALAR_BINDING Base.inv; evaluated SHA256=",bytes2hex(sha256(compatible)))
        Base.include_string(OxiBLAS,compatible,path)
    end
end
include("full_backend.jl")
const SVD_SIZES=parse.(Int,split(get(ENV,"OXIBLAS_SVD_SIZES","64,192"),','))
const SVD_SAMPLES=parse(Int,get(ENV,"OXIBLAS_SVD_SAMPLES","3"))
const SVD_SECONDS=parse(Float64,get(ENV,"OXIBLAS_SVD_SECONDS","0.20"))
const SVD_SELECT=filter(!isempty,split(get(ENV,"OXIBLAS_SVD_SELECT",""),','))
const SVD_SHAPES=filter(!isempty,split(get(ENV,"OXIBLAS_SVD_SHAPES",""),','))
const SVD_TYPES=filter(!isempty,split(get(ENV,"OXIBLAS_SVD_TYPES",""),','))
_js(x::AbstractString)="\""*replace(x,"\\"=>"\\\\","\""=>"\\\"","\n"=>"\\n", "\t"=>"\\t")*"\""
_js(x::Bool)=x ? "true" : "false"
_js(x::Integer)=string(x)
_js(x::AbstractFloat)=isfinite(x) ? string(Float64(x)) : "null"
_js(::Nothing)="null"
_js(x::NamedTuple)="{"*join((_js(string(k))*":"*_js(v) for (k,v) in pairs(x)),",")*"}"
_js(x)=_js(string(x))
function diagnostics(F,A,values,ref)
    S=values ? (ref ? F : F.S) : F.S
    k=min(size(A)...);R=real(eltype(A))
    err=LA.norm(S-ref_singular(A))/max(LA.norm(S),floatmin(R))
    residual=left=right=nothing
    if !values
        residual=LA.norm(A-F.U*LA.Diagonal(S)*F.Vt)/max(LA.norm(A),floatmin(R))
        left=LA.norm(F.U'*F.U-Matrix{eltype(A)}(LA.I,k,k))/sqrt(max(1,k))
        right=LA.norm(F.Vt*F.Vt'-Matrix{eltype(A)}(LA.I,k,k))/sqrt(max(1,k))
    end
    bound=128eps(R)*max(size(A)...,1)
    valid=issorted(S;rev=true) && all(x->isfinite(x)&&x>=0,S) && err<=bound &&
          (values || max(residual,left,right)<=bound)
    (;valid,spectrum_error=err,residual,left_orthogonality=left,right_orthogonality=right)
end
# Reference computed once per matrix, outside every timed call.
const CURRENT_REFERENCE=Ref{Any}()
ref_singular(A)=CURRENT_REFERENCE[]
function sample(f::F) where F
    warm=@timed f()
    ts=Float64[];bytes=Int[];allocs=Int[];last=warm.value;elapsed=0.0
    for _ in 1:SVD_SAMPLES
        r=@timed f();last=r.value
        push!(ts,r.time);push!(bytes,r.bytes);push!(allocs,Base.gc_alloc_count(r.gcstats));elapsed+=r.time
        elapsed>=SVD_SECONDS && break
    end
    sort!(ts);q=length(ts);med=isodd(q) ? ts[(q+1)÷2] : (ts[q÷2]+ts[q÷2+1])/2
    (;value=last,median_s=med,minimum_s=first(ts),bytes=minimum(bytes),allocations=minimum(allocs),samples=q,first_s=warm.time)
end
function run_case(A,algo,values)
    if FULL_BACKEND=="julia"
        if algo=="jacobi"
            algorithm=FULL_THREADS==1 ? :jacobi : :parallel_jacobi
            f=()->svd_factor(A;algorithm,vectors=!values,full=false,threaded=FULL_THREADS>1)
        else
            f=()->svd_bidiagonal(A;algorithm=:golub_kahan,vectors=!values,full=false,threaded=FULL_THREADS>1)
        end
    elseif values
        f=()->LA.svdvals(A)
    else
        algorithm=algo=="gesvd" ? LA.QRIteration() : LA.DivideAndConquer()
        f=()->LA.svd(A;full=false,alg=algorithm)
    end
    sample(f)
end
failures=0
for T in (Float32,Float64,ComplexF32,ComplexF64),n in SVD_SIZES,shape in ("square","tall","wide")
    !isempty(SVD_SHAPES) && !(shape in SVD_SHAPES) && continue
    !isempty(SVD_TYPES) && !(string(T) in SVD_TYPES) && continue
    m,p=shape=="square" ? (n,n) : shape=="tall" ? (3n,n) : (n,3n)
    seed=0x5ad+13n+(T<:Complex ? 1 : 0);A=randn(MersenneTwister(seed),T,m,p)/sqrt(real(T)(max(m,p)))
    original=copy(A);fingerprint=bytes2hex(sha256(reinterpret(UInt8,vec(A))))
    CURRENT_REFERENCE[]=LA.svdvals(A)
    for values in (false,true)
        algorithms=FULL_BACKEND=="julia" ? ("jacobi","golub_kahan") : values ? ("gesdd_values",) : ("gesdd","gesvd")
        for algorithm in algorithms
            FULL_BACKEND=="julia" && !isempty(SVD_SELECT) && !(algorithm in SVD_SELECT) && continue
            meta=(;backend=FULL_BACKEND,algorithm,type=string(T),n,m,columns=p,shape,values_only=values,threads=FULL_THREADS,fingerprint)
            try
                if FULL_BACKEND!="julia"
                    prefix=T===Float32 ? "s" : T===Float64 ? "d" : T===ComplexF32 ? "c" : "z"
                    ok,provider=full_providers([prefix*(algorithm=="gesvd" ? "gesvd_" : "gesdd_")])
                    ok || error("SVD symbol is not backed by the requested backend: "*provider)
                end
                r=run_case(A,algorithm,values);d=diagnostics(r.value,A,values,FULL_BACKEND!="julia")
                d.valid && A==original || error("SVD correctness guard failed: "*string(d))
                iterations=FULL_BACKEND=="julia" ? r.value.sweeps : nothing
                println("@@DSVD\t",_js((;meta...,status="OK",median_s=r.median_s,minimum_s=r.minimum_s,bytes=r.bytes,allocations=r.allocations,samples=r.samples,first_s=r.first_s,iterations,d...)))
            catch err
                global failures+=1
                showerror(stderr,err,catch_backtrace());println(stderr)
                println("@@DSVD\t",_js((;meta...,status=err isa OxiBLAS.ConvergenceError ? "NONCONVERGED" : "ERROR",reason=sprint(showerror,err))))
            end
            flush(stdout)
        end
    end
end
exit(failures==0 ? 0 : 1)
