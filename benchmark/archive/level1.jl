using OxiBLAS, BenchmarkTools, Random, InteractiveUtils
import LinearAlgebra as LA
include(joinpath(@__DIR__,"..","..","test","helpers","level1_references.jl"))
LA.BLAS.set_num_threads(1)
const backend=get(ENV,"OXIBLAS_REFERENCE","openblas")
if backend=="accelerate"
    @eval using AppleAccelerate
    AppleAccelerate.set_num_threads(1)
end
const reportdir=get(ENV,"OXIBLAS_REPORT_DIR",joinpath(@__DIR__,"..","..","reports"))
const methods=backend=="openblas" ? (:native,:reference_blas) : (:reference_blas,)
const sizes="--smoke" in ARGS ? (33,) : (32,4096,1_000_000)
Random.seed!(20261102)
open(joinpath(reportdir,"level1-$(backend)-environment.txt"),"w") do io
    versioninfo(io)
    println(io,"BLAS reference: ",LA.BLAS.get_config(),"; reference threads=",LA.BLAS.get_num_threads())
    interface=sizeof(LA.BlasInt)==8 ? :ilp64 : :lp64
    for symbol in ("srotm_","drotm_","srotmg_","drotmg_","dsdot_","sdsdot_")
        println(io,symbol," provider: ",LA.BLAS.lbt_find_backing_library(symbol,interface))
    end
    println(io,"BenchmarkTools ",pkgversion(BenchmarkTools))
    println(io,"Median/minimum, 11 samples max/0.15 s, evals=1, warmup/check outside timing.")
    println(io,"ROTM output resets outside timing. ROTMG includes parameter/state allocations in BLAS wrapper.")
    println(io,"ROTMG times are batches of n=256 varied inputs, with observable preallocated outputs.")
    println(io,"Native widened dots use ordered Float64 accumulation, with no fast-math/reassociation.")
end
function emit(io,op,T,n,step,flag,method,f,verify;setup=()->nothing)
    setup(); first=@timed f(); err=verify(first.value)
    limit=T===Float32 ? 5e-5 : 5e-12
    err<=limit || error("level1 mismatch: $((op,T,n,step,flag,method,err))")
    # Keep read-only results observable; mutable inputs below prevent constant
    # folding of scalar generation cases by the compiler.
    sink=Ref(first.value)
    measured=()->(sink[]=f())
    trial=run(@benchmarkable($measured(),setup=($setup()),evals=1);seconds=.15,samples=11)
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    println(io,join((op,T,n,step,flag,method,BenchmarkTools.median(trial).time/1e9,minimum(trial).time/1e9,trial.memory,trial.allocs,length(trial),first.time,compile,err),','));flush(io)
    println((op,T,n,step,flag,method));flush(stdout)
end
function strided(v,n,step)
    step>0 ? view(v,1:step:1+(n-1)*step) : view(v,1+(n-1)*abs(step):step:1)
end
function generation_error(F,d1,d2,x,y)
    H=BigFloat.(rotm_matrix(F.params)); a,b=BigFloat(F.d1),BigFloat(F.d2)
    p,q=BigFloat(x),BigFloat(y)
    r=abs(H[2,1]*p+H[2,2]*q)/max(abs(H[2,1]*p),abs(H[2,2]*q))
    s=abs(H[1,1]*p+H[1,2]*q-BigFloat(F.x1))/max(abs(H[1,1]*p),abs(H[1,2]*q),abs(BigFloat(F.x1)))
    c1=abs(a*H[1,1]^2+b*H[2,1]^2-BigFloat(d1))/abs(BigFloat(d1))
    c2=abs(a*H[1,2]^2+b*H[2,2]^2-BigFloat(d2))/abs(BigFloat(d2))
    Float64(max(r,s,c1,c2))
end
function generate_batch!(out,inputs,f)
    @inbounds for j in axes(inputs,2)
        out[j]=f(inputs[1,j],inputs[2,j],inputs[3,j],inputs[4,j])
    end
    out
end
open(joinpath(reportdir,"level1-$(backend).csv"),"w") do io
    println(io,"operation,type,n,stride,flag,algorithm,median_s,minimum_s,bytes,allocations,samples,first_call_s,compile_s,relative_error")
    for T in (Float32,Float64), n in sizes, step in (1,3,-2), flag in (-1,0,1)
        x=strided(randn(T,1+(n-1)*abs(step)),n,step); y=strided(randn(T,1+(n-1)*abs(step)),n,step)
        oldx,oldy=copy(x),copy(y); coeff=T[flag,.7,-.3,.4,1.2]; p=RotmParams(coeff)
        H=rotm_matrix(p); expected=H*vcat(transpose(oldx),transpose(oldy)); scale=max(1,LA.norm(expected))
        setup=()->(copyto!(x,oldx);copyto!(y,oldy))
        verify(_)=LA.norm(vcat(transpose(x),transpose(y))-expected)/scale
        for method in methods
            f=method===:native ? ()->rotm!(x,y,p) : ()->reference_rotm!(x,y,coeff)
            emit(io,:rotm,T,n,step,flag,method,f,verify;setup)
        end
    end
    for n in sizes, step in (1,3,-2), op in (:dsdot,:sdsdot)
        x=strided(randn(Float32,1+(n-1)*abs(step)),n,step); y=strided(randn(Float32,1+(n-1)*abs(step)),n,step)
        initial=op===:sdsdot ? Float64(.7f0) : 0.0
        # BigFloat products and accumulation are reference-only and untimed.
        expected=setprecision(BigFloat,256) do
            BigFloat(initial)+sum(BigFloat(x[i])*BigFloat(y[i]) for i in eachindex(x);init=BigFloat(0))
        end
        verify(z)=Float64(abs(BigFloat(z)-expected)/max(1,abs(expected)))
        for method in methods
            f=op===:dsdot ? (method===:native ? ()->dsdot(x,y) : ()->reference_dsdot(x,y)) :
                              (method===:native ? ()->sdsdot(.7f0,x,y) : ()->reference_sdsdot(.7f0,x,y))
            emit(io,op,op===:dsdot ? Float64 : Float32,n,step,0,method,f,verify)
        end
    end
    for T in (Float32,Float64), d1 in (T(1),T(2.0^48)), method in methods
        # Vary inputs and retain every output: scalar constants can otherwise
        # be folded, and one call is below the local timer's resolution.
        Random.seed!(20261102+(T===Float32 ? 0 : 100)+(d1==1 ? 0 : 10))
        batch=256; inputs=rand(T,4,batch)
        inputs[1,:].=(inputs[1,:].+T(.5)).*d1
        inputs[2,:].+=T(.5); inputs[3,:].+=T(2); inputs[4,:].+=T(3)
        out=Vector{typeof(rotmg(one(T),one(T),T(3),T(4)))}(undef,batch)
        func=method===:native ? rotmg : reference_rotmg
        f=()->generate_batch!(out,inputs,func)
        verify(results)=maximum(generation_error(results[j],inputs[:,j]...) for j in 1:batch)
        # A faulty external result is kept as a separate correctness observation,
        # not timed as if it were a valid competitor.
        check=verify(f())
        if check>(T===Float32 ? 5e-5 : 5e-12)
            println("Excluded invalid ROTMG reference: ",(backend,T,d1,check));flush(stdout)
            continue
        end
        emit(io,:rotmg,T,batch,1,d1==1 ? 1 : -1,method,f,verify)
    end
end
