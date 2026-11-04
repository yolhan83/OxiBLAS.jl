# PREPARED ONLY. No execution before an explicitly approved campaign.
using OxiBLAS, BenchmarkTools, InteractiveUtils, SHA
import LinearAlgebra as LA
include(joinpath(@__DIR__,"..","..","test","helpers","optional_float_types.jl"))
const PRECISION_OPTIONAL_TYPES=optional_float_types()
length(ARGS) in (2,3) || error("usage: precision_complete.jl NEW_DIR native|reference [openblas|accelerate]")
out=abspath(ARGS[1]); mode=Symbol(ARGS[2]); mode in (:native,:reference) || error("unknown mode")
backend=length(ARGS)==3 ? ARGS[3] : "openblas"; backend in ("openblas","accelerate") || error("unknown backend")
mode===:reference && backend=="accelerate" && (@eval using AppleAccelerate)
ispath(out) && error("refusing to overwrite results");mkpath(out)
bits=parse(Int,get(ENV,"OXIBLAS_GENERIC_BITS","256"));bits>=64 || error("working precision must be >=64 bits")
LA.BLAS.set_num_threads(mode===:native ? 1 : Threads.nthreads())
ref(x::BigFloat)=x
ref(x::Real)=BigFloat(x)
ref(x::Complex)=complex(ref(real(x)),ref(imag(x)))
ref(A::AbstractArray)=ref.(A)
worktype(::Type{T}) where T=T===Float16 ? Float32 : T===Complex{Float16} ? ComplexF32 : T
relative(a,b)=maximum(abs,ref(a)-b;init=big"0")/max(one(BigFloat),maximum(abs,b;init=big"0"))
csv(io,values...)=println(io,join(("\""*replace(string(value),'"'=>"\"\"")*"\"" for value in values),','))
function measure(io,name,T,n,f,verify;tolerance,workspace=nothing)
    first=try
        @timed f()
    catch error
        if mode===:reference && (error isa MethodError || error isa ArgumentError)
            csv(io,name,T,n,precision(real(worktype(T))),Threads.nthreads(),"unsupported",sprint(showerror,error),NaN,NaN,NaN,NaN,NaN,NaN)
            flush(io);return
        end
        rethrow()
    end
    metric=setprecision(BigFloat,max(512,2bits)) do
        BigFloat(verify(first.value))
    end
    isfinite(metric) && metric<=tolerance || error("precision guard failed: $((name,T,n,metric,tolerance))")
    trial=run(@benchmarkable($f(),evals=1);seconds=.1,samples=3)
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    csv(io,name,T,n,precision(real(worktype(T))),Threads.nthreads(),"measured",string(metric),first.time,compile,
        BenchmarkTools.median(trial).time/1e9,minimum(trial).time/1e9,trial.memory,trial.allocs)
    if workspace!==nothing
        open(joinpath(out,"contexts.txt"),"a") do context
            println(context,(;operation=name,type=T,n,context_type=typeof(workspace),retained_size_estimate=Base.summarysize(workspace)))
        end
    end
    flush(io)
end
# Canonical value text at oracle precision, not raw pointers or sizeof(BigFloat).
function input_hash(name,A)
    text=setprecision(BigFloat,max(512,2bits)) do
        sprint() do io
            println(io,eltype(A),";",size(A),";working bits=",bits)
            for value in A
                show(IOContext(io,:compact=>false,:limit=>false),ref(value));println(io)
            end
        end
    end
    open(joinpath(out,"inputs.sha256"),"a") do io
        println(io,bytes2hex(sha256(text)),"  ",name)
    end
end
open(joinpath(out,"environment.txt"),"w") do io
    versioninfo(io;verbose=true)
    println(io,"mode=",mode,"; BigFloat working bits=",bits,"; oracle bits=",max(512,2bits))
    println(io,"Actual reference backend=",LA.BLAS.get_config(),"; BLAS threads=",LA.BLAS.get_num_threads())
    println(io,"Same storage/work type in paired native/reference cases; no Float64 baseline mislabeled as generic high precision.")
    println(io,"Optional providers=",get(ENV,"OXIBLAS_OPTIONAL_FLOATS","none"))
    println(io,"Provider scalar arithmetic is external to OxiBLAS; unsupported reference algorithms are recorded, not timed.")
    println(io,"3 samples; 0.1 s target/case, individual calls and compilation may exceed it.")
end
open(joinpath(out,"sources.sha256"),"w") do io
    root=dirname(dirname(@__DIR__))
    for dir in ("src","test","benchmark"), (path,_,files) in walkdir(joinpath(root,dir)), name in sort(files)
        endswith(name,".jl") || endswith(name,".rs") || endswith(name,".toml") || continue
        file=joinpath(path,name);println(io,bytes2hex(sha256(read(file))),"  ",relpath(file,root))
    end
end
setprecision(BigFloat,bits) do
    types=vcat(DataType[Float16,Float32,Float64,Complex{Float16},ComplexF32,ComplexF64,BigFloat,Complex{BigFloat}],PRECISION_OPTIONAL_TYPES)
    open(joinpath(out,"precision.csv"),"w") do io
        csv(io,"operation","type","n","working_bits","threads","status","guard_or_reason","first_s","compile_s","median_s","min_s","bytes","allocations")
        for T in types,n in (8,24)
            W=worktype(T); R=real(T)
            working_epsilon=eps(real(W))
            tolerance=if R===Float16
                big".02"
            else
                setprecision(BigFloat,max(512,2bits)) do
                    BigFloat(working_epsilon)*100000
                end
            end
            A=T.([i==j ? 4+i/32 : mod(3i+j,7)/64 for i in 1:n,j in 1:n]); B=reverse(A;dims=1)
            if !(R in (Float16,Float32,Float64))
                A[1,1]+=T(ldexp(one(R),-max(60,precision(R)÷2)))
                B=reverse(A;dims=1)
            end
            H=T.([i==j ? 5.0 : abs(i-j)==1 ? .125 : 0 for i in 1:n,j in 1:n]); rhs=ones(T,n,2)
            for (name,data) in ((:A,A),(:B,B),(:H,H))
                input_hash("$name-$T-$n",data)
            end
            expected=setprecision(BigFloat,max(512,2bits)) do
                ref(A)*ref(B)
            end
            if mode===:reference
                wa,wb=W.(A),W.(B);output=zeros(W,n,n)
                compute=()->begin LA.mul!(output,wa,wb);T.(output) end
                measure(io,:gemm,T,n,compute,C->relative(C,expected);tolerance)
                for (name,make,matrix) in ((:lu,()->LA.lu(wa),A),(:qr,()->LA.qr(wa),A),(:cholesky,()->LA.cholesky(LA.Hermitian(W.(H))),H))
                    measure(io,name,T,n,make,F->relative(ref(matrix)*ref(F\W.(rhs)),ref(rhs));tolerance)
                end
                n==8 && measure(io,:svd,T,n,()->LA.svd(wa),F->relative(ref(F.U)*LA.Diagonal(ref(F.S))*ref(F.Vt),ref(A));tolerance)
                continue
            end
            workspace=GemmWorkspace(T,n,n,n);output=zeros(T,n,n)
            measure(io,:gemm,T,n,()->gemm!(output,A,B;workspace),C->relative(C,expected);tolerance,workspace)
            for (name,make,matrix) in ((:lu,()->lu_factor(A),A),(:rook,()->lu_rook(A),A),(:qr,()->qr_factor(A),A),(:cholesky,()->cholesky_factor(H),H),(:ldl,()->ldlt_factor(H),H))
                measure(io,Symbol(name,"_factor"),T,n,make,F->relative(ref(matrix)*ref(solve(F,rhs)),ref(rhs));tolerance)
                F=make();measure(io,Symbol(name,"_solve"),T,n,()->solve(F,rhs),X->relative(ref(matrix)*ref(X),ref(rhs));tolerance)
            end
            measure(io,:mixed_explicit_low,T,n,()->mixed_solve(A,rhs;lowtype=T<:Complex ? ComplexF32 : Float32,maxiter=40),
                r->r.converged ? relative(ref(A)*ref(r.x),ref(rhs)) : BigFloat(Inf);tolerance)
            tensor=reshape(copy(A),n,n,1);plan=plan_einsum("ikb,kjb->ijb",tensor,reshape(B,n,n,1))
            measure(io,:einsum,T,n,()->einsum(plan,tensor,reshape(B,n,n,1)),C->relative(C[:,:,1],expected);tolerance)
            n==8 || continue
            for algorithm in (:jacobi,:parallel_jacobi,:golub_kahan,:divide_conquer)
                measure(io,Symbol("svd_",algorithm),T,n,()->svd_factor(A;algorithm),F->relative(ref(F.U)*LA.Diagonal(ref(F.S))*ref(F.Vt),ref(A));tolerance)
            end
            for (name,make) in ((:evd,()->eigen_hermitian(H)),(:evd_dc,()->eigen_hermitian_dc(H)),(:evd_mrrr,()->eigen_hermitian_mrrr(H)))
                measure(io,name,T,n,make,E->relative(ref(H)*ref(E.vectors),ref(E.vectors)*LA.Diagonal(ref(E.values)));tolerance)
            end
        end
    end
end
open(joinpath(out,"results.sha256"),"w") do io
    for name in sort(readdir(out))
        name=="results.sha256" && continue
        println(io,bytes2hex(sha256(read(joinpath(out,name)))),"  ",name)
    end
end
