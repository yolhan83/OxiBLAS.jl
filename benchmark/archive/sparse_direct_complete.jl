# PREPARED, NOT EXECUTED. Native and external references run in separate processes.
using OxiBLAS, BenchmarkTools, Random, InteractiveUtils, SHA
import LinearAlgebra as LA
length(ARGS) in (2,3) || error("usage: sparse_direct_complete.jl NEW_DIR native|reference [openblas|accelerate]")
output=abspath(ARGS[1]); mode=Symbol(ARGS[2]); mode in (:native,:reference) || error("unknown mode")
backend=length(ARGS)==3 ? ARGS[3] : "openblas"
backend in ("openblas","accelerate") || error("unknown reference backend")
if mode===:reference && backend=="accelerate"
    @eval using AppleAccelerate
end
ispath(output) && error("refusing to overwrite results"); mkpath(output)
Random.seed!(601104); LA.BLAS.set_num_threads(mode===:native ? 1 : Threads.nthreads())
open(joinpath(output,"environment.txt"),"w") do io
    versioninfo(io;verbose=true)
    println(io,"mode=",mode,"; requested backend=",backend,"; actual reference=",LA.BLAS.get_config())
    println(io,"Julia threads=",Threads.nthreads(),"; BLAS threads=",LA.BLAS.get_num_threads())
    println(io,"Rust pin=8d82bb6bf1df133aa90525aa9f478ebe41514f9b; seed=601104")
    println(io,"Reference factors are dense LAPACK, not a sparse multifrontal performance equivalent.")
    println(io,"Native disk E2E includes creation, factor, solve, checksums, IO and close; OS cache is warm/uncontrolled.")
    println(io,"3 samples, <=0.1 s target/case, evals=1; one call/compilation/IO may exceed this target.")
    println(io,"Memory budget is numeric panel/tile/packing payload, not CSR copies/output/metadata/RSS.")
end
open(joinpath(output,"sources.sha256"),"w") do io
    root=dirname(dirname(@__DIR__))
    for dir in ("src","test","benchmark"), (path,_,files) in walkdir(joinpath(root,dir)), name in sort(files)
        endswith(name,".jl") || continue
        full=joinpath(path,name); println(io,bytes2hex(sha256(read(full))),"  ",relpath(full,root))
    end
end
function direct_measure(io,operation,T,n,pattern,hash,f,verify)
    first=@timed f(); metric=verify(first.value)
    isfinite(metric) && metric< (real(T)===Float32 ? 2e-3 : 2e-10) || error("guard failed: $((operation,T,n,metric))")
    trial=run(@benchmarkable($f(),evals=1);seconds=.1,samples=3)
    compilation=hasproperty(first,:compile_time) ? first.compile_time : NaN
    println(io,join((operation,T,n,pattern,hash,Threads.nthreads(),metric,first.time,compilation,
        BenchmarkTools.median(trial).time/1e9,minimum(trial).time/1e9,trial.memory,trial.allocs,length(trial)),',')); flush(io)
end
if mode===:native
    write(joinpath(output,"disk-storage.csv"),"type,n,pattern,operation,payload_bytes,disk_bytes\n")
end
relative_residual(A,x,b)=LA.norm(A*x-b)/max(LA.norm(b),floatmin(real(eltype(b))))
open(joinpath(output,"sparse-direct.csv"),"w") do io
    println(io,"operation,type,n,pattern,input_sha256,threads,guard_residual,first_s,compile_s,median_s,min_s,bytes,allocations,samples")
    for T in (Float32,Float64,ComplexF32,ComplexF64), n in (32,128,384), pattern in (:band,:arrow,:blocks)
        H=zeros(T,n,n)
        for i in 1:n
            H[i,i]=T(6)
        end
        for i in 2:n
            linked=pattern===:band || (pattern===:blocks && i%8!=1)
            j=pattern===:arrow ? 1 : i-1
            if linked || pattern===:arrow
                value=T<:Complex ? T(.25+.125im) : T(.25)
                H[i,j]=value; H[j,i]=conj(value)
            end
        end
        indefinite=copy(H)
        for i in 1:n
            indefinite[i,i]=isodd(i) ? T(-6) : T(6)
        end
        A=H[collect(n:-1:1),:]; A[:,1].*=T(1.125)
        b=randn(T,n,3)
        for (name,dense) in ((:lu,A),(:ldl,indefinite),(:cholesky,H))
            C=CSRMatrix(dense); hash=bytes2hex(sha256(reinterpret(UInt8,vec(dense))))
            if mode===:reference
                constructor=name===:lu ? ()->LA.lu(dense) : name===:ldl ? ()->LA.bunchkaufman(LA.Hermitian(dense)) : ()->LA.cholesky(LA.Hermitian(dense))
                F=constructor(); verify=F->relative_residual(dense,F\b,b)
                direct_measure(io,Symbol("dense_reference_",name,"_factor"),T,n,pattern,hash,constructor,verify)
                direct_measure(io,Symbol("dense_reference_",name,"_solve"),T,n,pattern,hash,()->F\b,x->relative_residual(dense,x,b))
            else
                constructors=name===:lu ? ((:multifrontal,()->multifrontal_lu(C)),(:threshold_lu,()->sparse_lu(C;pivot_threshold=.1)),(:static,()->sparse_lu_static(C))) :
                    name===:ldl ? ((:sparse_ldl,()->sparse_ldlt(C)),) : ((:multifrontal_cholesky,()->multifrontal_cholesky(C)),)
                for (algorithm,constructor) in constructors
                    # Static LU uses the strongly diagonal H, as the reversed A
                    # may require large perturbations: report its actual input.
                    actual=name===:lu && algorithm===:static ? H : dense
                    actualC=CSRMatrix(actual); actualhash=bytes2hex(sha256(reinterpret(UInt8,vec(actual))))
                    make=algorithm===:static ? ()->sparse_lu_static(actualC) : constructor
                    F=make(); verify=F->relative_residual(actual,solve(F,b),b)
                    direct_measure(io,Symbol(algorithm,"_factor"),T,n,pattern,actualhash,make,verify)
                    direct_measure(io,Symbol(algorithm,"_solve"),T,n,pattern,actualhash,()->solve(F,b),x->relative_residual(actual,x,b))
                    if algorithm!==:multifrontal_cholesky
                        x=solve(F,b)
                        direct_measure(io,Symbol(algorithm,"_diagnostics"),T,n,pattern,actualhash,
                            ()->sparse_direct_diagnostics(actualC,F,b,x),d->maximum(d.componentwise_backward_error;init=0))
                    end
                end
                if name in (:lu,:cholesky)
                    # Fresh owned directory every evaluation; no accumulating files.
                    function disk_e2e()
                        F=name===:lu ? out_of_core_lu(C;blocksize=32,threaded=true) : out_of_core_cholesky(C;blocksize=32,threaded=true)
                        try
                            x=solve(F,b)
                            return (;x,payload=out_of_core_memory_usage(F).payload_bytes,
                                disk_bytes=sum(filesize(joinpath(F.store.directory,file)) for file in readdir(F.store.directory)))
                        finally
                            close(F)
                        end
                    end
                    direct_measure(io,Symbol("disk_",name,"_e2e"),T,n,pattern,hash,disk_e2e,r->relative_residual(dense,r.x,b))
                    detail=disk_e2e()
                    open(joinpath(output,"disk-storage.csv"),"a") do storage
                        println(storage,join((T,n,pattern,name,detail.payload,detail.disk_bytes),','))
                    end
                end
            end
        end
    end
end
# Final digest of outputs (does not overwrite any historical v0.5 evidence).
open(joinpath(output,"results.sha256"),"w") do io
    for name in sort(readdir(output))
        name=="results.sha256" && continue
        println(io,bytes2hex(sha256(read(joinpath(output,name)))),"  ",name)
    end
end
