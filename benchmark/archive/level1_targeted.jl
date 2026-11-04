using OxiBLAS, BenchmarkTools, SHA
VERSION.major==1 && VERSION.minor==14 || error("This campaign requires Julia 1.14; select OXIBLAS_JULIA")
# Campaign mode, selected explicitly with OXIBLAS_L1_MODE:
#   sequential (default) — one Julia thread, N=512/4096/65536, four types,
#                          all BLAS 1 operations, batch=32.
#   parallel             — large vectors, any Julia thread count; each parallel
#                          variant is measured with its sequential counterpart
#                          in the same process. Defaults: N=32768/131072/
#                          524288/2097152 (around the 131072 threshold), four
#                          types, batch=4.
# Optional overrides (both modes): OXIBLAS_L1_SIZES="262144,1048576",
# OXIBLAS_L1_TYPES="Float32,Float64,ComplexF32,ComplexF64", OXIBLAS_L1_BATCH,
# OXIBLAS_L1_SAMPLES, OXIBLAS_L1_SECONDS, and OXIBLAS_L1_SELECT="op1,op2" as before.
# Correctness guards run before and after every measurement in both modes.
const L1_MODE=get(ENV,"OXIBLAS_L1_MODE","sequential")
L1_MODE in ("sequential","parallel") || error("OXIBLAS_L1_MODE must be sequential or parallel, got $L1_MODE")
L1_MODE=="sequential" && Threads.nthreads()!=1 && error("Sequential campaign requires one Julia thread (use OXIBLAS_L1_MODE=parallel)")
include("full_backend.jl")
include(joinpath(@__DIR__,"..","..","test","helpers","level1_references.jl"))
# Naming/provenance is shared with the full catalogue, without constructing it.
full_symbol(T,op)=op=="nrm2" && T<:Complex ? (T===ComplexF32 ? "scnrm2_" : "dznrm2_") : op=="asum" && T<:Complex ? (T===ComplexF32 ? "scasum_" : "dzasum_") : string(T===Float32 ? "s" : T===Float64 ? "d" : T===ComplexF32 ? "c" : "z",op,"_")
include("level1_targeted_cases.jl")
_l1_env_list(name,default)=(v=filter(!isempty,strip.(split(get(ENV,name,""),',')));isempty(v) ? default : v)
const L1_TYPE_NAMES=Dict("Float32"=>Float32,"Float64"=>Float64,"ComplexF32"=>ComplexF32,"ComplexF64"=>ComplexF64)
const PARALLEL=L1_MODE=="parallel"
const L1_TYPES=Tuple(L1_TYPE_NAMES[t] for t in _l1_env_list("OXIBLAS_L1_TYPES",["Float32","Float64","ComplexF32","ComplexF64"]))
const L1_SIZES=Tuple(parse(Int,s) for s in _l1_env_list("OXIBLAS_L1_SIZES",PARALLEL ? ["32768","131072","524288","2097152"] : ["512","4096","65536"]))
# Parallel mode: each *_parallel operation and the sequential kernel it is
# compared with (dot_parallel computes dotu).
const L1_PARALLEL_OPERATIONS=(:axpy,:axpy_parallel,:scal,:scal_parallel,:dotu,:dot_parallel,
    :nrm2,:nrm2_parallel,:asum,:asum_parallel)
const L1_OPS=PARALLEL ? L1_PARALLEL_OPERATIONS : L1_OPERATIONS
const SELECT=filter(!isempty,split(get(ENV,"OXIBLAS_L1_SELECT",""),','))
# Independent buffers prevent both repeated mutation and loop-invariant
# hoisting of read-only reductions. One benchmark evaluation is one full batch.
# The batch is fixed for a whole campaign: it must not change with threads.
const L1_BATCH=parse(Int,get(ENV,"OXIBLAS_L1_BATCH",PARALLEL ? "4" : "32"))
const L1_SAMPLES=parse(Int,get(ENV,"OXIBLAS_L1_SAMPLES","9"))
# Time budget per measurement; targeted trials can lower it (e.g. 0.5).
const L1_SECONDS=parse(Float64,get(ENV,"OXIBLAS_L1_SECONDS",PARALLEL ? "10" : "2"))
L1_BATCH>=1 && L1_SAMPLES>=1 || error("batch and samples must be positive")
println("L1 campaign mode=",L1_MODE,"; Julia threads=",Threads.nthreads(),"; types=",join(L1_TYPES,','),
    "; sizes=",join(L1_SIZES,','),"; batch=",L1_BATCH,"; samples=",L1_SAMPLES)
# A failed guard aborts an OxiBLAS measurement. A reference library is still
# timed but flagged (status INACCURATE): OpenBLAS ARM64 dnrm2/scnrm2/dznrm2 use
# an unscaled running sum and exceed 64eps from N~65536.
function measure_l1(cases::Vector{C}) where C
    setup=()->foreach(c->c.setup(),cases)
    guard(ok,what)=(ok || FULL_BACKEND!="julia" || error(what*" correctness guard");ok)
    setup();first=cases[1].f();accurate=guard(cases[1].verify(first),"initial")
    sink=Ref(first)
    f=()->(for c in cases;sink[]=c.f();end;nothing)
    trial=run(@benchmarkable($f(),setup=($setup()),evals=1,overhead=0);samples=L1_SAMPLES,seconds=L1_SECONDS)
    for c in cases
        c.setup();accurate&=guard(c.verify(c.f()),"final")
    end
    (median(trial).time/L1_BATCH,minimum(trial).time/L1_BATCH,
     trial.memory/L1_BATCH,trial.allocs/L1_BATCH,length(trial),accurate)
end
errors=0
for T in L1_TYPES,n in L1_SIZES
    data=l1_input(T,n)
    hash=bytes2hex(sha256(vcat(reinterpret(UInt8,data.x),reinterpret(UInt8,data.y))))
    for op in L1_OPS
        !isempty(SELECT) && !(string(op) in SELECT) && continue
        status="N/A";med=NaN;low=NaN;bytes=0;allocs=0;samples=0;provider="";note=""
        try
            c=l1_case(op,T,n,data;reference=FULL_BACKEND!="julia");note=c.note
            if c.f!==nothing
                compatible=true
                if FULL_BACKEND!="julia"
                    compatible,provider=full_providers(c.symbols)
                end
                if compatible
                    cases=Vector{typeof(c)}(undef,L1_BATCH);cases[1]=c
                    for j in 2:L1_BATCH
                        cases[j]=l1_case(op,T,n,l1_input(T,n);reference=FULL_BACKEND!="julia")
                    end
                    med,low,bytes,allocs,samples,accurate=measure_l1(cases)
                    note*="; mode=$L1_MODE; julia_threads=$(Threads.nthreads()); requested_blas_threads=$FULL_THREADS; batch=$L1_BATCH independent buffers; time/allocations normalized per call"
                    accurate || (note*="; reference result outside the 64eps oracle tolerance")
                    status=med<=0.001 ? "UNRESOLVED_TIMER" : accurate ? "OK" : "INACCURATE"
                    med<=0.001 && (note*="; timer floor: excluded from comparisons";med=NaN;low=NaN)
                else
                    note*="; requested backend not backing symbol"
                end
            else
                note*="; no adapter with equivalent arithmetic contract or inapplicable type"
            end
        catch err
            global errors+=1;status="ERROR";note=sprint(showerror,err)
            showerror(stderr,err,catch_backtrace());println(stderr)
        end
        println("@@L1\t",join((op,T,n,FULL_BACKEND,status,med,low,bytes,allocs,samples,hash,replace(provider,'\t'=>' '),replace(note,'\n'=>' ')),'\t'));flush(stdout)
    end
end
exit(errors==0 ? 0 : 1)
