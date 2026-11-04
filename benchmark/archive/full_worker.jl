# PREPARED ONLY. This script is launched in a fresh process for each backend/thread request.
using OxiBLAS, SHA
include("full_backend.jl")
include(joinpath(@__DIR__,"..","..","test","helpers","compact_references.jl"))
include(joinpath(@__DIR__,"..","..","test","helpers","optional_float_types.jl"))
const FULL_OPTIONAL_TYPES=optional_float_types() # load before compiled benchmark closures
include("full_cases.jl")
include("full_extra_cases.jl")
include("full_qz_schur_cases.jl")
include("selected_svd_reference.jl")
include("full_selected_svd_cases.jl")
include("full_arnoldi_xtrace_cases.jl")
include("full_memory_triangular_cases.jl")
include(joinpath(@__DIR__,"..","..","test","helpers","level1_references.jl"))
const FULL_SAMPLES=parse(Int,get(ENV,"OXIBLAS_FULL_SAMPLES","5"))
const FULL_SIZES=parse.(Int,split(get(ENV,"OXIBLAS_FULL_SIZES","16,32,64,128,256,512,1024,2048,4096"),','))
const FULL_BITS=parse(Int,get(ENV,"OXIBLAS_GENERIC_BITS","256"))
const FULL_MEMORY=parse(Int,get(ENV,"OXIBLAS_FULL_MEMORY_MIB","1536"))*1024^2
# Targeted campaigns (benchmark/archive/blas_targeted.py): comma-separated filters.
# A group named in OXIBLAS_FULL_GROUPS is measured at every requested size,
# without the size caps of the complete report.
_full_env_list(name)=filter(!isempty,strip.(split(get(ENV,name,""),',')))
const FULL_GROUP_FILTER=_full_env_list("OXIBLAS_FULL_GROUPS")
const FULL_SELECT=Symbol.(_full_env_list("OXIBLAS_FULL_SELECT"))
const FULL_TYPE_FILTER=_full_env_list("OXIBLAS_FULL_TYPES")
const FULL_SECONDS=parse(Float64,get(ENV,"OXIBLAS_FULL_SECONDS","1.0"))
FULL_SAMPLES>0 && all(>(0),FULL_SIZES) && FULL_BITS>=64 && FULL_MEMORY>0 && FULL_SECONDS>0 || error("Invalid campaign configuration")
clean(x)=replace(string(x),'\t'=>" ",'\n'=>" ",'\r'=>" ")
function emit(group,operation,T,n,status;median=NaN,minimum=NaN,bytes=0,allocs=0,first=NaN,compile=NaN,gflops=NaN,provider="",note="",fingerprint="")
    println("@@OXI_ROW\t",join(clean.((group,operation,T,n,FULL_THREADS,FULL_BACKEND,status,median,minimum,bytes,allocs,first,compile,gflops,provider,note,fingerprint)),'\t'));flush(stdout)
end
function fingerprint(data,T)
    if T<:OxiBLAS.NativeFloat && ENDIAN_BOM==0x04030201
        h=UInt64(0xcbf29ce484222325)
        for A in (data.A,data.B,data.H,data.x), byte in reinterpret(UInt8,vec(A))
            h=(h ⊻ UInt64(byte))*UInt64(0x100000001b3)
        end
        return "fnv64:"*string(h;base=16,pad=16)
    end
    text=sprint() do io
        println(io,T,";",FULL_BITS)
        for A in (data.A,data.B,data.H,data.x), value in A
            show(IOContext(io,:compact=>false,:limit=>false),value);println(io)
        end
    end
    "sha256:"*bytes2hex(sha256(text))
end
function eligible(group,operation,n)
    isempty(FULL_SELECT) || operation in FULL_SELECT || return false
    # Unblocked O(n^3) variants (scalar LDL/Aasen, complete/rook pivoting)
    # take minutes per call at 2048: capped at 1024 even in targeted runs.
    group=="LAPACK" && operation in (:ldl,:lu_complete,:lu_rook,:aasen) && n>1024 && return false
    group in FULL_GROUP_FILTER && return true
    # All requested GEMM sizes are represented; expensive families are capped.
    group=="EXTENDED PRECISION" && return n in (8,24)
    group=="TENSOR" && return n<= (operation in (:hosvd,:tensor_train,:cp_als,:kron,:kron_sum) ? 32 : 128)
    group=="SPARSE" && return n<=256
    group=="LAPACK" && return n<=512
    group=="SPECTRAL / SOLVERS / GRAPHS" && return n<= (operation in (:cg,:gmres,:bicgstab,:minimum_degree,:rcm,:nested_dissection,:connected_components) ? 512 : 128)
    group=="BLAS LEVEL 3" && return operation in (:gemm,) || n<=512
    group=="BLAS LEVEL 2" && return n<=1024
    true
end
# Resets are outside each timed call (evals=1). The guarded first call already
# warmed code and caches; one more untimed call is made only when a call is
# shorter than its share of the budget. Samples stop at FULL_SAMPLES or once
# FULL_SECONDS have elapsed (at least one), so a long operation costs the
# first call plus the samples that fit. A plain loop replaces BenchmarkTools:
# each @benchmarkable compiled a new function and its trial ran full GC
# sweeps, about 2.5 s per case on this worker's heap.
function full_sample(f::F,setup::S,firsttime,firstbytes=0,firstallocs=0) where {F,S}
    # A call longer than the whole budget is measured by the guarded first
    # call itself (compile time removed) rather than repeated.
    firsttime>=FULL_SECONDS && return (;median=firsttime,minimum=firsttime,bytes=firstbytes,allocs=firstallocs)
    firsttime<FULL_SECONDS/FULL_SAMPLES && (setup();f())
    times=Float64[];bytes=0;allocs=0;start=time()
    while length(times)<FULL_SAMPLES && (isempty(times) || time()-start<FULL_SECONDS)
        setup()
        s=@timed f()
        push!(times,s.time);bytes=s.bytes;allocs=Base.gc_alloc_count(s.gcstats)
    end
    sort!(times)
    m=length(times)
    (;median=isodd(m) ? times[(m+1)÷2] : (times[m÷2]+times[m÷2+1])/2,minimum=times[1],bytes,allocs)
end
failures=Ref(0)
setprecision(BigFloat,FULL_BITS) do
    for (group,operations) in FULL_OPERATIONS
        isempty(FULL_GROUP_FILTER) || group in FULL_GROUP_FILTER || continue
        types=group=="EXTENDED PRECISION" ? vcat(DataType[Float16,Float32,Float64,Complex{Float16},ComplexF32,ComplexF64,BigFloat,Complex{BigFloat}],FULL_OPTIONAL_TYPES) : DataType[Float32,Float64,ComplexF32,ComplexF64]
        isempty(FULL_TYPE_FILTER) || (types=filter(T->string(T) in FULL_TYPE_FILTER,types))
        sizes=group=="EXTENDED PRECISION" ? (8,24) : FULL_SIZES
        for T in types,n in sizes
            selected=filter(operation->eligible(group,operation,n),operations)
            isempty(selected) && continue
            validated=Set(split(get(ENV,"OXIBLAS_VALIDATED_TYPES",""),'\n'))
            if !(string(T) in validated)
                for operation in selected
                    emit(group,operation,T,n,"SKIP";note="numerical type not validated by the complete test gate")
                end
                continue
            end
            # Conservative scratch estimate; skips remain rows with explicit reason.
            # Kronecker sums materialize three n²-by-n² results, unlike GEMM.
            slot=isbitstype(T) ? sizeof(T) : max(64,cld(FULL_BITS,8)+64)
            if BigInt(8)*n*n*slot>FULL_MEMORY
                for operation in selected
                    emit(group,operation,T,n,"SKIP";note="estimated scratch exceeds configured memory budget")
                end
                continue
            end
            data=try
                full_input(T,n)
            catch error
                failures[]+=1;showerror(stderr,error,catch_backtrace());println(stderr)
                for operation in selected;emit(group,operation,T,n,"ERROR";note="input construction failed");end
                continue
            end
            hash=fingerprint(data,T)
            for operation in selected
                if operation==:kron_sum && 3*BigInt(n)^4*slot>FULL_MEMORY
                    emit(group,operation,T,n,"SKIP";note="Kronecker sum n^4 payload exceeds configured memory budget",fingerprint=hash)
                    continue
                end
                ref=FULL_BACKEND!="julia"
                try
                    case=full_case(operation,T,n,data;reference=ref)
                    if case.f===nothing
                        emit(group,operation,T,n,"N/A";note="no equivalent adapter with matching result/precision contract",fingerprint=hash);continue
                    end
                    provider="native Julia OxiBLAS"
                    if ref
                        if isempty(case.symbols)
                            if FULL_BACKEND=="accelerate"
                                emit(group,operation,T,n,"N/A";note="LinearAlgebra generic operation; no verified Accelerate kernel",fingerprint=hash);continue
                            end
                            provider=T<:OxiBLAS.NativeFloat ? "LinearAlgebra (internal routing unaudited; not credited to OpenBLAS)" : "LinearAlgebra (generic scalar path; no native BLAS attribution)"
                        else
                            ok,provider=full_providers(case.symbols)
                            if !ok
                                emit(group,operation,T,n,"N/A";provider,note="requested provider not backing every required symbol; fallback not credited",fingerprint=hash);continue
                            end
                        end
                    end
                    case.setup();first=@timed case.f()
                    case.verify(first.value) || error("Correctness guard failed: $((group,operation,T,n,FULL_BACKEND))")
                    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
                    trial=full_sample(case.f,case.setup,first.time-(isnan(compile) ? 0.0 : compile),first.bytes,Base.gc_alloc_count(first.gcstats))
                    med=trial.median
                    emit(group,operation,T,n,"OK";median=med,minimum=trial.minimum,bytes=trial.bytes,allocs=trial.allocs,
                         first=first.time,compile,gflops=case.flops/med/1e9,provider,note=case.note*"; BLAS reported threads="*string(LA.BLAS.get_num_threads())*"; active kernel workers unmeasured",fingerprint=hash)
                catch error
                    # Only a missing reference method is an unavailable comparison.
                    # Native exceptions, invalid options and failed guards are failures.
                    unavailable=ref && error isa MethodError
                    !unavailable && (failures[]+=1)
                    showerror(stderr,error,catch_backtrace());println(stderr)
                    emit(group,operation,T,n,unavailable ? "N/A" : "ERROR";note=sprint(showerror,error),fingerprint=hash)
                end
            end
        end
    end
end
println("@@OXI_WORKER_ERRORS\t",failures[])
exit(failures[]==0 ? 0 : 1)
