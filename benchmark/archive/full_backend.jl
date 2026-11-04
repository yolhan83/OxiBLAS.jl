# Shared benchmark-only backend configuration. Never included by src/.
import LinearAlgebra as LA
using InteractiveUtils
const FULL_BACKEND=get(ENV,"OXIBLAS_FULL_BACKEND","julia")
const FULL_THREADS=parse(Int,get(ENV,"OXIBLAS_FULL_THREADS",string(Threads.nthreads())))
if FULL_BACKEND=="accelerate"
    Sys.isapple() || error("Accelerate is only available on macOS")
    @eval import AppleAccelerate
    if isdefined(AppleAccelerate,:set_num_threads)
        try
            AppleAccelerate.set_num_threads(FULL_THREADS)
        catch error
            println("Accelerate thread-control request unavailable: ",sprint(showerror,error),"; actual per-call count remains unmeasured")
        end
    else
        println("Accelerate threading API unavailable; request is not an actual thread count")
    end
elseif FULL_BACKEND in ("openblas","julia","linearalgebra")
    # Explicitly clear forwarding inherited from startup/configuration; workers
    # also use --startup-file=no. OpenBLAS_jll is supplied by Julia's stdlib.
    @eval import OpenBLAS_jll
    LA.BLAS.lbt_forward(OpenBLAS_jll.libopenblas_path;clear=true)
else
    error("Unknown full-report backend $FULL_BACKEND")
end
FULL_BACKEND!="accelerate" && LA.BLAS.set_num_threads(FULL_BACKEND=="julia" ? 1 : FULL_THREADS)
println("Requested backend=",FULL_BACKEND,"; requested threads=",FULL_THREADS,
        "; Julia threads=",Threads.nthreads(),"; BLAS reported threads=",LA.BLAS.get_num_threads())
println("BLAS configuration: ",LA.BLAS.get_config())
if FULL_BACKEND=="accelerate"
    reported=try
        isdefined(AppleAccelerate,:get_num_threads) ? AppleAccelerate.get_num_threads() : "unknown"
    catch error
        "unknown: "*sprint(showerror,error)
    end
    println("Accelerate reported mode/count=",reported,
            "; >1 is a managed multithreading request, not proof of exactly N workers. Actual per-call count is unmeasured.")
end
versioninfo(verbose=true)
function full_providers(symbols)
    interface=sizeof(LA.BlasInt)==8 ? :ilp64 : :lp64
    providers=String[]
    for symbol in symbols
        if !(symbol in LA.BLAS.get_config().exported_symbols)
            push!(providers,string(symbol,"=>unexported"));return false,join(providers,';')
        end
        library=LA.BLAS.lbt_find_backing_library(symbol,interface)
        push!(providers,string(symbol,"=>",library))
        wanted=FULL_BACKEND=="accelerate" ? "accelerate" : "openblas"
        occursin(wanted,lowercase(string(library))) || return false,join(providers,';')
    end
    !isempty(symbols),join(providers,';')
end
