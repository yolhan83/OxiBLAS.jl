# Record actual libblastrampoline providers for all timed BLAS 2/3 symbols.
using InteractiveUtils
import LinearAlgebra as LA
backend=get(ENV,"OXIBLAS_REFERENCE","openblas")
if backend=="accelerate"
    @eval using AppleAccelerate
    AppleAccelerate.set_num_threads(1)
end
LA.BLAS.set_num_threads(1)
open(joinpath(get(ENV,"OXIBLAS_REPORT_DIR",joinpath(@__DIR__,"..","..","reports")),"blas2-3-providers-$(backend).txt"),"w") do io
    versioninfo(io); println(io,LA.BLAS.get_config())
    interface=sizeof(LA.BlasInt)==8 ? :ilp64 : :lp64
    symbols=String[]
    for prefix in ("s","d"), op in ("gbmv","sbmv","tbmv","tbsv","spmv","tpmv","tpsv","spr","spr2","symm","syrk","syr2k")
        push!(symbols,prefix*op*"_")
    end
    for prefix in ("c","z"), op in ("gbmv","hbmv","tbmv","tbsv","hpmv","tpmv","tpsv","hpr","hpr2","symm","hemm","syrk","syr2k","herk","her2k")
        push!(symbols,prefix*op*"_")
    end
    for name in sort(symbols)
        println(io,name," => ",LA.BLAS.lbt_find_backing_library(name,interface))
    end
end
