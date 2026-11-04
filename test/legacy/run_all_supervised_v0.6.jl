# PREPARED ONLY. Reuse every include from runtests.jl, continuing independent files.
using Test, Random, OxiBLAS
import LinearAlgebra as LA
Random.seed!(0x0b1a5); LA.BLAS.set_num_threads(1)
println("TEST ENV: Julia=",VERSION," arch=",Sys.ARCH," cpu=",Sys.CPU_NAME,
        " Julia_threads=",Threads.nthreads()," reference_BLAS_threads=",LA.BLAS.get_num_threads(),
        " bounds=",Base.JLOptions().check_bounds)
println("REFERENCE BLAS CONFIG: ",LA.BLAS.get_config());flush(stdout)
const TYPES=(Float32,Float64,ComplexF32,ComplexF64)
tol(::Type{T}) where T=real(T)===Float32 ? 2f-4 : 3e-12
op(A,t)=t===:N ? A : t===:T ? transpose(A) : adjoint(A)
files=[m.captures[1] for m in eachmatch(r"include\(\"([^\"]+)\"\)",read(joinpath(@__DIR__,"runtests.jl"),String))]
# DefaultTestSet stores passing assertions separately; nested failures/errors and
# Broken (skips) live in results. This also retains full Test stack traces.
function counts(ts::Test.DefaultTestSet)
    hasproperty(ts,:n_passed) || error("Unsupported Julia Test layout: cannot report trustworthy counts")
    total=[Int(ts.n_passed),0,0,0]
    for result in ts.results
        if result isa Test.DefaultTestSet
            total .+= counts(result)
        elseif result isa Test.Pass
            total[1]+=1
        elseif result isa Test.Fail
            total[2]+=1
        elseif result isa Test.Error
            total[3]+=1
        elseif result isa Test.Broken
            total[4]+=1
        else
            error("Unknown Test result $(typeof(result)); refusing fabricated counts")
        end
    end
    total
end
# Use the supported @testset API: Test.push_testset/pop_testset disappeared
# in newer Julia releases. Capture the active root before finish can throw.
selected=get(ENV,"OXIBLAS_TEST_FILES","")
if !isempty(selected)
    requested=split(selected,',')
    all(file->file in files,requested) || error("Unknown requested test file")
    dependencies=Dict(
        "audit_priority.jl"=>["tensors.jl"],
        "extended_blas.jl"=>["extended_scalar.jl"],
        "extended_lapack.jl"=>["extended_blas.jl"],
        "extended_spectral.jl"=>["extended_blas.jl"],
        "generic_tensors.jl"=>["extended_blas.jl"],
        "tensor_memory.jl"=>["tensors.jl"],
        "tensor_planning.jl"=>["tensors.jl","extended_blas.jl"],
        "spectral_rust_conformance.jl"=>["schur.jl"],
        "qz_schur_complete.jl"=>["schur.jl","extended_blas.jl"],
        "svd_selected.jl"=>["schur.jl","extended_blas.jl"],
        "restarted_arnoldi_xtrace.jl"=>["schur.jl"],
        "sparse_memory_triangular.jl"=>["audit_priority.jl","extended_blas.jl"])
    needed=Set(String.(requested))
    function add_dependencies(file)
        for dependency in get(dependencies,file,String[])
            dependency in needed && continue
            push!(needed,dependency); add_dependencies(dependency)
        end
    end
    foreach(add_dependencies,collect(needed))
    files=filter(file->file in needed,files)
end
suite=Ref{Any}(nothing)
failed=Ref(false)
try
    @testset "Complete OxiBLAS suite ($(Threads.nthreads()) requested Julia threads)" begin
        suite[]=Test.get_testset()
        for file in files
            println("TEST FILE: ",file);flush(stdout)
            @testset "$file" begin
                include(joinpath(@__DIR__,file))
            end
        end
    end
catch error
    error isa Test.TestSetException || rethrow()
    failed[]=true
end
result=counts(suite[])
if isdefined(Main,:EXT_TYPES)
    println("@@OXI_VALIDATED_TYPES\t",join(string.(EXT_TYPES),'\t'))
end
println("@@OXI_TEST\t",join(result,'\t'))
flush(stdout)
exit(failed[] || result[2]>0 || result[3]>0 ? 1 : 0)
