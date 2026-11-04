# OxiBLAS test suite.
#
#   julia --project=. -e 'using Pkg; Pkg.test()'                             # core + QA
#   OXIBLAS_TEST_GROUP=extended julia --project=. -e 'using Pkg; Pkg.test()'  # heavy campaigns
#   OXIBLAS_TEST_GROUP=all      ...                                          # everything
#   OXIBLAS_TEST_GROUP=sparse   ...                                          # one family (see groups.jl)
#   OXIBLAS_TEST_FILES=blas/gemm.jl,api/linalg_api.jl ...                    # explicit files
#
# Every file runs in its own testset after the shared prelude (testsetup.jl);
# a failure in one file does not stop the others, and a progress line with the
# file's duration is printed after each one.
include("testsetup.jl")
include("groups.jl")

const GROUP=get(ENV,"OXIBLAS_TEST_GROUP","core")
const FILES=haskey(ENV,"OXIBLAS_TEST_FILES") ? String.(split(ENV["OXIBLAS_TEST_FILES"],',')) : selected_test_files(GROUP)
const WITH_QA=!haskey(ENV,"OXIBLAS_TEST_FILES") && GROUP in ("core","all","qa")

println("OxiBLAS tests: group=",GROUP,", ",length(FILES)," files, Julia ",VERSION,", ",
        Threads.nthreads()," thread(s), reference BLAS threads=",LA.BLAS.get_num_threads())
flush(stdout)

@testset "OxiBLAS" begin
    for file in FILES
        t=@elapsed @testset "$file" begin
            include(joinpath(@__DIR__,file))
        end
        println("  done ",rpad(file,44)," ",round(t;digits=1)," s"); flush(stdout)
    end
    # Quality checks, inside the root testset so that they run (and are
    # reported) even when a numerical file failed. Aqua and ExplicitImports
    # come from the test target (Pkg.test); a direct `julia test/runtests.jl`
    # without them skips the QA.
    if WITH_QA
        if Base.find_package("Aqua")===nothing
            @info "Aqua is not available in this environment: QA skipped (run through Pkg.test)"
        else
            include(joinpath(@__DIR__,"qa","qa.jl"))
            println("  done qa/qa.jl"); flush(stdout)
        end
    end
end
