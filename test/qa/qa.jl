# Quality checks. Can be run standalone or included from runtests.jl.
#
# - Aqua: type piracy, method ambiguities, unbound type parameters. Blocking.
# - ExplicitImports: implicit imports and hidden dependencies. Reported only,
#   not blocking, until the source tree is final.
# - Test.detect_closure_boxes: closures that capture a variable in a
#   Core.Box (type instability, and an obstacle for juliac --trim). Reported
#   only, not blocking, on Julia 1.14 and later, where the function exists.
using Test, OxiBLAS, Aqua

@testset "QA" begin

@testset "Aqua" begin
    Aqua.test_piracies(OxiBLAS)
    Aqua.test_ambiguities(OxiBLAS)
    Aqua.test_unbound_args(OxiBLAS)
end

@testset "ExplicitImports (report, not blocking)" begin
    if Base.find_package("ExplicitImports")===nothing
        @info "ExplicitImports is not installed: check skipped"
    else
        @eval using ExplicitImports
        for check in (:check_no_implicit_imports,:check_no_stale_explicit_imports,
                      :check_all_explicit_imports_via_owners,:check_all_qualified_accesses_via_owners,
                      :check_no_self_qualified_accesses)
            try
                # Inside the try: a check missing from the installed version
                # is reported like a finding.
                f=getfield(ExplicitImports,check)
                Base.invokelatest(f,OxiBLAS)
            catch err
                @warn "ExplicitImports.$check (not blocking)" exception=err
            end
        end
    end
    @test true
end

@testset "closure boxes (report, not blocking)" begin
    if isdefined(Test,:detect_closure_boxes)
        boxes=Test.detect_closure_boxes(OxiBLAS)
        for (m,vars) in boxes
            @warn "Core.Box (not blocking)" method=m variables=vars
        end
        isempty(boxes) || @info "$(length(boxes)) methods allocate a Core.Box"
    else
        @info "Test.detect_closure_boxes needs Julia 1.14: check skipped"
    end
end

end
