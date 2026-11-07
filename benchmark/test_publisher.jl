# Unit tests for benchmark/publish_results.jl using synthetic fixtures.
using Test, TOML

include(joinpath(@__DIR__, "publish_results.jl"))

@testset "Benchmark Publisher Tests" begin
    test_dir = mktempdir()

    # Helper to write synthetic CSV and MD pair
    function create_fixture(prefix, threads, cases_count, cases;
                            julia="1.13.3", cpu="apple-m1", ref="openblas (libopenblas64_.dylib)")
        csv_file = joinpath(test_dir, "$prefix.csv")
        md_file = joinpath(test_dir, "$prefix.md")

        open(csv_file, "w") do io
            println(io, join(EXPECTED_COLUMNS, ","))
            for c in cases
                println(io, join([
                    c.name, c.op, c.family, c.form, c.input, c.ref_input, c.eltype, c.size,
                    string(c.order), c.status,
                    isnan(c.oxiblas_ms) ? "" : string(c.oxiblas_ms),
                    isnan(c.reference_ms) ? "" : string(c.reference_ms),
                    isnan(c.ratio) ? "" : string(c.ratio),
                    c.oxiblas_allocs < 0 ? "" : string(c.oxiblas_allocs),
                    c.reference_allocs < 0 ? "" : string(c.reference_allocs),
                    c.oxiblas_bytes < 0 ? "" : string(c.oxiblas_bytes),
                    c.reference_bytes < 0 ? "" : string(c.reference_bytes),
                    c.note
                ], ","))
            end
        end

        open(md_file, "w") do io
            println(io, "# High-level benchmark\n")
            println(io, "Julia $julia, $threads thread(s), reference $ref, $cpu, $cases_count cases\n")
            println(io, "Body text...")
        end

        csv_file, md_file
    end

    c1 = (name="mul/dense/f64/s", op="mul", family="products", form="allocating", input="dense",
          ref_input="Matrix", eltype="f64", size="s", order=16, status="ok",
          oxiblas_ms=0.002, reference_ms=0.004, ratio=0.5,
          oxiblas_allocs=2, reference_allocs=2, oxiblas_bytes=128, reference_bytes=128, note="")
    c2 = (name="mul/dense/f64/m", op="mul", family="products", form="allocating", input="dense",
          ref_input="Matrix", eltype="f64", size="m", order=64, status="ok",
          oxiblas_ms=0.010, reference_ms=0.005, ratio=2.0,
          oxiblas_allocs=2, reference_allocs=2, oxiblas_bytes=512, reference_bytes=512, note="")
    c_unref = (name="mul/csr/f16/s", op="mul", family="products", form="allocating", input="csr",
               ref_input="", eltype="f16", size="s", order=16, status="ok",
               oxiblas_ms=0.001, reference_ms=NaN, ratio=NaN,
               oxiblas_allocs=1, reference_allocs=-1, oxiblas_bytes=64, reference_bytes=-1, note="no ref")
    c_err = (name="inv/dense/f64/s", op="inv", family="solves", form="allocating", input="dense",
             ref_input="Matrix", eltype="f64", size="s", order=16, status="mismatch",
             oxiblas_ms=0.005, reference_ms=0.004, ratio=1.25,
             oxiblas_allocs=5, reference_allocs=2, oxiblas_bytes=256, reference_bytes=128, note="norm diff 1e-2")

    @testset "1. Valid CSV/MD pair and Metadata Extraction" begin
        csv1, md1 = create_fixture("valid_1t", 1, 3, [c1, c2, c_unref])
        meta = parse_md_metadata(md1)
        @test meta.julia_version == "1.13.3"
        @test meta.threads == 1
        @test meta.cpu == "apple-m1"
        @test meta.declared_cases == 3

        cases = parse_csv_results(csv1)
        @test length(cases) == 3
        @test cases[1].name == "mul/dense/f64/s"
        @test cases[1].ratio == 0.5
        @test isnan(cases[3].reference_ms)
        @test cases[3].oxiblas_allocs == 1
        @test cases[3].reference_allocs == -1
    end

    @testset "2. Geometric Mean Calculation" begin
        # Ratios 0.5 and 2.0 -> geomean = sqrt(0.5 * 2.0) = 1.0
        # Median = (0.5 + 2.0) / 2 = 1.25
        csv1, _ = create_fixture("geom_test", 1, 2, [c1, c2])
        cases = parse_csv_results(csv1)
        comp = filter(r -> r.status == "ok" && !isnan(r.ratio) && r.ratio > 0, cases)
        ratios = [r.ratio for r in comp]
        @test isapprox(exp(mean(log.(ratios))), 1.0)
        @test isapprox(median(ratios), 1.25)
    end

    @testset "3. Complete Snapshot Publication" begin
        csv1, _ = create_fixture("pub_1t", 1, 4, [c1, c2, c_unref, c_err])
        c1_17 = merge(c1, (oxiblas_ms=0.001, ratio=0.25))
        csv17, _ = create_fixture("pub_17t", 17, 1, [c1_17])

        manifest_file = joinpath(test_dir, "pub.toml")
        out_json = joinpath(test_dir, "results.json")
        open(manifest_file, "w") do io
            println(io, "[snapshot]\ntitle = \"Synthetic Test\"\nrevision = \"abc1234\"\n")
            println(io, "[[campaigns]]\nthreads = 1\ncsv = \"$(escape_string(csv1))\"\n")
            println(io, "[[campaigns]]\nthreads = 17\ncsv = \"$(escape_string(csv17))\"\n")
        end

        publish_benchmarks(; manifest_path=manifest_file, output_path=out_json)
        @test isfile(out_json)
        content = read(out_json, String)
        @test occursin("\"title\":\"Synthetic Test\"", content)
        @test occursin("\"revision\":\"abc1234\"", content)
        @test occursin("\"threads\":[1,17]", content)
        @test occursin("\"comparable_cases\":2", content)  # for 1T: c1 and c2 are comparable, c_unref has no ref, c_err is mismatch
        @test occursin("\"mismatch_cases\":1", content)
    end

    @testset "4. Incompatible Environment Rejection" begin
        csv_m1, _ = create_fixture("env_m1", 1, 1, [c1]; cpu="apple-m1")
        csv_m2, _ = create_fixture("env_m2", 17, 1, [c1]; cpu="apple-m2")

        manifest_file = joinpath(test_dir, "bad_env.toml")
        open(manifest_file, "w") do io
            println(io, "[snapshot]\ntitle = \"Mismatch\"\nrevision = \"bad\"\n")
            println(io, "[[campaigns]]\nthreads = 1\ncsv = \"$(escape_string(csv_m1))\"\n")
            println(io, "[[campaigns]]\nthreads = 17\ncsv = \"$(escape_string(csv_m2))\"\n")
        end

        @test_throws ErrorException publish_benchmarks(; manifest_path=manifest_file, output_path=joinpath(test_dir, "out.json"))
    end

    @testset "5. Thread Count Mismatch Rejection" begin
        csv_th, _ = create_fixture("bad_th", 1, 1, [c1]) # Header says 1 thread
        manifest_file = joinpath(test_dir, "bad_th.toml")
        open(manifest_file, "w") do io
            println(io, "[snapshot]\ntitle = \"Bad Thread\"\nrevision = \"bad\"\n")
            println(io, "[[campaigns]]\nthreads = 17\ncsv = \"$(escape_string(csv_th))\"\n") # TOML says 17
        end

        @test_throws ErrorException publish_benchmarks(; manifest_path=manifest_file, output_path=joinpath(test_dir, "out.json"))
    end

    @testset "6. Case Count Mismatch Rejection" begin
        csv_cnt, _ = create_fixture("bad_cnt", 1, 999, [c1]) # Header declared 999 cases, only 1 in CSV
        manifest_file = joinpath(test_dir, "bad_cnt.toml")
        open(manifest_file, "w") do io
            println(io, "[snapshot]\ntitle = \"Bad Count\"\nrevision = \"bad\"\n")
            println(io, "[[campaigns]]\nthreads = 1\ncsv = \"$(escape_string(csv_cnt))\"\n")
        end

        @test_throws ErrorException publish_benchmarks(; manifest_path=manifest_file, output_path=joinpath(test_dir, "out.json"))
    end

    @testset "7. Conflicting Duplicate Cases Rejection" begin
        c1_dup = merge(c1, (oxiblas_ms=99.9,)) # Conflicting time
        csv_dup1, _ = create_fixture("dup1", 1, 1, [c1])
        csv_dup2, _ = create_fixture("dup2", 1, 1, [c1_dup])

        manifest_file = joinpath(test_dir, "dup.toml")
        open(manifest_file, "w") do io
            println(io, "[snapshot]\ntitle = \"Dup\"\nrevision = \"bad\"\n")
            println(io, "[[campaigns]]\nthreads = 1\ncsv = \"$(escape_string(csv_dup1))\"\n")
            println(io, "[[campaigns]]\nthreads = 1\ncsv = \"$(escape_string(csv_dup2))\"\n")
        end

        @test_throws ErrorException publish_benchmarks(; manifest_path=manifest_file, output_path=joinpath(test_dir, "out.json"))
    end
end
