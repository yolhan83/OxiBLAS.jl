# benchmark/publish_results.jl
# Standalone publication tool for OxiBLAS benchmark documentation.
# Reads benchmark/publication.toml, loads benchmark CSV and paired Markdown files,
# validates data integrity and environment consistency, and generates the static
# docs/src/assets/benchmarks/results.json snapshot.
#
# Usage:
#   julia +release --project=benchmark benchmark/publish_results.jl
#   julia +release --project=benchmark benchmark/publish_results.jl --manifest=benchmark/publication.toml

using TOML, Dates, Printf, Statistics

# --- Pure Julia Zero-Dependency JSON Serializer -----------------------------
function to_json(io::IO, ::Nothing)
    print(io, "null")
end
function to_json(io::IO, b::Bool)
    print(io, b ? "true" : "false")
end
function to_json(io::IO, n::Integer)
    print(io, n)
end
function to_json(io::IO, f::AbstractFloat)
    if isnan(f) || isinf(f)
        print(io, "null")
    else
        @printf(io, "%.5g", f)
    end
end
function to_json(io::IO, s::AbstractString)
    print(io, '"')
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        elseif c == '\b'
            print(io, "\\b")
        elseif c == '\f'
            print(io, "\\f")
        elseif c < ' '
            @printf(io, "\\u%04x", Int(c))
        else
            print(io, c)
        end
    end
    print(io, '"')
end
function to_json(io::IO, v::Union{AbstractVector, Tuple})
    print(io, '[')
    for (i, item) in enumerate(v)
        i > 1 && print(io, ',')
        to_json(io, item)
    end
    print(io, ']')
end
function to_json(io::IO, d::Union{AbstractDict, NamedTuple})
    print(io, '{')
    first_item = true
    keys_iter = d isa NamedTuple ? keys(d) : keys(d)
    for k in keys_iter
        val = d isa NamedTuple ? getfield(d, k) : d[k]
        first_item || print(io, ',')
        first_item = false
        to_json(io, string(k))
        print(io, ':')
        to_json(io, val)
    end
    print(io, '}')
end

# --- Metadata Parsing ------------------------------------------------------
struct CampaignMeta
    julia_version::String
    threads::Int
    reference_backend::String
    cpu::String
    declared_cases::Int
end

function parse_md_metadata(md_path::String)::CampaignMeta
    isfile(md_path) || error("Paired Markdown report not found: $md_path")
    lines = readlines(md_path)
    header_idx = findfirst(l -> startswith(l, "Julia "), lines)
    header_idx !== nothing || error("Could not find environment header line in $md_path")
    header = lines[header_idx]

    # Regex: Julia 1.13.3, 17 thread(s), reference openblas (libopenblas64_.0.3.30.dylib), apple-m1, 936 cases
    m = match(r"^Julia\s+([^,]+),\s+(\d+)\s+thread\(s\),\s+reference\s+([^,]+),\s+([^,]+),\s+(\d+)\s+cases", header)
    m !== nothing || error("Malformed header in $md_path: $header")

    CampaignMeta(
        String(m.captures[1]),
        parse(Int, m.captures[2]),
        String(m.captures[3]),
        String(m.captures[4]),
        parse(Int, m.captures[5])
    )
end

# --- CSV Parsing & Validation ----------------------------------------------
const EXPECTED_COLUMNS = [
    "name", "op", "family", "form", "input", "ref_input", "eltype", "size",
    "order", "status", "oxiblas_ms", "reference_ms", "ratio",
    "oxiblas_allocs", "reference_allocs", "oxiblas_bytes", "reference_bytes", "note"
]

struct BenchmarkCase
    name::String
    op::String
    family::String
    form::String
    input::String
    ref_input::String
    eltype::String
    size::String
    order::Int
    status::String
    oxiblas_ms::Float64
    reference_ms::Float64
    ratio::Float64
    oxiblas_allocs::Int
    reference_allocs::Int
    oxiblas_bytes::Int
    reference_bytes::Int
    note::String
end

function parse_csv_results(csv_path::String)::Vector{BenchmarkCase}
    isfile(csv_path) || error("Benchmark CSV file not found: $csv_path")
    lines = readlines(csv_path)
    !isempty(lines) || error("Benchmark CSV file is empty: $csv_path")

    header_cols = split(lines[1], ',')
    if length(header_cols) < 18 || header_cols[1:18] != EXPECTED_COLUMNS
        error("Invalid CSV schema in $csv_path.\nExpected: $(join(EXPECTED_COLUMNS, ','))\nGot: $(lines[1])")
    end

    cases = BenchmarkCase[]
    for (line_idx, l) in enumerate(lines[2:end])
        cols = split(l, ',')
        length(cols) >= 18 || error("Truncated line $(line_idx + 1) in $csv_path: $l")

        name = cols[1]
        op = cols[2]
        family = cols[3]
        form = cols[4]
        input = cols[5]
        ref_input = cols[6]
        eltype = cols[7]
        size = cols[8]
        order = try parse(Int, cols[9]) catch; error("Invalid order at line $(line_idx+1) in $csv_path: $(cols[9])") end
        status = cols[10]

        oxiblas_ms = isempty(cols[11]) ? NaN : (try parse(Float64, cols[11]) catch; NaN end)
        reference_ms = isempty(cols[12]) ? NaN : (try parse(Float64, cols[12]) catch; NaN end)
        ratio = isempty(cols[13]) ? NaN : (try parse(Float64, cols[13]) catch; NaN end)

        oxiblas_allocs = isempty(cols[14]) ? -1 : (try parse(Int, cols[14]) catch; -1 end)
        reference_allocs = isempty(cols[15]) ? -1 : (try parse(Int, cols[15]) catch; -1 end)
        oxiblas_bytes = isempty(cols[16]) ? -1 : (try parse(Int, cols[16]) catch; -1 end)
        reference_bytes = isempty(cols[17]) ? -1 : (try parse(Int, cols[17]) catch; -1 end)

        note = join(cols[18:end], ",")

        push!(cases, BenchmarkCase(
            name, op, family, form, input, ref_input, eltype, size, order, status,
            oxiblas_ms, reference_ms, ratio,
            oxiblas_allocs, reference_allocs, oxiblas_bytes, reference_bytes, note
        ))
    end
    cases
end

# --- Consolidated Publication Builder --------------------------------------
function publish_benchmarks(;
    manifest_path::String = joinpath(@__DIR__, "publication.toml"),
    output_path::String = joinpath(dirname(@__DIR__), "docs", "src", "assets", "benchmarks", "results.json")
)
    isfile(manifest_path) || error("Manifest not found: $manifest_path")
    manifest = TOML.parsefile(manifest_path)

    snapshot_cfg = get(manifest, "snapshot", Dict{String, Any}())
    snapshot_title = get(snapshot_cfg, "title", "OxiBLAS Benchmark Snapshot")
    snapshot_rev = get(snapshot_cfg, "revision", "unspecified")

    campaigns_cfg = get(manifest, "campaigns", [])
    !isempty(campaigns_cfg) || error("No campaigns listed in $manifest_path")

    manifest_dir = dirname(abspath(manifest_path))

    # Consistency checks across campaigns
    common_julia = ""
    common_cpu = ""
    common_backend = ""

    # Group cases by thread count
    threads_set = Set{Int}()
    campaigns_by_thread = Dict{Int, Vector{BenchmarkCase}}()

    for c in campaigns_cfg
        t = get(c, "threads", nothing)
        t !== nothing || error("Missing 'threads' key in campaign entry of $manifest_path")
        t_int = Int(t)
        push!(threads_set, t_int)

        csv_rel = get(c, "csv", "")
        !isempty(csv_rel) || error("Missing 'csv' path in campaign entry of $manifest_path")

        csv_path = isabspath(csv_rel) ? csv_rel : normpath(joinpath(manifest_dir, csv_rel))
        md_path = replace(csv_path, r"\.csv$" => ".md")

        meta = parse_md_metadata(md_path)
        cases = parse_csv_results(csv_path)

        # Validate case count match
        if length(cases) != meta.declared_cases
            error("Case count mismatch in $csv_path: parsed $(length(cases)) cases, but markdown declared $(meta.declared_cases)")
        end

        # Validate thread match
        if meta.threads != t_int
            error("Thread mismatch for $csv_path: TOML declared $t_int threads, but markdown report header declared $(meta.threads) thread(s)")
        end

        # Validate environment consistency
        if isempty(common_julia)
            common_julia = meta.julia_version
            common_cpu = meta.cpu
            common_backend = meta.reference_backend
        else
            if meta.julia_version != common_julia
                error("Incompatible Julia versions in snapshot: $common_julia vs $(meta.julia_version) (in $csv_path)")
            end
            if meta.cpu != common_cpu
                error("Incompatible CPU architectures in snapshot: $common_cpu vs $(meta.cpu) (in $csv_path)")
            end
            if meta.reference_backend != common_backend
                error("Incompatible reference backends in snapshot: $common_backend vs $(meta.reference_backend) (in $csv_path)")
            end
        end

        # Accumulate and check duplicate keys
        thread_cases = get!(campaigns_by_thread, t_int, BenchmarkCase[])
        seen_names = Dict(c.name => c for c in thread_cases)

        for case in cases
            if haskey(seen_names, case.name)
                prev = seen_names[case.name]
                if prev.status != case.status || prev.oxiblas_ms != case.oxiblas_ms || prev.order != case.order
                    error("Conflicting duplicate case '$(case.name)' in $t_int-thread configuration: $(csv_path)")
                end
            else
                push!(thread_cases, case)
                seen_names[case.name] = case
            end
        end
    end

    sorted_threads = sort(collect(threads_set))

    # Build thread summaries
    summaries = Dict{String, Any}()
    all_cases_by_thread = Dict{String, Any}()

    all_families = Set{String}()
    all_ops = Set{String}()
    all_inputs = Set{String}()
    all_eltypes = Set{String}()
    all_sizes = Set{String}()

    for t in sorted_threads
        cases = campaigns_by_thread[t]
        t_str = string(t)

        for c in cases
            push!(all_families, c.family)
            push!(all_ops, c.op)
            push!(all_inputs, c.input)
            push!(all_eltypes, c.eltype)
            push!(all_sizes, c.size)
        end

        # Comparable cases: status == "ok", finite positive times & ratio
        comp = filter(r -> r.status == "ok" && !isnan(r.ratio) && r.ratio > 0 && !isnan(r.oxiblas_ms) && !isnan(r.reference_ms), cases)
        ratios = [r.ratio for r in comp]

        geo_ratio = isempty(ratios) ? NaN : exp(mean(log.(ratios)))
        med_ratio = isempty(ratios) ? NaN : median(ratios)
        faster_count = count(<(1.0), ratios)
        faster_pct = isempty(ratios) ? 0.0 : round(100.0 * faster_count / length(ratios); digits=1)

        summaries[t_str] = (
            threads = t,
            total_cases = length(cases),
            ok_cases = count(r -> r.status == "ok", cases),
            unchecked_cases = count(r -> r.status == "unchecked", cases),
            unsupported_cases = count(r -> r.status == "unsupported", cases),
            mismatch_cases = count(r -> r.status == "mismatch", cases),
            error_cases = count(r -> r.status == "error", cases),
            comparable_cases = length(comp),
            missing_reference_cases = length(cases) - length(comp),
            geomean_ratio = geo_ratio,
            median_ratio = med_ratio,
            faster_count = faster_count,
            faster_percent = faster_pct
        )

        all_cases_by_thread[t_str] = [
            (
                name = c.name,
                op = c.op,
                family = c.family,
                form = c.form,
                input = c.input,
                ref_input = c.ref_input,
                eltype = c.eltype,
                size = c.size,
                order = c.order,
                status = c.status,
                oxiblas_ms = c.oxiblas_ms,
                reference_ms = c.reference_ms,
                ratio = c.ratio,
                oxiblas_allocs = c.oxiblas_allocs,
                reference_allocs = c.reference_allocs,
                oxiblas_bytes = c.oxiblas_bytes,
                reference_bytes = c.reference_bytes,
                note = c.note
            ) for c in cases
        ]
    end

    # Standard order for filters
    eltype_order = ["f16", "f32", "f64", "cf32", "cf64", "big"]
    sorted_eltypes = [e for e in eltype_order if e in all_eltypes]
    append!(sorted_eltypes, sort(collect(setdiff(all_eltypes, eltype_order))))

    size_order = ["s", "m", "l"]
    sorted_sizes = [s for s in size_order if s in all_sizes]

    payload = (
        snapshot = (
            title = snapshot_title,
            revision = snapshot_rev,
            generated_at = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            julia_version = common_julia,
            cpu = common_cpu,
            reference_backend = common_backend,
            threads = sorted_threads
        ),
        filters = (
            families = sort(collect(all_families)),
            operations = sort(collect(all_ops)),
            inputs = sort(collect(all_inputs)),
            eltypes = sorted_eltypes,
            sizes = sorted_sizes
        ),
        summaries = summaries,
        cases = all_cases_by_thread
    )

    # Atomic write to destination
    mkpath(dirname(output_path))
    tmp_path = output_path * ".tmp"

    open(tmp_path, "w") do io
        to_json(io, payload)
    end

    # Quick sanity check on temporary file
    filesize(tmp_path) > 100 || error("Generated JSON file appears corrupted or empty: $tmp_path")
    mv(tmp_path, output_path; force=true)

    # Print publication summary
    println("="^72)
    println("  OxiBLAS Benchmark Publication Snapshot")
    println("="^72)
    println("Title:              ", snapshot_title)
    println("Revision:           ", snapshot_rev)
    println("Julia Version:      ", common_julia)
    println("CPU Architecture:   ", common_cpu)
    println("Reference Backend:  ", common_backend)
    println("Thread Configs:     ", join(string.(sorted_threads), ", "))
    println("-"^72)
    for t in sorted_threads
        s = summaries[string(t)]
        @printf("  [%2d Thread(s)] Total: %4d | Comparable: %4d | Geomean: %5.2f | Median: %5.2f | Faster: %4d (%4.1f%%)\n",
                t, s.total_cases, s.comparable_cases, s.geomean_ratio, s.median_ratio, s.faster_count, s.faster_percent)
    end
    println("-"^72)
    @printf("Dataset generated:   %s (%0.1f KB)\n", output_path, filesize(output_path) / 1024)
    println("="^72)
end

function main(args=ARGS)
    manifest_path = joinpath(@__DIR__, "publication.toml")
    output_path = joinpath(dirname(@__DIR__), "docs", "src", "assets", "benchmarks", "results.json")

    for a in args
        if startswith(a, "--manifest=")
            manifest_path = split(a, '='; limit=2)[2]
        elseif startswith(a, "--output=")
            output_path = split(a, '='; limit=2)[2]
        elseif a in ("-h", "--help")
            println("Usage: julia --project=benchmark benchmark/publish_results.jl [--manifest=path] [--output=path]")
            return
        end
    end

    publish_benchmarks(; manifest_path=manifest_path, output_path=output_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
