# Automated runner for all 1-thread benchmark campaigns.
# Runs: products, factorizations, solves, spectral, tensors, functions.
# Updates benchmark/publication.toml and docs/src/assets/benchmarks/results.json after each.

using Dates, TOML

const ROOT = dirname(@__DIR__)
const RESULTS_DIR = joinpath(ROOT, "benchmark", "results")
const PUB_PATH = joinpath(ROOT, "benchmark", "publication.toml")
const BUILD_RESULTS_DIR = joinpath(ROOT, "docs", "build", "assets", "benchmarks")
const SRC_RESULTS_FILE = joinpath(ROOT, "docs", "src", "assets", "benchmarks", "results.json")

# Ensure we use Julia 1.13.3
const JULIA_1_13 = "/Users/yolhan/.julia/juliaup/julia-1.13.3+0.aarch64.apple.darwin14/Julia-1.13.app/Contents/Resources/julia/bin/julia"
const JULIA_BIN = isfile(JULIA_1_13) ? JULIA_1_13 : (Sys.which("julia") !== nothing ? Sys.which("julia") : "julia")

const FAMILIES = [
    ("products", 1),
    ("factorizations", 3),
    ("solves", 5),
    ("spectral", 7),
    ("tensors", 9),
    ("functions", 11),
]

function get_csv_files()
    Set(filter(f -> endswith(f, ".csv"), readdir(RESULTS_DIR)))
end

function update_publication(campaign_idx::Int, new_rel_csv::String)
    content = read(PUB_PATH, String)
    data = TOML.parse(content)
    old_csv = data["campaigns"][campaign_idx]["csv"]
    println("Updating campaign #$campaign_idx in publication.toml: $old_csv -> $new_rel_csv")
    
    target = "csv = \"$old_csv\""
    replacement = "csv = \"$new_rel_csv\""
    new_content = replace(content, target => replacement, count=1)
    write(PUB_PATH, new_content)
end

function run_campaign(family::String, campaign_idx::Int)
    println("\n" * "="^70)
    println("  STARTING CAMPAIGN: $family (1 thread)")
    println("  Time: ", now())
    println("  Julia: ", JULIA_BIN)
    println("="^70)
    
    before = get_csv_files()
    run_file = joinpath(ROOT, "benchmark", "run.jl")
    cmd = `$JULIA_BIN -t 1 --project=benchmark $run_file --families=$family`
    run(cmd)
    
    after = get_csv_files()
    diff = collect(setdiff(after, before))
    isempty(diff) && error("No new CSV file generated for family $family!")
    
    sort!(diff, by=f -> stat(joinpath(RESULTS_DIR, f)).mtime, rev=true)
    new_csv = diff[1]
    rel_path = "results/$new_csv"
    println("Generated new CSV: $rel_path")
    
    update_publication(campaign_idx, rel_path)
    
    println("Publishing results snapshot...")
    pub_file = joinpath(ROOT, "benchmark", "publish_results.jl")
    run(`$JULIA_BIN $pub_file`)

    if isdir(BUILD_RESULTS_DIR) && isfile(SRC_RESULTS_FILE)
        cp(SRC_RESULTS_FILE, joinpath(BUILD_RESULTS_DIR, "results.json"); force=true)
        println("Synced results.json to docs/build/assets/benchmarks/results.json")
    end
    
    println("Campaign $family completed and published at $(now()).\n")
end

function main()
    total_start = now()
    println("Launching complete 1T benchmark suite at $total_start using Julia 1.13")
    
    for (fam, idx) in FAMILIES
        run_campaign(fam, idx)
    end
    
    println("\n" * "="^70)
    println("  ALL 1T BENCHMARKS COMPLETED SUCCESSFULLY!")
    println("  Total elapsed: ", canonicalize(Dates.CompoundPeriod(now() - total_start)))
    println("="^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
