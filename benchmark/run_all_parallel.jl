# Automated runner for all 17-thread benchmark campaigns.
# Runs: products, factorizations, spectral, functions, solves.
# Updates benchmark/publication.toml and docs/src/assets/benchmarks/results.json after each.

using Dates, TOML

const ROOT = dirname(@__DIR__)
const RESULTS_DIR = joinpath(ROOT, "benchmark", "results")
const PUB_PATH = joinpath(ROOT, "benchmark", "publication.toml")

const FAMILIES = [
    ("products", 2),
    ("factorizations", 4),
    ("solves", 6),
    ("spectral", 8),
    ("tensors", 10),
    ("functions", 12),
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
    println("  STARTING CAMPAIGN: $family (17 threads)")
    println("  Time: ", now())
    println("="^70)
    
    before = get_csv_files()
    run_file = joinpath(ROOT, "benchmark", "run.jl")
    cmd = `$(Base.julia_cmd()) -t 17 --project=benchmark $run_file --families=$family`
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
    run(`$(Base.julia_cmd()) $pub_file`)
    println("Campaign $family completed and published at $(now()).\n")
end

function main()
    total_start = now()
    println("Launching complete 17T benchmark suite at $total_start")
    
    for (fam, idx) in FAMILIES
        run_campaign(fam, idx)
    end
    
    println("\n" * "="^70)
    println("  ALL 17T BENCHMARKS COMPLETED SUCCESSFULLY!")
    println("  Total elapsed: ", canonicalize(Dates.CompoundPeriod(now() - total_start)))
    println("="^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
