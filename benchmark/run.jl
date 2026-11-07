# High-level benchmark suite: every high-level operation, on every input type
# it accepts (LinearAlgebra and OxiBLAS types), for Float16, Float32, Float64,
# ComplexF32, ComplexF64 and BigFloat, at three sizes; against the matching
# reference (LinearAlgebra with OpenBLAS or Accelerate, or its generic code
# for Float16/BigFloat; SparseArrays' UMFPACK/CHOLMOD/SPQR; TensorOperations).
#
#   julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#   julia --project=benchmark --threads=1 benchmark/run.jl                      # whole grid
#   julia --project=benchmark --threads=1 benchmark/run.jl lu cholesky          # names starting with lu or cholesky
#   julia --project=benchmark --threads=1 benchmark/run.jl 'ldiv!/*/f64/*'      # glob on op/input/type/size
#   julia --project=benchmark benchmark/run.jl --types=f32,f64 --sizes=s,m --families=solves --inputs=csr,dense
#   julia --project=benchmark benchmark/run.jl --allocs                         # allocation campaign only (no timing)
#   julia --project=benchmark benchmark/run.jl --list                           # print the selected case names
#   OXIBLAS_REFERENCE=accelerate julia --project=benchmark benchmark/run.jl     # macOS
#
# Case names are op/input/type/size, e.g. lu/csr/f64/m. Sizes s, m, l depend
# on the input kind and the element type (benchmark/suite/inputs.jl).
#
# Method:
#   - same data and same thread count for both sides (the reference BLAS gets
#     Threads.nthreads() threads);
#   - one warm-up call each compiles; the OxiBLAS result is then checked (the
#     check does not need the reference, so it also runs when there is none);
#   - allocations: count and bytes of one warm call (Base.@allocations,
#     Base.@allocated), for both sides;
#   - times: median of warm samples (Chairmarks), OXIBLAS_BENCH_SECONDS each;
#   - only the selected cases are built and compiled (invokelatest per case).
# Status of each case:
#   ok            checked and timed;
#   unchecked     timed, nothing could validate the result (no reference);
#   unsupported   OxiBLAS has no method for this input (MethodError on its API);
#   mismatch      the check failed;
#   error         OxiBLAS raised another exception (message in `note`).
# A missing reference (LinearAlgebra has no method, or no BLAS/LAPACK for the
# type) leaves the reference columns empty with the reason in `note`.
# Output: benchmark/results/<stamp>.csv and .md.
using OxiBLAS, Chairmarks, SparseArrays, Random, Printf, Dates, Statistics
import LinearAlgebra as LA
const REFERENCE=get(ENV,"OXIBLAS_REFERENCE","openblas")
if REFERENCE=="accelerate"
    Sys.isapple() || error("Accelerate is only available on macOS")
    @eval using AppleAccelerate
end
using TensorOperations
LA.BLAS.set_num_threads(Threads.nthreads())
const SECONDS=parse(Float64,get(ENV,"OXIBLAS_BENCH_SECONDS","0.5"))
const O=OxiBLAS

include(joinpath(@__DIR__,"suite","inputs.jl"))
include(joinpath(@__DIR__,"suite","operations.jl"))

struct Case
    name::String
    op::Op
    kind::Symbol
    T::Type
    size_label::String
    n::Int                # size parameter (order, grid side or tensor extent)
end
function all_cases()
    cases=Case[]
    for op in OPS, kind in op.kinds, (T,_) in ELTYPES
        for (lab,n) in zip(SIZE_LABELS,sizes_for(kind,op.class,T))
            push!(cases,Case(join((op.name,kind,eltype_label(T),lab),'/'),op,kind,T,lab,n))
        end
    end
    cases
end

# ---- command line ---------------------------------------------------------
function parse_args(args)
    opts=Dict{String,Vector{String}}(); pats=String[]; flags=Set{String}()
    for a in args
        if startswith(a,"--")
            kv=split(a[3:end],'=';limit=2)
            length(kv)==1 ? push!(flags,kv[1]) : (opts[kv[1]]=split(kv[2],','))
        else
            push!(pats,a)
        end
    end
    pats,opts,flags
end
globre(p)=Regex("^"*replace(p,r"[.+^$()|\[\]{}\\]"=>s"\\\0","*"=>"[^/]*")*"\$")
function selected_cases(pats,opts)
    sel=all_cases()
    isempty(pats) || filter!(c->any(p->occursin('*',p) ? occursin(globre(p),c.name) : startswith(c.name,p),pats),sel)
    haskey(opts,"types") && filter!(c->eltype_label(c.T) in opts["types"],sel)
    haskey(opts,"sizes") && filter!(c->c.size_label in opts["sizes"],sel)
    haskey(opts,"families") && filter!(c->c.op.family in opts["families"],sel)
    haskey(opts,"inputs") && filter!(c->String(c.kind) in opts["inputs"],sel)
    sel
end

# ---- one case -------------------------------------------------------------
short(e)=first(replace(sprint(showerror,e),r"\s+"=>" "),140)
function is_unsupported(e,api)
    if e isa MethodError
        f=e.f
        f===api && return true
        m=try parentmodule(f) catch; nothing end
        return m===OxiBLAS && !startswith(String(nameof(f)),"_")
    end
    e isa ArgumentError && occursin(r"not (supported|available|implemented)|refuse"i,e.msg)
end
function allocs_of(f::F) where F
    a=Base.@allocations f()
    b=Base.@allocated f()
    a,b
end
function median_time(f::F) where F
    b=@be f() seconds=SECONDS
    median(s.time for s in b.samples)
end

const SUITESPARSE_FAMILIES=("factorizations","solves")
const SUITESPARSE_OPS=("det","logabsdet")
no_suitesparse(c)=(c.kind in SPARSE_KINDS || c.kind in BANDED_KINDS) && !(c.T in (Float64,ComplexF64)) &&
                  (c.op.family in SUITESPARSE_FAMILIES || c.op.name in SUITESPARSE_OPS)
# LinearAlgebra.sylvester / lyap recurse infinitely in generic trsyl! without LAPACK.
no_lapack_sylvester(c)=c.op.name in ("sylvester","lyap") && generic_type(c.T)
chol_det_ref(c,R)=c.op.name in ("det","logabsdet") && R isa LA.Hermitian{<:Any,<:SparseMatrixCSC}

blank(c,ref_label)=(;name=c.name,op=c.op.name,family=c.op.family,form=c.op.form,input=String(c.kind),ref_input=ref_label,
    eltype=eltype_label(c.T),size=c.size_label,order=c.kind===:tensor ? c.n : order(c.kind,c.n),status="",
    oxiblas_s=NaN,reference_s=NaN,oxiblas_allocs=-1,reference_allocs=-1,oxiblas_bytes=-1,reference_bytes=-1,note="")

function run_case(c::Case,timing::Bool)
    r=rng(); T=c.T
    A,R,ref_label=c.kind===:tensor ? (nothing,nothing,"TensorOperations") : make_input(c.kind,T,c.n,r)
    if chol_det_ref(c,R)
        ref_label="cholesky(Hermitian{SparseMatrixCSC})"
    end
    row=blank(c,ref_label)
    built=try
        c.op.build(A,R,T,row.order,r)
    catch e
        return merge(row,(;status=is_unsupported(e,c.op.api) ? "unsupported" : "error",note="setup: "*short(e)))
    end
    oxi,ref0,check=built
    # SuiteSparse (UMFPACK, CHOLMOD, SPQR) only handles Float64 and ComplexF64:
    # on other element types, LinearAlgebra falls back to generic dense code
    # on the sparse matrix, which does not finish at the larger sizes.
    if no_suitesparse(c) || no_lapack_sylvester(c)
        ref0=nothing
    end
    xo=try oxi() catch e
        return merge(row,(;status=is_unsupported(e,c.op.api) ? "unsupported" : "error",note=short(e)))
    end
    note=""; xr=nothing; ref=ref0
    if ref===nothing
        note=no_suitesparse(c) ? "no reference: SuiteSparse has no $(eltype_label(c.T))" :
             no_lapack_sylvester(c) ? "no reference: LinearAlgebra.$(c.op.name) requires LAPACK trsyl!" :
             (c.op.name == "norm" && c.kind === :tridiagonal) ? "no reference: LinearAlgebra has no specialized norm for Tridiagonal" : "no reference"
    else
        try
            xr=ref()
            if chol_det_ref(c,R)
                note="ref: $(c.op.name)(cholesky(R)) (LA.$(c.op.name) densifies Hermitian{SparseMatrixCSC})"
            end
        catch e; note="no reference: "*short(e); ref=nothing end
    end
    ok=try check(xo,xr) catch e; note*=(isempty(note) ? "" : "; ")*"check: "*short(e); false end
    status=ok===missing ? "unchecked" : ok ? "ok" : "mismatch"
    ao,bo=allocs_of(oxi)
    ar,br=ref===nothing ? (-1,-1) : allocs_of(ref)
    to=timing ? median_time(oxi) : NaN
    tr=timing && ref!==nothing ? median_time(ref) : NaN
    merge(row,(;status,oxiblas_s=to,reference_s=tr,oxiblas_allocs=ao,reference_allocs=ar,oxiblas_bytes=bo,reference_bytes=br,note))
end

# ---- reports --------------------------------------------------------------
fmt_ms(t)=isnan(t) ? "" : @sprintf("%.4g",1e3t)
fmt_int(x)=x<0 ? "" : string(x)
ratio(r)=r.oxiblas_s/r.reference_s
csvfield(s)=replace(string(s),','=>';','"'=>"'")
function write_csv(path,rows)
    open(path,"w") do io
        println(io,"name,op,family,form,input,ref_input,eltype,size,order,status,oxiblas_ms,reference_ms,ratio,oxiblas_allocs,reference_allocs,oxiblas_bytes,reference_bytes,note")
        for r in rows
            q=ratio(r)
            println(io,join(csvfield.((r.name,r.op,r.family,r.form,r.input,r.ref_input,r.eltype,r.size,r.order,r.status,
                fmt_ms(r.oxiblas_s),fmt_ms(r.reference_s),isnan(q) ? "" : @sprintf("%.3f",q),
                fmt_int(r.oxiblas_allocs),fmt_int(r.reference_allocs),fmt_int(r.oxiblas_bytes),fmt_int(r.reference_bytes),r.note)),','))
        end
    end
end
geomean(v)=isempty(v) ? NaN : exp(mean(log.(v)))
function summary_table(io,rows,key,title)
    println(io,"\n## By ",lowercase(title),"\n")
    println(io,"| $title | Cases | Ran | Unsupported | Mismatch / error | Timed vs reference | Geometric mean ratio | Median ratio | OxiBLAS faster |")
    println(io,"|---|---|---|---|---|---|---|---|---|")
    for k in unique(key.(rows))
        g=filter(r->key(r)==k,rows)
        q=[ratio(r) for r in g if r.status=="ok" && !isnan(ratio(r))]
        @printf(io,"| %s | %d | %d | %d | %d | %d | %s | %s | %s |\n",k,length(g),count(r->r.status in ("ok","unchecked"),g),
            count(r->r.status=="unsupported",g),count(r->r.status in ("mismatch","error"),g),length(q),
            isempty(q) ? "" : @sprintf("%.2f",geomean(q)),isempty(q) ? "" : @sprintf("%.2f",median(q)),
            isempty(q) ? "" : "$(count(<(1),q))/$(length(q))")
    end
end
function write_md(path,rows,header,timing)
    open(path,"w") do io
        println(io,"# High-level benchmark\n\n$header\n")
        println(io,"Ratio = OxiBLAS time / reference time (below 1: OxiBLAS is faster); ratios only count checked cases. ",
                   "Allocations are those of one warm call.\n")
        println(io,"Status counts: ",join(("$s $(count(r->r.status==s,rows))" for s in ("ok","unchecked","unsupported","mismatch","error")),", "),".")
        if timing
            summary_table(io,rows,r->r.family,"Family")
            summary_table(io,rows,r->r.op,"Operation")
            summary_table(io,rows,r->r.eltype,"Element type")
            summary_table(io,rows,r->r.input,"Input")
        end
        done=filter(r->r.oxiblas_allocs>=0,rows)
        println(io,"\n## Allocations\n")
        println(io,"BigFloat arithmetic allocates by itself: BigFloat cases are counted in the tables but not in the in-place list.\n")
        inplace=filter(r->r.form=="in-place" && r.eltype!="big" && r.oxiblas_allocs>0,done)
        println(io,"In-place cases that allocate (BigFloat excluded): $(length(inplace)) of $(count(r->r.form=="in-place" && r.eltype!="big",done)).\n")
        more=filter(r->r.reference_allocs>=0 && r.oxiblas_allocs>r.reference_allocs,done)
        println(io,"Cases where OxiBLAS allocates more often than the reference: $(length(more)) of $(count(r->r.reference_allocs>=0,done)).\n")
        println(io,"| Case | Form | OxiBLAS allocs | Reference allocs | OxiBLAS bytes | Reference bytes |")
        println(io,"|---|---|---|---|---|---|")
        for r in sort(unique(vcat(inplace,more));by=r->-r.oxiblas_allocs)
            @printf(io,"| `%s` | %s | %d | %s | %d | %s |\n",r.name,r.form,r.oxiblas_allocs,fmt_int(r.reference_allocs),r.oxiblas_bytes,fmt_int(r.reference_bytes))
        end
        bad=filter(r->r.status in ("mismatch","error"),rows)
        if !isempty(bad)
            println(io,"\n## Mismatches and errors\n")
            println(io,"| Case | Status | Note |\n|---|---|---|")
            for r in bad; println(io,"| `$(r.name)` | $(r.status) | $(replace(r.note,'|'=>'/')) |"); end
        end
        println(io,"\n## All cases\n")
        println(io,"| Case | Reference input | Order | Status | OxiBLAS (ms) | Reference (ms) | Ratio | OxiBLAS allocs | Reference allocs | OxiBLAS bytes | Reference bytes |")
        println(io,"|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows
            q=ratio(r)
            @printf(io,"| `%s` | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |\n",r.name,r.ref_input,r.order,r.status,
                fmt_ms(r.oxiblas_s),fmt_ms(r.reference_s),isnan(q) ? "" : @sprintf("%.2f",q),
                fmt_int(r.oxiblas_allocs),fmt_int(r.reference_allocs),fmt_int(r.oxiblas_bytes),fmt_int(r.reference_bytes))
        end
    end
end

# ---- main -----------------------------------------------------------------
function main(args)
    pats,opts,flags=parse_args(args)
    sel=selected_cases(pats,opts)
    isempty(sel) && error("no case matches $(args); see --list")
    if "list" in flags
        foreach(c->println(c.name),sel); println(length(sel)," cases"); return
    end
    timing=!("allocs" in flags)
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    stamp=Dates.format(now(),"yyyymmdd_HHMMSS")*(timing ? "" : "_allocs")
    header="Julia $(VERSION), $(Threads.nthreads()) thread(s), reference $(REFERENCE) ($(basename(LA.BLAS.get_config().loaded_libs[1].libname))), $(Sys.CPU_NAME), $(length(sel)) cases"*(timing ? "" : ", allocation campaign (no timing)")
    println(header)
    rows=[]
    for (i,c) in enumerate(sel)
        row=Base.invokelatest(run_case,c,timing); push!(rows,row)
        @printf("[%d/%d] %-40s %-11s oxi %9s ms  ref %9s ms  allocs %s / %s %s\n",i,length(sel),row.name,row.status,
                fmt_ms(row.oxiblas_s),fmt_ms(row.reference_s),fmt_int(row.oxiblas_allocs),fmt_int(row.reference_allocs),
                isempty(row.note) ? "" : "  ($(first(row.note,70)))")
        flush(stdout)
        if i%25==0 || i==length(sel)           # partial results survive an interrupted run
            write_csv(joinpath(outdir,"$stamp.csv"),rows)
        end
    end
    write_md(joinpath(outdir,"$stamp.md"),rows,header,timing)
    println("Results: ",joinpath(outdir,"$stamp.csv"),", ",joinpath(outdir,"$stamp.md"))
end
main(ARGS)
