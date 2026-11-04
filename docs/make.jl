# Documentation build. Local use, from the repository root:
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#   julia --project=docs docs/make.jl
# The pages are written in docs/src; the API reference of each source family
# (docs/src/reference/<family>.md, from src/<family>/) is generated here,
# before makedocs, from the docstrings and the exports, and is not kept in the
# repository.
# No secret is needed to build. Deployment (deploydocs) only runs on GitHub
# Actions, where the workflow provides GITHUB_TOKEN.
using Documenter, OxiBLAS
const ROOT=dirname(@__DIR__)
const SRC=joinpath(@__DIR__,"src")

# ---- API reference by source family ---------------------------------------------
const FAMILIES=[
    ("api","API haut niveau (non exportée)","Fonctions homonymes de `LinearAlgebra`, propres à OxiBLAS et non exportées : on les appelle `OxiBLAS.lu`, `OxiBLAS.ldiv`, etc. Elles sont distinctes des fonctions de `LinearAlgebra` (`OxiBLAS.lu !== LinearAlgebra.lu`). Les types de factorisation sont des sous-types de `LinearAlgebra.Factorization`."),
    ("blas","BLAS 1, 2, 3 et GEMM",""),
    ("lapack","Factorisations denses et LAPACK",""),
    ("structured","Matrices structurées (compactes et bandes)",""),
    ("sparse","Matrices creuses",""),
    ("graphs","Graphes et ordonnancements",""),
    ("spectral","Algorithmes spectraux et itératifs",""),
    ("tensors","Tenseurs",""),
    ("utils","Utilitaires, matériel et précision",""),
]
# The documented names of the high-level API are not exported: they are
# collected from the docstrings whose source file is in src/api/.
function family_of(path)
    rel=relpath(path,joinpath(ROOT,"src"))
    parts=splitpath(rel)
    length(parts)>=2 ? parts[1] : "utils"
end
function definition_files(f)
    ms=f isa Union{Function,Type} ? [m for m in methods(f) if parentmodule(m)===OxiBLAS] : Method[]
    unique(String(m.file) for m in ms)
end
function reference_pages()
    meta=Base.Docs.meta(OxiBLAS)
    documented=Dict{String,Vector{String}}()
    exported=Set(filter(!=(:OxiBLAS),names(OxiBLAS)))
    foreign=Dict{String,Vector{String}}()
    for (binding,multidoc) in meta
        name=binding.var
        docs=collect(values(multidoc.docs))
        isempty(docs) && continue
        fam=family_of(String(docs[1].data[:path]))
        if binding.mod!==OxiBLAS
            # A docstring of an OxiBLAS method of another module's function
            # (LinearAlgebra.ldiv!, SparseArrays.fkeep!...): listed by name,
            # since the binding belongs to that module.
            push!(get!(foreign,fam,String[]),"$(binding.mod).$(name)")
            continue
        end
        # Exported names, and every documented name of the high-level API.
        (name in exported || fam=="api") || continue
        push!(get!(documented,fam,String[]),String(name))
    end
    undocumented=Dict{String,Vector{String}}()
    for name in exported
        Base.Docs.hasdoc(OxiBLAS,name) && continue
        files=definition_files(getfield(OxiBLAS,name))
        fam=isempty(files) ? "utils" : family_of(files[1])
        push!(get!(undocumented,fam,String[]),String(name))
    end
    out=joinpath(SRC,"reference"); mkpath(out)
    pages=String[]
    for (fam,title,intro) in FAMILIES
        docnames=sort(get(documented,fam,String[])); bare=sort(get(undocumented,fam,String[]))
        isempty(docnames) && isempty(bare) && isempty(get(foreign,fam,String[])) && continue
        open(joinpath(out,"$fam.md"),"w") do io
            println(io,"# $title\n")
            isempty(intro) || println(io,intro,"\n")
            println(io,"Sources : `src/$fam/`.\n")
            if !isempty(docnames)
                println(io,"```@docs")
                foreach(n->println(io,"OxiBLAS.$n"),docnames)
                println(io,"```\n")
            end
            ext=sort(unique(get(foreign,fam,String[])))
            if !isempty(ext)
                println(io,"## Méthodes OxiBLAS de fonctions d'autres modules\n")
                println(io,"Ces fonctions appartiennent à `LinearAlgebra`, `SparseArrays` ou `Base` ; OxiBLAS leur ajoute des méthodes pour ses propres types (voir les guides) :\n")
                println(io,join(("`$n`" for n in ext),", "),"\n")
            end
            if !isempty(bare)
                println(io,"## Fonctions exportées sans docstring\n")
                println(io,"Leur convention d'appel est décrite dans les guides ; une docstring reste à écrire.\n")
                println(io,join(("`$n`" for n in bare),", "),"\n")
            end
        end
        push!(pages,"reference/$fam.md")
    end
    "Référence de l'API"=>vcat(["reference/index.md"],pages)
end

reference=reference_pages()
# The repository is fixed. In a Git checkout (GitHub Actions) Documenter takes
# the revision from Git; in a local copy without .git, source links point to
# the main branch.
const GITHUB_REPO=Documenter.Remotes.GitHub("yolhan83","OxiBLAS.jl")
const REMOTE_SETTINGS=isdir(joinpath(ROOT,".git")) ? (;repo=GITHUB_REPO) :
                      (;remotes=Dict(ROOT=>(GITHUB_REPO,"main")))
makedocs(;
    REMOTE_SETTINGS...,
    sitename="OxiBLAS.jl",
    modules=[OxiBLAS],
    authors="OxiBLAS.jl contributors",
    format=Documenter.HTML(;prettyurls=get(ENV,"CI","false")=="true",edit_link=nothing,
                            size_threshold=nothing,size_threshold_warn=1024^2,search_size_threshold_warn=1024^2,collapselevel=1),
    pages=[
        "Accueil"=>"index.md",
        "Guide"=>[
            "guide/blas.md",
            "guide/factorizations.md",
            "guide/highlevel.md",
            "guide/structured.md",
            "guide/sparse.md",
            "guide/spectral.md",
            "guide/tensors.md",
            "guide/performance.md",
        ],
        reference,
        "Développement"=>vcat(["dev/index.md"],
            ["dev/$f" for f in sort(filter(f->endswith(f,".md") && f!="index.md",readdir(joinpath(SRC,"dev"))))]),
    ],
    # Exported names must all appear (with or without docstring, see the
    # reference pages); internal helpers with a docstring are not listed.
    checkdocs=:exports,
    # The archived development pages name files of the repository that are
    # not pages (logs, CSV, source files): reported, not fatal.
    warnonly=[:cross_references,:missing_docs],
)
if get(ENV,"GITHUB_ACTIONS","false")=="true"
    deploydocs(;repo="github.com/yolhan83/OxiBLAS.jl.git",devbranch="main",push_preview=false)
end
