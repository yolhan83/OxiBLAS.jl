# Tests

`runtests.jl` loads the shared prelude (`testsetup.jl`, which includes
`helpers/`) and runs the files selected in `groups.jl`, each in its own
testset, printing one progress line per file. QA (`qa/qa.jl`: Aqua, ExplicitImports, closure
boxes on Julia 1.14+) runs with the `core`, `all` and `qa` groups.

| `OXIBLAS_TEST_GROUP` | Files |
|---|---|
| `core` (default) | representative coverage of every family; run by CI on every PR |
| `extended` | heavy campaigns, large problems, exotic element types, compilation-heavy suites |
| `all` | both |
| `blas`, `structured`, `lapack`, `sparse`, `graphs`, `spectral`, `tensors`, `generic`, `api`, `qa` | one family, both levels |

`OXIBLAS_TEST_FILES=blas/gemm.jl,api/linalg_api.jl` runs explicit files.

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
OXIBLAS_TEST_GROUP=extended julia --project=. -e 'using Pkg; Pkg.test(julia_args=["--threads=4"])'
```

`legacy/` keeps the former runners: the campaign wrappers now delegate to
`runtests.jl`, and the original files are kept with a `_v0.6` suffix.
`optional/` is the environment of the optional extended-precision providers.
