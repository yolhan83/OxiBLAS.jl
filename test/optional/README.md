# Optional scalar compatibility environment

Prepared only. No dependency installation, package loading, test, benchmark or
LLVM inspection is authorized during the current implementation phase.
The package's own Project.toml does not depend on these providers.

After explicit authorization, from the OxiBLAS package directory, prepare this
environment using `julia --project=test/optional` and
`using Pkg; Pkg.develop(path="."); Pkg.instantiate()`.
This can download dependencies and precompile them; it has not been run.

Use `OXIBLAS_OPTIONAL_FLOATS=DoubleFloats,Quadmath` to require both providers, or
`available` to log unsupported/missing providers and run those available.
Omitting the variable never imports either provider. Double64, Float128 and
their complex types join the same parameterized numerical test cases.

Future correctness campaign: set
`OXIBLAS_CORRECTNESS_PROJECT=test/optional`, then invoke
`benchmark/run_precision_campaign.sh NEW_DIRECTORY correctness`.
Future benchmark/inspection campaigns: set
`OXIBLAS_PRECISION_PROJECT=test/optional` and invoke the separate performance
or inspect phase. The launcher never prepares/installs environments implicitly.
Each phase needs authorization; no phase automatically starts the next one.

BigFloat benchmark precision is controlled by `OXIBLAS_GENERIC_BITS` (default
256); native and reference cases use the same working precision. Oracle
precision is at least 512 bits and twice the working precision. No provider
compatibility or performance claim is established by the prepared scripts.
