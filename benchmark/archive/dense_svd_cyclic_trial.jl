# Process-local alternative to the allocating Hermitian eigensolver.
using OxiBLAS
@eval OxiBLAS _svd_block_rotation!(G,W,rtol,m)=_svd_block_rotation_cyclic!(G,W,rtol,m)
include("dense_svd_jacobi_tuning.jl")
