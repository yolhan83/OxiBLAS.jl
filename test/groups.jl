# Test files by family, with their level:
#   :core      run on every pull request (OXIBLAS_TEST_GROUP=core, the default);
#   :extended  heavy campaigns, large problems, exotic element types,
#              compilation-heavy suites (OXIBLAS_TEST_GROUP=extended).
# OXIBLAS_TEST_GROUP=all runs both; a family name runs that family at both
# levels. Paths are relative to test/. The QA (qa/qa.jl) runs with core, all
# and qa.
const TEST_GROUPS=[
    "blas"=>[
        ("blas/gemm.jl",:core),
        ("blas/blas.jl",:core),
        ("blas/level1_extended.jl",:core),
        ("blas/level1_optimization.jl",:core),
        ("blas/batched.jl",:core),
        ("blas/level3_parallel.jl",:core),
        ("blas/continuation_parallel.jl",:core),
        ("blas/level3_blocked.jl",:extended),
    ],
    "structured"=>[
        ("structured/storage_sparse.jl",:core),
        ("structured/compact_kernels.jl",:core),
    ],
    "lapack"=>[
        ("lapack/factorizations.jl",:core),
        ("lapack/lapack_dense.jl",:core),
        ("lapack/hessenberg.jl",:core),
        ("lapack/schur.jl",:core),
        ("lapack/hermitian_eigen.jl",:core),
        ("lapack/svd.jl",:core),
        ("lapack/advanced_factorizations.jl",:core),
        ("lapack/svd_selected.jl",:core),
        ("lapack/dense_svd_blocked.jl",:core),
        ("lapack/dense_svd_block_jacobi.jl",:core),
        ("lapack/dense_svd_qr_precondition.jl",:core),
        ("lapack/dense_svd_complex_subnormal.jl",:core),
        ("lapack/dense_svd_api_compat.jl",:core),
        ("lapack/dense_svd_campaign.jl",:core),
        ("lapack/qz_schur_complete.jl",:extended),
    ],
    "sparse"=>[
        ("sparse/sparse_extended.jl",:core),
        ("sparse/sparse_pivot_regression.jl",:core),
        ("sparse/sparse_direct_kernels.jl",:core),
        ("sparse/ainv.jl",:core),
        ("sparse/sparse_indefinite.jl",:core),
        ("sparse/multifrontal_lu.jl",:core),
        ("sparse/out_of_core.jl",:core),
        ("sparse/sparse_interop.jl",:core),
        ("sparse/sparse_supertypes.jl",:core),
        ("sparse/sparse_base_ops.jl",:core),
        ("sparse/sparse_memory_triangular.jl",:extended),
    ],
    "graphs"=>[
        ("graphs/weighted_matching.jl",:core),
        ("graphs/sparse_ordering_advanced.jl",:core),
        ("graphs/multilevel_partition.jl",:core),
        ("graphs/graph_matching.jl",:core),
        ("graphs/graph_refinement.jl",:core),
        ("graphs/graph_structure.jl",:core),
    ],
    "spectral"=>[
        ("spectral/stochastic.jl",:core),
        ("spectral/block_spectral.jl",:core),
        ("spectral/sparse_svd_iterative.jl",:core),
        ("spectral/incremental_svd.jl",:core),
        ("spectral/spectral_filter.jl",:core),
        ("spectral/interval_spectral.jl",:core),
        ("spectral/restarted_arnoldi_xtrace.jl",:core),
    ],
    "tensors"=>[
        ("tensors/tensors.jl",:core),
        ("tensors/tensor_memory.jl",:core),
        ("tensors/tensor_planning.jl",:core),
    ],
    # Generic and extended-precision element types (Float16, BigFloat,
    # optional providers): compilation-heavy by nature.
    "generic"=>[
        ("generic/extended_scalar.jl",:extended),
        ("generic/extended_blas.jl",:extended),
        ("generic/extended_lapack.jl",:extended),
        ("generic/extended_spectral.jl",:extended),
        ("generic/generic_contract.jl",:extended),
        ("generic/generic_tensors.jl",:extended),
    ],
    "api"=>[
        ("api/mul_layer.jl",:core),
        ("api/linalg_api.jl",:core),
        ("api/linalg_api_structured.jl",:core),
        ("api/linalg_api_solve.jl",:core),
    ],
    "qa"=>[
        ("qa/unbound_args_regression.jl",:core),
        ("qa/audit_priority.jl",:core),
    ],
]
const TEST_LEVELS=("core","extended","all")

function selected_test_files(group::AbstractString)
    names=first.(TEST_GROUPS)
    group in TEST_LEVELS || group in names ||
        error("unknown test group $(repr(group)); use one of $(join(vcat(collect(TEST_LEVELS),names),", "))")
    files=String[]
    for (family,entries) in TEST_GROUPS, (file,level) in entries
        keep=group=="all" || group==family || (group=="core" && level===:core) || (group=="extended" && level===:extended)
        keep && push!(files,file)
    end
    files
end
