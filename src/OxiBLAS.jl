# Native Julia adaptation of OxiBLAS. See NOTICE and LICENSE.
# SPDX-License-Identifier: Apache-2.0
module OxiBLAS

# Wrapper types only. No LinearAlgebra numerical routine is an implementation
# backend; test/benchmark references are isolated outside src/.
import LinearAlgebra: Transpose, Adjoint
# The standard exceptions are reused (re-exported as the same bindings), so
# `using LinearAlgebra, OxiBLAS` has no name conflict.
import LinearAlgebra: SingularException, PosDefException
# SparseArrays interoperability: conversions, OxiBLAS sparse routines and
# tensor contractions accept its matrices and vectors (sparse_arrays.jl).
import SparseArrays
import SparseArrays: SparseMatrixCSC, SparseVector

export graph_bandwidth_profile, refine_partition, bipartite_graph_matching,
       SparseLDLFactor, sparse_ldlt, ldl_inertia, SparseStaticLUFactor, sparse_lu_static,
       sparse_direct_diagnostics, MultifrontalLUFactor, multifrontal_lu,
       OutOfCoreFactor, out_of_core_lu, out_of_core_cholesky, out_of_core_memory_usage,
       gemm!, gemm_reference!, gemm_blocked!, gemv!, GemmConfig, GemmWorkspace,
       kernel_info, axpy!, scal!, copy_vector!, swap!, dotu, dotc, dot_kahan,
       dot_pairwise, dsdot, sdsdot, nrm2, asum, iamax, iamin, rot!, rotg,
       RotmParams, rotm!, rotmg, rotmg!,
       ger!, gerc!, symv!, hemv!, trmv!, trsv!, syr!, syr2!, her!, her2!,
       symm!, hemm!, trmm!, trsm!, triangular_workspace, syrk!, syr2k!, herk!, her2k!, gemmt!,
       lu_factor, cholesky_factor, qr_factor, solve, least_squares,
       LUFactor, CholeskyFactor, QRFactor, SingularException, PosDefException,
       gemm_batched!, gemm_strided_batched!, mixed_solve,
       PackedMatrix, BandedMatrix, pack_matrix, band_matrix,
       spmv!, hpmv!, tpmv!, tpsv!, spr!, spr2!, hpr!, hpr2!,
       gbmv!, sbmv!, hbmv!, tbmv!, tbsv!, CSRMatrix, sparse_mul!, cg,
       HessenbergFactor, hessenberg_factor, hessenberg_q, apply_hessenberg_q!,
       SchurFactor, SchurWorkspace, schur_factor, eigenvalues, ConvergenceError,
       trevc_right, trevc_left, trsna_s, trsna_sep, schur_eigenvectors,
       qz_reorder, qz_condition, schur_condition, generalized_sylvester,
       TridiagonalFactor, tridiagonal_factor, tridiagonal_q,
       EigenFactor, eigen_hermitian, eigen_tridiagonal,
       SVDFactor, svd_factor, singular_values,
       CompleteLUFactor, lu_complete, lu_rook, PivotedQRFactor, qr_pivoted,
       LDLFactor, ldlt_factor, bunchkaufman_factor, eigen_generalized, expert_solve,
       CSCMatrix, COOMatrix, sparse_matmul, ILUFactor, ilu0, gmres, bicgstab, lanczos,
       EinsumPlan, SlicedEinsumPlan, EinsumWorkspace, SparseTensor, plan_einsum, plan_einsum_sliced, einsum_layout_report, einsum, einsum!, contract, contract!, outer_product, batched_matmul, batched_matmul!,
       mode_product, TuckerFactor, hosvd, TensorTrainFactor, tensor_train, CPFactor, cp_als, reconstruct,
       BidiagonalFactor, bidiagonal_factor, schur_reorder, eigen_general,
       GemmTuning, autotune_gemm, save_gemm_tuning, load_gemm_tuning,
       eigen_tridiagonal_dc, eigen_hermitian_dc, svd_bidiagonal, svd_divide_conquer,
       svd_truncated, svd_selected, svd_randomized, QZFactor, qz_factor, eigen_pencil,
       OrthogonalFactor, lq_factor, ql_factor, rq_factor, CompleteOrthogonalFactor,
       complete_orthogonal, apply_q!, matrix_norm, native_inverse, cholesky_inverse, pseudoinverse,
       numerical_rank, null_space, left_null_space, column_space, row_space,
       condition_number, determinant, logabsdet_factor, equilibrate, backward_error,
       kron_product, khatri_rao, kron_sum, kronecker_action,
       matrix_exp, matrix_sqrt, matrix_log, matrix_power, matrix_sign, matrix_sin,
       matrix_cos, frechet_matrix, frechet_exp, frechet_log, frechet_sqrt,
       DIAMatrix, ELLMatrix, BSRMatrix, BSCMatrix, HYBMatrix, SELLMatrix,
       SparseLUFactor, sparse_lu, ilut, SparseCholeskyFactor, sparse_cholesky,
       ic0, ict, sparse_triangular_solve, sparse_transpose, sparse_add, sparse_permute,
       sparse_diagonal, sparse_norm, connected_components, level_sets,
       reverse_cuthill_mckee, minimum_degree_ordering, nested_dissection,
       symbolic_cholesky, amd_ordering, colamd_ordering, bandwidth_profile, bipartite_matching,
       read_matrix_market, write_matrix_market, SparseQRFactor, sparse_qr,
       sparse_apply_q!, sparse_svd, qr_q, RelaxationPreconditioner,
       relaxation_preconditioner, PolynomialPreconditioner, polynomial_preconditioner,
       ApproxInversePreconditioner, spai, AdditiveSchwarzPreconditioner,
       additive_schwarz, AMGPreconditioner, amg, fgmres, pcg, minres, tfqmr,
       arnoldi, iram, shift_invert, lobpcg, eigen_tridiagonal_mrrr, eigen_hermitian_mrrr,
       BandLUFactor, band_lu, BandCholeskyFactor, band_cholesky,
       PackedCholeskyFactor, packed_cholesky, AasenFactor, aasen_factor,
       MultifrontalCholeskyFactor, multifrontal_cholesky, supernodal_cholesky,
       SparseBlockedLUFactor, sparse_lu_blocked, gemm_cache_oblivious!,
       gemm_strassen!, gemm_winograd!, axpy_parallel!, scal_parallel!,
       dot_parallel, nrm2_parallel, asum_parallel, schur_multishift,
       BalanceFactor, balance_matrix, balance_backtransform, qmr, idrs,
       block_cg, block_gmres, GenericWorkspace, generic_threadsafe, gemm_extended!, dot_extended,
       solve_transposed, inverse_norm_estimate, reciprocal_condition,
       refine_solve, expert_cholesky, expert_symmetric, AINVPreconditioner, ainv,
       StochasticConfig, trace_hutchinson, trace_hutch_plus_plus, trace_xtrace, trace_xtrace_loo,
       stochastic_diagonal, stochastic_frobenius, stochastic_logdet,
       BlockKrylovFactor, block_arnoldi_factor, block_lanczos_factor,
       block_arnoldi, block_lanczos, thick_restart_lanczos, thick_restart_arnoldi, sparse_svd_lanczos, SparseSVDWorkspace, sparse_svd_workspace_bytes,
       IncrementalSVDFactor, incremental_svd, svd_add_rows!, svd_add_columns!,
       svd_snapshot, svd_rank, initialize_svd!, reorthogonalize_svd!,
       ChebyshevFilter, FilterWorkspace, polynomial_filter, apply_filter!, filter_response,
       hermitian_spectral_bounds, polynomial_filtered_eigenvalues,
       eigenvalues_in_interval, count_eigenvalues_in_interval,
       weighted_bipartite_matching, bipartite_coloring, weighted_graph_matching,
       degree_sequence, is_structurally_symmetric, pseudo_peripheral_vertex,
       multiple_minimum_degree, column_minimum_degree, sparse_ordering, ordering_metrics,
       compare_orderings, select_sparse_ordering, multilevel_partition, multilevel_nested_dissection

const NativeFloat = Union{Float32,Float64,ComplexF32,ComplexF64}
const RealFloat = Union{Float32,Float64}

# OxiBLAS.inv (high-level API, linalg_api.jl) is its own function, distinct
# from Base.inv; kernels invert scalars with an explicit Base.inv. On a
# number it is Base.inv, like exp/log/sqrt/sin/cos below (the API accepts
# scalars; no other type falls back to Base).
function inv end
inv(x::Number)=Base.inv(x)
# OxiBLAS.exp/log/sqrt/sin/cos (matrix functions, linalg_api.jl) are also
# OxiBLAS's own functions. Kernels throughout the package (svd*.jl included)
# call them on scalars, so they are declared before any file and forward
# Number arguments only to Base. No other type falls back to Base: a matrix
# type without an OxiBLAS method gets a MethodError, never LinearAlgebra.
for f in (:exp,:log,:sqrt,:sin,:cos)
    @eval function $f end
    @eval $f(x::Number)=Base.$f(x)
end
log(b::Number,x::Number)=Base.log(b,x)

# Source layout: one module, files grouped by family under src/<family>/.
# The include order below is the dependency order (types, constants and
# generated methods used at load time); it interleaves families on purpose
# and must be kept as it is when files are added.
#
# Scalars, hardware detection, SIMD and GEMM.
include("utils/numeric_types.jl")
include("utils/hardware.jl")
include("blas/simd.jl")
include("blas/gemm.jl")
include("utils/scheduling.jl")
# BLAS 1-3 and batched products.
include("blas/level1.jl")
include("blas/modified_rotations.jl")
include("blas/level2.jl")
include("blas/level3.jl")
include("blas/triangular_microkernels.jl")
include("lapack/factorizations.jl")
include("blas/batched.jl")
# Packed and banded storage.
include("structured/storage.jl")
include("structured/compact_blas.jl")
# Sparse formats, conversions, direct solvers and preconditioners.
include("sparse/sparse.jl")
include("sparse/sparse_extended.jl")
include("sparse/sparse_formats.jl")
include("sparse/sparse_arrays.jl")
include("sparse/sparse_direct.jl")
include("sparse/sparse_operations.jl")
include("sparse/sparse_amd.jl")
include("sparse/sparse_colamd.jl")
include("sparse/matrix_market.jl")
include("sparse/sparse_qr.jl")
include("sparse/sparse_qr_multifrontal.jl")
include("sparse/sparse_preconditioners.jl")
# Dense eigenvalue and singular value decompositions, dense utilities.
include("lapack/hessenberg.jl")
include("lapack/schur.jl")
include("lapack/hermitian_eigen.jl")
include("lapack/svd.jl")
include("lapack/advanced_factorizations.jl")
include("lapack/bidiagonal.jl")
include("lapack/schur_advanced.jl")
include("lapack/divide_conquer.jl")
include("lapack/svd_advanced.jl")
include("lapack/svd_selected.jl")
include("lapack/qz.jl")
include("lapack/qz_real.jl")
include("lapack/schur_blocks.jl")
include("lapack/orthogonal_decompositions.jl")
include("lapack/dense_utilities.jl")
include("lapack/matrix_functions.jl")
# Iterative solvers and spectral methods.
include("sparse/krylov_extended.jl")
include("spectral/sparse_spectral.jl")
# MRRR, packed/banded factorizations, supernodal Cholesky.
include("lapack/mrrr.jl")
include("structured/compact_factorizations.jl")
include("sparse/multifrontal.jl")
include("sparse/supernodal_cholesky.jl")
# GEMM variants, parallel BLAS 1, multishift QR, block Krylov.
include("blas/gemm_variants.jl")
include("blas/level1_parallel.jl")
include("lapack/schur_multishift.jl")
include("sparse/block_krylov.jl")
# Generic and extended-precision paths.
include("utils/extended_precision.jl")
include("blas/extended_blas.jl")
include("lapack/extended_lapack.jl")
include("lapack/expert_refinement.jl")
# Tensors (dense planner first, then the sparse einsum engine).
include("tensors/tensors.jl")
include("tensors/tensor_planning.jl")
include("tensors/sparse_tensors.jl")
# OxiBLAS.mul layer and sparse Base operations.
include("api/mul.jl")
include("sparse/sparse_base.jl")
include("sparse/sparse_base_ops.jl")
# Autotuning, approximate inverses, stochastic and block spectral methods.
include("utils/autotuning.jl")
include("sparse/ainv.jl")
include("spectral/stochastic.jl")
include("spectral/block_spectral.jl")
include("spectral/restarted_block_arnoldi.jl")
include("spectral/xtrace_optimized.jl")
include("spectral/sparse_svd_iterative.jl")
include("spectral/sparse_svd_golub_kahan.jl")
include("spectral/incremental_svd.jl")
include("spectral/spectral_filter.jl")
include("spectral/interval_spectral.jl")
# Graphs and orderings.
include("graphs/weighted_matching.jl")
include("graphs/sparse_ordering_advanced.jl")
include("graphs/multilevel_partition.jl")
include("graphs/graph_refinement.jl")
include("graphs/graph_matching.jl")
include("graphs/graph_structure.jl")
# Sparse indefinite, multifrontal/supernodal LU and LDLt, out of core.
include("sparse/sparse_indefinite.jl")
include("sparse/multifrontal_lu.jl")
include("sparse/supernodal_lu.jl")
include("sparse/umf_lu.jl")
include("sparse/supernodal_ldlt.jl")
include("sparse/out_of_core.jl")
# High-level LinearAlgebra-style API over the engines above (not exported).
include("api/linalg_api.jl")
include("api/linalg_api_sparse.jl")
include("api/linalg_api_structured.jl")
include("api/linalg_api_solve.jl")
# Precompilation workload (PrecompileTools), last: it calls the API above.
include("precompile.jl")

end
