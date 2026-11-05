# Tensors

## [`einsum`](@ref) and contractions

`einsum(spec, A...)` evaluates an Einstein notation on dense arrays,
[`SparseTensor`](@ref)s or sparse matrices. Pairwise contractions go through the
native GEMM (TTGT).

```@example tens
using OxiBLAS
A = randn(16, 16, 16); B = randn(16, 16)
C = einsum("abc,cd->abd", A, B)            # allocating
P = plan_einsum("abc,cd->abd", A, B)       # reusable plan
D = similar(C); ws = EinsumWorkspace(Float64)
einsum!(D, P, A, B; workspace=ws)          # in place, reused workspace
D ≈ C
```

- **Plans.** `plan_einsum(spec, A...; optimize)` chooses the contraction
  order: `:greedy` by default, `:beam`, `:optimal`, or `:none` for the direct
  contraction. A plan is immutable and thread-safe.
- **In-place evaluation.** `einsum!(C, plan, A...; α, β, workspace)` writes
  into `C`. Temporaries come from the [`EinsumWorkspace`](@ref).
- **Sliced plans.** [`plan_einsum_sliced`](@ref) bounds the memory by slicing the
  computation; [`einsum_layout_report`](@ref) describes the chosen memory layouts.
- **Explicit contractions.** `contract(A, B, axesA, axesB)` and [`contract!`](@ref)
  contract the given axes (explicit conjugations).
- **Specialized products.** [`outer_product`](@ref), `mode_product(A, U, mode)`,
  [`batched_matmul`](@ref) and [`batched_matmul!`](@ref).

## Sparse tensors

`SparseTensor(dims, coordinates, values)` stores sorted linear indices
(column-major order).

```@example tens
X = SparseTensor((30, 30, 30), [(rand(1:30), rand(1:30), rand(1:30)) for _ in 1:500], randn(500))
U = randn(30, 4)
Y = einsum("abc,cz->abz", X, U)            # sparse × dense
size(Y)
```

- Sparse × sparse products use a column-wise SpGEMM on the unfoldings.
- Matrix operands (CSR, CSC, `SparseMatrixCSC`, 2-D [`SparseTensor`](@ref)) stay in
  native CSC.
- [`einsum!`](@ref) into a sparse destination receives the CSC result directly.
- The result of a sparse computation is a [`SparseTensor`](@ref).

## Decompositions

| Function | Result |
|---|---|
| `hosvd(A; ranks)` | Tucker ([`TuckerFactor`](@ref)) |
| `tensor_train(A; maxrank, rtol)` | tensor train ([`TensorTrainFactor`](@ref)) |
| `cp_als(A, rank)` | CP by alternating least squares ([`CPFactor`](@ref)) |
| `reconstruct(F)` | reconstructed tensor |

## Performance

The high-level suite compares [`einsum`](@ref) with TensorOperations
([Performance](performance.md)).
