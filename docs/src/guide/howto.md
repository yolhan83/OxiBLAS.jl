# How do I…?

Task-oriented recipes. Each example runs when the documentation is built.
Every function named here has its full description in the
[API reference](../reference/index.md) (search box, top left).

```@example howto
using OxiBLAS, LinearAlgebra, SparseArrays
nothing # hide
```

The sparse examples use a 2-D Laplacian, symmetric positive definite:

```@example howto
k = 30
T1 = spdiagm(-1 => -ones(k-1), 0 => 2ones(k), 1 => -ones(k-1))
L = kron(sparse(1.0I, k, k), T1) + kron(T1, sparse(1.0I, k, k))   # 900×900
A = CSRMatrix(L)                                                 # OxiBLAS format
b = ones(size(A, 1))
nothing # hide
```

## Solve a linear system

### Dense, direct

```@example howto
M = randn(200, 200) + 200I; c = randn(200)
x = OxiBLAS.ldiv(M, c)              # factorize + solve (LU here)
F = OxiBLAS.lu(M)                   # keep the factorization to reuse it
x2 = F \ c                          # or OxiBLAS.ldiv(F, c)
y = similar(c); OxiBLAS.ldiv!(y, F, c)   # in place, no allocation of the result
norm(M * y - c) / norm(c)
```

The engine-level form is [`lu_factor`](@ref) + [`solve`](@ref); use
[`cholesky_factor`](@ref) for Hermitian positive definite matrices and
[`ldlt_factor`](@ref) / [`bunchkaufman_factor`](@ref) for symmetric
indefinite ones. `OxiBLAS.factorize(M)` chooses from the structure.

### Sparse, direct

```@example howto
x = A \ b                           # multifrontal LU or Cholesky, chosen from the structure
F = OxiBLAS.factorize(A)            # reuse: Cholesky here (symmetric positive definite)
x2 = OxiBLAS.ldiv(F, b)
norm(L * x2 - b) / norm(b)
```

The engines are [`multifrontal_lu`](@ref), [`multifrontal_cholesky`](@ref),
[`sparse_ldlt`](@ref) (indefinite) and [`sparse_qr`](@ref) (least squares).

### Iteratively (Krylov methods)

Every solver returns a named tuple `(; x, converged, iterations, residual,
reason)`; `residual` is the recomputed true residual norm. Common keywords:
`x0` (initial guess), `rtol`, `atol`, `maxiter`, `preconditioner`.

```@example howto
r = cg(A, b; rtol=1e-10)                              # Hermitian positive definite
(r.converged, r.iterations, r.residual)
```

```@example howto
r = pcg(A, b; preconditioner=ic0(A), rtol=1e-10)      # incomplete Cholesky
(r.converged, r.iterations)
```

```@example howto
N = CSRMatrix(L + 0.3 * spdiagm(1 => ones(size(L,1)-1)))  # nonsymmetric
r = gmres(N, b; preconditioner=ilu0(N), restart=30, rtol=1e-10)
(r.converged, r.iterations)
```

| Solver | Matrix |
|---|---|
| [`cg`](@ref), [`pcg`](@ref) | Hermitian positive definite |
| [`minres`](@ref) | Hermitian, possibly indefinite |
| [`gmres`](@ref), [`fgmres`](@ref) (flexible: variable preconditioner) | general |
| [`bicgstab`](@ref), [`tfqmr`](@ref), [`qmr`](@ref), [`idrs`](@ref) | general, short recurrences (less memory than GMRES) |
| [`block_cg`](@ref), [`block_gmres`](@ref) | several right-hand sides at once |

Preconditioners: [`ilu0`](@ref), [`ilut`](@ref), [`ic0`](@ref),
[`ict`](@ref), [`amg`](@ref) (algebraic multigrid), [`spai`](@ref),
[`ainv`](@ref), [`relaxation_preconditioner`](@ref) (Gauss–Seidel, SOR,
SSOR), [`polynomial_preconditioner`](@ref),
[`additive_schwarz`](@ref), a vector (Jacobi: the diagonal), or a sparse
direct factor.

```@example howto
r = pcg(A, b; preconditioner=amg(A), rtol=1e-10)      # multigrid: few iterations
(r.converged, r.iterations)
```

```@example howto
B = randn(size(A, 1), 4)                             # four right-hand sides
r = block_cg(A, B; rtol=1e-8)
(r.converged, r.iterations, maximum(r.residuals))
```

### Least squares

```@example howto
Tall = randn(300, 40); d = randn(300)
x = OxiBLAS.qr(Tall) \ d                     # or OxiBLAS.ldiv(Tall, d)
norm(Tall' * (Tall * x - d))                 # normal equations satisfied
```

Rank-deficient problems: [`qr_pivoted`](@ref) or [`complete_orthogonal`](@ref)
(minimum-norm solution). Sparse: [`sparse_qr`](@ref) then [`solve`](@ref).

### Check a solution

```@example howto
x = OxiBLAS.ldiv(M, c)
backward_error(M, x, c).error                # componentwise backward error
```

[`refine_solve`](@ref) adds iterative refinement with condition estimates;
[`expert_solve`](@ref) adds equilibration.

## Compute an SVD

### Full SVD

```@example howto
X = randn(120, 80)
S = OxiBLAS.svd(X)                     # S.U, S.S, S.Vt (and S.V)
norm(S.U * Diagonal(S.S) * S.Vt - X) / norm(X)
```

```@example howto
F = svd_factor(X; algorithm=:golub_kahan)   # engine: :jacobi (default, most accurate),
F.S[1:3]                                    # :golub_kahan, :divide_conquer (fastest)
```

### Singular values only

```@example howto
OxiBLAS.svdvals(X)[1:3]                # or singular_values(X)
```

### A few singular triplets

```@example howto
T5 = svd_truncated(X, 5)               # the 5 largest, without the full SVD
R5 = svd_randomized(X, 5)              # randomized range finder: fastest for low ranks
I3 = svd_selected(X; indices=1:3)      # chosen indices (or interval=(lo, hi))
(T5.S, R5.S[1:2], I3.S)
```

### Sparse matrix

```@example howto
s = sparse_svd_lanczos(A, 4)           # Golub–Kahan–Lanczos with thick restart
(s.converged, s.S)
```

[`sparse_svd`](@ref) is the randomized variant. For data that arrives by rows
or columns, [`incremental_svd`](@ref) is updated by
[`svd_add_rows!`](@ref) / [`svd_add_columns!`](@ref).

## Compute eigenvalues

### Dense

```@example howto
H = Symmetric(randn(100, 100))
E = OxiBLAS.eigen(H)                   # Hermitian: real values, orthonormal vectors
G = OxiBLAS.eigen(randn(100, 100))     # general: complex values
(E.values[1:2], G.values[1:2])
```

Engines: [`eigen_hermitian`](@ref) (with `algorithm`), [`eigen_hermitian_dc`](@ref)
(divide and conquer), [`eigen_hermitian_mrrr`](@ref) (subset by `indices` or
`interval`), [`eigen_general`](@ref), [`schur_factor`](@ref). Generalized
problems: `OxiBLAS.eigen(A, B)`, [`eigen_generalized`](@ref) (Hermitian
definite), [`eigen_pencil`](@ref) (QZ).

### A few eigenvalues of a large (sparse) matrix

```@example howto
r = lobpcg(A; nev=3, which=:smallest, preconditioner=ic0(A))   # Hermitian; preconditioned
(r.converged, r.iterations, r.values)
```

```@example howto
r = lanczos(A; nev=3, which=:largest, maxiter=300)   # Hermitian Lanczos; maxiter = subspace size
(r.converged, r.values)
```

```@example howto
r = shift_invert(A; sigma=1.0, nev=3)           # eigenvalues closest to sigma
real.(r.values)
```

Check `r.converged`: the iteration budgets (`maxiter`, `maxrestarts`) are
finite, and an unconverged result is returned, not raised.
Non-Hermitian: [`arnoldi`](@ref), [`iram`](@ref) (implicit restarts),
[`thick_restart_arnoldi`](@ref). All eigenvalues in an interval:
[`eigenvalues_in_interval`](@ref) (Chebyshev filtering).

## Other matrix computations

```@example howto
P = Symmetric(randn(50, 50)) + 60I
(OxiBLAS.exp(P) ≈ exp(Matrix(P)), OxiBLAS.sqrt(P)^2 ≈ P, matrix_power(P, 3) ≈ Matrix(P)^3)
```

```@example howto
D = randn(30, 10) * randn(10, 20)              # rank 10
(numerical_rank(D), size(null_space(D)), condition_number(randn(5, 5)) > 1)
```

Also: `OxiBLAS.det`, `logabsdet`, `inv`, `pinv` ([`pseudoinverse`](@ref)),
`OxiBLAS.log`, `sin`, `cos`, `sylvester`, `lyap`, [`frechet_exp`](@ref),
[`kron_product`](@ref), [`kronecker_action`](@ref).

## Multiply matrices

```@example howto
P1 = randn(300, 200); P2 = randn(200, 100); C = zeros(300, 100)
w = GemmWorkspace(Float64, 300, 100, 200)      # reused buffers: no allocation per call
gemm!(C, P1, P2, 1.0, 0.0; workspace=w)        # C ← 1·P1·P2 + 0·C
OxiBLAS.mul!(C, P1, P2)                        # high-level: also structured and sparse operands
C ≈ P1 * P2
```

```@example howto
y = zeros(size(A, 1)); OxiBLAS.mul!(y, A, b)   # sparse × dense (A*b)
z = OxiBLAS.mul(A, A)                          # sparse × sparse: sparse result
(y ≈ L * b, nnz(SparseMatrixCSC(z)))
```

## Work with sparse matrices

```@example howto
As = sparse_permute(A, sortperm(rand(size(A, 1))))   # the Laplacian, randomly renumbered
p = reverse_cuthill_mckee(As)                        # bandwidth-reducing ordering
(bandwidth_profile(As).bandwidth, bandwidth_profile(sparse_permute(As, p)).bandwidth)
```

```@example howto
path = tempname() * ".mtx"
write_matrix_market(path, A; symmetry=:symmetric)
A2 = read_matrix_market(path)                  # CSRMatrix by default (format=:csc, :coo, :dense)
A2 ≈ A
```

Formats: [`CSRMatrix`](@ref), [`CSCMatrix`](@ref), [`COOMatrix`](@ref),
[`DIAMatrix`](@ref), [`ELLMatrix`](@ref), [`BSRMatrix`](@ref),
[`HYBMatrix`](@ref), [`SELLMatrix`](@ref); each converts from a
`SparseMatrixCSC` (`CSRMatrix(S)`) and back (`SparseMatrixCSC(A)`). Orderings:
[`amd_ordering`](@ref), [`nested_dissection`](@ref),
[`select_sparse_ordering`](@ref) (compares several).

## Work with tensors

```@example howto
X3 = randn(8, 9, 10); Y2 = randn(10, 7)
Z = einsum("abc,cd->abd", X3, Y2)              # Einstein summation
Pl = plan_einsum("abc,cd->abd", X3, Y2)        # plan once, run many times
Z2 = similar(Z); einsum!(Z2, Pl, X3, Y2)
(size(Z), Z ≈ Z2, contract(X3, Y2, (3,), (1,)) ≈ Z)
```

```@example howto
Tk = hosvd(randn(6, 7, 8); ranks=(3, 3, 3))     # Tucker decomposition
size(reconstruct(Tk))
```

Also [`tensor_train`](@ref), [`cp_als`](@ref), sparse tensors
([`SparseTensor`](@ref)) and [`batched_matmul!`](@ref).
