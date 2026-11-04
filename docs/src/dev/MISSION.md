# Mission: Reimplement OxiBLAS in Native Julia — Full Low-Level Performance

You are an expert in Julia, LLVM, CPU architecture, numerical linear algebra, SIMD programming, cache optimization, BLAS, LAPACK, and high-performance computing.

Your mission is to **port the entire OxiBLAS Rust project to Julia**, preserving its algorithms, features, numerical stability, architecture-specific optimizations, and high-performance design.

Original repository: https://github.com/cool-japan/oxiblas

New package: `OxiBLAS.jl`

## 1. Philosophy

Julia is not merely a high-level numerical language. It provides low-level capabilities comparable to systems programming languages.

**Exploit the entire Julia compilation stack, from high-level multiple dispatch down to LLVM IR and machine instructions.**

Do not artificially restrict the implementation to ordinary Julia loops.

The goal is not to demonstrate that simple Julia code is reasonably fast. The goal is to build a genuine high-performance BLAS/LAPACK implementation written in Julia.

Performance, correctness, portability, and compilation efficiency are fundamental requirements.

## 2. Allowed low-level techniques

You are explicitly encouraged to use:

- Native Julia loops, functions, and multiple dispatch.
- `@inbounds`, `@simd`, `@inline`, and `@noinline`.
- `@fastmath` where its numerical tradeoffs are justified.
- `muladd` for fused multiply-add operations.
- `Ptr{T}`, `pointer`, `unsafe_load`, and `unsafe_store!`.
- `GC.@preserve` for safe pointer lifetimes.
- `VecElement{T}` and SIMD-compatible tuples.
- `Base.llvmcall` / `Core.Intrinsics.llvmcall`.
- LLVM vector instructions and intrinsics.
- Architecture-specific LLVM intrinsics.
- LLVM inline assembly where justified and supported.
- Architecture-specific kernel specialization.
- Manual memory alignment and buffer management.
- Explicit matrix packing.
- Cache blocking and register blocking.
- Manual loop unrolling.
- Native Julia multithreading.
- Preallocated thread-local workspaces.
- Runtime hardware feature detection.
- Compile-time specialization and generated functions when genuinely beneficial.

When porting Rust SIMD intrinsics, investigate their LLVM equivalents.

For instance, reproduce the behavior of Rust's AVX2, AVX-512, and NEON intrinsics using Julia and LLVM.

Do not assume ordinary Julia loops are optimal. Inspect their generated instructions.

Conversely, do not assume handwritten LLVM IR is faster. Always compare against compiler-generated code.

**The fastest correct implementation wins, regardless of which Julia language level it uses.**

## 3. External library restrictions

All numerical computations must be implemented in Julia.

Forbidden for numerical execution:

- OpenBLAS
- MKL
- Apple Accelerate
- BLIS
- Rust OxiBLAS binaries
- External C/C++/Fortran numerical kernels
- `LinearAlgebra.BLAS` and `LinearAlgebra.LAPACK` as implementation backends

Do not introduce dependencies on SIMD.jl or LoopVectorization.jl. Their techniques may be studied, but required SIMD kernels must be implemented within the package.

`ccall` is permitted for necessary operating-system services such as hardware queries, memory management, or other platform facilities.

`ccall` must not be used to delegate numerical calculations to external libraries.

LLVM intrinsics and architecture-specific LLVM code are explicitly permitted. They do not count as external numerical dependencies.

Standard Julia libraries and packages may be used for testing, benchmarking, and verification.

## 4. Study OxiBLAS before implementing it

Clone the original repository and inspect its complete source tree.

Focus on the real implementations, not just the README.

Study:

- `oxiblas-core`: SIMD abstractions, scalar traits, memory allocation, hardware detection.
- `oxiblas-matrix`: matrix representations, views, strides, alignment.
- `oxiblas-blas`: BLAS levels 1, 2, and 3.
- `oxiblas-lapack`: dense decompositions and solvers.
- `oxiblas-sparse`: sparse data structures and algorithms.
- Architecture-specific GEMM microkernels.
- Memory packing strategies.
- Parallel scheduling.
- Cache-aware blocking.
- Automatic kernel selection and tuning.

Map each Rust component to an idiomatic Julia implementation.

Do not perform a superficial line-by-line syntax translation.

Preserve useful algorithmic structure while adapting the memory model and optimization techniques to Julia.

Respect the upstream Apache-2.0 license and copyright requirements.

## 5. GEMM: highest priority

Implement:

\[
C \leftarrow \alpha\,op(A)\,op(B)+\beta C
\]

Support Float32, Float64, ComplexF32, and ComplexF64.

Support normal, transposed, and conjugate-transposed operands.

Develop and benchmark multiple GEMM implementations:

1. Simple scalar reference kernel.
2. Cache-blocked kernel.
3. Packed GEMM.
4. Register-blocked microkernel.
5. Explicit SIMD microkernel.
6. Architecture-specific microkernels.
7. Multithreaded GEMM.

Study the original OxiBLAS blocking strategy, including MC, NC, KC, MR, and NR.

Port its actual optimizations where applicable.

Investigate:

- Contiguous memory access.
- Packed A and B panels.
- Cache-line alignment.
- Register reuse.
- SIMD vector width.
- FMA throughput.
- Loop unrolling.
- Software prefetching.
- Cache hierarchy.
- Memory bandwidth.
- Small-matrix overhead.
- Tall, skinny, and rectangular matrices.

Implement correct tail handling when dimensions are not multiples of the microkernel dimensions.

Ensure that α and β are applied correctly, including special cases where the output must not be read.

Do not allocate new packing buffers for every microkernel invocation.

Provide reusable workspaces whenever appropriate.

Never sacrifice memory safety, including pointer lifetime, overlapping-array behavior, or thread safety, to obtain artificial benchmark improvements.

## 6. SIMD and hardware-specific kernels

Support at least:

- AArch64 NEON.
- x86_64 AVX2 and FMA.
- x86_64 AVX-512 where available.
- Generic scalar fallback.

Primary development hardware: Apple Silicon M5 Max.

Secondary targets: Linux and Windows x86_64, and other AArch64 processors.

The implementation must not assume that all machines support the same instruction sets.

Use hardware feature detection and dispatch to appropriate kernels.

Avoid executing unsupported instructions.

Study each Rust SIMD implementation and reproduce its intended operations through Julia.

Use `VecElement` and LLVM vector types where appropriate.

Use `llvmcall` for operations that LLVM does not reliably generate from ordinary Julia code.

Consider architecture-specific LLVM inline assembly only when it delivers a verified improvement over intrinsics or compiler-generated code.

Inspect generated LLVM IR and assembly using:

```julia
@code_llvm
@code_native
```

Check actual SIMD instructions, FMA usage, loads, stores, register pressure, and unwanted function calls.

Do not confuse vectorized source code with vectorized machine code.

Avoid excessive kernel specialization that creates unmanageable compilation times or code-size explosions.

## 7. Memory and data representation

Use Julia's column-major storage convention.

Prefer standard Julia arrays:

```julia
Matrix{T}
Vector{T}
StridedMatrix{T}
StridedVector{T}
```

Support compatible views and wrappers such as transpose and adjoint.

Provide generic fallback implementations for layouts unsupported by optimized kernels.

Use raw pointers where beneficial in performance-critical kernels.

When using pointer arithmetic:

- Preserve object lifetimes correctly.
- Respect bounds and alignment.
- Avoid invalid assumptions about strides.
- Handle aliasing explicitly.
- Avoid out-of-bounds vector loads.
- Ensure that thread-local buffers never race.

Do not create custom matrix container types without a compelling performance or functionality reason.

## 8. Parallelism

Replace Rayon with native Julia threading.

Investigate:

- Static partitioning.
- Task scheduling overhead.
- Output-tile parallelism.
- Thread-local scratch buffers.
- Packing reuse.
- False sharing.
- NUMA considerations on applicable platforms.
- Single-thread versus multithread thresholds.

Do not spawn a task for every microkernel.

Ensure that large matrix operations scale across cores while small operations avoid parallel overhead.

Expose configuration for controlling parallel execution.

## 9. Complete BLAS and LAPACK port

Once GEMM is correct and optimized, progressively port the remaining OxiBLAS functionality.

BLAS Level 1:
- Vector operations, reductions, norms, rotations.

BLAS Level 2:
- Matrix-vector multiplication, triangular solves, symmetric and Hermitian operations, banded and packed variants.

BLAS Level 3:
- GEMM, SYMM, HEMM, TRMM, TRSM, SYRK, SYR2K, HERK, HER2K.

LAPACK:
- LU with pivoting.
- Cholesky.
- QR.
- Hessenberg reduction.
- Schur decomposition.
- Eigenvalue algorithms.
- SVD.
- Least-squares solvers.
- Blocked, recursive, and parallel variants.

Also port the original project's relevant sparse, tensor, batched, mixed-precision, and advanced solver capabilities.

Use the original implementation as the feature checklist.

Do not silently omit advanced algorithms.

Favor optimized blocked algorithms built on the package's own native BLAS kernels.

## 10. Julia integration

Implement a Julia-native public interface compatible in behavior with common `LinearAlgebra` operations.

For example:

```julia
OxiBLAS.gemm!(C, A, B, α, β)
OxiBLAS.gemv!(y, A, x, α, β)
```

Support standard `mul!` semantics.

Initially, avoid pirating or overriding existing `LinearAlgebra` methods.

Investigate optional integration with Julia's standard linear algebra interfaces after the independent package is correct and performant.

Do not assume a Julia-native library can be directly registered as a BLAS ABI backend without implementing the necessary ABI compatibility layer.

Treat integration with `libblastrampoline` as a separate optional phase.

## 11. Testing and verification

Write comprehensive tests for every implemented operation.

Use `LinearAlgebra` and the original Rust implementation as references where appropriate.

Test:

- Every supported scalar type.
- Rectangular and square matrices.
- Transpose and adjoint.
- Non-contiguous views.
- Zero-size and small matrices.
- Non-multiple SIMD dimensions.
- Nontrivial α and β.
- Singular or ill-conditioned inputs where applicable.
- Single-thread and multithread operation.
- Pointer alignment and aliasing cases.

Verify numerical accuracy with appropriate absolute and relative tolerances.

Test numerical invariants and residuals for decompositions.

Do not enable unsafe floating-point transformations indiscriminately.

Distinguish performance-oriented floating-point reassociation from transformations that compromise numerical correctness.

## 12. Performance benchmarks

Use BenchmarkTools.jl.

Compare against OpenBLAS and Apple Accelerate where available, and the original Rust OxiBLAS when executable.

Benchmark:

- Sizes 16 through 4096 and larger when practical.
- Rectangular matrices.
- Float32, Float64, and complex types.
- Single-thread and multithread execution.
- Small repeated operations.
- Large throughput-oriented operations.

Record:

- Execution time.
- GFLOP/s.
- Allocations.
- Memory traffic when measurable.
- Thread scalability.
- Compilation time separately from execution time.

Compute FLOP counts independently and verify benchmark correctness.

Ensure that all implementations use equivalent workloads, warm-up, numerical precision, and thread configurations.

Prevent CPU BLAS thread oversubscription.

Run benchmarks after every major kernel optimization, and preserve performance regression results.

Do not claim speedups without measurements.

## 13. Development priorities

Implement in this order:

**Phase 1 — GEMM correctness**
- Julia package structure.
- Reference GEMM.
- Complete GEMM tests.

**Phase 2 — GEMM optimization**
- Blocking.
- Packing.
- SIMD microkernels.
- LLVM intrinsics.
- Architecture dispatch.
- Parallel execution.
- Benchmarks and assembly inspection.

**Phase 3 — BLAS**
- Complete Level 1.
- Complete Level 2.
- Complete Level 3.

**Phase 4 — LAPACK**
- Dense factorizations and solvers.
- Advanced decompositions.
- Recursive and parallel algorithms.

**Phase 5 — Extended functionality**
- Sparse algorithms.
- Batched operations.
- Tensor operations.
- Mixed precision.
- Autotuning.
- Additional features found in OxiBLAS.

**Phase 6 — Integration**
- Public Julia API.
- Documentation.
- Optional standard-library integration.
- Optional BLAS ABI adapter.

Maintain a checklist mapping upstream OxiBLAS functionality to the Julia implementation.

Distinguish implemented, tested, optimized, and unfinished features.

## 14. Execution instructions

Do not stop after proposing a design.

Actually create the package, write source code, run tests, inspect generated assembly, benchmark, and improve the implementation.

Do not substitute an unrelated simple GEMM for the upstream architecture.

Do not stop merely because a microkernel requires unsafe Julia or LLVM-level programming.

Use the lower-level facilities when they are appropriate.

If a low-level optimization cannot be implemented portably, provide an architecture-specific version and a correct fallback.

Keep the repository in a compiling and tested state after every development phase.

If the complete port cannot be finished in one execution, prioritize working code and precise progress documentation.

**Final objective: create a genuine, production-quality, fully native Julia implementation of OxiBLAS that exploits Julia's low-level capabilities as aggressively as Rust exploits its own.**