# Case definitions shared by the four-column report; no work at include time.
const FULL_GROUPS=("BLAS LEVEL 1","BLAS LEVEL 2","BLAS LEVEL 3","LAPACK","SPARSE","SPECTRAL / SOLVERS / GRAPHS","TENSOR","EXTENDED PRECISION")
const FULL_OPERATIONS=(
    ("BLAS LEVEL 1",(:axpy,:scal,:copy,:swap,:dotu,:dotc,:dot_kahan,:dot_pairwise,:nrm2,:asum,:iamax,:iamin,:dot_extended)),
    ("BLAS LEVEL 2",(:gemv,:ger,:gerc,:symv,:hemv,:trmv,:trsv,:gbmv,:sbmv,:hbmv,:spmv,:hpmv,:tbmv,:tbsv,:tpmv,:tpsv)),
    ("BLAS LEVEL 3",(:gemm,:gemm_batched,:gemm_strided_batched,:symm,:hemm,:trmm,:trsm,:syrk,:syr2k,:herk,:her2k,:gemmt,:gemm_cache_oblivious,:gemm_strassen,:gemm_winograd)),
    ("LAPACK",(:lu,:lu_complete,:lu_rook,:cholesky,:ldl,:qr,:qr_pivoted,:lq,:ql,:rq,:cod,:solve,:mixed_solve,:expert_solve,:inverse,:pseudoinverse,:rank,:condition,:determinant,:band_lu,:band_cholesky,:packed_cholesky,:aasen)),
    ("SPARSE",(:sparse_mul,:sparse_mul_csc,:sparse_mul_csc_transpose,:sparse_matmul,:sparse_lu,:sparse_cholesky,:sparse_qr,:ilu0,:ilut,:ic0,:ict,:sparse_ldl,:multifrontal_lu,:multifrontal_cholesky,:supernodal_cholesky,:sparse_lu_blocked,:sparse_lu_static,:ainv)),
    ("SPECTRAL / SOLVERS / GRAPHS",(:hessenberg,:evd,:evd_dc,:evd_mrrr,:eigen_generalized,:schur,:schur_multishift,:qz,:eigen_pencil,:svd,:svd_parallel,:svd_golub_kahan,:svd_dc,:svd_selected,:svd_randomized,:cg,:gmres,:bicgstab,:matrix_exp,:matrix_log,:matrix_sqrt,:matrix_sign,:minimum_degree,:rcm,:nested_dissection,:connected_components)),
    ("TENSOR",(:einsum,:einsum_inplace,:contract,:batched_matmul,:hosvd,:tensor_train,:cp_als,:kron,:khatri_rao,:einsum_beam,:einsum_optimal,:einsum_sliced,:einsum_sliced_inplace,:plan_einsum_beam,:plan_einsum_optimal,:plan_einsum_sliced)),
    ("EXTENDED PRECISION",(:gemm,:dot_extended,:lu,:cholesky,:qr,:solve,:mixed_solve,:einsum)),
    ("BLAS LEVEL 1",(:rot,:rotg,:rotm,:rotmg,:dsdot,:sdsdot,:axpy_parallel,:scal_parallel,:dot_parallel,:nrm2_parallel,:asum_parallel)),
    ("BLAS LEVEL 2",(:syr,:syr2,:her,:her2,:spr,:spr2,:hpr,:hpr2)),
    ("LAPACK",(:solve_transposed,:refine_solve,:expert_cholesky,:expert_symmetric,:reciprocal_condition,:inverse_norm_estimate,:equilibrate,:backward_error,:logabsdet,:matrix_norm,:null_space,:left_null_space,:column_space,:row_space,:least_squares,:qr_q,:apply_q,:bidiagonal,:tridiagonal,:potrs,:potri,:qr_tall,:qr_wide,:lu_tall)),
    ("SPARSE",(:sparse_transpose,:sparse_add,:sparse_permute,:sparse_diagonal,:sparse_norm,:dia,:ell,:bsr,:bsc,:hyb,:sell,:relaxation_preconditioner,:polynomial_preconditioner,:spai,:additive_schwarz,:amg)),
    ("SPECTRAL / SOLVERS / GRAPHS",(:pcg,:minres,:tfqmr,:fgmres,:qmr,:idrs,:block_cg,:block_gmres,:matrix_power,:matrix_sin,:matrix_cos,:frechet_exp,:frechet_log,:frechet_sqrt)),
    ("SPECTRAL / SOLVERS / GRAPHS",(:schur_reorder,:qz_reorder,:trevc_right,:trevc_left,:trsna_s,:trsna_sep,:schur_condition,:qz_condition,:generalized_sylvester,:schur_multishift_workspace,:schur_bulge_train,:qz_complex)),
    ("SPECTRAL / SOLVERS / GRAPHS",(:svd_selected_values,:svd_selected_middle,:svd_selected_smallest,:svd_selected_interval,:svd_selected_tall,:svd_selected_wide,:svd_selected_quarter,:svd_selected_half,:svd_selected_gram,:svd_truncated)),
    ("SPECTRAL / SOLVERS / GRAPHS",(:arnoldi_restart,:arnoldi_restart_selective,:arnoldi_restart_fullspace,:xtrace_optimized,:xtrace_loo,:xtrace_resphere,:xtrace_rank_deficient)),
    ("SPARSE",(:sparse_svd_gk,:sparse_svd_gk_reuse,:sparse_svd_dilation)),
    ("BLAS LEVEL 3",(:trsm_right_leaf,:trsm_left_leaf,:trmm_right_leaf,:trmm_left_leaf,:trsm_reuse,:trmm_reuse)),
    ("TENSOR",(:outer_product,:mode_product,:kron_sum,:kronecker_action,:contract_inplace,:batched_matmul_inplace)))

full_relative(a,b)=LA.norm(a-b)/max(LA.norm(b),eps(real(eltype(b))))
function full_input(::Type{T},n) where T
    R=real(T)
    value(i,j,seed)=convert(R,mod(17*(i-1)+13*(j-1)+seed,31)-15)/convert(R,32n)
    A=T[i==j ? convert(R,4)+convert(R,mod(i-1,7))/16 : value(i,j,1) for i in 1:n,j in 1:n]
    B=T[value(i,j,3) for i in 1:n,j in 1:n]
    if T<:Complex
        for j in 1:n,i in 1:n
            A[i,j]+=complex(zero(R),value(i,j,7));B[i,j]+=complex(zero(R),value(i,j,9))
        end
    end
    if !(R in (Float16,Float32,Float64))
        A[1,1]+=ldexp(one(R),-max(60,precision(R)÷2))
    end
    H=zeros(T,n,n)
    for i in 1:n
        H[i,i]=convert(T,4)
        i<n && (H[i,i+1]=H[i+1,i]=convert(T,1)/8)
    end
    x=T[convert(R,mod(17*(i-1)+3,31)-15)/16 for i in 1:n]
    (;A,B,H,x)
end
full_symbol(T,op)=op=="nrm2" && T<:Complex ? (T===ComplexF32 ? "scnrm2_" : "dznrm2_") : op=="asum" && T<:Complex ? (T===ComplexF32 ? "scasum_" : "dzasum_") : string(T===Float32 ? "s" : T===Float64 ? "d" : T===ComplexF32 ? "c" : "z",op,"_")
# Benchmark comparisons are paired only with an explicit equivalent adapter.
# A missing adapter remains visible as N/A, rather than a different algorithm
# being silently substituted (notably sparse versus dense factorization).
# One method covers every group; specializing it on T re-inferred all of it
# for each type (about 50 s per complex type). The returned closures capture
# concrete values, and full_sample is the function barrier for timing.
function full_case(operation,@nospecialize(T),n,@nospecialize(data);reference=false)
    operation in (:svd_selected,:svd_selected_values,:svd_selected_middle,:svd_selected_smallest,:svd_selected_interval,:svd_selected_tall,:svd_selected_wide,:svd_selected_quarter,:svd_selected_half,:svd_selected_gram,:svd_truncated) && return full_selected_svd_case(operation,T,n,data;reference)
    operation in (:arnoldi_restart,:arnoldi_restart_selective,:arnoldi_restart_fullspace,:xtrace_optimized,:xtrace_loo,:xtrace_resphere,:xtrace_rank_deficient) && return full_arnoldi_xtrace_case(operation,T,n,data;reference)
    operation in (:sparse_svd_gk,:sparse_svd_gk_reuse,:sparse_svd_dilation,:trsm_right_leaf,:trsm_left_leaf,:trmm_right_leaf,:trmm_left_leaf,:trsm_reuse,:trmm_reuse) && return full_memory_triangular_case(operation,T,n,data;reference)
    reference && T<:OxiBLAS.HalfFloat && return full_half_reference_case(operation,T,n,data)
    (;A,B,H,x)=data; C=zeros(T,n,n); y=zeros(T,n); R=real(T)
    native=T<:OxiBLAS.NativeFloat; prefix(op)=full_symbol(T,op)
    setup=()->nothing; symbols=String[];flops=NaN; note="same inputs; full operation including returned result allocations"
    verify=result->all(isfinite,result)
    expect(result,expected)=full_relative(result,expected)<=500max(n,1)*eps(R)
    residual(result,matrix=A)=expect(matrix*result,x)
    f=nothing
    if operation in (:axpy,:scal,:copy,:swap)
        y=copy(x);z=reverse(x);setup=()->(copyto!(y,x);copyto!(z,reverse(x)))
        if operation==:axpy
            f=reference ? ()->LA.BLAS.axpy!(one(T),x,y) : ()->axpy!(y,x)
            verify=_ ->expect(y,2x);symbols=[prefix("axpy")];flops=(T<:Complex ? 8 : 2)*n
        elseif operation==:scal
            f=reference ? ()->LA.BLAS.scal!(T(2),y) : ()->scal!(y,T(2))
            verify=_ ->expect(y,2x);symbols=[prefix("scal")];flops=n
        elseif operation==:copy
            f=reference ? ()->LA.BLAS.blascopy!(n,z,1,y,1) : ()->copy_vector!(y,z)
            verify=_ -> y==reverse(x);symbols=[prefix("copy")]
        else
            f=reference ? nothing : ()->swap!(y,z)
            verify=_ ->y==reverse(x) && z==x
        end
    elseif operation in (:dotu,:dotc,:dot_kahan,:dot_pairwise,:dot_extended,:nrm2,:asum,:iamax,:iamin)
        expected=operation in (:nrm2,:asum,:iamax,:iamin) ? nothing : sum(x.*x)
        if operation==:nrm2
            f=reference ? ()->LA.BLAS.nrm2(x) : ()->nrm2(x);expected=sqrt(sum(abs2,x));symbols=[prefix("nrm2")]
        elseif operation==:asum
            f=reference ? ()->LA.BLAS.asum(x) : ()->asum(x);expected=sum(t->abs(real(t))+abs(imag(t)),x);symbols=[prefix("asum")]
        elseif operation in (:iamax,:iamin)
            g=operation==:iamax ? iamax : iamin;f=reference ? nothing : ()->g(x)
            expected=operation==:iamax ? findmax(abs.(x))[2] : findmin(abs.(x))[2]
        else
            g=operation==:dotu ? dotu : operation==:dotc ? dotc : operation==:dot_kahan ? dot_kahan : operation==:dot_pairwise ? dot_pairwise : dot_extended
            # Compensated/extended dots are different arithmetic contracts.
            f=reference ? (operation==:dotc ? ()->LA.dot(x,x) : operation==:dotu ? (T<:Complex ? ()->LA.BLAS.dotu(x,x) : ()->LA.dot(x,x)) : nothing) : ()->g(x,x)
            expected=operation==:dotc ? sum(conj.(x).*x) : sum(x.*x)
            symbols=[prefix(T<:Complex ? (operation==:dotu ? "dotu" : "dotc") : "dot")];flops=(T<:Complex ? 8 : 2)*n
        end
        verify=result->abs(result-expected)<=500max(n,1)*eps(R)*max(one(R),abs(expected))
    elseif operation in (:gemv,:symv,:hemv,:trmv,:trsv,:ger,:gerc,:gbmv,:sbmv,:hbmv,:spmv,:hpmv,:tbmv,:tbsv,:tpmv,:tpsv)
        M=operation in (:symv,:hemv,:sbmv,:hbmv,:spmv,:hpmv) ? H : A
        if operation==:gemv
            f=reference ? ()->LA.BLAS.gemv!('N',one(T),A,x,zero(T),y) : ()->gemv!(y,A,x)
            verify=_ ->expect(y,A*x);symbols=[prefix("gemv")];flops=(T<:Complex ? 8 : 2)*n*n
        elseif operation in (:ger,:gerc)
            conjugate=operation==:gerc
            f=reference ? (conjugate ? ()->LA.BLAS.ger!(one(T),x,x,C) : ()->LA.BLAS.geru!(one(T),x,x,C)) : (conjugate ? ()->gerc!(C,x,x) : ()->ger!(C,x,x))
            setup=()->fill!(C,zero(T));verify=_ ->expect(C,x*(conjugate ? adjoint(x) : transpose(x)));symbols=[prefix(T<:Complex ? (conjugate ? "gerc" : "geru") : "ger")];flops=(T<:Complex ? 8 : 2)*n*n
        elseif operation in (:symv,:hemv)
            herm=operation==:hemv
            f=reference ? (herm && T<:Complex ? ()->LA.BLAS.hemv!('U',one(T),M,x,zero(T),y) : ()->LA.BLAS.symv!('U',one(T),M,x,zero(T),y)) : ()->(herm ? hemv!(y,M,x) : symv!(y,M,x))
            verify=_ ->expect(y,M*x);symbols=[prefix(herm && T<:Complex ? "hemv" : "symv")];flops=(T<:Complex ? 8 : 2)*n*n
        elseif operation in (:trmv,:trsv)
            U=Matrix(LA.UpperTriangular(A));setup=()->copyto!(y,x);solving=operation==:trsv
            f=reference ? (solving ? ()->LA.BLAS.trsv!('U','N','N',U,y) : ()->LA.BLAS.trmv!('U','N','N',U,y)) : (solving ? ()->trsv!(y,U) : ()->trmv!(y,U))
            verify=_ ->solving ? expect(U*y,x) : expect(y,U*x);symbols=[prefix(string(operation))]
        else
            triangular=operation in (:tbmv,:tbsv,:tpmv,:tpsv);packed=operation in (:spmv,:hpmv,:tpmv,:tpsv)
            kind=operation==:gbmv ? :general : triangular ? :triangular : operation in (:hbmv,:hpmv) ? :hermitian : :symmetric
            matrix=triangular ? Matrix(LA.UpperTriangular(H)) : H
            P=packed ? pack_matrix(matrix;kind,uplo=:U) : band_matrix(matrix;kl=triangular ? 0 : 1,ku=1,kind)
            g=getfield(OxiBLAS,Symbol(operation,"!"));solving=operation in (:tbsv,:tpsv)
            f=reference ? (triangular ? ()->getfield(Main,Symbol("reference_",operation,"!"))(y,P,'N') : ()->reference_compact_mv!(operation,y,P,x,one(T),zero(T),'N')) : triangular ? ()->g(y,P) : ()->g(y,P,x)
            symbols=[prefix(string(operation))]
            if reference && ((T<:Complex && operation in (:sbmv,:spmv)) || (T<:Real && operation in (:hbmv,:hpmv)))
                f=nothing
            end
            setup=()->copyto!(y,x);verify=_ ->solving ? expect(matrix*y,x) : expect(y,matrix*x)
        end
    elseif operation in (:gemm,:gemm_cache_oblivious,:gemm_strassen,:gemm_winograd,:symm,:hemm,:trmm,:trsm,:syrk,:syr2k,:herk,:her2k,:gemmt,:gemm_batched,:gemm_strided_batched)
        if operation in (:gemm,:gemm_cache_oblivious,:gemm_strassen,:gemm_winograd)
            w=GemmWorkspace(T,n,n,n);g=getfield(OxiBLAS,Symbol(operation,"!"))
            f=reference ? ()->LA.mul!(C,A,B) : operation==:gemm ? ()->g(C,A,B;workspace=w) : ()->g(C,A,B)
            # O(n²) projection checks every output row; no extra O(n³) oracle.
            verify=_ ->expect(C*ones(T,n),A*(B*ones(T,n)));symbols=[prefix("gemm")];flops=(T<:Complex ? 8 : 2)*Float64(n)^3
            operation!=:gemm && reference && (f=nothing)
            note="preallocated GEMM output; reusable workspace for :gemm; projection correctness guard"
        elseif operation in (:gemm_batched,:gemm_strided_batched)
            batch=4;aa=repeat(reshape(A,n,n,1),1,1,batch);bb=repeat(reshape(B,n,n,1),1,1,batch);cc=similar(aa)
            if operation==:gemm_batched
                as=[view(aa,:,:,i) for i in 1:batch];bs=[view(bb,:,:,i) for i in 1:batch];cs=[view(cc,:,:,i) for i in 1:batch]
                f=reference ? ()->foreach(i->LA.mul!(cs[i],as[i],bs[i]),1:batch) : ()->gemm_batched!(cs,as,bs)
            else
                f=reference ? ()->foreach(i->LA.mul!(view(cc,:,:,i),A,B),1:batch) : ()->gemm_strided_batched!(cc,aa,bb)
            end
            verify=_ ->all(i->expect(cc[:,:,i]*ones(T,n),A*(B*ones(T,n))),1:batch);symbols=[prefix("gemm")];note="four independent products; reference is explicit loop, not batched BLAS API"
        elseif operation in (:symm,:hemm)
            g=operation==:symm ? symm! : hemm!;op=operation==:hemm && T<:Complex ? "hemm" : "symm"
            f=reference ? ()->getfield(LA.BLAS,Symbol(op,"!"))('L','U',one(T),H,B,zero(T),C) : ()->g(C,H,B)
            verify=_ ->expect(C,H*B);symbols=[prefix(op)]
        elseif operation in (:trmm,:trsm)
            U=Matrix(LA.UpperTriangular(A));setup=()->copyto!(C,B);solving=operation==:trsm;g=solving ? trsm! : trmm!
            f=reference ? ()->getfield(LA.BLAS,Symbol(operation,"!"))('L','U','N','N',one(T),U,C) : ()->g(C,U)
            verify=_ ->solving ? expect(U*C,B) : expect(C,U*B);symbols=[prefix(string(operation))]
        else
            herm=operation in (:herk,:her2k);rank2=operation in (:syr2k,:her2k);g=getfield(OxiBLAS,Symbol(operation,"!"))
            expected=operation==:gemmt ? A*B : rank2 ? A*(herm ? B' : transpose(B))+B*(herm ? A' : transpose(A)) : A*(herm ? A' : transpose(A))
            f=reference ? nothing : rank2 || operation==:gemmt ? ()->g(C,A,B) : ()->g(C,A)
            if reference && operation!=:gemmt
                if herm && T<:Complex
                    f=rank2 ? ()->LA.BLAS.her2k!('U','N',one(T),A,B,zero(R),C) : ()->LA.BLAS.herk!('U','N',one(R),A,zero(R),C)
                elseif !herm
                    f=rank2 ? ()->LA.BLAS.syr2k!('U','N',one(T),A,B,zero(T),C) : ()->LA.BLAS.syrk!('U','N',one(T),A,zero(T),C)
                end
            end
            verify=_ ->expect(LA.UpperTriangular(C),LA.UpperTriangular(expected));symbols=[prefix(string(operation))]
        end
    elseif operation in (:lu,:lu_complete,:lu_rook,:cholesky,:ldl,:qr,:qr_pivoted,:lq,:ql,:rq,:cod,:band_lu,:band_cholesky,:packed_cholesky,:aasen)
        M=operation in (:cholesky,:ldl,:band_lu,:band_cholesky,:packed_cholesky,:aasen) ? H : A
        g=operation==:lu ? lu_factor : operation==:lu_complete ? lu_complete : operation==:lu_rook ? lu_rook : operation==:cholesky ? cholesky_factor : operation==:ldl ? ldlt_factor : operation==:qr ? qr_factor : operation==:qr_pivoted ? qr_pivoted : operation==:lq ? lq_factor : operation==:ql ? ql_factor : operation==:rq ? rq_factor : operation==:cod ? complete_orthogonal : operation==:aasen ? aasen_factor : operation==:band_lu ? band_lu : operation==:band_cholesky ? band_cholesky : packed_cholesky
        input=operation in (:band_lu,:band_cholesky) ? band_matrix(M;kl=1,ku=1,kind=operation==:band_cholesky ? :hermitian : :general) : operation==:packed_cholesky ? pack_matrix(M;kind=:hermitian) : M
        if reference
            # lq_factor returns explicit L and Q (OrthogonalFactor), so its
            # reference also materializes Q rather than timing compact gelqf.
            f=operation==:lu ? ()->LA.lu(M) : operation==:cholesky ? ()->LA.cholesky(LA.Hermitian(M)) : operation==:qr ? ()->LA.qr(M) : operation==:qr_pivoted ? ()->LA.qr(M,LA.ColumnNorm()) : operation==:ldl ? ()->LA.bunchkaufman(LA.Hermitian(M)) : operation==:lq ? ()->(F=LA.lq(M);(F.L,Matrix(F.Q))) : nothing
            verify=operation==:lq ? F->expect(F[1]*F[2],M) : F->residual(F\x,M)
            operation==:lq && (note="explicit L and Q on both sides (gelqf + orglq/unglq)")
        else
            f=()->g(input)
            verify=operation in (:lq,:rq) ? F->expect(F.triangular*F.Q,M) : operation==:ql ? F->expect(F.Q*F.L,M) : F->residual(solve(F,x),M)
        end
        symbols=[prefix(operation==:lu ? "getrf" : operation==:cholesky ? "potrf" : operation==:ldl ? (T<:Complex ? "hetrf" : "sytrf") : operation==:qr_pivoted ? "geqp3" : operation==:lq ? "gelqf" : "geqrf")]
    elseif operation in (:solve,:mixed_solve,:expert_solve,:inverse,:pseudoinverse,:rank,:condition,:determinant)
        if operation==:solve
            F=reference ? LA.lu(A) : lu_factor(A);f=reference ? ()->F\x : ()->solve(F,x);verify=residual;symbols=[prefix("getrs")]
        elseif operation in (:mixed_solve,:expert_solve)
            g=operation==:mixed_solve ? mixed_solve : expert_solve;f=reference ? nothing : ()->g(A,x);verify=result->residual(result.x)
        elseif operation==:inverse
            f=reference ? ()->inv(A) : ()->native_inverse(A);verify=result->expect(A*result,Matrix{T}(LA.I,n,n));symbols=[prefix("getrf"),prefix("getri")]
        elseif operation==:pseudoinverse
            f=reference ? ()->LA.pinv(A) : ()->pseudoinverse(A);verify=result->expect(A*result*A,A);symbols=[prefix("gesdd")]
        elseif operation==:rank
            f=reference ? ()->LA.rank(A) : ()->numerical_rank(A);verify=result->result==n;symbols=[prefix("gesdd")]
        elseif operation==:condition
            f=reference ? ()->LA.cond(A) : ()->condition_number(A);verify=result->isfinite(result) && result>=1;symbols=[prefix("gesdd")]
        else
            f=reference ? ()->LA.det(A) : ()->determinant(A);expected=LA.det(A);verify=result->(!isfinite(expected) && !isfinite(result)) || abs(result-expected)<=1000n*eps(R)*max(one(R),abs(expected));symbols=[prefix("getrf")]
            # Single-precision det of these inputs overflows (real Inf, complex NaN parts): an overflowed reference requires an overflowed result.
        end
    elseif operation in (:hessenberg,:evd,:evd_dc,:evd_mrrr,:eigen_generalized,:schur,:schur_multishift,:qz,:eigen_pencil,:svd,:svd_parallel,:svd_golub_kahan,:svd_dc,:svd_randomized)
        if operation==:hessenberg
            f=reference ? ()->LA.hessenberg(A) : ()->hessenberg_factor(A);verify=F->expect(F.Q*F.H*F.Q',A);symbols=[prefix("gehrd")]
        elseif operation in (:evd,:evd_dc,:evd_mrrr,:eigen_generalized)
            generalized=operation==:eigen_generalized
            g=operation==:evd ? eigen_hermitian : operation==:evd_dc ? eigen_hermitian_dc : operation==:evd_mrrr ? eigen_hermitian_mrrr : eigen_generalized
            f=reference ? (generalized ? ()->LA.eigen(LA.Hermitian(H),LA.Hermitian(2H)) : ()->LA.eigen(LA.Hermitian(H))) : (generalized ? ()->g(H,2H) : ()->g(H))
            verify=E->expect(H*E.vectors,(generalized ? 2H*E.vectors : E.vectors)*LA.Diagonal(E.values));symbols=[prefix(generalized ? (T<:Complex ? "hegvd" : "sygvd") : (T<:Complex ? "heevr" : "syevr"))]
        elseif operation in (:schur,:schur_multishift,:qz,:eigen_pencil)
            generalized=operation in (:qz,:eigen_pencil)
            if generalized
                f=reference ? ()->LA.schur(A,H) : operation==:qz ? ()->qz_factor(A,H) : ()->eigen_pencil(A,H)
                verify=F->operation==:eigen_pencil && !reference ? expect(A*F.vectors,H*F.vectors*LA.Diagonal(F.values)) : expect(F.Q*F.S*F.Z',A) && expect(F.Q*F.T*F.Z',H)
            else
                g=operation==:schur ? schur_factor : schur_multishift;f=reference ? ()->LA.schur(A) : ()->g(A);verify=F->expect(F.Z*F.T*F.Z',A)
            end
            operation==:eigen_pencil && reference && (f=nothing);symbols=[prefix(generalized ? "gges" : "gees")]
        else
            algorithm=operation==:svd_parallel ? :parallel_jacobi : operation==:svd_golub_kahan ? :golub_kahan : operation==:svd_dc ? :divide_conquer : :jacobi
            f=reference ? ()->LA.svd(A;full=false) : operation==:svd_randomized ? ()->svd_randomized(A,n) : ()->svd_factor(A;algorithm)
            verify=F->expect(F.U*LA.Diagonal(F.S)*F.Vt,A);symbols=[prefix("gesdd")]
            note="full thin singular triplets; algorithm differences explicit; randomized API requests full rank here, no partial-spectrum efficiency claim"
        end
    elseif operation in (:matrix_exp,:matrix_log,:matrix_sqrt,:matrix_sign)
        # Scalar diagonal analytic oracle, independent of the matrix algorithm.
        D=Matrix(LA.Diagonal(T[one(T)+T(i)/T(2n) for i in 1:n]));scalar=operation==:matrix_exp ? exp : operation==:matrix_log ? log : operation==:matrix_sqrt ? sqrt : sign
        g=getfield(OxiBLAS,operation)
        g=getfield(OxiBLAS,operation)
        # Use the actual stdlib operation; do not synthesize an eigensolver-based
        # algorithm merely to obtain an Accelerate column. Its internal routing
        # is not attributed to either BLAS provider without a symbol audit.
        output_type=operation==:matrix_exp ? T : Complex{R}
        f=reference ? (operation==:matrix_sign ? nothing : ()->output_type.(scalar(D))) : ()->g(D)
        symbols=String[]
        verify=result->expect(result,LA.Diagonal(scalar.(LA.diag(D))));note="direct LinearAlgebra matrix function; no verified per-call BLAS attribution; principal output conversion explicit"

    elseif operation in (:cg,:gmres,:bicgstab,:sparse_mul,:sparse_matmul,:sparse_lu,:sparse_cholesky,:sparse_qr,:ilu0,:ilut,:ic0,:ict,:sparse_ldl,:multifrontal_lu,:multifrontal_cholesky,:supernodal_cholesky,:sparse_lu_blocked,:sparse_lu_static,:ainv,:minimum_degree,:rcm,:nested_dissection,:connected_components)
        S=CSRMatrix(H)
        if operation==:sparse_mul
            f=reference ? nothing : ()->sparse_mul!(y,S,x);verify=_ ->expect(y,H*x)
        elseif operation==:sparse_matmul
            f=reference ? nothing : ()->sparse_matmul(S,S);verify=result->expect(Matrix(result),H*H)
        elseif operation in (:cg,:gmres,:bicgstab)
            g=getfield(OxiBLAS,operation);f=reference ? nothing : ()->g(S,x;rtol=8eps(R));verify=result->result.converged && residual(result.x,H)
        elseif operation in (:minimum_degree,:rcm,:nested_dissection)
            g=operation==:minimum_degree ? minimum_degree_ordering : operation==:rcm ? reverse_cuthill_mckee : nested_dissection
            f=reference ? nothing : ()->g(S);verify=result->sort(result)==collect(1:n)
        elseif operation==:connected_components
            f=reference ? nothing : ()->connected_components(S);verify=result->!isempty(result)
        elseif operation in (:ilu0,:ilut,:ic0,:ict,:ainv)
            g=getfield(OxiBLAS,operation);f=reference ? nothing : ()->g(S);verify=result->result!==nothing
            note="preconditioner construction only; quality/residual tests remain in complete test suite"
        else
            g=operation==:sparse_ldl ? sparse_ldlt : getfield(OxiBLAS,operation)
            f=reference ? nothing : ()->g(S);verify=result->residual(solve(result,x),H)
        end
    elseif operation in (:einsum,:einsum_inplace,:contract,:batched_matmul,:hosvd,:tensor_train,:cp_als,:kron,:khatri_rao)
        if operation in (:einsum,:einsum_inplace,:contract)
            plan=plan_einsum("ik,kj->ij",A,B);w=EinsumWorkspace(T)
            f=reference ? ()->A*B : operation==:einsum ? ()->einsum(plan,A,B) : operation==:einsum_inplace ? ()->einsum!(C,plan,A,B;workspace=w) : ()->contract(A,B,(2,),(1,))
            verify=result->expect(result*ones(T,n),A*(B*ones(T,n)));symbols=[prefix("gemm")];note="same matrix contraction; LinearAlgebra matrix product reference"
        elseif operation==:batched_matmul
            aa=reshape(A,n,n,1);bb=reshape(B,n,n,1)
            f=reference ? ()->reshape(A*B,n,n,1) : ()->batched_matmul(aa,bb);verify=result->expect(result[:,:,1],A*B);symbols=[prefix("gemm")]
        elseif operation in (:hosvd,:tensor_train,:cp_als)
            cube=T[T(i)*T(j)*T(k)/T(n^3) for i in 1:n,j in 1:n,k in 1:2]
            g=operation==:hosvd ? hosvd : tensor_train
            initial=[reshape(T.(1:n),n,1),reshape(T.(1:n),n,1),reshape(T.(1:2)./T(n^3),2,1)]
            f=reference ? nothing : operation==:cp_als ? ()->cp_als(cube,1;initial,maxiter=5,regularization=zero(R)) : ()->g(cube)
            verify=result->expect(reconstruct(result),cube)
        elseif operation==:kron
            f=reference ? ()->kron(x,x) : ()->kron_product(reshape(x,:,1),reshape(x,:,1));verify=result->expect(vec(result),kron(x,x))
        else
            f=reference ? nothing : ()->khatri_rao(reshape(x,:,1),reshape(x,:,1));verify=result->expect(vec(result),kron(x,x))
        end
    else
        return full_extra_case(operation,T,n,data;reference)
    end
    # Generic arithmetic never gets a hardware BLAS attribution. Most generic
    # LA algorithms can still serve as a separately labelled fallback reference.
    !native && (symbols=String[])
    (;f,setup,verify,symbols,flops,note)
end
