# Additional exported numerical surfaces; construction/application are separate.
function full_extra_case(operation,T,n,data;reference=false)
    (;A,B,H,x)=data;R=real(T);prefix(op)=full_symbol(T,op)
    setup=()->nothing;symbols=String[];flops=NaN
    note="full returned-result operation; no direct external equivalent unless explicitly adapted"
    f=nothing;verify=result->result!==nothing
    expect(a,b)=full_relative(a,b)<=1000max(1,n)*eps(R)
    if operation in (:schur_reorder,:qz_reorder,:trevc_right,:trevc_left,:trsna_s,:trsna_sep,:schur_condition,:qz_condition,:generalized_sylvester,:schur_multishift_workspace,:schur_bulge_train,:qz_complex)
        return full_qz_schur_case(operation,T,n,data;reference)
    elseif operation in (:syr,:syr2,:her,:her2,:spr,:spr2,:hpr,:hpr2)
        herm=operation in (:her,:her2,:hpr,:hpr2);rank2=operation in (:syr2,:her2,:spr2,:hpr2);packed=operation in (:spr,:spr2,:hpr,:hpr2)
        C=zeros(T,n,n);P=pack_matrix(C;kind=herm ? :hermitian : :symmetric);z=reverse(x)
        setup=()->(fill!(C,zero(T));fill!(P.data,zero(T)))
        right=herm ? adjoint : transpose
        expected=x*right(x)+(rank2 ? z*right(x)+x*right(z)-x*right(x) : zeros(T,n,n))
        g=getfield(OxiBLAS,Symbol(operation,"!"));operand=packed ? P : C
        if reference
            if packed && ((herm && T<:Complex) || (!herm && T<:Real))
                f=rank2 ? ()->getfield(Main,Symbol("reference_",operation,"!"))(P,x,z,one(T)) : ()->getfield(Main,Symbol("reference_",operation,"!"))(P,x,herm ? one(R) : one(T))
            elseif !packed && !(herm && T<:Real)
                refg=isdefined(LA.BLAS,Symbol(operation,"!")) ? getfield(LA.BLAS,Symbol(operation,"!")) :
                     isdefined(Main,Symbol("reference_",operation,"!")) ? getfield(Main,Symbol("reference_",operation,"!")) : nothing
                f=refg===nothing ? nothing : rank2 ? ()->refg('U',one(T),x,z,C) : ()->refg('U',herm ? one(R) : one(T),x,C)
            end
        else
            f=rank2 ? ()->g(operand,x,z) : ()->g(operand,x)
        end
        verify=_ ->expect(LA.UpperTriangular(packed ? Matrix(P) : C),LA.UpperTriangular(expected));symbols=[prefix(string(operation))]
    elseif operation in (:rot,:rotg,:rotm,:rotmg,:dsdot,:sdsdot)
        y=reverse(x);z=copy(x);oldy=copy(y);setup=()->(copyto!(y,oldy);copyto!(z,x))
        if operation==:rot
            c=one(R);s=zero(T);f=reference ? ()->LA.BLAS.rot!(n,z,1,y,1,c,s) : ()->rot!(z,y,c,s)
            verify=_ ->z==x && y==oldy;symbols=[T<:Complex ? (T===ComplexF32 ? "csrot_" : "zdrot_") : prefix("rot")]
        elseif operation==:rotg
            f=reference ? nothing : ()->rotg(T(3),T(4));verify=result->result!==nothing
        elseif T<:Real && operation in (:rotm,:rotmg)
            if operation==:rotm
                params=RotmParams(T[-1,1,0,0,1]);f=reference ? nothing : ()->rotm!(z,y,params)
                # Provider fixture stores parameters in field order explicitly.
                f=reference ? ()->reference_rotm!(z,y,T[-1,1,0,0,1]) : f
                verify=_ ->z==x && y==oldy;symbols=[prefix("rotm")]
            else
                f=reference ? ()->reference_rotmg(T(1),T(1),T(3),T(4)) : ()->rotmg(T(1),T(1),T(3),T(4));symbols=[prefix("rotmg")]
            end
        elseif operation in (:dsdot,:sdsdot) && T===Float32
            f=reference ? (operation==:dsdot ? ()->reference_dsdot(x,x) : ()->reference_sdsdot(0f0,x,x)) : (operation==:dsdot ? ()->dsdot(x,x) : ()->sdsdot(0f0,x,x))
            expected=sum(Float64.(x).^2);verify=result->abs(result-expected)<=100eps(Float32)*max(1,expected);symbols=[operation==:dsdot ? "dsdot_" : "sdsdot_"]
        end
    elseif operation in (:axpy_parallel,:scal_parallel,:dot_parallel,:nrm2_parallel,:asum_parallel)
        z=copy(x);setup=()->copyto!(z,x)
        if operation==:axpy_parallel
            f=reference ? ()->LA.BLAS.axpy!(one(T),x,z) : ()->axpy_parallel!(z,x);verify=_ ->expect(z,2x);symbols=[prefix("axpy")]
        elseif operation==:scal_parallel
            f=reference ? ()->LA.BLAS.scal!(T(2),z) : ()->scal_parallel!(z,T(2));verify=_ ->expect(z,2x);symbols=[prefix("scal")]
        elseif operation==:dot_parallel
            f=reference ? (T<:Complex ? ()->LA.BLAS.dotu(x,x) : ()->LA.dot(x,x)) : ()->dot_parallel(x,x);expected=sum(x.*x);verify=result->abs(result-expected)<=1000n*eps(R)*max(one(R),abs(expected));symbols=[prefix(T<:Complex ? "dotu" : "dot")]
        elseif operation==:nrm2_parallel
            f=reference ? ()->LA.BLAS.nrm2(x) : ()->nrm2_parallel(x);expected=sqrt(sum(abs2,x));verify=result->abs(result-expected)<=1000n*eps(R)*max(one(R),abs(expected));symbols=[prefix("nrm2")]
        else
            f=reference ? ()->LA.BLAS.asum(x) : ()->asum_parallel(x);expected=sum(abs,x);verify=result->abs(result-expected)<=1000n*eps(R)*max(one(R),abs(expected));symbols=[prefix("asum")]
        end
    elseif operation in (:solve_transposed,:refine_solve,:expert_cholesky,:expert_symmetric,:reciprocal_condition,:inverse_norm_estimate,:equilibrate,:backward_error,:logabsdet,:matrix_norm,:null_space,:left_null_space,:column_space,:row_space,:least_squares,:qr_q,:apply_q,:bidiagonal,:tridiagonal,:potrs,:potri,:qr_tall,:qr_wide,:lu_tall)
        F=lu_factor(A)
        if operation==:solve_transposed
            refF=reference ? LA.lu(A) : nothing;f=reference ? ()->transpose(refF)\x : ()->solve_transposed(F,x;trans=:T);verify=result->expect(transpose(A)*result,x);symbols=[prefix("getrs")]
        elseif operation==:refine_solve
            f=reference ? nothing : ()->refine_solve(A,F,x);verify=result->expect(A*result.x,x)
        elseif operation in (:expert_cholesky,:expert_symmetric)
            g=getfield(OxiBLAS,operation);f=reference ? nothing : ()->g(H,x);verify=result->expect(H*result.x,x)
        elseif operation==:reciprocal_condition
            f=reference ? nothing : ()->reciprocal_condition(A);verify=result->result>=0 && isfinite(result)
        elseif operation==:inverse_norm_estimate
            f=reference ? nothing : ()->inverse_norm_estimate(F,T,n);verify=result->result>=0 && isfinite(result)
        elseif operation==:equilibrate
            f=reference ? nothing : ()->equilibrate(A)
        elseif operation==:backward_error
            solution=solve(F,x);f=reference ? nothing : ()->backward_error(A,solution,x);verify=result->isfinite(result) && result<=1000n*eps(R)
        elseif operation==:logabsdet
            f=reference ? ()->LA.logabsdet(A) : ()->logabsdet_factor(F);expected=LA.logabsdet(A);verify=result->abs(result[1]-expected[1])<=1000n*eps(R)*max(one(R),abs(expected[1]));symbols=[prefix("getrf")];reference && (note="factorization included in LA reference; not an equivalent precomputed-factor timing";f=nothing)
        elseif operation==:matrix_norm
            f=reference ? ()->LA.BLAS.nrm2(vec(A)) : ()->matrix_norm(A);symbols=[prefix("nrm2")];expected=sqrt(sum(abs2,A));verify=result->abs(result-expected)<=1000n*eps(R)*max(one(R),expected)
        elseif operation in (:null_space,:left_null_space,:column_space,:row_space)
            g=getfield(OxiBLAS,operation);f=reference ? nothing : ()->g(A)
            verify=result->operation==:null_space ? expect(A*result,zeros(T,n,size(result,2))) : operation==:left_null_space ? expect(result'*A,zeros(T,size(result,2),n)) : expect(result'*result,Matrix{T}(LA.I,size(result,2),size(result,2)))
        elseif operation in (:least_squares,:qr_q,:apply_q)
            Q=qr_factor(A);refQ=reference ? LA.qr(A) : nothing
            if operation==:least_squares
                f=reference ? ()->refQ\x : ()->least_squares(Q,x);verify=result->expect(A*result,x);symbols=[prefix("ormqr"),prefix("trtrs")];T<:Complex && (symbols=[prefix("unmqr"),prefix("trtrs")])
            elseif operation==:qr_q
                f=reference ? ()->Matrix(refQ.Q) : ()->qr_q(Q);verify=result->expect(result'*result,Matrix{T}(LA.I,n,n));symbols=[prefix(T<:Complex ? "ungqr" : "orgqr")]
            else
                Z=copy(A);setup=()->copyto!(Z,A);f=reference ? ()->LA.lmul!(refQ.Q,Z) : ()->apply_q!(Z,Q)
                verify=_ ->expect(Z,(reference ? Matrix(refQ.Q) : Q.Q)*A);symbols=[prefix(T<:Complex ? "unmqr" : "ormqr")]
                note="in place Q*B with a precomputed factorization; reference applies LAPACK's compact WY Q (gemqrt)"
            end
        elseif operation in (:potrs,:potri,:qr_tall,:qr_wide,:lu_tall)
            if operation==:potri
                Fc=reference ? LA.cholesky(LA.Hermitian(H)) : cholesky_factor(H)
                f=reference ? ()->inv(Fc) : ()->cholesky_inverse(Fc);verify=result->expect(H*result,Matrix{T}(LA.I,n,n));symbols=[prefix("potri")]
                note="inverse from a precomputed Cholesky factor; full Hermitian result on both sides"
            elseif operation==:potrs
                Fc=reference ? LA.cholesky(LA.Hermitian(H)) : cholesky_factor(H)
                f=reference ? ()->Fc\x : ()->solve(Fc,x);verify=result->expect(H*result,x);symbols=[prefix("potrs")]
                note="solve with a precomputed Cholesky factor; vector right-hand side"
            else
                # Rectangular inputs from the shared data: tall [A;B] or wide [A B].
                M=operation==:qr_wide ? hcat(A,B) : vcat(A,B)
                if operation==:lu_tall
                    f=reference ? ()->LA.lu(M) : ()->lu_factor(M);symbols=[prefix("getrf")]
                    verify=F->expect(F.L*F.U,M[F.p,:])
                else
                    f=reference ? ()->LA.qr(M) : ()->qr_factor(M);symbols=[prefix("geqrf")]
                    k=min(size(M)...)
                    verify=reference ? F->expect(F.Q*F.R,M) : F->expect(F.Q[:,1:k]*F.R[1:k,:],M)
                end
                note="$(size(M,1))x$(size(M,2)) input; compact factorization on both sides"
            end
        else
            g=operation==:bidiagonal ? bidiagonal_factor : tridiagonal_factor;f=reference ? nothing : ()->g(H)
            verify=operation==:bidiagonal ? F->expect(F.U*F.B*F.V',H) : F->expect(tridiagonal_q(F)*Matrix(LA.SymTridiagonal(F.diagonal,F.offdiagonal))*tridiagonal_q(F)',H)
        end
    elseif operation in (:sparse_mul_csc,:sparse_mul_csc_transpose)
        S=CSCMatrix(CSRMatrix(H));out=zeros(T,n);tr=operation==:sparse_mul_csc ? :N : :C
        f=reference ? nothing : ()->sparse_mul!(out,S,x;trans=tr)
        verify=result->expect(result,(tr===:N ? H : adjoint(H))*x)
        note="native compressed-column multiply; parallel N may retain CSR conversion; transpose uses disjoint CSC column dots"
    elseif operation in (:sparse_transpose,:sparse_add,:sparse_permute,:sparse_diagonal,:sparse_norm,:dia,:ell,:bsr,:bsc,:hyb,:sell,:relaxation_preconditioner,:polynomial_preconditioner,:spai,:additive_schwarz,:amg,:pcg,:minres,:tfqmr,:fgmres,:qmr,:idrs,:block_cg,:block_gmres)
        S=CSRMatrix(H)
        if operation in (:pcg,:minres,:tfqmr,:fgmres,:qmr,:idrs,:block_cg,:block_gmres)
            g=getfield(OxiBLAS,operation);rhs=operation in (:block_cg,:block_gmres) ? reshape(x,:,1) : x
            f=reference ? nothing : ()->g(S,rhs;rtol=8eps(R));verify=result->expect(H*result.x,rhs)
        elseif operation in (:relaxation_preconditioner,:polynomial_preconditioner,:spai,:additive_schwarz,:amg)
            g=getfield(OxiBLAS,operation);f=reference ? nothing : ()->g(S);note="preconditioner construction; independent numerical quality tests in full suite"
        elseif operation in (:dia,:ell,:bsr,:bsc,:hyb,:sell)
            g=getfield(OxiBLAS,operation==:dia ? :DIAMatrix : operation==:ell ? :ELLMatrix : operation==:bsr ? :BSRMatrix : operation==:bsc ? :BSCMatrix : operation==:hyb ? :HYBMatrix : :SELLMatrix)
            formatted=g(S);y=zeros(T,n);f=reference ? nothing : ()->sparse_mul!(y,formatted,x);verify=_ ->expect(y,H*x)
        elseif operation==:sparse_transpose
            f=reference ? nothing : ()->sparse_transpose(S);verify=result->expect(Matrix(result),transpose(H))
        elseif operation==:sparse_add
            f=reference ? nothing : ()->sparse_add(S,S);verify=result->expect(Matrix(result),2H)
        elseif operation==:sparse_permute
            p=reverse(1:n);f=reference ? nothing : ()->sparse_permute(S,p);verify=result->expect(Matrix(result),H[p,p])
        elseif operation==:sparse_diagonal
            f=reference ? nothing : ()->sparse_diagonal(x);verify=result->expect(Matrix(result),LA.Diagonal(x))
        else
            f=reference ? nothing : ()->sparse_norm(S);expected=sqrt(sum(abs2,H));verify=result->abs(result-expected)<=1000n*eps(R)*max(one(R),expected)
        end
    elseif operation in (:outer_product,:mode_product,:kron_sum,:kronecker_action,:contract_inplace,:batched_matmul_inplace,:matrix_power,:matrix_sin,:matrix_cos,:frechet_exp,:frechet_log,:frechet_sqrt)
        if operation==:outer_product
            f=reference ? ()->x*transpose(x) : ()->outer_product(x,x)
            verify=result->expect(result,x*transpose(x));symbols=String[];note="direct stdlib outer product; internal provider not attributed without per-call audit"
        elseif operation==:mode_product
            cube=reshape(B,n,n,1);f=reference ? ()->reshape(A*B,n,n,1) : ()->mode_product(cube,A,1);verify=result->expect(result[:,:,1],A*B);symbols=[prefix("gemm")]
        elseif operation==:contract_inplace
            C=zeros(T,n,n);f=reference ? ()->LA.mul!(C,A,B) : ()->contract!(C,A,B,(2,),(1,));verify=_ ->expect(C*ones(T,n),A*(B*ones(T,n)));symbols=[prefix("gemm")]
        elseif operation==:batched_matmul_inplace
            C=zeros(T,n,n,1);aa=reshape(A,n,n,1);bb=reshape(B,n,n,1);f=reference ? ()->LA.mul!(view(C,:,:,1),A,B) : ()->batched_matmul!(C,aa,bb);verify=_ ->expect(C[:,:,1],A*B);symbols=[prefix("gemm")]
        elseif operation==:kron_sum
            f=reference ? ()->kron(A,Matrix{T}(LA.I,n,n))+kron(Matrix{T}(LA.I,n,n),B) : ()->kron_sum(A,B)
            expected=kron(A,Matrix{T}(LA.I,n,n))+kron(Matrix{T}(LA.I,n,n),B);verify=result->expect(result,expected)
        elseif operation==:kronecker_action
            rhs=repeat(x,n);f=reference ? nothing : ()->kronecker_action(A,B,rhs);expected=vec(B*reshape(rhs,n,n)*transpose(A));verify=result->expect(result,expected)
        elseif operation in (:frechet_exp,:frechet_log,:frechet_sqrt)
            D=Matrix(LA.Diagonal(fill(T(2),n)));E=Matrix{T}(LA.I,n,n);g=getfield(OxiBLAS,operation)
            f=reference ? nothing : ()->g(D,E);scale=operation==:frechet_exp ? exp(R(2)) : operation==:frechet_log ? one(R)/2 : one(R)/(2sqrt(R(2)));verify=result->expect(result.derivative,scale*E)
        else
            D=Matrix(LA.Diagonal(fill(T(2),n)));g=getfield(OxiBLAS,operation)
            output_type=operation==:matrix_power ? T : Complex{R}
            f=reference ? (operation==:matrix_power ? ()->D^3 : operation==:matrix_sin ? ()->output_type.(sin(D)) : ()->output_type.(cos(D))) : operation==:matrix_power ? ()->g(D,3) : ()->g(D)
            symbols=String[];note="direct stdlib matrix power/sin/cos; no synthesized reference algorithm or unverified provider attribution"
            scale=operation==:matrix_power ? T(8) : operation==:matrix_sin ? sin(T(2)) : cos(T(2));verify=result->expect(result,scale*Matrix{T}(LA.I,n,n))
        end
    elseif operation in (:einsum_beam,:einsum_optimal,:einsum_sliced,:einsum_sliced_inplace,:plan_einsum_beam,:plan_einsum_optimal,:plan_einsum_sliced)
        spec="ij,jk,kl->il";sliced=operation in (:einsum_sliced,:einsum_sliced_inplace,:plan_einsum_sliced)
        planning=operation in (:plan_einsum_beam,:plan_einsum_optimal,:plan_einsum_sliced)
        optimize=operation in (:einsum_optimal,:plan_einsum_optimal) ? :optimal : :beam
        make=()->sliced ? plan_einsum_sliced(spec,A,B,H;chunks=(i=max(1,n÷2),j=max(1,n÷2)),scratch_limit=2n*n*sizeof(T)) : plan_einsum(spec,A,B,H;optimize)
        if planning
            f=reference ? nothing : make
            verify=plan->sliced ? plan.peak_intermediate_bytes<=2n*n*sizeof(T) : length(plan.steps)==2
            note="planning only; metadata cost; no equivalent external planner adapter"
        else
            plan=make();workspace=EinsumWorkspace(T);out=zeros(T,n,n)
            f=reference ? ()->A*B*H : operation==:einsum_sliced_inplace ? ()->einsum!(out,plan,A,B,H;workspace) : ()->einsum(plan,A,B,H;workspace)
            verify=result->expect(result*ones(T,n),A*(B*(H*ones(T,n))))
            symbols=[prefix("gemm")];note="same three-matrix contraction; advanced plan/sliced execution versus explicit LinearAlgebra products"
        end
    else
        error("Missing benchmark adapter $operation")
    end
    !(T<:OxiBLAS.NativeFloat) && (symbols=String[])
    (;f,setup,verify,symbols,flops,note)
end

# Explicit Float16 storage / Float32 working arithmetic. Conversion and result
# storage are inside each reference call, as in the native half adapters.
function full_half_reference_case(operation,T,n,data)
    (;A,B,H,x)=data;W=T===Float16 ? Float32 : ComplexF32;R=real(T)
    setup=()->nothing;f=nothing;symbols=String[];flops=NaN
    note="Float16 storage; Float32 working arithmetic; conversion/storage included; no native half BLAS claimed"
    expect(a,b)=full_relative(a,b)<=500max(1,n)*eps(R)
    verify=result->result!==nothing
    if operation in (:gemm,:einsum)
        wa=zeros(W,n,n);wb=similar(wa);wc=similar(wa);out=zeros(T,n,n)
        f=()->begin
            copyto!(wa,A);copyto!(wb,B);LA.mul!(wc,wa,wb);copyto!(out,wc);out
        end
        verify=result->expect(W.(result)*ones(W,n),W.(A)*(W.(B)*ones(W,n)))
        symbols=[full_symbol(W,"gemm")]
        operation==:gemm && (flops=(T<:Complex ? 8 : 2)*Float64(n)^3)
    elseif operation in (:lu,:cholesky,:qr)
        matrix=operation==:cholesky ? H : A
        f=operation==:lu ? ()->LA.lu(W.(matrix)) : operation==:qr ? ()->LA.qr(W.(matrix)) : ()->LA.cholesky(LA.Hermitian(W.(matrix)))
        verify=F->expect(W.(matrix)*(F\W.(x)),W.(x))
        symbols=[full_symbol(W,operation==:lu ? "getrf" : operation==:cholesky ? "potrf" : "geqrf")]
    elseif operation==:solve
        F=LA.lu(W.(A));f=()->F\W.(x);verify=result->expect(W.(A)*result,W.(x));symbols=[full_symbol(W,"getrs")]
    end
    (;f,setup,verify,symbols,flops,note)
end
