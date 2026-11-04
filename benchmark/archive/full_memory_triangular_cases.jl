# Prepared whole operations in the single four-column campaign.
function full_memory_triangular_case(operation,T,n,data;reference=false)
    R=real(T); symbols=String[]; flops=NaN; setup=()->nothing
    if operation in (:sparse_svd_gk,:sparse_svd_gk_reuse,:sparse_svd_dilation)
        base=copy(data.B)
        for j in 1:n
            base[j,j]+=T(j<=3 ? 40-5j : 1+j/(32n))
        end
        A=vcat(base,data.B); S=CSRMatrix(A); rank=min(3,n); cap=min(n,16)
        w=operation===:sparse_svd_gk_reuse ? SparseSVDWorkspace(S,rank;subspace=cap) : nothing
        f=reference ? nothing : operation===:sparse_svd_dilation ?
            ()->sparse_svd_lanczos(S,rank;algorithm=:dilation,subspace=2cap,keep=min(2cap-1,2rank)) :
            ()->sparse_svd_lanczos(S,rank;subspace=cap,keep=min(cap-1,2rank),workspace=w)
        expected=LA.svdvals(A)[1:rank]
        verify=F->F.converged && length(F.S)==rank && LA.norm(F.S-expected)<=5000n*eps(R)*max(one(R),LA.norm(expected)) &&
            LA.norm(A*F.Vt'-F.U*LA.Diagonal(F.S))<=5000n*sqrt(eps(R))*LA.norm(A) &&
            LA.norm(A'*F.U-F.Vt'*LA.Diagonal(F.S))<=5000n*sqrt(eps(R))*LA.norm(A)
        note="native sparse iterative triplets; physical size $(2n)x$(n); separate GK/dilation algorithm cases (different subspace conventions); no dense SVD substituted as equivalent timing; retained memory in dedicated campaign"
    else
        solving=operation in (:trsm_right_leaf,:trsm_left_leaf,:trsm_reuse)
        right=operation in (:trsm_right_leaf,:trmm_right_leaf)
        reuse=operation in (:trsm_reuse,:trmm_reuse)
        side=right ? :R : :L; order=reuse ? n : min(8,n)
        rows,cols=right ? (max(19,n),order) : (order,max(11,n))
        A=Matrix(LA.UpperTriangular(data.A[1:order,1:order])); B=T[convert(T,mod(3i+7j,19))/32 for i in 1:rows,j in 1:cols]; C=copy(B)
        w=triangular_workspace(T,rows,cols;side); setup=()->copyto!(C,B)
        g=solving ? trsm! : trmm!
        f=reference ? ()->getfield(LA.BLAS,solving ? :trsm! : :trmm!)(right ? 'R' : 'L','U','N','N',one(T),A,C) :
            ()->g(C,A;side,blocksize=8,workspace=reuse ? w : nothing)
        expected=solving ? (right ? B/A : A\B) : (right ? B*A : A*B)
        verify=_ ->full_relative(C,expected)<=5000max(rows,cols)*eps(R)
        symbols=[full_symbol(T,solving ? "trsm" : "trmm")]
        note="triangular $(side) order=$(order), output $(rows)x$(cols); native contiguous SIMD/right or four-RHS/left leaf; reusable context=$(reuse); no reciprocal diagonal; setup excluded"
    end
    (;f,setup,verify,symbols,flops,note)
end
