# Prepared partial-spectrum cases in the single four-backend report.
function full_selected_svd_case(operation,T,n,data;reference=false)
    R=real(T); A=operation===:svd_selected_tall ? vcat(data.A,data.B) : operation===:svd_selected_wide ? hcat(data.A,data.B) : data.A
    m,p=size(A); r=min(m,p); k=min(r,max(1,min(8,r÷8)))
    operation===:svd_selected_quarter && (k=max(1,r÷4))
    operation===:svd_selected_half && (k=max(1,r÷2))
    indices=operation===:svd_selected_smallest ? (r-k+1:r) : operation===:svd_selected_middle ? (max(1,r÷2-k÷2):max(1,r÷2-k÷2)+k-1) : (1:k)
    interval=operation===:svd_selected_interval ? (R(129)/32,R(133)/32) : nothing
    vectors=operation!==:svd_selected_values; representation=operation===:svd_selected_gram ? :gram : :dilation
    # Independent full-spectrum oracle is setup only, never the measured path.
    oracle=LA.svdvals(A); expected=interval===nothing ? oracle[indices] : filter(s->interval[1]<s<=interval[2],oracle)
    f=reference ? ()->reference_selected_svd(A;indices,interval,vectors) : operation===:svd_truncated ? ()->svd_truncated(A,k;vectors) : interval===nothing ? ()->svd_selected(A;indices,vectors,representation) : ()->svd_selected(A;interval,vectors,representation)
    verify=F->begin
        length(F.S)==length(expected) || return false
        LA.norm(F.S-expected)<=5000max(m,p)*eps(R)*max(one(R),LA.norm(oracle)) || return false
        !vectors && return F.U===nothing && F.Vt===nothing
        q=length(F.S); V=F.Vt'
        bound=5000max(m,p)*eps(R)*max(one(R),LA.norm(A))
        LA.norm(A*V-F.U*LA.Diagonal(F.S))<=bound && LA.norm(A'*F.U-V*LA.Diagonal(F.S))<=bound &&
            LA.norm(F.U'*F.U-Matrix{T}(LA.I,q,q))<=bound && LA.norm(F.Vt*F.Vt'-Matrix{T}(LA.I,q,q))<=bound
    end
    note="genuine selected SVD; physical size=$(m)x$(p); requested="*(interval===nothing ? string(indices) : string(interval))*"; vectors=$(vectors); native representation=$(representation); external GESVDX; input copies/query/allocations included; oracle excluded"
    (;f,setup=()->nothing,verify,symbols=[full_symbol(T,"gesvdx")],flops=NaN,note)
end
