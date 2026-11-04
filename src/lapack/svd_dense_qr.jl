# SPDX-License-Identifier: Apache-2.0
# QR preprocessing for rectangular dense SVDs. The QR primitives are reused
# unchanged. Jacobi normalizes columns before QR and restores their weights in
# R, preserving column-relative backward errors on strongly graded matrices.
function _svd_dense_tall_qr(A::AbstractMatrix{T},scaled_columns,threaded) where {T<:NativeFloat}
    C,scale=_normalized_copy(A);m,n=size(C);R=real(T)
    norms=scaled_columns ? Vector{R}(undef,n) : nothing
    if scaled_columns
        @inbounds for j in 1:n
            z=nrm2(view(C,:,j));norms[j]=z
            if !iszero(z)
                @simd for i in 1:m
                    C[i,j]=_realdiv(C[i,j],z)
                end
            end
        end
    end
    Q=qr_factor(C;threaded);small=zeros(T,n,n)
    @inbounds for j in 1:n
        weight=scaled_columns ? norms[j] : one(R)
        for i in 1:j
            small[i,j]=Q.factors[i,j]*weight
        end
    end
    Q,small,scale
end

# Conservative eligibility check on the column-equilibrated QR. Small pivots
# indicate a rank-deficient or poorly conditioned panel: preserve the direct
# SVD path rather than exposing its null space to the QR backtransform.
function _svd_dense_qr_eligible(Q)
    A=Q.factors;n=size(A,2);R=real(eltype(A));lo=typemax(R);hi=zero(R)
    @inbounds for j in 1:n
        d=abs(A[j,j]);lo=min(lo,d)
        for i in 1:j
            hi=max(hi,abs(A[i,j]))
        end
    end
    isfinite(hi) && lo>sqrt(sqrt(eps(R)))*hi
end

function _svd_dense_qr_result(Q,G,scale,m,n,full,threaded)
    T=eltype(Q.factors);R=real(T);G.S .*= scale
    G.U===nothing && return SVDFactor{T,R}(nothing,G.S,nothing,G.sweeps)
    U=zeros(T,m,n);copyto!(view(U,1:n,:),G.U)
    apply_q!(U,Q;threaded)
    result=SVDFactor(U,G.S,G.Vt,G.sweeps)
    full ? _full_svd(result,m,n) : result
end
