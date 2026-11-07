# SPDX-License-Identifier: Apache-2.0
"""
    AINVPreconditioner{T}

Factored sparse approximate inverse Z*Diagonal(1/d)*W'.
Biconjugation without pivoting; a zero/nonfinite pivot is reported explicitly.
Dropping is relative to each updated column and retains its unit diagonal.
"""
struct AINVPreconditioner{T}
    Z::CSRMatrix{T,Int}
    W::CSRMatrix{T,Int}
    diagonal::Vector{T}
    work::Vector{T}
    guard::ReentrantLock
end
function _ainv_drop!(column::Dict{Int,T},diagonal,tolerance,maxnnz) where T
    scale=maximum(abs,Base.values(column);init=zero(real(T)))
    for row in collect(keys(column))
        row==diagonal && continue
        abs(column[row])<=tolerance*scale && delete!(column,row)
    end
    off=sort!([i for i in keys(column) if i!=diagonal];by=i->(-abs(column[i]),i))
    for k in maxnnz:length(off)
        delete!(column,off[k])
    end
end
function _ainv_matrix(columns,::Type{T}) where T
    n=length(columns); rows=Int[]; cols=Int[]; vals=T[]
    for j in 1:n, i in sort!(collect(keys(columns[j])))
        push!(rows,i); push!(cols,j); push!(vals,columns[j][i])
    end
    CSRMatrix(COOMatrix(n,n,rows,cols,vals))
end
function ainv(A::NativeSparse;droptol=0.1,maxnnz=20)
    n=_square(A); T=eltype(A); C=CSRMatrix(A)
    isfinite(droptol) && droptol>=0 && maxnnz>=1 || throw(ArgumentError("invalid AINV dropping options"))
    Z=[Dict{Int,T}(j=>one(T)) for j in 1:n]; W=[Dict{Int,T}(j=>one(T)) for j in 1:n]
    diagonal=zeros(T,n); z=zeros(T,n); w=similar(z); az=similar(z); ahw=similar(z)
    for i in 1:n
        fill!(z,zero(T)); fill!(w,zero(T))
        for (row,value) in Z[i]
            z[row]=value
        end
        for (row,value) in W[i]
            w[row]=value
        end
        sparse_mul!(az,C,z;threaded=false); sparse_mul!(ahw,C,w;trans=:C,threaded=false)
        pivot=dotc(w,az)
        isfinite(pivot) && !iszero(pivot) || throw(SingularException(i))
        diagonal[i]=pivot
        for j in i+1:n
            alpha=sum((conj(ahw[row])*value for (row,value) in Z[j]);init=zero(T))/pivot
            beta=sum((conj(value)*az[row] for (row,value) in W[j]);init=zero(T))/pivot
            for (row,value) in Z[i]
                Z[j][row]=get(Z[j],row,zero(T))-alpha*value
            end
            for (row,value) in W[i]
                W[j][row]=get(W[j],row,zero(T))-conj(beta)*value
            end
            _ainv_drop!(Z[j],j,droptol,maxnnz); _ainv_drop!(W[j],j,droptol,maxnnz)
        end
    end
    AINVPreconditioner(_ainv_matrix(Z,T),_ainv_matrix(W,T),diagonal,zeros(T,n),ReentrantLock())
end
function _precondition!(z,r,P::AINVPreconditioner)
    n=length(P.diagonal); length(z)==length(r)==n || throw(DimensionMismatch("AINV dimensions differ"))
    _check_vector_output(z)
    (Base.mightalias(z,P.Z) || Base.mightalias(z,P.W) || Base.mightalias(z,P.diagonal) || Base.mightalias(z,P.work)) && throw(ArgumentError("output must not alias AINV storage"))
    lock(P.guard) do
        sparse_mul!(P.work,P.W,r;trans=:C,threaded=false)
        for i in 1:n
            P.work[i]/=P.diagonal[i]
        end
        sparse_mul!(z,P.Z,P.work;threaded=false)
    end
    z
end
_check_preconditioner(P::AINVPreconditioner,n)=length(P.diagonal)==n ? nothing : throw(DimensionMismatch("preconditioner dimensions differ"))
