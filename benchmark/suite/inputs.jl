# Element types, sizes and inputs of the high-level suite.
#
# Each input kind gives the OxiBLAS operand and the LinearAlgebra /
# SparseArrays operand of the same matrix (the reference). For the OxiBLAS
# storage types without a LinearAlgebra counterpart, the reference is the
# closest standard type: SparseMatrixCSC for BandedMatrix, Hermitian{Matrix}
# for PackedMatrix. The `ref_input` column of the results records it.

const ELTYPES=[Float16=>"f16",Float32=>"f32",Float64=>"f64",ComplexF32=>"cf32",ComplexF64=>"cf64",BigFloat=>"big"]
eltype_label(T)=last(ELTYPES[findfirst(p->first(p)===T,ELTYPES)])
# Float16 and BigFloat have no BLAS/LAPACK: LinearAlgebra runs its generic
# code, so their sizes are smaller.
generic_type(T)=real(T)===Float16 || real(T)===BigFloat

const SIZE_LABELS=("s","m","l")
# Sizes per class: (BLAS types, Float16, BigFloat). n is the matrix order,
# except for :sparse (side k of the k×k grid, n = k²) and the tensor classes
# (extent of each index).
const SIZES=Dict(
    :cubic    =>((32,256,1024),(16,64,256),(16,48,128)),
    :spectral =>((32,128,512),(16,48,128),(8,24,64)),
    :kron     =>((8,24,64),(4,12,24),(4,8,16)),
    :linear   =>((1_000,100_000,1_000_000),(100,1_000,10_000),(100,1_000,10_000)),
    :linear_spectral=>((64,512,2048),(32,128,512),(16,64,128)),
    :banded   =>((256,4096,65536),(64,512,4096),(64,512,4096)),
    :sparse   =>((16,64,192),(8,16,32),(8,16,32)),
    :tensor   =>((16,48,96),(8,16,32),(8,12,16)),
    :tensor4  =>((8,16,28),(4,8,12),(4,6,8)),
)
const DENSE_KINDS=(:dense,:dense_unit,:tall,:hermitian,:indefinite,:upper,:lower,:packed)
const LINEAR_KINDS=(:diagonal,:bidiagonal,:tridiagonal,:symtridiagonal)
const BANDED_KINDS=(:banded,:banded_spd)
const SPARSE_KINDS=(:sparse,:csr,:csc,:coo,:sparse_spd,:csr_spd,:csc_spd)

function size_class(kind::Symbol,opclass::Symbol)
    kind in DENSE_KINDS && return opclass
    kind in LINEAR_KINDS && return opclass===:cubic ? :linear : :linear_spectral
    kind in BANDED_KINDS && return :banded
    kind in SPARSE_KINDS && return :sparse
    kind===:tensor && return opclass
    error("unknown input kind $kind")
end
function sizes_for(kind,opclass,T)
    s=SIZES[size_class(kind,opclass)]
    real(T)===Float16 ? s[2] : real(T)===BigFloat ? s[3] : s[1]
end

rng()=Xoshiro(20261104)
# randn in Float64/ComplexF64, then converted: same values for every type.
rnd(r,T,dims...)=T.(randn(r,T<:Complex ? ComplexF64 : Float64,dims...))

# 2-D Laplacian L and skew convection K: L + c·K is a nonsymmetric system;
# L + i·c·K is Hermitian positive definite.
function lap2d(k)
    e=ones(k-1); T1=spdiagm(-1=>-e,0=>fill(2.0,k),1=>-e); C=spdiagm(-1=>-e,1=>e)
    Id=sparse(1.0LA.I,k,k)
    kron(Id,T1)+kron(T1,Id),kron(Id,C)+kron(C,Id)
end
function sparse_general(T,k)
    L,K=lap2d(k)
    S=SparseMatrixCSC{T,Int}(L+0.3K)
    T<:Complex ? S+T(0.2im)*sparse(T(1)*LA.I,k*k,k*k) : S
end
function sparse_spd(T,k)
    L,K=lap2d(k)
    # i·K is Hermitian with eigenvalues in ±1.2: the shift 2I keeps L+0.3i·K+2I
    # positive definite (λmin(L) alone is O(1/k²)).
    T<:Complex ? SparseMatrixCSC{T,Int}(L+0.3im*K+2.0LA.I) : SparseMatrixCSC{T,Int}(L)
end

function spd_dense(r,T,n)
    X=rnd(r,T,n,n)
    H=X*X'/T(n)+LA.I
    LA.Hermitian((H+H')/2)
end

function banded_general(r,T,n;kd=4)
    kl=ku=kd; data=rnd(r,T,kl+ku+1,n)
    I,J,V=Int[],Int[],T[]
    for j in 1:n, q in 1:kl+ku+1
        i=j+q-ku-1
        if 1<=i<=n
            q==ku+1 && (data[q,j]+=T(4kd))
            push!(I,i); push!(J,j); push!(V,data[q,j])
        else
            data[q,j]=zero(T)
        end
    end
    BandedMatrix(data,n;kl,ku),sparse(I,J,V,n,n)
end
function banded_spd(r,T,n;kd=4)
    data=rnd(r,T,kd+1,n)
    I,J,V=Int[],Int[],T[]
    for j in 1:n, q in 1:kd+1
        i=j+q-kd-1
        if 1<=i<=n
            q==kd+1 && (data[q,j]=T(4kd+1))
            push!(I,i); push!(J,j); push!(V,data[q,j])
        else
            data[q,j]=zero(T)
        end
    end
    S=sparse(I,J,V,n,n)
    BandedMatrix(data,n;ku=kd,kind=T<:Complex ? :hermitian : :symmetric,uplo=:U),LA.Hermitian(S,:U)
end

# (oxi, ref, ref_label): the two operands of the same matrix of order ~n.
function make_input(kind::Symbol,T,n,r)
    if kind===:dense
        A=rnd(r,T,n,n)+T(n)*LA.I; (A,A,"Matrix")
    elseif kind===:dense_unit           # spectrum of order 1 (matrix functions)
        A=rnd(r,T,n,n)/T(sqrt(n)); (A,A,"Matrix")
    elseif kind===:tall
        A=rnd(r,T,2n,n); (A,A,"Matrix")
    elseif kind===:hermitian
        H=spd_dense(r,T,n); (H,H,"Hermitian{Matrix}")
    elseif kind===:indefinite
        X=rnd(r,T,n,n); H=LA.Hermitian(X+X'); (H,H,"Hermitian{Matrix}")
    elseif kind===:upper
        U=LA.UpperTriangular(LA.triu(rnd(r,T,n,n))/T(n)+2LA.I); (U,U,"UpperTriangular")
    elseif kind===:lower
        L=LA.LowerTriangular(LA.tril(rnd(r,T,n,n))/T(n)+2LA.I); (L,L,"LowerTriangular")
    elseif kind===:packed
        H=spd_dense(r,T,n)
        P=pack_matrix(Matrix(H);uplo=:U,kind=T<:Complex ? :hermitian : :symmetric); (P,H,"Hermitian{Matrix}")
    elseif kind===:diagonal
        D=LA.Diagonal(rnd(r,T,n).+T(4)); (D,D,"Diagonal")
    elseif kind===:bidiagonal
        B=LA.Bidiagonal(rnd(r,T,n).+T(4),rnd(r,T,n-1),:U); (B,B,"Bidiagonal")
    elseif kind===:tridiagonal
        M=LA.Tridiagonal(rnd(r,T,n-1),rnd(r,T,n).+T(4),rnd(r,T,n-1)); (M,M,"Tridiagonal")
    elseif kind===:symtridiagonal       # strictly diagonally dominant (|e| ≤ 1/2, d ≥ 4): positive definite when real
        e=rnd(r,T,n-1); e=e./T(2max(1,maximum(abs,e;init=0)))
        M=LA.SymTridiagonal(abs.(real(rnd(r,T,n))).+T(4),e); (M,M,"SymTridiagonal")
    elseif kind===:banded
        B,S=banded_general(r,T,n); (B,S,"SparseMatrixCSC")
    elseif kind===:banded_spd
        B,S=banded_spd(r,T,n); (B,S,"Hermitian{SparseMatrixCSC}")
    elseif kind===:sparse
        S=sparse_general(T,n); (S,S,"SparseMatrixCSC")
    elseif kind===:csr
        S=sparse_general(T,n); (CSRMatrix(S),S,"SparseMatrixCSC")
    elseif kind===:csc
        S=sparse_general(T,n); (CSCMatrix(S),S,"SparseMatrixCSC")
    elseif kind===:coo
        S=sparse_general(T,n); (COOMatrix(S),S,"SparseMatrixCSC")
    elseif kind===:sparse_spd
        S=LA.Hermitian(sparse_spd(T,n)); (S,S,"Hermitian{SparseMatrixCSC}")
    elseif kind===:csr_spd
        S=sparse_spd(T,n); (LA.Hermitian(CSRMatrix(S)),LA.Hermitian(S),"Hermitian{SparseMatrixCSC}")
    elseif kind===:csc_spd
        S=sparse_spd(T,n); (LA.Hermitian(CSCMatrix(S)),LA.Hermitian(S),"Hermitian{SparseMatrixCSC}")
    else
        error("unknown input kind $kind")
    end
end
# Matrix order of the input built from size parameter n.
order(kind,n)=kind in SPARSE_KINDS ? n*n : n
