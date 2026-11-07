# Operations of the high-level suite.
#
# An operation is benchmarked on every (input kind, element type, size) of its
# grid. Its builder receives (A, R, T, N, r):
#   A  the OxiBLAS operand, R the reference operand (same matrix),
#   T  the element type, N the matrix order, r a seeded RNG,
# and returns (oxi, ref, check):
#   oxi, ref     zero-argument closures over the same data (ref may be nothing);
#   check(xo,xr) true/false, or missing when nothing can validate the result
#                (no reference and no cheap identity). xr is nothing when the
#                reference is unavailable.
# Checks are sanity checks (backward errors, reconstructions), not accuracy
# tests: the test suite owns accuracy.

struct Op
    name::String
    family::String
    form::String          # "allocating" or "in-place"
    api::Any              # the OxiBLAS function the case measures
    kinds::Vector{Symbol}
    class::Symbol         # size class of the dense kinds (see inputs.jl)
    build::Function
end

# ---- checks ---------------------------------------------------------------
rtol(T,N)=min(0.3,50*eps(real(T))*max(N,16))
cmp_form(x)=x isa SparseArrays.AbstractSparseMatrix ? SparseMatrixCSC(x) : x isa AbstractArray ? Array(x) : x
function relerr(x,y)
    x,y=cmp_form(x),cmp_form(y)
    LA.norm(x-y)/max(LA.norm(y),floatmin(Float64))
end
close_to(x,y,T,N)=isequal(x,y) || relerr(x,y)<=rtol(T,N)
# Frobenius norm without O(n²) generic iteration over structured wrappers.
fnorm(R::Union{LA.Hermitian,LA.Symmetric})=parent(R) isa SparseMatrixCSC ? 2LA.norm(parent(R)) : LA.norm(R)
fnorm(R::LA.Diagonal)=LA.norm(R.diag)
fnorm(R::Union{LA.Bidiagonal,LA.Tridiagonal,LA.SymTridiagonal})=LA.norm(sparse(R))
fnorm(R::Union{LA.UpperTriangular,LA.LowerTriangular})=LA.norm(parent(R))
fnorm(R)=LA.norm(R)
# Square: normwise backward error of the residual. Tall: least-squares
# optimality, ‖Rᴴ(Rx-b)‖ small against ‖R‖(‖R‖‖x‖+‖b‖).
function solves(R,x,b,T,N)
    res=R*x-b; nR=fnorm(R)
    size(R,1)==size(R,2) ? LA.norm(res)<=rtol(T,N)*(nR*LA.norm(x)+LA.norm(b)) :
                           LA.norm(R'*res)<=rtol(T,N)*nR*(nR*LA.norm(x)+LA.norm(b))
end
ref_or_missing(f,xr)=xr===nothing ? missing : f(xr)
sortc(v)=sort(collect(v);by=z->(real(z),imag(z)))

rhs(r,T,R,k=0)=k==0 ? rnd(r,T,size(R,1)) : rnd(r,T,size(R,1),k)
# Right-hand columns of a product: square for dense-like operands, 8 otherwise.
ncols(R)=R isa Union{Matrix,LA.Hermitian{<:Any,<:Matrix},LA.UpperTriangular,LA.LowerTriangular} ? size(R,2) : 8

# ---- products -------------------------------------------------------------
const PRODUCT_KINDS=[:dense,:hermitian,:upper,:diagonal,:bidiagonal,:tridiagonal,:symtridiagonal,:banded,:packed,:sparse,:csr,:csc,:coo]
mul_alloc(A,R,T,N,r)=(B=rnd(r,T,size(R,2),ncols(R));
    (()->O.mul(A,B), ()->R*B, (xo,xr)->ref_or_missing(y->close_to(xo,y,T,N),xr)))
mul_inplace(A,R,T,N,r)=(B=rnd(r,T,size(R,2),ncols(R)); C=zeros(T,size(R,1),size(B,2)); D=similar(C);
    (()->O.mul!(C,A,B), ()->LA.mul!(D,R,B), (xo,xr)->ref_or_missing(y->close_to(xo,y,T,N),xr)))
mulvec_inplace(A,R,T,N,r)=(x=rnd(r,T,size(R,2)); y=zeros(T,size(R,1)); z=similar(y);
    (()->O.mul!(y,A,x), ()->LA.mul!(z,R,x), (xo,xr)->ref_or_missing(v->close_to(xo,v,T,N),xr)))
spmm(A,R,T,N,r)=(()->O.mul(A,A), ()->R*R, (xo,xr)->ref_or_missing(y->close_to(xo,y,T,N),xr))
gemm_ws(A,R,T,N,r)=(B=rnd(r,T,N,N); C=zeros(T,N,N); D=similar(C); w=GemmWorkspace(T,N,N,N);
    (()->gemm!(C,A,B;workspace=w), ()->LA.mul!(D,R,B), (xo,xr)->ref_or_missing(y->close_to(xo,y,T,N),xr)))
function syrk_case(A,R,T,N,r)
    As=(T===Float16 && N==256) ? A./T(N) : A
    Rs=(T===Float16 && N==256) ? R./T(N) : R
    C=zeros(T,N,N); D=zeros(T,N,N)
    (()->syrk!(C,As), ()->LA.mul!(D,Rs,LA.transpose(Rs)),
     (xo,xr)->ref_or_missing(y->close_to(LA.triu(xo),LA.triu(y),T,N),xr))
end
trsm_case(A,R,T,N,r)=(B=rnd(r,T,N,N); X=similar(B); Y=similar(B); U=parent(A);
    (()->(copyto!(X,B); trsm!(X,U)), ()->(copyto!(Y,B); LA.ldiv!(R,Y)), (xo,xr)->solves(R,xo,B,T,N)))

# ---- factorizations -------------------------------------------------------
factor(f,g)=(A,R,T,N,r)->(b=rhs(r,T,R); (()->f(A), ()->g(R), (Fo,Fr)->solves(R,Fo\b,b,T,N)))
function factorize_case(A,R,T,N,r)
    b=rhs(r,T,R)
    # LinearAlgebra.factorize is a no-op identity (F === R) on triangular, diagonal, and bidiagonal.
    is_noop = R isa Union{LA.UpperTriangular, LA.LowerTriangular, LA.Diagonal, LA.Bidiagonal}
    g = is_noop ? nothing : () -> LA.factorize(R)
    (()->O.factorize(A), g, (Fo,_)->solves(R,Fo\b,b,T,N))
end
lu_inplace(A,R,T,N,r)=(b=rhs(r,T,R); W=similar(A); W2=similar(R);
    (()->(copyto!(W,A); O.lu!(W)), ()->(copyto!(W2,R); LA.lu!(W2)), (Fo,Fr)->solves(R,Fo\b,b,T,N)))
cholesky_inplace(A,R,T,N,r)=(b=rhs(r,T,R); P=parent(A); W=similar(P); W2=similar(P);
    (()->(copyto!(W,P); O.cholesky!(LA.Hermitian(W))), ()->(copyto!(W2,P); LA.cholesky!(LA.Hermitian(W2))),
     (Fo,Fr)->solves(R,Fo\b,b,T,N)))
qr_inplace(A,R,T,N,r)=(b=rhs(r,T,R); W=similar(A); W2=similar(R);
    (()->(copyto!(W,A); O.qr!(W)), ()->(copyto!(W2,R); LA.qr!(W2)), (Fo,Fr)->solves(R,Fo\b,b,T,N)))
# Sparse refactorization with the same pattern (symbolic analysis reused).
lu_refactor(A,R,T,N,r)=(b=rhs(r,T,R); Fo=O.lu(A); Fr=try LA.lu(R) catch; nothing end;
    (()->O.lu!(Fo,A), Fr===nothing ? nothing : ()->LA.lu!(Fr,R), (Fo2,_)->solves(R,Fo2\b,b,T,N)))
cholesky_refactor(A,R,T,N,r)=(b=rhs(r,T,R); Fo=O.cholesky(A); Fr=try LA.cholesky(R) catch; nothing end;
    (()->O.cholesky!(Fo,A), Fr===nothing ? nothing : ()->LA.cholesky!(Fr,R), (Fo2,_)->solves(R,Fo2\b,b,T,N)))

# ---- solves ---------------------------------------------------------------
ldiv_alloc(k)=(A,R,T,N,r)->(b=rhs(r,T,R,k); (()->O.ldiv(A,b), ()->R\b, (xo,xr)->solves(R,xo,b,T,N)))
backslash(A,R,T,N,r)=(b=rhs(r,T,R); (()->A\b, ()->R\b, (xo,xr)->solves(R,xo,b,T,N)))
function ldiv_inplace(A,R,T,N,r)
    b=rhs(r,T,R); Fo=O.factorize(A)
    skip_ref = (R isa AbstractSparseMatrix || R isa LA.Hermitian{<:Any,<:AbstractSparseMatrix} || R isa BandedMatrix) && !(T in (Float64, ComplexF64))
    Fr = skip_ref ? nothing : (try LA.factorize(R) catch; nothing end)
    y=zeros(T,size(R,2)); z=similar(y)
    (()->O.ldiv!(y,Fo,b), Fr===nothing ? nothing : ()->LA.ldiv!(z,Fr,b), (xo,xr)->solves(R,xo,b,T,N))
end

# ---- spectral -------------------------------------------------------------
svd_case(A,R,T,N,r)=(()->O.svd(A), ()->LA.svd(R),
    (F,_)->relerr(F.U*LA.Diagonal(F.S)*F.Vt,Matrix(R))<=rtol(T,N))
values_case(f,g)=(A,R,T,N,r)->(()->f(A), ()->g(R),
    (xo,xr)->ref_or_missing(y->close_to(sortc(xo),sortc(y),T,N),xr))
eigen_case(A,R,T,N,r)=(M=Matrix(R); (()->O.eigen(A), ()->LA.eigen(R),
    (F,_)->relerr(M*F.vectors,F.vectors*LA.Diagonal(F.values))<=rtol(T,N)))
schur_case(A,R,T,N,r)=(()->O.schur(A), ()->LA.schur(R),
    (F,_)->relerr(F.Z*F.T*F.Z',R)<=rtol(T,N))
hessenberg_case(A,R,T,N,r)=(M=Matrix(R); (()->O.hessenberg(A), ()->LA.hessenberg(R),
    (F,_)->(Q=Matrix(F.Q); relerr(Q*Matrix(F.H)*Q',M)<=rtol(T,N))))
function pencil_B(R,T,r)
    R isa LA.Hermitian ? spd_dense(r,T,size(R,1)) : rnd(r,T,size(R)...)+T(size(R,1))*LA.I
end
eigen_pencil(A,R,T,N,r)=(B=pencil_B(R,T,r); M=Matrix(R); Bm=Matrix(B);
    (()->O.eigen(A,B), ()->LA.eigen(R,B),
     (F,_)->relerr(M*F.vectors,Bm*F.vectors*LA.Diagonal(F.values))<=rtol(T,N)))
schur_pencil(A,R,T,N,r)=(B=pencil_B(R,T,r); (()->O.schur(A,B), ()->LA.schur(R,B),
    (F,_)->relerr(F.Q*F.S*F.Z',R)<=rtol(T,N)))

# ---- functions of a matrix ------------------------------------------------
scalar_case(f,g)=(A,R,T,N,r)->(()->f(A), ()->g(R), (xo,xr)->ref_or_missing(y->close_to(xo,y,T,N),xr))
function norm_case(A,R,T,N,r)
    if R isa LA.Tridiagonal
        # LinearAlgebra has no specialized norm for Tridiagonal (falls back to generic O(n²)).
        # Validate mathematical accuracy via sparse(R), but do not time a surrogate reference.
        return (()->O.norm(A), nothing, (xo,_)->close_to(xo, LA.norm(sparse(R)), T, N))
    end
    scalar_case(O.norm, LA.norm)(A, R, T, N, r)
end
scale_mat(A::LA.Hermitian,c)=LA.Hermitian(scale_mat(parent(A),c),Symbol(A.uplo))
scale_mat(A::LA.UpperTriangular,c)=LA.UpperTriangular(parent(A)*c)
scale_mat(A::LA.Diagonal,c)=LA.Diagonal(A.diag.*c)
scale_mat(A::LA.Bidiagonal,c)=LA.Bidiagonal(A.dv.*c,A.ev.*c,Symbol(A.uplo))
scale_mat(A::LA.Tridiagonal,c)=LA.Tridiagonal(A.dl.*c,A.d.*c,A.du.*c)
scale_mat(A::LA.SymTridiagonal,c)=LA.SymTridiagonal(A.dv.*c,A.ev.*c)
scale_mat(A::BandedMatrix,c)=BandedMatrix(A.data.*c,A.m,A.kl,A.ku,A.kind,A.upper,A.unit)
scale_mat(A::CSRMatrix,c)=CSRMatrix(A.m,A.n,copy(A.rowptr),copy(A.colind),A.values.*c)
scale_mat(A::AbstractMatrix,c)=A*c
# LinearAlgebra.det / logabsdet on Hermitian{SparseMatrixCSC} falls back to
# _factorize -> isdiag (O(N²)) -> dense bunchkaufman; use cholesky(R) for the
# SPD reference and record it explicitly in ref_input/note (see run.jl).
ref_det(R)=R isa LA.Hermitian{<:Any,<:SparseMatrixCSC} ? LA.det(LA.cholesky(R)) : LA.det(R)
ref_logabsdet(R)=R isa LA.Hermitian{<:Any,<:SparseMatrixCSC} ? LA.logabsdet(LA.cholesky(R)) : LA.logabsdet(R)
function det_case(A,R,T,N,r)
    L=Float64(O.logabsdet(A)[1])
    c=isfinite(L) ? inv(real(T)(exp(L/N))) : one(real(T))
    As=scale_mat(A,c); Rs=scale_mat(R,c)
    (()->O.det(As), ()->ref_det(Rs), (xo,xr)->ref_or_missing(y->close_to(xo,y,T,N),xr))
end
logabsdet_case(A,R,T,N,r)=(()->O.logabsdet(A), ()->ref_logabsdet(R),
    (xo,xr)->ref_or_missing(y->abs(xo[1]-y[1])<=rtol(T,N)*max(1,abs(y[1])) && isapprox(xo[2],y[2];rtol=0.01),xr))
function inv_case(A,R,T,N,r)
    (A isa LA.Bidiagonal || A isa LA.Tridiagonal) && N > 2048 && throw(ArgumentError("refuse to allocate dense inverse for size $N"))
    ((()->O.inv(A), ()->LA.inv(R), (X,_)->fnorm(R*X-LA.I)<=rtol(T,N)*fnorm(R)*fnorm(X)))
end
matfun(f,g)=(A,R,T,N,r)->(()->f(A), ()->g(R), (xo,xr)->ref_or_missing(y->close_to(xo,y,T,N),xr))
sqrt_case(A,R,T,N,r)=(M=Matrix(R); (()->O.sqrt(A), ()->sqrt(R), (X,_)->relerr(Matrix(X)*Matrix(X),M)<=rtol(T,N)))
rank_case(A,R,T,N,r)=(()->O.rank(A), ()->LA.rank(R), (xo,xr)->ref_or_missing(y->xo==y,xr))
pinv_case(A,R,T,N,r)=(()->O.pinv(A), ()->LA.pinv(R), (X,_)->relerr(R*X*R,R)<=rtol(T,N))
kron_case(A,R,T,N,r)=(B=rnd(r,T,N,N); (()->O.kron(A,B), ()->LA.kron(R,B), (xo,xr)->ref_or_missing(y->close_to(xo,y,T,N),xr)))
sylvester_case(A,R,T,N,r)=(B=rnd(r,T,N,N)+T(N)*LA.I; C=rnd(r,T,N,N);
    (()->O.sylvester(A,B,C), ()->LA.sylvester(R,B,C), (X,_)->relerr(R*X+X*B,-C)<=rtol(T,N)))
lyap_case(A,R,T,N,r)=(C0=rnd(r,T,N,N); C=C0+C0';
    (()->O.lyap(A,C), ()->LA.lyap(R,C), (X,_)->relerr(R*X+X*R',-C)<=rtol(T,N)))

# ---- tensors (operands built here; the input kind is :tensor) -------------
# TensorOperations is the reference; its macros need literal index names.
function einsum3(A,R,T,d,r)
    X=rnd(r,T,d,d,d); Y=rnd(r,T,d,d)
    (()->einsum("abc,cd->abd",X,Y), ()->(@tensor Z[a,b,e]:=X[a,b,c]*Y[c,e]), (xo,xr)->ref_or_missing(y->close_to(xo,y,T,d),xr))
end
function einsum3_inplace(A,R,T,d,r)
    X=rnd(r,T,d,d,d); Y=rnd(r,T,d,d); C=zeros(T,d,d,d); D=zeros(T,d,d,d)
    P=plan_einsum("abc,cd->abd",X,Y); ws=EinsumWorkspace(T)
    (()->einsum!(C,P,X,Y;workspace=ws), ()->(@tensor D[a,b,e]=X[a,b,c]*Y[c,e]), (xo,xr)->ref_or_missing(y->close_to(xo,y,T,d),xr))
end
function einsum4(A,R,T,d,r)
    X=rnd(r,T,d,d,d,d); Y=rnd(r,T,d,d,d,d)
    (()->einsum("abcd,ebfd->aecf",X,Y), ()->(@tensor Z[a,e,c,f]:=X[a,b,c,d]*Y[e,b,f,d]), (xo,xr)->ref_or_missing(y->close_to(xo,y,T,d),xr))
end

# ---- the grid -------------------------------------------------------------
const SOLVE_KINDS=[:dense,:tall,:hermitian,:indefinite,:upper,:lower,:diagonal,:bidiagonal,:tridiagonal,:symtridiagonal,
                   :banded,:banded_spd,:packed,:sparse,:csr,:csc,:coo,:sparse_spd,:csr_spd,:csc_spd]
const DET_KINDS=[:dense,:hermitian,:upper,:diagonal,:bidiagonal,:tridiagonal,:symtridiagonal,:banded,:sparse,:csr,:sparse_spd]

const OPS=[
    # products
    Op("mul","products","allocating",O.mul,PRODUCT_KINDS,:cubic,mul_alloc),
    Op("mul!","products","in-place",O.mul!,PRODUCT_KINDS,:cubic,mul_inplace),
    Op("mulvec!","products","in-place",O.mul!,PRODUCT_KINDS,:cubic,mulvec_inplace),
    Op("spmm","products","allocating",O.mul,[:sparse,:csr,:csc],:cubic,spmm),
    Op("gemm!","products","in-place",gemm!,[:dense],:cubic,gemm_ws),
    Op("syrk!","products","in-place",syrk!,[:dense],:cubic,syrk_case),
    Op("trsm!","products","in-place",trsm!,[:upper],:cubic,trsm_case),
    # factorizations
    Op("lu","factorizations","allocating",O.lu,[:dense,:tridiagonal,:banded,:sparse,:csr,:csc,:coo],:cubic,factor(O.lu,LA.lu)),
    Op("lu!","factorizations","in-place",O.lu!,[:dense],:cubic,lu_inplace),
    Op("lu!refactor","factorizations","in-place",O.lu!,[:sparse,:csr,:csc],:cubic,lu_refactor),
    Op("cholesky","factorizations","allocating",O.cholesky,[:hermitian,:packed,:symtridiagonal,:banded_spd,:sparse_spd,:csr_spd,:csc_spd],:cubic,factor(O.cholesky,LA.cholesky)),
    Op("cholesky!","factorizations","in-place",O.cholesky!,[:hermitian],:cubic,cholesky_inplace),
    Op("cholesky!refactor","factorizations","in-place",O.cholesky!,[:sparse_spd,:csr_spd],:cubic,cholesky_refactor),
    Op("ldlt","factorizations","allocating",O.ldlt,[:hermitian,:symtridiagonal,:banded_spd,:sparse_spd,:csr_spd],:cubic,factor(O.ldlt,LA.ldlt)),
    Op("bunchkaufman","factorizations","allocating",O.bunchkaufman,[:indefinite,:hermitian],:cubic,factor(O.bunchkaufman,LA.bunchkaufman)),
    Op("qr","factorizations","allocating",O.qr,[:dense,:tall,:sparse,:csr,:csc],:cubic,factor(O.qr,LA.qr)),
    Op("qr_colnorm","factorizations","allocating",O.qr,[:dense,:tall],:cubic,factor(A->O.qr(A,LA.ColumnNorm()),R->LA.qr(R,LA.ColumnNorm()))),
    Op("qr!","factorizations","in-place",O.qr!,[:dense],:cubic,qr_inplace),
    Op("lq","factorizations","allocating",O.lq,[:dense],:cubic,factor(O.lq,LA.lq)),
    Op("factorize","factorizations","allocating",O.factorize,SOLVE_KINDS,:cubic,factorize_case),
    # solves
    Op("ldiv","solves","allocating",O.ldiv,SOLVE_KINDS,:cubic,ldiv_alloc(0)),
    Op("ldiv_8rhs","solves","allocating",O.ldiv,[:dense,:hermitian,:tridiagonal,:banded,:packed,:sparse,:csr,:sparse_spd],:cubic,ldiv_alloc(8)),
    Op("ldiv!","solves","in-place",O.ldiv!,SOLVE_KINDS,:cubic,ldiv_inplace),
    Op("backslash","solves","allocating",(\),[:csr,:csc,:coo,:csr_spd],:cubic,backslash),
    # spectral
    Op("svd","spectral","allocating",O.svd,[:dense,:tall,:bidiagonal],:spectral,svd_case),
    Op("svdvals","spectral","allocating",O.svdvals,[:dense,:tall,:bidiagonal,:upper],:spectral,values_case(O.svdvals,LA.svdvals)),
    Op("eigen","spectral","allocating",O.eigen,[:hermitian,:dense,:symtridiagonal],:spectral,eigen_case),
    Op("eigvals","spectral","allocating",O.eigvals,[:hermitian,:dense,:symtridiagonal,:upper],:spectral,values_case(O.eigvals,LA.eigvals)),
    Op("eigen_pencil","spectral","allocating",O.eigen,[:hermitian,:dense],:spectral,eigen_pencil),
    Op("schur","spectral","allocating",O.schur,[:dense],:spectral,schur_case),
    Op("schur_pencil","spectral","allocating",O.schur,[:dense],:spectral,schur_pencil),
    Op("hessenberg","spectral","allocating",O.hessenberg,[:dense,:hermitian],:spectral,hessenberg_case),
    # matrix functions and scalars
    Op("det","functions","allocating",O.det,DET_KINDS,:cubic,det_case),
    Op("logabsdet","functions","allocating",O.logabsdet,DET_KINDS,:cubic,logabsdet_case),
    Op("inv","functions","allocating",O.inv,[:dense,:hermitian,:upper,:lower,:diagonal,:bidiagonal,:tridiagonal],:cubic,inv_case),
    Op("norm","functions","allocating",O.norm,[:dense,:hermitian,:tridiagonal,:sparse,:csr],:cubic,norm_case),
    Op("opnorm1","functions","allocating",O.opnorm,[:dense,:hermitian,:tridiagonal,:sparse,:csr],:cubic,scalar_case(A->O.opnorm(A,1),R->LA.opnorm(R,1))),
    Op("opnorm2","functions","allocating",O.opnorm,[:dense],:spectral,scalar_case(O.opnorm,LA.opnorm)),
    Op("cond","functions","allocating",O.cond,[:dense],:spectral,scalar_case(O.cond,LA.cond)),
    Op("rank","functions","allocating",O.rank,[:dense,:tall],:spectral,rank_case),
    Op("pinv","functions","allocating",O.pinv,[:dense,:tall],:spectral,pinv_case),
    Op("exp","functions","allocating",O.exp,[:dense_unit,:hermitian],:spectral,matfun(O.exp,exp)),
    Op("log","functions","allocating",O.log,[:hermitian],:spectral,matfun(O.log,log)),
    Op("sqrt","functions","allocating",O.sqrt,[:hermitian],:spectral,sqrt_case),
    Op("sin","functions","allocating",O.sin,[:dense_unit],:spectral,matfun(O.sin,sin)),
    Op("cos","functions","allocating",O.cos,[:dense_unit],:spectral,matfun(O.cos,cos)),
    Op("kron","functions","allocating",O.kron,[:dense],:kron,kron_case),
    Op("sylvester","functions","allocating",O.sylvester,[:dense],:spectral,sylvester_case),
    Op("lyap","functions","allocating",O.lyap,[:dense],:spectral,lyap_case),
    # tensors
    Op("einsum_abc_cd","tensors","allocating",einsum,[:tensor],:tensor,einsum3),
    Op("einsum!_abc_cd","tensors","in-place",einsum!,[:tensor],:tensor,einsum3_inplace),
    Op("einsum_abcd_ebfd","tensors","allocating",einsum,[:tensor],:tensor4,einsum4),
]
