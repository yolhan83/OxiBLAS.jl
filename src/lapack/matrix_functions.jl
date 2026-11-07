# SPDX-License-Identifier: Apache-2.0
function _exponent_magnitude(p::Integer)
    isbitstype(typeof(p)) || throw(ArgumentError("a fixed-width integer exponent is required"))
    p<0 ? unsigned(-(p+one(p)))+one(unsigned(p)) : unsigned(p)
end
function _mm(A::AbstractMatrix{T},B::AbstractMatrix{T};threaded=true) where T
    C=Matrix{T}(undef,size(A,1),size(B,2)); gemm!(C,A,B;threaded); C
end
function _right_solve(A,B)
    Matrix(transpose(solve(lu_factor(transpose(B)),Matrix(transpose(A)))))
end

"""
    matrix_exp(A::AbstractMatrix{T}; threaded=true) -> Matrix{T}

Matrix exponential by degree-13 Padé scaling and squaring (Higham).
All products and the rational solve use the native Julia kernels.
"""
function matrix_exp(A::AbstractMatrix{T};threaded=true) where {T<:NativeFloat}
    n=_square(A); _finite_scale(A); R=real(T); norm=matrix_norm(A,:one)
    isfinite(norm) || throw(OverflowError("matrix 1-norm is not representable"))
    theta=R(5.371920351148152)
    s=norm<=theta ? 0 : ceil(Int,log2(norm/theta))
    B=Matrix(A).*ldexp(one(R),-s); Iden=_identity(T,n)
    B2=_mm(B,B;threaded); B4=_mm(B2,B2;threaded); B6=_mm(B4,B2;threaded)
    b=R.((64764752532480000,32382376266240000,7771770303897600,1187353796428800,
          129060195264000,10559470521600,670442572800,33522128640,1323241920,
          40840800,960960,16380,182,1))
    U=_mm(B,_mm(B6,b[14].*B6+b[12].*B4+b[10].*B2;threaded)+b[8].*B6+b[6].*B4+b[4].*B2+b[2].*Iden;threaded)
    V=_mm(B6,b[13].*B6+b[11].*B4+b[9].*B2;threaded)+b[7].*B6+b[5].*B4+b[3].*B2+b[1].*Iden
    F=solve(lu_factor(V-U;threaded),V+U)
    for _ in 1:s
        F=_mm(F,F;threaded)
    end
    F
end

function _triangular_sqrt(T)
    n=size(T,1); R=zeros(eltype(T),n,n)
    for i in 1:n
        R[i,i]=sqrt(T[i,i])
    end
    for gap in 1:n-1, i in 1:n-gap
        j=i+gap; s=T[i,j]
        for k in i+1:j-1
            s-=R[i,k]*R[k,j]
        end
        den=R[i,i]+R[j,j]
        iszero(den) && !iszero(s) && throw(DomainError(T,"no compatible principal triangular square root"))
        R[i,j]=iszero(den) ? zero(eltype(T)) : s/den
    end
    R
end
"""
    matrix_sqrt(A::AbstractMatrix{T}; threaded=true) -> Matrix

Principal matrix square root through complex Schur and triangular recurrence.
The result is complex even for real input; negative eigenvalues are supported.
"""
function matrix_sqrt(A::AbstractMatrix{T};threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    _square(A); F=schur_factor(Complex{real(T)}.(A)); R=_triangular_sqrt(F.T)
    _mm(_mm(F.Z,R;threaded),adjoint(F.Z);threaded)
end
"""
    matrix_log(A::AbstractMatrix{T}; maxiter=200, rtol=8eps(real(T)), threaded=true) -> Matrix

Principal matrix logarithm by complex Schur, inverse scaling/squaring and
an atanh series. Singular input is rejected. A finite iteration limit is enforced.
"""
function matrix_log(A::AbstractMatrix{T};maxiter=200,rtol=8eps(real(T)),threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    n=_square(A); maxiter>=1 && 0<rtol<1 || throw(ArgumentError("invalid matrix logarithm options"))
    C=Complex{real(T)}; F=schur_factor(C.(A)); B=copy(F.T); Iden=_identity(C,n)
    # F.T, not B: B is reassigned below and the generator captured it.
    any(iszero(F.T[i,i]) for i in 1:n) && throw(DomainError(A,"matrix logarithm requires nonsingular input"))
    scaling=0
    while matrix_norm(B-Iden,:one)>real(T)(.5)
        scaling>=maxiter && throw(ConvergenceError(:logarithm_scaling,scaling,1:n))
        B=_triangular_sqrt(B); scaling+=1
    end
    X=_right_solve(B-Iden,B+Iden); X2=_mm(X,X;threaded); term=copy(X); result=copy(X); converged=iszero(matrix_norm(term,:one))
    spare=similar(term)   # term and spare swap roles: no n×n allocation per term
    for k in 1:maxiter
        gemm!(spare,term,X2;threaded); term,spare=spare,term
        c=2k+1; result .+= term./c          # fused: no temporary
        if matrix_norm(term,:one)/c<=rtol*max(floatmin(real(T)),matrix_norm(result,:one))
            converged=true; break
        end
    end
    converged || throw(ConvergenceError(:logarithm_series,maxiter,1:n))
    result .*= ldexp(real(T)(2),scaling)
    _mm(_mm(F.Z,result;threaded),adjoint(F.Z);threaded)
end
function matrix_power(A::AbstractMatrix{T},p::Integer;threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    n=_square(A); B=p<0 ? native_inverse(A) : Matrix(A); q=_exponent_magnitude(p)
    result=_identity(T,n)
    while q>0
        isodd(q) && (result=_mm(result,B;threaded))
        q>>=1
        q>0 && (B=_mm(B,B;threaded))
    end
    result
end
function matrix_power(A::AbstractMatrix{T},p::Number;threaded=true,kwargs...) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    matrix_exp(p.*matrix_log(A;threaded,kwargs...);threaded)
end
"""
    matrix_sign(A::AbstractMatrix{T}; threaded=true) -> Matrix

Matrix sign via complex Schur. Eigenvalues on the imaginary axis are excluded.
"""
function matrix_sign(A::AbstractMatrix{T};threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    n=_square(A); F=schur_factor(Complex{real(T)}.(A)); C=eltype(F.T); S=zeros(C,n,n)
    for i in 1:n
        iszero(real(F.T[i,i])) && throw(DomainError(A,"matrix sign undefined on the imaginary axis"))
        S[i,i]=real(F.T[i,i])>0 ? one(C) : -one(C)
    end
    for gap in 1:n-1, i in 1:n-gap
        j=i+gap
        if S[i,i]==S[j,j]
            s=zero(C)
            for k in i+1:j-1
                s+=S[i,k]*S[k,j]
            end
            S[i,j]=-s/(S[i,i]+S[j,j])
        else
            s=(S[i,i]-S[j,j])*F.T[i,j]
            for k in i+1:j-1
                s+=S[i,k]*F.T[k,j]-F.T[i,k]*S[k,j]
            end
            S[i,j]=s/(F.T[i,i]-F.T[j,j])
        end
    end
    _mm(_mm(F.Z,S;threaded),adjoint(F.Z);threaded)
end
function matrix_sin(A::AbstractMatrix{T};kwargs...) where {T<:GenericFloat}
    C=Complex{real(T)}; B=im.*C.(A)
    (matrix_exp(B;kwargs...)-matrix_exp(-B;kwargs...))./(2im)
end
function matrix_cos(A::AbstractMatrix{T};kwargs...) where {T<:GenericFloat}
    C=Complex{real(T)}; B=im.*C.(A)
    (matrix_exp(B;kwargs...)+matrix_exp(-B;kwargs...))./2
end
"""
    frechet_matrix(f, A::AbstractMatrix{T}, E::AbstractMatrix{T}; kwargs...) -> NamedTuple

Fréchet derivative L_f(A,E) from the upper block of f([A E;0 A]).
"""
function frechet_matrix(f,A::AbstractMatrix{T},E::AbstractMatrix{T};kwargs...) where {T<:GenericFloat}
    n=_square(A); Base.require_one_based_indexing(E)
    size(E)==size(A) || throw(DimensionMismatch("Frechet direction dimensions differ"))
    B=zeros(T,2n,2n); B[1:n,1:n]=A; B[n+1:2n,n+1:2n]=A; B[1:n,n+1:2n]=E
    F=f(B;kwargs...)
    (;value=F[1:n,1:n],derivative=F[1:n,n+1:2n])
end
frechet_exp(A,E;kwargs...)=frechet_matrix(matrix_exp,A,E;kwargs...)
frechet_log(A,E;kwargs...)=frechet_matrix(matrix_log,A,E;kwargs...)
frechet_sqrt(A,E;kwargs...)=frechet_matrix(matrix_sqrt,A,E;kwargs...)
