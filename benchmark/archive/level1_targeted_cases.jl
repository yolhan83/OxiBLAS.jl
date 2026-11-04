function generation_error(F,d1,d2,x,y)
    H=BigFloat.(rotm_matrix(F.params)); a,b=BigFloat(F.d1),BigFloat(F.d2)
    p,q=BigFloat(x),BigFloat(y)
    r=abs(H[2,1]*p+H[2,2]*q)/max(abs(H[2,1]*p),abs(H[2,2]*q))
    s=abs(H[1,1]*p+H[1,2]*q-BigFloat(F.x1))/max(abs(H[1,1]*p),abs(H[1,2]*q),abs(BigFloat(F.x1)))
    c1=abs(a*H[1,1]^2+b*H[2,1]^2-BigFloat(d1))/abs(BigFloat(d1))
    c2=abs(a*H[1,2]^2+b*H[2,2]^2-BigFloat(d2))/abs(BigFloat(d2))
    Float64(max(r,s,c1,c2))
end

# BLAS 1 only: vector storage, nontrivial inputs, independent untimed oracles.
const L1_OPERATIONS=(:axpy,:scal,:copy,:swap,:dotu,:dotc,:dot_kahan,:dot_pairwise,
    :dot_extended,:nrm2,:asum,:iamax,:iamin,:rot,:rotg,:rotm,:rotmg,:rotmg_inplace,
    :dsdot,:sdsdot,:axpy_parallel,:scal_parallel,:dot_parallel,:nrm2_parallel,:asum_parallel)
function l1_input(::Type{T},n) where T
    R=real(T)
    x=T[R(mod(17i+3,61)-30)/32 for i in 1:n]
    y=T[R(mod(13i+7,59)-29)/32 for i in 1:n]
    if T<:Complex
        x .+= T[complex(0,R(mod(11i+1,53)-26)/32) for i in 1:n]
        y .+= T[complex(0,R(mod(7i+5,47)-23)/32) for i in 1:n]
    end
    (;x,y)
end
# Batch copies share l1_input(T,n): their BigFloat oracles are computed once.
const L1_ORACLES=Dict{Any,Any}()
l1_oracle(f,key...)=get!(f,L1_ORACLES,key)
function l1_case(op,::Type{T},n,data;reference=false) where T
    x,y=data.x,data.y; a=copy(x);b=copy(y);R=real(T)
    setup=()->(copyto!(a,x);copyto!(b,y))
    symbols=String[]; note="vector inputs; evals=1; reset outside timing";f=nothing
    # Cancellation-aware absolute error; oracle arithmetic independent of backend.
    hi(z)=BigFloat(real(z))+im*BigFloat(imag(z))
    close(z,w)=all(isfinite,z) && maximum(abs.(z.-w);init=0)<=64eps(R)*max(1,maximum(abs,w;init=0))
    verify=z->true
    prefix(s)=full_symbol(T,s)
    base=op in (:axpy_parallel,:scal_parallel,:dot_parallel,:nrm2_parallel,:asum_parallel) ? Symbol(first(split(string(op),"_parallel"))) : op
    if base==:axpy
        axpy_alpha=T(.625);axpy_g=op==:axpy_parallel ? axpy_parallel! : axpy!
        f=reference ? ()->LA.BLAS.axpy!(axpy_alpha,x,b) : ()->axpy_g(b,x,axpy_alpha)
        verify=_ ->close(b,y+axpy_alpha*x);symbols=[prefix("axpy")]
    elseif base==:scal
        scal_alpha=T(.625);scal_g=op==:scal_parallel ? scal_parallel! : scal!
        f=reference ? ()->LA.BLAS.scal!(scal_alpha,a) : ()->scal_g(a,scal_alpha)
        verify=_ ->close(a,scal_alpha*x);symbols=[prefix("scal")]
    elseif base==:copy
        f=reference ? ()->LA.BLAS.blascopy!(n,x,1,b,1) : ()->copy_vector!(b,x)
        verify=_ ->b==x;symbols=[prefix("copy")]
    elseif base==:swap
        f=reference ? ()->l1_reference_swap!(a,b) : ()->swap!(a,b)
        verify=_ ->a==y && b==x;symbols=[prefix("swap")]
    elseif base in (:dotu,:dotc,:dot,:dot_kahan,:dot_pairwise,:dot_extended,:dsdot,:sdsdot)
        conjugate=base==:dotc
        expected,scale=l1_oracle(:dot,conjugate,T,n) do
            (sum((conjugate ? conj(hi(x[i])) : hi(x[i]))*hi(y[i]) for i in 1:n),
             sum(abs(hi(x[i]))*abs(hi(y[i])) for i in 1:n))
        end
        if base in (:dsdot,:sdsdot)
            if T===Float32
                f=base==:dsdot ? (reference ? ()->reference_dsdot(x,y) : ()->dsdot(x,y)) :
                    (reference ? ()->reference_sdsdot(.375f0,x,y) : ()->sdsdot(.375f0,x,y))
                base==:sdsdot && (expected+=BigFloat(.375f0))
                symbols=[string(base,"_")]
            end
        elseif reference
            if base in (:dotu,:dotc,:dot)
                f=T<:Complex ? (conjugate ? ()->LA.BLAS.dotc(x,y) : ()->LA.BLAS.dotu(x,y)) : ()->LA.BLAS.dot(x,y)
                symbols=[prefix(T<:Complex ? (conjugate ? "dotc" : "dotu") : "dot")]
            end
        else
            dot_g=op==:dot_parallel ? dot_parallel : getfield(OxiBLAS,base)
            f=()->dot_g(x,y)
        end
        precision=base==:dsdot ? eps(Float64) : eps(R)
        verify=z->isfinite(z) && abs(hi(z)-expected)<=64precision*max(1,scale)
        base in (:dot_kahan,:dot_pairwise,:dot_extended) && (note*="; distinct accumulation contract: no ordinary-BLAS ratio")
    elseif base in (:nrm2,:asum,:iamax,:iamin)
        mags=abs.(real.(x)).+abs.(imag.(x))
        expected=l1_oracle(base,T,n) do
            base==:nrm2 ? sqrt(sum(abs2(hi(v)) for v in x)) : base==:asum ? sum(BigFloat.(mags)) : base==:iamax ? argmax(mags) : argmin(mags)
        end
        if reference
            f=base==:nrm2 ? ()->LA.BLAS.nrm2(x) : base==:asum ? ()->LA.BLAS.asum(x) : base==:iamax ? ()->LA.BLAS.iamax(x) : nothing
            symbols=[base==:iamax ? string(T===Float32 ? "is" : T===Float64 ? "id" : T===ComplexF32 ? "ic" : "iz","amax_") : prefix(string(base))]
        else
            norm_g=getfield(OxiBLAS,op);f=()->norm_g(x)
        end
        verify=z->base in (:iamax,:iamin) ? z==expected : isfinite(z) && abs(BigFloat(z)-expected)<=64eps(R)*max(1,expected)
    elseif base==:rot
        c=R(.6);s=T(.8)
        f=reference ? ()->LA.BLAS.rot!(n,a,1,b,1,c,s) : ()->rot!(a,b,c,s)
        verify=_ ->close(a,c*x+s*y) && close(b,c*y-conj(s)*x)
        symbols=[prefix("rot")]
    elseif base==:rotm && T<:Real
        raw=T[-1,.7,-.3,.4,1.2];params=RotmParams(raw)
        f=reference ? ()->reference_rotm!(a,b,raw) : ()->rotm!(a,b,params)
        verify=_ ->close(a,.7x+.4y) && close(b,-.3x+1.2y);symbols=[prefix("rotm")]
    elseif base in (:rotg,:rotmg,:rotmg_inplace)
        # Scalar generators are batched with varying, mutable inputs. No
        # literal scalar closure can be constant-folded into a fake picosecond.
        if base==:rotg
            rotg_out=Vector{Tuple{R,T,T}}(undef,n)
            rotg_g=reference ? l1_reference_rotg : rotg
            f=()->(for i in 1:n;rotg_out[i]=rotg_g(x[i],y[i]);end;rotg_out)
            symbols=[prefix("rotg")]
            verify=z->all(i->begin c,s,r=z[i];abs(c*x[i]+s*y[i]-r)<=64eps(R)*max(1,abs(r)) && abs(c*y[i]-conj(s)*x[i])<=64eps(R)*max(1,abs(r)) end,1:n)
        elseif T<:Real
            states=[T[1+abs(x[i]),1+abs(y[i]),2+x[i]] for i in 1:n];initial=copy.(states)
            out=Vector{typeof(rotmg(one(T),one(T),T(3),T(4)))}(undef,n)
            setup=()->foreach(i->copyto!(states[i],initial[i]),1:n)
            mg_g=reference ? reference_rotmg : rotmg
            f=base==:rotmg_inplace ? ()->(for i in 1:n
                if reference
                    result=mg_g(states[i][1],states[i][2],states[i][3],T(3)+y[i])
                    states[i][1]=result.d1;states[i][2]=result.d2;states[i][3]=result.x1;out[i]=result
                else
                    p=rotmg!(states[i],T(3)+y[i]);out[i]=(;d1=states[i][1],d2=states[i][2],x1=states[i][3],params=p)
                end
            end;out) : ()->(for i in 1:n;out[i]=mg_g(initial[i][1],initial[i][2],initial[i][3],T(3)+y[i]);end;out)
            verify=z->all(i->generation_error(z[i],initial[i]...,T(3)+y[i])< (T===Float32 ? 5e-5 : 5e-12),1:n)
            symbols=[prefix("rotmg")]
        end
        note*="; scalar-generator batch of N calls; result/state retained"
    end
    if f!==nothing && op in (:axpy,:scal,:copy,:swap,:rot,:rotm,:axpy_parallel,:scal_parallel)
        f=let call=f
            ()->(call();nothing)
        end
    end
    (;f,setup,verify,symbols,note)
end
for (T,symbol) in ((Float32,:sswap_),(Float64,:dswap_),(ComplexF32,:cswap_),(ComplexF64,:zswap_))
    @eval function l1_reference_swap!(x::Vector{$T},y::Vector{$T})
        ccall((LA.BLAS.@blasfunc($symbol),LA.BLAS.libblastrampoline),Cvoid,
            (Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt}),length(x),x,1,y,1)
        (x,y)
    end
end
for (T,R,symbol) in ((Float32,Float32,:srotg_),(Float64,Float64,:drotg_),(ComplexF32,Float32,:crotg_),(ComplexF64,Float64,:zrotg_))
    @eval function l1_reference_rotg(a::$T,b::$T)
        aa,bb,c,s=Ref(a),Ref(b),Ref{$R}(),Ref{$T}()
        ccall((LA.BLAS.@blasfunc($symbol),LA.BLAS.libblastrampoline),Cvoid,
            (Ref{$T},Ref{$T},Ref{$R},Ref{$T}),aa,bb,c,s)
        (c[],s[],aa[])
    end
end
