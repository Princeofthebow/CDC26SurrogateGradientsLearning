const F = Float64
abstract type AbstractCurrent end

function get_idx(objs,nᵥ=1)
    len = [length(obj) for obj in objs]
    cum_len = [0;cumsum(len)*nᵥ]
    vals = Tuple(Tuple(1+cum_len[n]+(k-1)*nᵥ:cum_len[n]+k*nᵥ for k=1:len[n]) for n in 1:length(objs))
    return NamedTuple{keys(objs)}(vals)
end
# [x[(k-1)*n+1:k*n] for k in idx]...
include("./nonlinearities.jl")

"""
    Gating variable dynamics
"""
struct Gating{G,L}
    τ::G
    σ::L
    p::Int
end

function f!(g::Gating,dx,v,x,idx)
    for i in 1:length(v)
        dx[idx[i]] = (-x[idx[i]] + g.σ(v[i])) / g.τ(v[i])
    end
end

function ∇f!(g::Gating,∇ᵥf,∇ₓf,v,x,idx)
    for i in 1:length(v)
        τ = g.τ(v[i])
        ∇ᵥf[idx[i]] = -(-x[idx[i]] + g.σ(v[i]))/τ^2 * d(g.τ,v[i]) + d(g.σ,v[i])/τ   
        ∇ₓf[idx[i]] = -1/τ
    end
end

function initialCondition(g::Gating,v)
    return g.σ.(v)
end

"""
    Voltage-gated current dynamics
"""
struct OhmicCurrent{X<:Union{Tuple,Nothing}} <: AbstractCurrent
    μ::Vector{F}
    ν::Vector{F}
    gates::X
end

# Gated currents
const GatedCurrent = OhmicCurrent{T} where T<:Tuple
Base.length(I::GatedCurrent) = length(I.gates)
GatedCurrent(μ::Real,ν::Real,gates::Tuple; n=1) = OhmicCurrent(fill(F(μ),n),fill(F(ν),n),gates)
initialCondition(I::GatedCurrent,v) = reduce(vcat,[initialCondition(I.gates[i],v) for i in 1:length(I.gates)]; init=F[])

function f!(I::GatedCurrent,dx,v,x,idx)
    for i in 1:length(I.gates)
        f!(I.gates[i],dx,v,x,idx[i])
    end
end

function ∇f!(I::GatedCurrent,∇ᵥf,∇ₓf,v,x,idx)
    for i in 1:length(I.gates)
        ∇f!(I.gates[i],∇ᵥf,∇ₓf,v,x,idx[i])
    end
end

function gating(I::GatedCurrent, x, idx, i)
    g = x[idx[1][i]]^I.gates[1].p
    for k in 2:length(I.gates)
        g *= x[idx[k][i]]^I.gates[k].p
    end
    return g
end

function ∇gating(I::GatedCurrent, x, idx, idx_gate, i)
    ∇g = I.gates[idx_gate].p*x[idx[idx_gate][i]]^(I.gates[idx_gate].p-1)
    for k in 1:length(I.gates)
        if k != idx_gate
            ∇g *= x[idx[k][i]].^I.gates[k].p
        end
    end
    return ∇g
end

# Allocating versions
function (I::GatedCurrent)(v::AbstractVector,x::AbstractVector,idx)
    current = similar(v)
    for i in eachindex(v)
        current[i] = I.μ[i] * (v[i] - I.ν[i]) * gating(I,x,idx,i)
    end
    return current
end
(I::GatedCurrent)(v::AbstractMatrix,x::AbstractMatrix,idx) = reduce(hcat,[I(v[:,i],x[:,i],idx) for i in 1:size(v,2)]) 

function ∇v(I::GatedCurrent, v::AbstractVector, x::AbstractVector, idx)
    ∇v = similar(v)
    for i in eachindex(v)
        ∇v[i] = I.μ[i] * gating(I,x,idx,i)
    end
    return ∇v
end
∇v(I::GatedCurrent,v::AbstractMatrix,x::AbstractMatrix,idx) = reduce(hcat,[∇v(I,v[:,i],x[:,i],idx) for i in 1:size(v,2)]) 

function gating(I::GatedCurrent,x::AbstractVector,idx)
    return [gating(I,x,idx,i) for i in eachindex(idx[1])]
end

# Non-allocating versions
function add!(I::GatedCurrent,Σcurrents,v,x,idx)
    for i in eachindex(v)
        Σcurrents[i] += I.μ[i] * (v[i] - I.ν[i]) * gating(I,x,idx,i)
    end
    return nothing
end

function add∇v!(I::GatedCurrent,∇v,v,x,idx)
    for i in eachindex(v)
        ∇v[i] += I.μ[i] * gating(I,x,idx,i)
    end
    return nothing
end

function ∇x!(I::GatedCurrent,∇x,v,x,idx)
    for k in 1:length(I.gates)
        for i in eachindex(v)
            ∇x[idx[k][i]] = I.μ[i] * ∇gating(I,x,idx,k,i) * (v[i] - I.ν[i])
        end
    end
    return nothing
end

∂μ(I::GatedCurrent,v,x,idx,i) = gating(I,x,idx,i) * (v[i] - I.ν[i])

# Leak current
const LeakCurrent = OhmicCurrent{Nothing}
Base.length(I::LeakCurrent) = 0
LeakCurrent(μ::Real,ν::Real; n=1) = OhmicCurrent(fill(F(μ),n),fill(F(ν),n),nothing)
initialCondition(I::LeakCurrent,v) = F[]

f!(I::LeakCurrent,dx,v,x,idx) = nothing
∇f!(I::LeakCurrent,∇ᵥf,∇ₓf,v,x,idx) = nothing

# Allocating version
(I::LeakCurrent)(v,x,idx) = I.μ .* (v .- I.ν)

# Non-allocating version
function add!(I::LeakCurrent, Σcurrents, v, x, idx)
    Σcurrents .+= I.μ .* (v .- I.ν)
    return nothing
end

# Non-allocating
function add∇v!(I::LeakCurrent,∇v,v,x,idx)
    ∇v .+= I.μ
    return nothing
end
∇x!(I::LeakCurrent,∇x,v,x,idx) = nothing
∂μ(I::LeakCurrent,v,x,idx,i) = v[i] - I.ν[i]

"""
    Parallel currents
"""
struct Currents{C<:NamedTuple,X<:NamedTuple}
    currents::C                    # Named tuple of GatedCurrents (or other current types)
    idx_x::X                       # Indices of intrinsic gating variable states
end

Base.length(c::Currents) = length(c.currents)
Base.getindex(c::Currents, curr::Symbol) = c.currents[curr]
length_x(c::Currents) = sum(sum(length(idx[i]) for i in 1:length(idx);init=0) for idx in c.idx_x)
length_μ(c::Currents) = reduce(+, length(c.currents[i].μ) for i in 1:length(c); init=0)

function Currents(currents::NamedTuple; n=1)
    return Currents(currents,get_idx(currents,n))
end

function f!(c::Currents,dx,v,x)
    map((curr,idx) -> f!(curr,dx,v,x,idx), c.currents, c.idx_x)
    return nothing
end

function ∇f!(c::Currents,∇ᵥf,∇ₓf,v,x)
    map((curr,idx) -> ∇f!(curr,∇ᵥf,∇ₓf,v,x,idx), c.currents, c.idx_x)
    return nothing
end

# Recover values and gradients of specific currents (allocating)
(c::Currents)(name::Symbol,v,x) = c.currents[name](v,x,c.idx_x[name])
∇v(c::Currents,name::Symbol,v,x) = ∇v(c.currents[name],v,x,c.idx_x[name])

# Sum total of currents (allocating)
function (c::Currents)(v,x)
    Σ = zeros(eltype(v), length(v))
    map((curr,idx) -> add!(curr,Σ,v,x,idx), c.currents, c.idx_x)
    return Σ
end

# Sum total of currents (non-allocating)
function Σ!(c::Currents,Σ,v,x)
    fill!(Σ,0.0)
    map((curr,idx) -> add!(curr,Σ,v,x,idx), values(c.currents), values(c.idx_x))
    return nothing
end

# Sum total of currents that also computes the sum of a subset of currents (non-allocating)
function Σ!(c::Currents,Σv,Σw,v,x,selection::Tuple)
    fill!(Σv,0.0)
    fill!(Σw,0.0)
    for ion in keys(c.currents)
        if ion ∈ selection
            add!(c.currents[ion], Σw, v, x, c.idx_x[ion])
        else
            add!(c.currents[ion], Σv, v, x, c.idx_x[ion])
        end
    end
    Σv .+= Σw
    return nothing
end

# Non-allocating
function ∇!(c::Currents,∇ᵥΣ,∇ₓΣ,v,x)
    fill!(∇ᵥΣ,0.0)
    map((curr,idx) -> add∇v!(curr,∇ᵥΣ,v,x,idx), values(c.currents), values(c.idx_x))
    map((curr,idx) -> ∇x!(curr,∇ₓΣ,v,x,idx), values(c.currents), values(c.idx_x))
    return nothing
end

# Non-allocating
function Jμ!(c::Currents,Jμ,v,x)
    # Access the matrix array in column-major order for performance
    @inbounds for j in 1:length(c.currents)
        @inbounds for i in eachindex(v)
            Jμ[i,j] = ∂μ(c.currents[j], v, x, c.idx_x[j], i)
        end
    end
end

function initialCondition(c::Currents,v)
    return vcat([initialCondition(c.currents[i],v) for i in 1:length(c.currents)]...)
end

"""
    Concentration dynamics:
    a first-order filter
"""
struct Concentration{T<:Tuple}
    currents::T     # Tuple of symbols with current names 
    τ::Vector{F}
    # b::Vector{F}
end

function f!(c::Concentration,dw,w,Σ)
    dw .= (.-w .- Σ) ./ c.τ # .+ m.concentration.b 
end

"""
    Membrane dynamics:
    c is either a float or an n-dimensional vector of capacitances, with n the number of neurons.
    β is an n×m matrix [θᵢₒₙ₁ θᵢₒₙ₂ ... ] of maximal conductances, where m is the number of currents per neuron.
    currents is a Currents struct containing the currents of the membrane.
"""
struct Membrane{C<:Union{F,Vector{F}},I<:Currents,K<:Union{Concentration,Nothing}}
    c::C
    currents::I
    concentration::K
    function Membrane(c::C, currents::I, conc::K) where {C<:Union{AbstractFloat,AbstractVector}, I<:Currents, K<:Union{Concentration,Nothing}}
        c = F.(c)
        for curr in values(currents.currents)
            @assert length(curr.μ) == length(c) "The length of each maximal conductance vector must match the number of neurons defined by the capacitance vector."
        end
        if !(conc isa Nothing)
            for curr in conc.currents
                @assert curr ∈ keys(currents.currents) "Concentration currents must correspond to existing currents in the membrane."
            end
        end
        return new{typeof(c),I,K}(c,currents,conc)
    end
end
Base.length(m::Membrane) = length_v(m) + length_x(m)
length_v(m::Membrane) = length(m.c)
length_x(m::Membrane) = length_x(m.currents)
length_μ(m::Membrane) = length_μ(m.currents)

# Constructor ignoring concentratin dynamics
Membrane(c,currents) = Membrane(c,currents,nothing)

function f!(m::Membrane,dv,dx,v,x,u)
    Σ!(m.currents,dv,v,x)
    dv .= (u .- dv) ./ m.c
    f!(m.currents,dx,v,x)
    return nothing
end

function ∇f!(m::Membrane,∇ᵥfᵥ,∇ₓfᵥ,∇ᵥfₓ,∇ₓfₓ,v,x)
    ∇!(m.currents,∇ᵥfᵥ,∇ₓfᵥ,v,x)
    ∇ᵥfᵥ .= ∇ᵥfᵥ ./ (-m.c)
    for idx_x in m.currents.idx_x
        for idx_gate in idx_x
            ∇ₓfᵥ[idx_gate] .= ∇ₓfᵥ[idx_gate] ./ (-m.c) 
        end
    end
    ∇f!(m.currents,∇ᵥfₓ,∇ₓfₓ,v,x)
    return nothing
end

# Parameters
function Jμfᵥ!(m::Membrane,Jμfᵥ,v,x)
    Jμ!(m.currents,Jμfᵥ,v,x)
    Jμfᵥ .= Jμfᵥ ./ (-m.c)
end

function ∇ufᵥ!(m::Membrane,∇ufᵥ)
    ∇ufᵥ .= 1.0 ./ m.c
end

# With concentration dynamics
function f!(m::Membrane,dv,dx,dw,v,x,w,u)
    Σ!(m.currents,dv,dw,v,x,m.concentration.currents)
    dv .= (u .- dv) ./ m.c
    f!(m.currents,dx,v,x)
    f!(m.concentration,dw,w,dw)
    return nothing
end

function initialCondition(m::Membrane,v)
    return initialCondition(m.currents,v)
end

function J_prototype(mem::Membrane)
    n = length_v(mem)
    m = length_x(mem)÷n
    M = zeros(n*(1+m),n*(1+m))
    M[diagind(M)[1:n]] .= 1.0
    M[vcat((diagind(M,k*n)[1:n] for k in 1:m)...)] .= 1.0
    M[vcat((diagind(M,-k*n)[1:n] for k in 1:m)...)] .= 1.0
    M[diagind(M)[n+1:end]] .= 1.0
    return sparse(M)
end

function test_alloc_f!(hh::Membrane, dv0, dx0, v0, x0, u0)
    @allocated f!(hh, dv0, dx0, v0, x0, u0)
end

function test_alloc_∇f!(hh::Membrane,∇ᵥfᵥ,∇ₓfᵥ,∇ᵥfₓ,∇ₓfₓ,v,x)
    @allocated ∇f!(hh,∇ᵥfᵥ,∇ₓfᵥ,∇ᵥfₓ,∇ₓfₓ,v,x)
end