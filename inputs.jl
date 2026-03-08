function highpass(x,xf)
    return x .- xf
end

struct Bumps{F<:AbstractFloat,T<:NTuple{N,F} where N}
    A::T
    t₀::T
    α::T
    τ::T
    output::Vector{F}
end

function Bumps(; A::T, t::T, α::T, τ::T, n::Int) where {T<:NTuple{N,F} where {N,F<:AbstractFloat}}
    Anorm = ntuple(i->A[i]*(1/τ[i])^(α[i])/Distributions.gamma(α[i]),length(A))
    println(typeof(Anorm))
    return Bumps(Anorm, t, α, τ, fill(one(F), n))
end

function (b::Bumps)(t::F) where F<:AbstractFloat
    fill!(b.output, zero(F))
    @inbounds for k in eachindex(b.A)
        if t > b.t₀[k]
            b.output .+= b.A[k]*(t-b.t₀[k])^(b.α[k]-1)*exp(-(t-b.t₀[k])/b.τ[k])
        end
    end
    return b.output
end

# function (b::Bumps)(t)
#     o = zero(F)
#     @inbounds for k in eachindex(b.A)
#         if t > b.t₀[k]
#             o += b.A[k]*(t-b.t₀[k])^(b.α[k]-1)*exp(-(t-b.t₀[k])/b.τ[k])
#         end
#     end
#     return o*b.output
# end

(b::Bumps)(t::AbstractVector) = reduce(hcat,[b(ti) for ti in t])

struct PinkSoS{T}
    A::T
    f::Vector{T}
    a::Vector{T}
    ϕ::Vector{T}
    μ::T
end

function PinkSoS(; A=1.0, fmin=0.01, fmax=100.0, K=32, μ=0.0, rng=Random.default_rng())
    logf = range(log(fmin), log(fmax), length=K+1)
    f = exp.((logf[1:end-1] .+ logf[2:end]) ./ 2)
    a = 1 ./ sqrt.(f)
    ϕ = 2π .* rand(rng, K)
    PinkSoS(A, f, a, ϕ, μ)
end

@inline function (p::PinkSoS)(t)
    s = 0.0
    @inbounds for k in eachindex(p.f)
        s += p.A * p.a[k] * sin(2π * p.f[k] * t + p.ϕ[k])
    end
    return s / sqrt(length(p.f)) + p.μ
end