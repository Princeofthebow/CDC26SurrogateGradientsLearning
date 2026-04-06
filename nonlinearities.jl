struct Logistic
    η::Vector{F}
    κ::Vector{F}
end

function Logistic(η::F, κ::F)
    return Logistic([η;], [κ;])
end

function (σ::Logistic)(v::Real; i::Int=1)
    return 1/(1+exp(-(v-σ.η[i])/σ.κ[i]))
end

function (σ::Logistic)(v::AbstractVector)
    return 1 ./ (1 .+ exp.(-(v.-σ.η)./σ.κ))
end

function d(σ::Logistic,v::Real; i::Int=1)
    s = σ(v; i=i)
    return s * (1 - s) / σ.κ[i]
end

function d(σ::Logistic,v::AbstractVector)
    s = σ(v)
    return s .* (1 .- s) ./ σ.κ
end

struct Gaussian
    Cbase::Vector{F}
    Camp::Vector{F}
    Vmax::F
    std::F
end

function Gaussian(Cbase,Camp,Vmax,std)
    return Gaussian([Cbase;], [Camp;], Vmax, std)
end

function (τ::Gaussian)(v::Real; i::Int=1)
    return τ.Cbase[i] + τ.Camp[i]*exp(-(v-τ.Vmax)^2/τ.std^2)
end

function (τ::Gaussian)(v::AbstractVector)
    return τ.Cbase .+ τ.Camp .* exp.(-(v .- τ.Vmax).^2 ./ τ.std^2)
end

function d(τ::Gaussian,v::Real; i::Int=1)
    return τ.Camp[i] * exp(-((v - τ.Vmax)^2) / (τ.std^2)) * (-2 * (v - τ.Vmax) / (τ.std^2))
end