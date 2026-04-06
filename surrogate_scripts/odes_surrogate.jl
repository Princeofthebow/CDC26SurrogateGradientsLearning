ℓ(v,v̂,T) = (v - v̂)^2/2/T
∂ᵥℓ(v,v̂,T) = -(v - v̂)/T
∇ₓℓ(x,x̂,T) = -(x - x̂)/T

function hh_ode!(dz, z, p, t)
    u =          p[1]
    hh =         p[2]
    idx_v = 1:length_v(hh)
    idx_x = length_v(hh)+1:length(z)
    dv = @view dz[idx_v]
    dx = @view dz[idx_x]
    v = @view z[idx_v]
    x = @view z[idx_x]
    uₜ =  u(t)
    f!(hh,dv,dx,v,x,uₜ)
    return nothing
end

function hh_jac!(J,z,p,t)
    u =          p[1]
    hh =         p[2]
    n = length_v(hh)
    m = length_x(hh)÷n

    ∇ᵥfᵥ = @view J[diagind(J)[1:n]]
    ∇ₓfᵥ = @view J[vcat((diagind(J,k*n)[1:n] for k in 1:m)...)]
    ∇ᵥfₓ = @view J[vcat((diagind(J,-k*n)[1:n] for k in 1:m)...)]
    ∇ₓfₓ = @view J[diagind(J)[n+1:end]]

    ∇f!(hh,∇ᵥfᵥ,∇ₓfᵥ,∇ᵥfₓ,∇ₓfₓ,view(z,1:n),view(z,n+1:length(z)))
    return nothing
end

function co_hh_ode!(dz,z,p,t)
    hh,sol,sôl,T,γ,(Cᵥ,Cₓ),grads = p
    ∇ᵥfᵥ,∇ₓfᵥ,∇ᵥfₓ,∇ₓfₓ,Jμfᵥ = grads
    nᵥ,nₓ = length_v(hh), length_x(hh)

    # Recover target states
    v = sol(T-t)[1:nᵥ]
    # x = sol(T-t)[nᵥ+1:nᵥ+nₓ]

    # Recover forward pass states
    v̂ = sôl(T-t)[1:nᵥ]
    x̂ = sôl(T-t)[nᵥ+1:nᵥ+nₓ]

    # Recover costates
    λᵥ = z[1:nᵥ]
    λₓ = z[nᵥ+1:nᵥ+nₓ]
    # λᵦ = z[nᵥ+nₓ+1:end]

    # Costate derivatives
    dλᵥ = @view dz[1:nᵥ]
    dλₓ = @view dz[nᵥ+1:nᵥ+nₓ]
    dλμ = @view dz[nᵥ+nₓ+1:end]
    
    # Gradients
    ∇f!(hh,∇ᵥfᵥ,∇ₓfᵥ,∇ᵥfₓ,∇ₓfₓ,v̂,x̂)

    # Costates
    dλᵥ .= ∇ᵥfᵥ.*λᵥ
    dλₓ .= ∇ₓfₓ.*λₓ
    for (ion,idx) in pairs(hh.currents.idx_x)
        for i in 1:length(idx)
            # Switching positive feedback off
            if ion == :Na && i==1 && γ > 0
                dλᵥ         .+= (∇ᵥfₓ[idx[i]] .- (1-γ) .* d(hh.currents[:Na].gates[1].σ,v̂) ./ hh.currents[:Na].gates[1].τ(v̂)
                                ).*λₓ[idx[i]]
                dλₓ[idx[i]] .+= (∇ₓfᵥ[idx[i]] .* γ 
                                ).*λᵥ
            else
                dλᵥ         .+= ∇ᵥfₓ[idx[i]].*λₓ[idx[i]]
                dλₓ[idx[i]] .+= ∇ₓfᵥ[idx[i]].*λᵥ
            end
        end
    end

    # Costate
    if γ >= 0
        dλᵥ .+= Cᵥ*∂ᵥℓ(v,v̂,T)
    else
        dλᵥ .+= Cᵥ*∂ᵥℓ(v,v̂,T) .+ γ*λᵥ
    end

    # Coparameters
    Jμfᵥ!(hh,Jμfᵥ,v̂,x̂)         # Preallocate this like the other gradients?
    for i in 1:length(hh.currents)
        dλμ[1+(i-1)*nᵥ:i*nᵥ] = Jμfᵥ[:,i].*λᵥ
    end
end