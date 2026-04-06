# Import the necessary packages
using DifferentialEquations, Random, Distributions, Plots, LinearAlgebra, DelimitedFiles, BenchmarkTools, SparseArrays
include("./../maincode.jl") 
include("./odes_surrogate.jl")   # Contains odes and backpropagation equations

# Decide on surrogate gradients
# If γ == 0 : no costate controller
# If γ > 0 : use gated controller
# If γ < 0 : use proportional controller 
# γ = 0.0
γ = -4.0
# γ = 0.7

# Define input type
inp_type = :sines

# ODE parameters 
T = (inp_type == :bumps ? 100 : 500)
dt = 0.01
tspan = (0.,T)
Solver = Tsit5
CoSolver = Tsit5
odeopts = (reltol=1e-6,abstol=1e-6,)

# Number of models is equal to number of ĝNa values
dg = 0.05            # 0.025 in the paper
ĝNa_vec = 100:dg:125 

# HH model
n = length(ĝNa_vec)
c = 1.0
μ = [120. 36. 0.3]
ν = [55.,-77.,-54.4]
hh = HH(c,μ,ν,n=length(ĝNa_vec))

# Define input
u = if inp_type == :bumps
        Bumps(; A=(15.0,25.0), t=(10.0,50.0), α=(2.0,2.0), τ=(5.0,5.0), n=n)
    else    
        PinkSoS(; A=2.0, fmin=0.1, fmax=10.0, K=48, μ=-1.0, rng=MersenneTwister(12))
    end

# Define states to compare in the cost function
Cᵥ = Matrix(I,length_v(hh),length_v(hh))    # only voltage readout
Cₓ = zeros(F,length_v(hh),length_x(hh))     # no internal state readout

## Initial conditions / test allocations
v0 = -64.0*ones(F,n)                            # remove this if evaluating gradient: + randn(F,n).*1.0
x0 = initialCondition(hh,v0)
z0 = vcat(v0,x0)

# Obtain solution of the true model
odefun = ODEFunction(hh_ode!; jac=hh_jac!, jac_prototype=J_prototype(hh))
prob = ODEProblem(odefun,z0,tspan,(u,hh);odeopts...)
sol = solve(prob,Solver(),saveat=dt) 
t = sol.t
v = sol[1:n,1,:];
x = sol[n+1:end,1,:];
y = v # Cᵥ*v + Cₓ*x;

# Plot true solution
neuron_number = 1
p1=plot(t,y[neuron_number,:], lw=2, label="y(t)")    # Plot true solution (all should be identical)
p2=plot(t,[u(tᵢ)[neuron_number] for tᵢ in t], lw=2,label="u(t)")  # Plot input
plot(p1,p2,layout=(2,1))

# Save trajectories to file
# data = hcat(t, y[neuron_number,:], [u(tᵢ)[neuron_number] for tᵢ in t])
# open("trajectories_sin.dat", "w") do io
#     println(io, "t v u")
#     writedlm(io, data)
# end

#######################################################################
## Compute cost function and its gradients for different values of ĝNa
#######################################################################
# Pre-allocate vector field gradients
∇ᵥfᵥ = Vector{F}(undef,length_v(hh))
∇ₓfᵥ = Vector{F}(undef,length_x(hh))
∇ᵥfₓ = Vector{F}(undef,length_x(hh))
∇ₓfₓ = Vector{F}(undef,length_x(hh))
Jμfᵥ = Matrix{F}(undef, length_v(hh), length(hh.currents))
grads = (∇ᵥfᵥ=∇ᵥfᵥ,∇ₓfᵥ=∇ₓfᵥ,∇ᵥfₓ=∇ᵥfₓ,∇ₓfₓ=∇ₓfₓ,Jμfᵥ=Jμfᵥ);
# @btime ∇f!(hh,∇ᵥfᵥ,∇ₓfᵥ,∇ᵥfₓ,∇ₓfₓ,v0,x0)  # check allocations

# Change parameters of estimator model
ĥh=deepcopy(hh)
for (i,ĝNa) in enumerate(ĝNa_vec)
    ĥh.currents[:Na].μ[i] = ĝNa
end

# Compute the cost function V
prob = ODEProblem(odefun,z0,tspan,(u,ĥh);odeopts...)
sôl = solve(prob,Solver(),saveat=dt)
v̂ = sôl[1:n,1,:];
x̂ = sôl[n+1:end,1,:];
ŷ = Cᵥ*v̂ + Cₓ*x̂
V = sum(Cᵥ*ℓ.(v,v̂,Ref(T))+(Cₓ*ℓ.(x,x̂,Ref(T))),dims=2)*dt;

## Compute the gradients using costates
coĥh = deepcopy(ĥh)
co_z0 = zeros(F,length(z0)+length_μ(coĥh))
co_prob = ODEProblem(co_hh_ode!,co_z0,tspan,(coĥh,sol,sôl,T,γ,(Cᵥ,Cₓ),grads);odeopts...)
co_sol = solve(co_prob,CoSolver(),saveat=dt)

## Recover gradient of specific conductance
idx_ion=1   # 1 for Na, 2 for K, 3 for L
δV = diff(V[:])/dg
∂Ṽ = co_sol(T)[length_v(coĥh)+length_x(coĥh).+(1+(idx_ion-1)*length_v(coĥh):idx_ion*length_v(coĥh))]
Ṽ = cumsum(∂Ṽ)*dg

# Plot cost function and gradients
plt1=plot(ĝNa_vec,V,title="Cost function",xlabel="ĝNa",label="C(gₙₐ)")
plt2=plot(ĝNa_vec[1:end-1],δV,xlabel="ĝNa",title="Cost gradient (finite difference)",label="∂C/∂gₙₐ",ylims=(-20,20))
plt3=plot(ĝNa_vec,Ṽ.-minimum(Ṽ),title="Surrogate Cost function",xlabel="ĝNa",label="C(gₙₐ)")
plt4=plot(ĝNa_vec,∂Ṽ,xlabel="ĝNa",xformatter=:auto,title="Surrogate gradient",label="∂ₛC/∂ₛgₙₐ",
                    # ylims=(-20,20)
                )
plot(plt1,plt2,plt3,plt4,layout=(4,1),size=(800,1000),margins=5Plots.mm,xformatter=:auto)

# plt2=plot(ĝNa_vec[1:end-1],δV-∂Ṽ[1:end-1],ylims=(-1,1),xlabel="ĝNa",title="Cost gradient (finite difference)",label="∂C/∂gₙₐ")

## Save cost functions to file
# Non-surrogate cost function
# ds=1
# data = hcat(ĝNa_vec[1:ds:end],V[1:ds:end],Ṽ[1:ds:end],∂Ṽ[1:ds:end])
# if γ == 0
#     mode = "orig"
# elseif γ>0 
#     mode = "gat_$γ"
# else
#     mode = "prop_$γ"
# end

# # open("cost_prop_γ=$(γ).dat", "w") do io
# open("cost_"*mode*"_sin.dat", "w") do io
#     println(io, "gNahat V Vsurr dVsurr")
#     writedlm(io, data)
# end
# minimum(Ṽ)

## Plot time series of costate trajectories
# Run this with no costate controler to see exploding gradients
plts = []
for ĝNa = [107.6,107.9]
    i = findfirst(ĝNa_vec .== ĝNa)
    
    co_v = reverse(co_sol[i,1,:])
    co_m = reverse(co_sol[length_v(ĥh)+i,1,:])
    co_h = reverse(co_sol[2*length_v(ĥh)+i,1,:])
    co_n = reverse(co_sol[3*length_v(ĥh)+i,1,:])
    co_gNa = reverse(co_sol[length_v(ĥh)+length_x(ĥh).+i,1,:])

    plt_v = plot(t,y[i,:],lw=2,label="y")
    plt_v = plot!(t,ŷ[i,:],lw=2,label="ŷ",linecolor="red",title="Output trajectories for ĝNa = $ĝNa")
    plt_co_v = plot(t,co_v,lw=2,label="λv",title="Covoltage trajectories for ĝNa = $ĝNa")
    plt_co_x = plot(t,co_m,lw=2,label="λm",title="Cogating variable trajectories for ĝNa = $ĝNa")
    plt_co_x = plot!(t,co_h,lw=2,label="λh")
    plt_co_x = plot!(t,co_n,lw=2,label="λn")
    plt_co_gNa = plot(t,co_gNa,lw=2,label="λgNa",title="Coparameter trajectories for ĝNa = $ĝNa")

    # Save trajectories to file
    # data = hcat(t, y[i,:], ŷ[i,:])
    # open("trajectories_$(ĝNa).dat", "w") do io
    #     println(io, "t v vhat")
    #     writedlm(io, data)
    # end

    push!(plts,plot(plt_v,plt_co_v,plt_co_x,plt_co_gNa,
                                layout=(4,1),xlabel="t"))
end
plot(plts...,layout=(1,2),size=(1200,800),margins=5Plots.mm)