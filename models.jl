"""
    Hodgkin-Huxley model with approximated gating nonlinearities
"""
## HH currents
function SodiumCurrent(μ,ν; n=1)
    m = Gating(Gaussian(0.04,0.46,-38.0,30.0), Logistic(-40., 9.),3)
    h = Gating(Gaussian(1.2,7.4,-67.0,20.0), Logistic(-62., -7.),1)
    return GatedCurrent(μ,ν,(m,h); n=n)
end

function PotassiumCurrent(μ,ν; n=1)
    m = Gating(Gaussian(1.1,4.7,-79.0,50.0), Logistic(-53., 15.),4)
    return GatedCurrent(μ,ν,(m,); n=n)
end

## HH model
function HH(c,μ,ν; n=1)
    currents = Currents((Na=SodiumCurrent(μ[1],ν[1],n=n),
                         K=PotassiumCurrent(μ[2],ν[2],n=n),
                         L=LeakCurrent(μ[3],ν[3],n=n)),
                         n=n)
    return Membrane(fill(c,n),currents)
end