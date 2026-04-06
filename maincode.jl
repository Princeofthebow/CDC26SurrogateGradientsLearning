using DifferentialEquations, Peaks, Interpolations, Random, Distributions, Plots, LinearAlgebra, DelimitedFiles, BenchmarkTools, SparseArrays
include("./inputs.jl")
include("./structs.jl")  # Model structs
include("./models.jl")   # Model constructors

default(
    titlefont = font(14),       # plot title
    guidefont = font(14),       # axis labels
    tickfont  = font(12),       # tick labels
    legendfont = font(12),       # legend
    legendforegroundcolor=:transparent, 
    legendbackgroundcolor=:transparent, 
    margins=2.5Plots.mm, 
    legend=(1.125,0.5),
    lw=2,
    xformatter=:none,
    yticks=:auto, 
    rightmargin=37.5Plots.mm
)

function autoticks(y; n=4, sigdigits=2)
    lo = minimum(skipmissing(replace(y, NaN => missing)))
    hi = maximum(skipmissing(replace(y, NaN => missing)))
    lo=lo+0.1*(hi-lo)
    hi=hi-0.1*(hi-lo)
    round.(range(lo, hi; length=n), sigdigits=sigdigits)
end