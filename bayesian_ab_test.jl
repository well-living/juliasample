#include(homedir() * "/パスを書いて/bayesian_ab_test.jl")

using Random
using Distributions
using Turing

using DataFrames
using CSV

#using Plots
using PlotlyJS

Random.seed!(1234)

"""
生成モデル(本来は実データを用いる)
"""
p₀ = 0.2
outcome = 0.1
sample_size = 1000

sample_data = Distributions.rand(sample_size)
y₀ = sample_data .<= p₀
p₁ = p₀ .+ outcome
y₁ = sample_data .<= p₁

z = Distributions.rand(Distributions.Bernoulli(0.5), sample_size)
p = p₀ .* (1 .- z) + p₁ .* z
y = y₀ .* (1 .- z) + y₁ .* z

sample0 = y[.!z]
sample1 = y[z]

"""
ベイズ推論
"""

@model function AB_test(data)
    p ~ Turing.Beta(1, 1)  # (1, 1)のときベータ分布は一様分布
    N = length(data)
    for n in 1:N
        data[n] ~ Turing.Bernoulli(p)  # データはgiven pのときのベルヌーイ分布に従う
    end
end


ϵ = 0.05
τ = 10
iterations = 1000

print("MCMC実行時間計測")

@time chain0 = sample(AB_test(sample0), HMC(ϵ, τ), iterations)
@time chain1 = sample(AB_test(sample1), HMC(ϵ, τ), iterations)

#describe(chain)

function chain_to_df(chain::MCMCChains.Chains)
    num_iters, num_params, c = size(chain.value)
    chain_reshape = zeros(num_iters * c, num_params)
    idx = 1
    for col_name in keys(chain)
        chain_reshape[:, idx] = reshape(convert(Array{Float64}, chain[col_name]), num_iters * c)
        idx += 1
    end
    col_names = keys(chain)
    return DataFrame(chain_reshape[:, :], col_names)
end

chain_df0 = chain_to_df(chain0)
chain_df0[!, :z] = zeros(size(chain_df0)[1])

chain_df1 = chain_to_df(chain1)
chain_df1[!, :z] = ones(size(chain_df1)[1])

CSV.write("./chain_df0.csv", chain_df0)
CSV.write("./chain_df1.csv", chain_df1)

chain_df = vcat(chain_df1, chain_df0)

print("HTML生成")

open("./bayesian_ab_test_graph.html", "w") do io
    PlotlyBase.to_html(
        io, 
        PlotlyJS.plot(
            chain_df[:, ["p", "z"]], 
            x=:p, 
            kind="histogram",
            color=:z,
            opacity=0.5,
            Layout(barmode="overlay")
        ).plot
    )
end


