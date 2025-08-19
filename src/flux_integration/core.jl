using Flux
using Flux: Dense, @layer
using Functors
using Statistics
using Random

flux_silu(x) = x .* sigmoid.(x)

function flux_expert_init(input_dim::Int, hidden_dim::Int, init_fn::Function = Flux.glorot_uniform)
    return init_fn
end

function count_flux_parameters(layer)
    return sum(length, Flux.trainables(layer))
end