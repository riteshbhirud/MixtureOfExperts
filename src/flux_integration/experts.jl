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

"""
    FluxStandardExpert(input_dim, hidden_dim, output_dim, activation; 
                      dropout=0.0f0, bias=true, init=Flux.glorot_uniform)

ULTRA-SIMPLE Flux-compatible standard expert - no custom forward pass.
Uses Flux Chain for maximum compatibility.
"""
struct FluxStandardExpert{C}
    chain::C
end

@layer FluxStandardExpert

function FluxStandardExpert(input_dim::Int, hidden_dim::Int, output_dim::Int, 
                           activation=Flux.relu; 
                           dropout::Float32=0.0f0, 
                           bias::Bool=true,
                           init=Flux.glorot_uniform)
    
    if dropout > 0
        chain = Flux.Chain(
            Dense(input_dim => hidden_dim; init=init, bias=bias),
            activation, 
            Flux.Dropout(dropout),
            Dense(hidden_dim => output_dim; init=init, bias=bias)
        )
    else
        chain = Flux.Chain(
            Dense(input_dim => hidden_dim; init=init, bias=bias),
            activation, 
            Dense(hidden_dim => output_dim; init=init, bias=bias)
        )
    end
    
    return FluxStandardExpert(chain)
end

function (expert::FluxStandardExpert)(x::AbstractVecOrMat; training::Bool=false)
    if training
        return expert.chain(x)
    else
        Flux.testmode!(expert.chain)
        result = expert.chain(x)
        Flux.trainmode!(expert.chain)
        return result
    end
end

function Base.show(io::IO, expert::FluxStandardExpert)
    print(io, "FluxStandardExpert($(expert.chain))")
end

"""
    FluxGatedExpert(input_dim, hidden_dim, output_dim, activation; 
                   bias=false, init=Flux.glorot_uniform)

ULTRA-SIMPLE Flux-compatible gated expert using basic operations.
"""
struct FluxGatedExpert{D1, D2, D3, A}
    w1::D1 
    w2::D2  
    w3::D3  
    activation::A
end

@layer FluxGatedExpert

function FluxGatedExpert(input_dim::Int, hidden_dim::Int, output_dim::Int, 
                        activation=flux_silu;
                        bias::Bool=false,
                        init=Flux.glorot_uniform)
    
    w1 = Dense(input_dim => hidden_dim; init=init, bias=bias)
    w2 = Dense(hidden_dim => output_dim; init=init, bias=bias) 
    w3 = Dense(input_dim => hidden_dim; init=init, bias=bias)
    
    return FluxGatedExpert(w1, w2, w3, activation)
end

function (expert::FluxGatedExpert)(x::AbstractVecOrMat; training::Bool=false)
    gate = expert.w1(x)
    up = expert.w3(x)
    
    gate = expert.activation(gate)
    
    h = gate .* up
    
    return expert.w2(h)
end

function Base.show(io::IO, expert::FluxGatedExpert)
    in_dim = size(expert.w1.weight, 2)
    hidden_dim = size(expert.w1.weight, 1)
    out_dim = size(expert.w2.weight, 1)
    print(io, "FluxGatedExpert($in_dim => $hidden_dim => $out_dim, gated)")
end