"""
    FluxMoEConfig

Configuration struct for Flux MoE layer with sensible defaults.
"""
Base.@kwdef struct FluxMoEConfig
    num_experts::Int = 8
    expert_type::Symbol = :standard  # :standard or :gated
    input_dim::Int = 768
    hidden_dim::Int = 3072
    output_dim::Int = 768
    activation::Function = Flux.relu
    expert_dropout::Float32 = 0.0f0
    expert_bias::Bool = true
    
    top_k::Int = 2
    noise_scale::Float32 = 0.0f0
    use_noise_network::Bool = false
    use_fp32_router::Bool = true
    
    balance_loss_weight::Float32 = 0.01f0
    z_loss_weight::Float32 = 0.001f0
    
    init::Function = Flux.glorot_uniform
end

"""
    FluxMoELayer(config::FluxMoEConfig)
    FluxMoELayer(input_dim, hidden_dim, output_dim; kwargs...)

Main Flux-compatible Mixture of Experts layer that combines router and experts.

# Arguments
- `config::FluxMoEConfig`: Complete configuration object
- OR individual dimensions with keyword arguments

# Returns
Tuple of (output, auxiliary_loss) where auxiliary_loss is 0 during inference.
"""
# Example
#=
# Using config
config = FluxMoEConfig(input_dim=768, hidden_dim=3072, output_dim=768, 
                       num_experts=8, top_k=2, expert_type=:gated)
moe = FluxMoELayer(config)

# Using convenience constructor  
moe = FluxMoELayer(768, 3072, 768; num_experts=8, top_k=2, expert_type=:standard)

# Forward pass
x = randn(Float32, 768, 32)
y, aux_loss = moe(x; training=true)
=#

struct FluxMoELayer{E, R, L}
    experts::E
    router::R
    balance_loss::L
    config::FluxMoEConfig
end

@layer FluxMoELayer trainable=(experts, router)

function FluxMoELayer(config::FluxMoEConfig)
    experts = map(1:config.num_experts) do i
        if config.expert_type == :gated
            FluxGatedExpert(
                config.input_dim, 
                config.hidden_dim, 
                config.output_dim,
                config.activation;
                bias=config.expert_bias,
                init=config.init
            )
        else
            FluxStandardExpert(
                config.input_dim, 
                config.hidden_dim, 
                config.output_dim,
                config.activation;
                dropout=config.expert_dropout,
                bias=config.expert_bias,
                init=config.init
            )
        end
    end
    
    gate_type = FluxTopKGating(config.top_k)
    router = FluxRouter(
        config.input_dim, 
        config.num_experts, 
        gate_type;
        noise_scale=config.noise_scale,
        use_noise_network=config.use_noise_network,
        use_fp32=config.use_fp32_router,
        init=config.init
    )
    
    balance_loss = FluxSwitchTransformerLoss(config.balance_loss_weight)
    
    return FluxMoELayer(experts, router, balance_loss, config)
end
"""
    process_expert_forward_clean(experts, x, expert_indices, expert_gates, training, output_dim)

Truly functional approach: no mutations, no accumulation variables.
"""
function process_expert_forward_clean(experts::AbstractVector,
                                     x::AbstractMatrix, 
                                     expert_indices::AbstractMatrix,
                                     expert_gates::AbstractMatrix, 
                                     training::Bool,
                                     output_dim::Int)
    batch_size = size(x, 2)
    
    token_outputs = map(1:batch_size) do token_idx
        token_input = x[:, token_idx:token_idx] 
        
        expert_contributions = map(1:size(expert_indices, 1)) do k
            expert_id = expert_indices[k, token_idx]
            weight = expert_gates[k, token_idx]
            
            if expert_id > 0 && expert_id <= length(experts) && weight > 0
                expert_output = experts[expert_id](token_input; training=training)
                return weight * expert_output[:, 1]
            else
                return zeros(Float32, output_dim)
            end
        end
        
        return sum(expert_contributions)
    end
    
    return reduce(hcat, token_outputs)
end

function (moe::FluxMoELayer)(x::AbstractVecOrMat; training::Bool=false)
    if x isa AbstractVector
        x = reshape(x, :, 1)
        squeeze_output = true
    else
        squeeze_output = false
    end
    
    batch_size = size(x, 2)
    config = moe.config
    
    expert_indices, expert_gates, router_probs, router_logits = 
        moe.router(x; training=training)
    
    output = process_expert_forward_clean(
        moe.experts, x, expert_indices, expert_gates, training, config.output_dim
    )
    
    aux_loss = 0.0f0
    if training
        aux_loss = compute_loss(moe.balance_loss, expert_indices, router_probs)
        
        if config.z_loss_weight > 0
            z_loss_fn = FluxZLoss(config.z_loss_weight)
            aux_loss += compute_loss(z_loss_fn, router_logits)
        end
    end
    
    if squeeze_output
        output = vec(output)
    end
    
    return output, aux_loss
end

function (moe::FluxMoELayer)(x::AbstractVecOrMat; training::Bool=false)
    if x isa AbstractVector
        x = reshape(x, :, 1)
        squeeze_output = true
    else
        squeeze_output = false
    end
    
    batch_size = size(x, 2)
    config = moe.config
    
    expert_indices, expert_gates, router_probs, router_logits = 
        moe.router(x; training=training)
    
    output = process_expert_forward_clean(
        moe.experts, x, expert_indices, expert_gates, training, config.output_dim
    )
    aux_loss = 0.0f0
    if training
        aux_loss = compute_loss(moe.balance_loss, expert_indices, router_probs)
        
        if config.z_loss_weight > 0
            z_loss_fn = FluxZLoss(config.z_loss_weight)
            aux_loss += compute_loss(z_loss_fn, router_logits)
        end
    end
    
    if squeeze_output
        output = vec(output)
    end
    
    return output, aux_loss
end

function Base.show(io::IO, moe::FluxMoELayer)
    config = moe.config
    print(io, "FluxMoELayer(")
    print(io, "$(config.input_dim) => $(config.output_dim), ")
    print(io, "$(config.num_experts) experts, ")
    print(io, "top_k=$(config.top_k), ")
    print(io, "$(config.expert_type))")
end

function FluxMoELayer(input_dim::Int, hidden_dim::Int, output_dim::Int;
                     num_experts::Int=8, top_k::Int=2, expert_type::Symbol=:standard,
                     kwargs...)
    config = FluxMoEConfig(;
        input_dim=input_dim,
        hidden_dim=hidden_dim, 
        output_dim=output_dim,
        num_experts=num_experts,
        top_k=top_k,
        expert_type=expert_type,
        kwargs...
    )
    return FluxMoELayer(config)
end

"""
    create_flux_moe_config(; kwargs...)

Convenience function to create FluxMoEConfig with keyword arguments.
"""
function create_flux_moe_config(; kwargs...)
    return FluxMoEConfig(; kwargs...)
end