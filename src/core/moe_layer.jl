"""
    MoEConfig - Complete configuration for MoE layer
"""
Base.@kwdef struct MoEConfig
    num_experts::Int = 8
    expert_type::Symbol = :standard 
    input_dim::Int = 768
    hidden_dim::Int = 3072
    output_dim::Int = 768
    activation::Function = gelu
    expert_dropout::Float32 = 0.0f0
    
    gate_type::GatingMechanism = TopKGating(2)
    top_k::Int = 2
    noise_scale::Float32 = 0.0f0
    use_noise_network::Bool = false
    use_fp32_router::Bool = true
    
    balance_loss::LoadBalancingLoss = SwitchTransformerLoss(0.01f0)
    z_loss_weight::Float32 = 0.001f0
    
    capacity_factor::Float32 = 1.25f0
    drop_tokens::Bool = false
    
    use_cur::Bool = false
    cur_rank::Union{Int, Nothing} = nothing
    
    num_shared_experts::Int = 0
end

"""
    MoELayer - Main Mixture of Experts layer
"""
struct MoELayer{E, R, L}
    experts::E
    router::R
    balance_loss::L
    config::MoEConfig
    training_stats::Dict{Symbol, Any}
end

Flux.@functor MoELayer (experts, router)

function MoELayer(config::MoEConfig)
    experts = create_experts(config)
    
    router = Router(
        config.input_dim, 
        config.num_experts - config.num_shared_experts, 
        config.gate_type;
        noise_scale = config.noise_scale,
        use_noise_network = config.use_noise_network,
        use_fp32 = config.use_fp32_router
    )
    
    stats = Dict{Symbol, Any}(
        :tokens_per_expert => zeros(Int, config.num_experts),
        :routing_entropy => Float32[],
        :capacity_overflow => 0
    )
    
    return MoELayer(experts, router, config.balance_loss, config, stats)
end

function create_experts(config::MoEConfig)
    expert_hidden_dim = config.hidden_dim  # FIXED: Don't divide by num_experts!
    
    experts = []
    for i in 1:config.num_experts
        if config.use_cur
            if config.expert_type == :gated
                expert = GatedCURExpert(
                    config.input_dim, expert_hidden_dim, config.output_dim,
                    config.activation; 
                    rank = something(config.cur_rank, expert_hidden_dim ÷ 4)
                )
            else
                expert = CURExpert(
                    config.input_dim, expert_hidden_dim, config.output_dim,
                    config.activation;
                    rank = something(config.cur_rank, expert_hidden_dim ÷ 4)
                )
            end
        else
            if config.expert_type == :gated
                expert = GatedExpert(
                    config.input_dim, expert_hidden_dim, config.output_dim,
                    config.activation
                )
            else
                expert = StandardExpert(
                    config.input_dim, expert_hidden_dim, config.output_dim,
                    config.activation;
                    dropout = config.expert_dropout
                )
            end
        end
        push!(experts, expert)
    end
    
    return experts
end

"""
    Forward pass through MoE layer
"""
function (moe::MoELayer)(x::AbstractMatrix; training::Bool = false, return_stats::Bool = false)
    batch_size = size(x, 2)
    config = moe.config
    
    if config.num_shared_experts > 0
        output, balance_loss = forward_with_shared_experts(moe, x, training)
    else
        output, balance_loss = forward_standard(moe, x, training)
    end
    
    if training && config.z_loss_weight > 0
        _, _, _, router_logits = moe.router(x; training=false)
        z_loss = compute_loss(ZLoss(config.z_loss_weight), router_logits)
        balance_loss += z_loss
    end
    
    if return_stats
        return output, balance_loss, moe.training_stats
    else
        return output, balance_loss
    end
end

function forward_standard(moe::MoELayer, x::AbstractMatrix, training::Bool)
    batch_size = size(x, 2)
    output = zeros(Float32, moe.config.output_dim, batch_size)
    
    expert_indices, expert_gates, router_probs, router_logits = 
        moe.router(x; training=training)
    
    if moe.balance_loss isa AuxiliaryFreeLoss
        biased_logits = get_biased_logits(moe.balance_loss, router_logits)
        expert_indices, expert_gates, _ = compute_gates(moe.router.gate_type, biased_logits)
    end
    
    if moe.config.drop_tokens && training
        expert_indices, expert_gates = apply_capacity_constraint(
            expert_indices, expert_gates, 
            moe.config.capacity_factor, moe.config.num_experts
        )
    end
    
    process_expert_forward!(output, moe.experts, x, expert_indices, expert_gates, training)
    
    balance_loss = training ? compute_loss(moe.balance_loss, expert_indices, router_probs) : 0.0f0
    
    update_stats!(moe.training_stats, expert_indices, router_probs)
    
    return output, balance_loss
end

function forward_with_shared_experts(moe::MoELayer, x::AbstractMatrix, training::Bool)
    batch_size = size(x, 2)
    config = moe.config
    
    shared_output = zeros(Float32, config.output_dim, batch_size)
    for i in 1:config.num_shared_experts
        expert_out = moe.experts[i](x; training=training)
        shared_output .+= expert_out ./ config.num_shared_experts
    end
    
    routed_experts = moe.experts[(config.num_shared_experts + 1):end]
    expert_indices, expert_gates, router_probs, router_logits = 
        moe.router(x; training=training)
    
    expert_indices .+= config.num_shared_experts
    
    routed_output = zeros(Float32, config.output_dim, batch_size)
    process_expert_forward!(routed_output, routed_experts, x, 
                           expert_indices .- config.num_shared_experts, 
                           expert_gates, training)
    
    α = 0.5f0  # Can be learned
    output = α .* shared_output .+ (1 - α) .* routed_output
    
    balance_loss = training ? compute_loss(moe.balance_loss, expert_indices, router_probs) : 0.0f0
    
    return output, balance_loss
end

function process_expert_forward!(output::AbstractMatrix, experts::AbstractVector,
                                x::AbstractMatrix, expert_indices::AbstractMatrix,
                                expert_gates::AbstractMatrix, training::Bool)
    batch_size = size(x, 2)
    
    expert_batches = [Int[] for _ in 1:length(experts)]
    expert_weights = [Float32[] for _ in 1:length(experts)]
    
    for token in 1:batch_size
        for k in 1:size(expert_indices, 1)
            expert_id = expert_indices[k, token]
            if expert_id > 0 && expert_id <= length(experts)
                push!(expert_batches[expert_id], token)
                push!(expert_weights[expert_id], expert_gates[k, token])
            end
        end
    end
    
    for expert_id in 1:length(experts)
        if !isempty(expert_batches[expert_id])
            tokens = expert_batches[expert_id]
            weights = expert_weights[expert_id]
            
            expert_input = x[:, tokens]
            
            expert_output = experts[expert_id](expert_input; training=training)
            
            for (i, token) in enumerate(tokens)
                output[:, token] .+= weights[i] .* expert_output[:, i]
            end
        end
    end
end

function apply_capacity_constraint(expert_indices, expert_gates, capacity_factor, num_experts)
    batch_size = size(expert_indices, 2)
    capacity = ceil(Int, capacity_factor * batch_size / num_experts)
    
    expert_counts = zeros(Int, num_experts)
    
    for j in 1:batch_size
        for k in 1:size(expert_indices, 1)
            expert = expert_indices[k, j]
            if expert > 0
                if expert_counts[expert] >= capacity
                    
                    expert_indices[k, j] = 0
                    expert_gates[k, j] = 0.0f0
                else
                    expert_counts[expert] += 1
                end
            end
        end
        
        gate_sum = sum(expert_gates[:, j])
        if gate_sum > 0
            expert_gates[:, j] ./= gate_sum
        end
    end
    
    return expert_indices, expert_gates
end

function update_stats!(stats::Dict, expert_indices, router_probs; reset_counters::Bool = false)
    if reset_counters
        fill!(stats[:tokens_per_expert], 0)
    end
    
    for idx in expert_indices
        if idx > 0 && idx <= length(stats[:tokens_per_expert])
            stats[:tokens_per_expert][idx] += 1
        end
    end
    
    # Compute entropy for current batch only
    batch_entropy = -sum(router_probs .* log.(router_probs .+ 1e-8), dims=1)
    push!(stats[:routing_entropy], mean(batch_entropy))
end

"""
    reset_stats!(moe::MoELayer)

Reset training statistics to start fresh.
"""
function reset_stats!(moe::MoELayer)
    fill!(moe.training_stats[:tokens_per_expert], 0)
    empty!(moe.training_stats[:routing_entropy])
    moe.training_stats[:capacity_overflow] = 0
    return nothing
end

"""
    Convenience functions
"""
function create_moe_config(; kwargs...)
    return MoEConfig(; kwargs...)
end

function create_moe_layer(input_dim::Int, hidden_dim::Int, output_dim::Int; kwargs...)
    config = MoEConfig(;
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        output_dim = output_dim,
        kwargs...
    )
    return MoELayer(config)
end