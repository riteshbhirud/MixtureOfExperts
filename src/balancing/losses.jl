"""
    SwitchTransformerLoss(α::Float32)

Stanford CS336 / Switch Transformer auxiliary loss:
loss = α · N · Σ(i=1 to N) f_i · P_i

where:
- f_i = fraction of tokens dispatched to expert i
- P_i = average probability assigned to expert i
"""
struct SwitchTransformerLoss <: LoadBalancingLoss
    α::Float32
end

function compute_loss(loss_fn::SwitchTransformerLoss, expert_indices::AbstractMatrix, router_probs::AbstractMatrix)
    N = size(router_probs, 1)  # num_experts
    total_assignments = length(expert_indices)
    
    # Count actual expert assignments (not argmax of probs)
    f = zeros(Float32, N)
    
    for expert_idx in expert_indices
        if expert_idx > 0 && expert_idx <= N
            f[expert_idx] += 1.0f0
        end
    end
    
    # Normalize by total assignments
    if total_assignments > 0
        f ./= total_assignments
    end
    
    P = mean(router_probs, dims=2)[:]
    
    return loss_fn.α * N * sum(f .* P)
end

"""
    DeepSeekLoss(α::Float32, balance_type::Symbol)

DeepSeek V1/V2 loss variants:
- :expert - Per-expert balancing (same as Switch)
- :device - Per-device balancing
- :communication - Balance incoming/outgoing communication
"""
struct DeepSeekLoss <: LoadBalancingLoss
    α::Float32
    balance_type::Symbol
    device_map::Union{Nothing, Vector{Int}}
end

DeepSeekLoss(α::Float32) = DeepSeekLoss(α, :expert, nothing)

function compute_loss(loss_fn::DeepSeekLoss, expert_indices::AbstractMatrix, 
                     router_probs::AbstractMatrix)
    if loss_fn.balance_type == :expert
        return compute_loss(SwitchTransformerLoss(loss_fn.α), expert_indices, router_probs)
    elseif loss_fn.balance_type == :device && !isnothing(loss_fn.device_map)
        return compute_device_balance_loss(loss_fn, expert_indices, router_probs)
    elseif loss_fn.balance_type == :communication
        return compute_communication_balance_loss(loss_fn, expert_indices, router_probs)
    else
        return compute_loss(SwitchTransformerLoss(loss_fn.α), expert_indices, router_probs)
    end
end

function compute_device_balance_loss(loss_fn::DeepSeekLoss, expert_indices, router_probs)
    num_devices = maximum(loss_fn.device_map)
    num_experts = size(router_probs, 1)
    
    device_counts = zeros(Float32, num_devices)
    device_probs = zeros(Float32, num_devices)
    
    for expert in 1:num_experts
        device = loss_fn.device_map[expert]
        
        for idx in expert_indices
            if idx == expert
                device_counts[device] += 1
            end
        end
        
        device_probs[device] += sum(router_probs[expert, :])
    end
    
    total_count = sum(device_counts)
    device_counts ./= max(total_count, 1)
    device_probs ./= size(router_probs, 2)
    
    return loss_fn.α * num_devices * sum(device_counts .* device_probs)
end

"""
    ZLoss(β::Float32)

Stanford CS336 Z-loss for preventing logit explosion:
L_z(x) = (1/B) Σ(i=1 to B) (log Σ(j=1 to N) e^{x_j^{(i)}})²
"""
struct ZLoss <: LoadBalancingLoss
    β::Float32
end

function compute_loss(loss_fn::ZLoss, router_logits::AbstractMatrix)
    log_sum_exp = logsumexp(router_logits; dims=1)
    
    z_loss = mean(log_sum_exp .^ 2)
    
    return loss_fn.β * z_loss
end

function logsumexp(x::AbstractMatrix; dims)
    max_x = maximum(x; dims=dims)
    return max_x .+ log.(sum(exp.(x .- max_x); dims=dims))
end

"""
    ImportanceWeightedLoss(base_loss::LoadBalancingLoss)

Weight the loss by token importance (e.g., based on attention scores).
"""
struct ImportanceWeightedLoss <: LoadBalancingLoss
    base_loss::LoadBalancingLoss
    importance_fn::Function
end

function compute_loss(loss_fn::ImportanceWeightedLoss, expert_indices::AbstractMatrix, 
                     router_probs::AbstractMatrix, x::AbstractMatrix)
    importance = loss_fn.importance_fn(x)
    
    weighted_probs = router_probs .* reshape(importance, 1, :)
    
    return compute_loss(loss_fn.base_loss, expert_indices, weighted_probs)
end