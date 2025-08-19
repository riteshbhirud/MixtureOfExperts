"""
    FluxSwitchTransformerLoss(α::Float32=0.01f0)

Flux-compatible Switch Transformer auxiliary loss for load balancing.
Loss = α × N × Σ(f_i × P_i) where f_i is fraction of tokens to expert i,
P_i is average probability assigned to expert i.
"""
struct FluxSwitchTransformerLoss <: LoadBalancingLoss
    α::Float32
end

FluxSwitchTransformerLoss() = FluxSwitchTransformerLoss(0.01f0)

function compute_loss(loss_fn::FluxSwitchTransformerLoss, 
                     expert_indices::AbstractMatrix, 
                     router_probs::AbstractMatrix)
    N = size(router_probs, 1)  # num_experts
    total_assignments = length(expert_indices)
    
    if total_assignments == 0
        return 0.0f0
    end
    
    # SIMPLE approach using broadcasting - avoid complex operations
    # Flatten expert_indices for easier processing
    flat_indices = vec(expert_indices)
    
    # Use broadcasting to create counting matrix (no loops, no complex maps)
    expert_range = reshape(1:N, N, 1)  # Column vector of expert IDs
    flat_indices_row = reshape(flat_indices, 1, :)  # Row vector of assignments
    
    # Count matches using broadcasting (much simpler than map)
    matches = (expert_range .== flat_indices_row)  # N x total_assignments boolean matrix
    f = sum(matches, dims=2)[:] / Float32(total_assignments)  # Count and normalize
    
    # Average probability assigned to each expert
    P = mean(router_probs, dims=2)[:]
    
    # Simple element-wise multiplication and sum
    return loss_fn.α * N * sum(f .* P)
end

"""
    FluxZLoss(β::Float32=0.001f0)

Flux-compatible Z-loss for preventing router logit explosion.
Loss = β × (1/B) × Σ(log Σe^x_i)²
"""
struct FluxZLoss <: LoadBalancingLoss
    β::Float32
end

FluxZLoss() = FluxZLoss(0.001f0)

function compute_loss(loss_fn::FluxZLoss, router_logits::AbstractMatrix)
    log_sum_exp = logsumexp_flux(router_logits; dims=1)
    z_loss = mean(log_sum_exp .^ 2)
    return loss_fn.β * z_loss
end

function logsumexp_flux(x::AbstractMatrix; dims)
    max_x = maximum(x; dims=dims)
    return max_x .+ log.(sum(exp.(x .- max_x); dims=dims))
end