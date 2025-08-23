"""
    FluxTopKGating(k::Int, use_softmax_renorm::Bool=true)

TopK gating mechanism compatible with Flux layers.
Implements the routing logic: g_{i,t} = s_{i,t} if s_{i,t} ∈ TopK, 0 otherwise
"""
struct FluxTopKGating <: GatingMechanism
    k::Int
    use_softmax_renorm::Bool
end

FluxTopKGating(k::Int) = FluxTopKGating(k, true)

function compute_gates(gate::FluxTopKGating, router_logits::AbstractMatrix)
    router_probs = Flux.softmax(router_logits; dims=1)
    
    num_experts, batch_size = size(router_logits)
    

    token_results = map(1:batch_size) do t
        topk_indices = partialsortperm(router_probs[:, t], 1:gate.k, rev=true)
        
        selected_probs = router_probs[topk_indices, t]
        
        if gate.use_softmax_renorm
            gates = selected_probs ./ sum(selected_probs)
        else
            gates = selected_probs
        end
        
        return (indices=topk_indices, gates=gates)
    end
    
    expert_indices = reduce(hcat, [result.indices for result in token_results])
    expert_gates = reduce(hcat, [result.gates for result in token_results])
    
    return expert_indices, expert_gates, router_probs
end

"""
    FluxSwitchGating()

Switch Transformer gating - special case of TopK with k=1.
"""
struct FluxSwitchGating <: GatingMechanism end

function compute_gates(gate::FluxSwitchGating, router_logits::AbstractMatrix)
    return compute_gates(FluxTopKGating(1), router_logits)
end


#2. ExpertChoice

"""
    FluxExpertChoiceGating(capacity_factor::Float32=1.25f0, specialization_strength::Float32=1.0f0)

Expert choice routing where experts select tokens instead of tokens selecting experts.
Each expert selects its top tokens up to capacity limit with proper specialization support.

# Arguments
- `capacity_factor::Float32`: Controls how many tokens each expert can process
- `specialization_strength::Float32`: Controls preference vs load balancing trade-off
  - Higher values (>1.0) allow more specialization based on scores
  - Lower values (<1.0) enforce more uniform load balancing  
  - 1.0 provides balanced behavior
"""
struct FluxExpertChoiceGating <: GatingMechanism
    capacity_factor::Float32
    specialization_strength::Float32
end

FluxExpertChoiceGating() = FluxExpertChoiceGating(1.25f0, 1.0f0)
FluxExpertChoiceGating(capacity_factor::Float32) = FluxExpertChoiceGating(capacity_factor, 1.0f0)

function compute_gates(gate::FluxExpertChoiceGating, router_logits::AbstractMatrix)
    num_experts, batch_size = size(router_logits)
    
    router_probs = Flux.softmax(router_logits; dims=1)
    
    base_capacity = ceil(Int, batch_size * gate.capacity_factor / num_experts)
    base_capacity = min(base_capacity, batch_size) 
    
    expert_token_preferences = Vector{Vector{Tuple{Int, Float32}}}(undef, num_experts)
    
    for expert in 1:num_experts
        raw_probs = router_probs[expert, :]
        
        if gate.specialization_strength != 1.0f0
            max_prob = maximum(raw_probs)
            min_prob = minimum(raw_probs)
            
            if max_prob > min_prob  
                normalized_probs = (raw_probs .- min_prob) ./ (max_prob - min_prob)
                specialized_probs = normalized_probs .^ gate.specialization_strength
                expert_scores = specialized_probs .* (max_prob - min_prob) .+ min_prob
            else
                expert_scores = raw_probs  
            end
        else
            expert_scores = raw_probs
        end
        

        capacity_flexibility = gate.specialization_strength >= 1.0f0 ? 
                               min(0.5f0, (gate.specialization_strength - 1.0f0) * 0.3f0) : 0.0f0
        
        expert_capacity = base_capacity + ceil(Int, base_capacity * capacity_flexibility)
        expert_capacity = min(expert_capacity, batch_size)
        
        token_score_pairs = [(token, expert_scores[token]) for token in 1:batch_size]
        sort!(token_score_pairs, by=x->x[2], rev=true) 
        
        selected_pairs = token_score_pairs[1:min(expert_capacity, length(token_score_pairs))]
        
        expert_token_preferences[expert] = selected_pairs
    end
    
    max_experts_per_token = min(base_capacity, num_experts)
    
    expert_indices = zeros(Int, max_experts_per_token, batch_size)
    expert_gates = zeros(Float32, max_experts_per_token, batch_size)
    token_expert_counts = zeros(Int, batch_size)
    
    global_preferences = Tuple{Int, Int, Float32}[]
    
    for expert in 1:num_experts
        for (token, score) in expert_token_preferences[expert]
            raw_strength = score * gate.specialization_strength
            
            if gate.specialization_strength > 1.0f0
                expert_scores = [pair[2] for pair in expert_token_preferences[expert]]
                max_expert_score = maximum(expert_scores)
                if max_expert_score > 0 && score >= 0.9f0 * max_expert_score 
                    raw_strength *= 1.2f0 
                end
            end
            
            push!(global_preferences, (token, expert, raw_strength))
        end
    end
    
    sort!(global_preferences, by=x->x[3], rev=true)
    
    expert_assignment_counts = zeros(Int, num_experts)
    
    for (token, expert, strength) in global_preferences
        can_assign = (
            token_expert_counts[token] < max_experts_per_token &&  
            expert_assignment_counts[expert] < base_capacity * 2  
        )
        
        if can_assign
            token_expert_counts[token] += 1
            expert_assignment_counts[expert] += 1
            slot = token_expert_counts[token]
            
            expert_indices[slot, token] = expert
            expert_gates[slot, token] = router_probs[expert, token]
        end
    end
    

    total_assignments = sum(expert_assignment_counts)
    if total_assignments > 0
        target_per_expert = total_assignments / num_experts
        max_imbalance = maximum(expert_assignment_counts) - minimum(expert_assignment_counts)
        
        if max_imbalance > target_per_expert * 0.7f0  
            # could Implement gentle rebalancing logic here if needed,  For now, we accept some imbalance for better specialization
        end
    end
    
    for token in 1:batch_size
        active_experts = token_expert_counts[token]
        if active_experts > 0
            active_gates = expert_gates[1:active_experts, token]
            gate_sum = sum(active_gates)
            
            if gate_sum > 0.0f0
                expert_gates[1:active_experts, token] ./= gate_sum
            else
                expert_gates[1:active_experts, token] .= 1.0f0 / active_experts
            end
        end
    end
    
    return expert_indices, expert_gates, router_probs
end

function Base.show(io::IO, gate::FluxExpertChoiceGating)
    print(io, "FluxExpertChoiceGating(capacity_factor=$(gate.capacity_factor), specialization_strength=$(gate.specialization_strength))")
end