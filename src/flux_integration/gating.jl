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