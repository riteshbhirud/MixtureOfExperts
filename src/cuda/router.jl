using CUDA
using NNlib
using LinearAlgebra

function cuda_router_forward!(router::CudaRouter, x::CuArray{Float32, 2}, 
                             router_logits::CuArray{Float32, 2},
                             router_probs::CuArray{Float32, 2},
                             expert_indices::CuArray{Int32, 2},
                             expert_gates::CuArray{Float32, 2};
                             training::Bool = false)
    
    batch_size = size(x, 2)
    num_experts = size(router.weight, 1)
    
    mul!(router_logits, router.weight, x)
    
    if training && router.noise_scale > 0.0f0
        if !isnothing(router.noise_weight)
            noise_logits = router.noise_weight * x
            noise = CUDA.randn(Float32, size(router_logits)) .* 
                   NNlib.softplus.(noise_logits) .* router.noise_scale
            router_logits .+= noise
        else
            noise = CUDA.randn(Float32, size(router_logits)) .* router.noise_scale
            router_logits .+= noise
        end
    end
    
    router_probs .= NNlib.softmax(router_logits; dims=1)
    
    try
        threads_per_block = min(batch_size, 256)
        blocks = cld(batch_size, threads_per_block)
        @cuda threads=threads_per_block blocks=blocks topk_kernel!(
            expert_indices, expert_gates, router_probs, router.k, num_experts, batch_size)
        CUDA.synchronize()
    catch e
        @warn "CUDA TopK kernel failed, falling back to CPU: $e"
        router_probs_cpu = Array(router_probs)
        expert_indices_cpu = zeros(Int32, router.k, batch_size)
        expert_gates_cpu = zeros(Float32, router.k, batch_size)
        
        for t in 1:batch_size
            probs = router_probs_cpu[:, t]
            indices = collect(1:num_experts)
            
            for i in 1:router.k
                max_idx = i
                for j in (i+1):num_experts
                    if probs[indices[j]] > probs[indices[max_idx]]
                        max_idx = j
                    end
                end
                if max_idx != i
                    indices[i], indices[max_idx] = indices[max_idx], indices[i]
                end
            end
            
            topk_indices = indices[1:router.k]
            expert_indices_cpu[:, t] = topk_indices
            
            selected_probs = probs[topk_indices]
            expert_gates_cpu[:, t] = selected_probs ./ sum(selected_probs)
        end
        
        expert_indices .= CuArray(expert_indices_cpu)
        expert_gates .= CuArray(expert_gates_cpu)
    end
    
    return router_logits, router_probs, expert_indices, expert_gates
end

function cuda_compute_balance_loss(expert_indices::CuArray{Int32, 2},
                                  router_probs::CuArray{Float32, 2},
                                  expert_counts::CuArray{Float32, 1},
                                  config::CudaMoEConfig)
    
    batch_size = size(expert_indices, 2)
    num_experts = config.num_experts
    
    expert_counts .= 0.0f0
    threads_per_block = 256
    blocks = cld(batch_size, threads_per_block)
    
    @cuda threads=threads_per_block blocks=blocks count_assignments_kernel!(
        expert_counts, expert_indices, config.top_k, num_experts, batch_size)
    CUDA.synchronize()
    
    total_assignments = sum(expert_counts)
    if total_assignments > 0
        expert_counts ./= total_assignments
    end
    
    P = mean(router_probs, dims=2)[:]
    
    balance_loss = config.balance_weight * num_experts * sum(expert_counts .* P)
    
    return balance_loss
end