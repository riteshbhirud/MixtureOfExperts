using CUDA

function cuda_moe_forward!(moe::CudaMoELayer, x::CuArray{Float32, 2}; 
                          training::Bool = false)
    
    batch_size = size(x, 2)
    config = moe.config
    
    if batch_size > size(moe.router_logits_buf, 2)
        error("Batch size $batch_size exceeds maximum buffer size")
    end
    
    router_logits = view(moe.router_logits_buf, :, 1:batch_size)
    router_probs = view(moe.router_probs_buf, :, 1:batch_size)
    expert_indices = view(moe.expert_indices_buf, :, 1:batch_size)
    expert_gates = view(moe.expert_gates_buf, :, 1:batch_size)
    
    cuda_router_forward!(moe.router, x, router_logits, router_probs, 
                        expert_indices, expert_gates; training=training)
    
    output = CUDA.zeros(Float32, config.output_dim, batch_size)
    cuda_process_experts!(output, moe.experts, x, expert_indices, 
                         expert_gates, config)
    
    balance_loss = 0.0f0
    if training
        balance_loss = cuda_compute_balance_loss(expert_indices, router_probs,
                                               moe.expert_counts_buf, config)
    end
    
    return output, balance_loss
end