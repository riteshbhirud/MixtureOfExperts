using CUDA
using NNlib

function cuda_standard_expert_forward!(expert::CudaStandardExpert, 
                                      x::CuArray{Float32, 2})
    h = expert.w1 * x
    if !isnothing(expert.b1)
        h .+= expert.b1
    end
    h = NNlib.gelu.(h)
    
    y = expert.w2 * h
    if !isnothing(expert.b2)
        y .+= expert.b2
    end
    
    return y
end

function cuda_gated_expert_forward!(expert::CudaGatedExpert, 
                                   x::CuArray{Float32, 2})
    gate = NNlib.swish.(expert.w1 * x)
    up = expert.w3 * x
    h = gate .* up
    return expert.w2 * h
end

function cuda_process_experts!(output::CuArray{Float32, 2},
                              experts::Vector,
                              x::CuArray{Float32, 2},
                              expert_indices::CuArray{Int32, 2},
                              expert_gates::CuArray{Float32, 2},
                              config::CudaMoEConfig)
    
    batch_size = size(x, 2)
    output .= 0.0f0
    
    for expert_id in 1:config.num_experts
        expert = experts[expert_id]
        
        if config.expert_type == :gated
            expert_output = cuda_gated_expert_forward!(expert, x)
        else
            expert_output = cuda_standard_expert_forward!(expert, x)
        end
        
        threads_per_block = min(batch_size, 256)
        blocks = cld(batch_size, threads_per_block)
        
        @cuda threads=threads_per_block blocks=blocks apply_gates_kernel!(
            output, expert_output, expert_indices, expert_gates, 
            expert_id, config.top_k, config.output_dim, batch_size)
        CUDA.synchronize()
    end
    
    return output
end