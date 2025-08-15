using AMDGPU
using NNlib

function amdgpu_standard_expert_forward!(expert::AMDGPUStandardExpert, 
                                        x::ROCArray{Float32, 2})
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

function amdgpu_gated_expert_forward!(expert::AMDGPUGatedExpert, 
                                     x::ROCArray{Float32, 2})
    gate = NNlib.swish.(expert.w1 * x)
    up = expert.w3 * x
    h = gate .* up
    return expert.w2 * h
end

function amdgpu_process_experts!(output::ROCArray{Float32, 2},
                                experts::Vector,
                                x::ROCArray{Float32, 2},
                                expert_indices::ROCArray{Int32, 2},
                                expert_gates::ROCArray{Float32, 2},
                                config::AMDGPUMoEConfig)
    
    batch_size = size(x, 2)
    output .= 0.0f0
    
    for expert_id in 1:config.num_experts
        expert = experts[expert_id]
        
        if config.expert_type == :gated
            expert_output = amdgpu_gated_expert_forward!(expert, x)
        else
            expert_output = amdgpu_standard_expert_forward!(expert, x)
        end
        
        threads_per_block = min(batch_size, 256)
        blocks = cld(batch_size, threads_per_block)
        
        @roc groupsize=threads_per_block gridsize=blocks apply_gates_kernel!(
            output, expert_output, expert_indices, expert_gates, 
            expert_id, config.top_k, config.output_dim, batch_size)
        AMDGPU.synchronize()
    end
    
    return output
end