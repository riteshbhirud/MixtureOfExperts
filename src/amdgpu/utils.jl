using AMDGPU
using Statistics

function create_amdgpu_moe(; kwargs...)
    config = AMDGPUMoEConfig(; kwargs...)
    return AMDGPUMoELayer(config)
end

function to_amdgpu(x::AbstractArray)
    return ROCArray(x)
end

function to_cpu(x::ROCArray)
    return Array(x)
end

function get_amdgpu_expert_stats(expert_indices::ROCArray{Int32, 2}, num_experts::Int)
    expert_counts = zeros(Int, num_experts)
    expert_indices_cpu = Array(expert_indices)
    
    for idx in expert_indices_cpu
        if idx > 0 && idx <= num_experts
            expert_counts[idx] += 1
        end
    end
    
    total_assignments = sum(expert_counts)
    usage_percentages = expert_counts ./ max(total_assignments, 1) .* 100
    
    return expert_counts, usage_percentages
end

function generate_realistic_input_amdgpu(input_dim::Int, batch_size::Int; 
                                        sequence_length::Int = 1024)
    base_std = 0.8f0
    
    num_base_patterns = min(input_dim ÷ 4, 128)
    base_patterns = randn(Float32, num_base_patterns, input_dim) .* base_std
    
    mixing_weights = randn(Float32, num_base_patterns, batch_size)
    mixing_weights = softmax(mixing_weights, dims=1)
    
    input_data = base_patterns' * mixing_weights
    
    noise_level = 0.1f0
    input_data .+= randn(Float32, size(input_data)) .* noise_level
    
    input_data = clamp.(input_data, -3.0f0, 3.0f0)
    
    return ROCArray(input_data)
end

function generate_sequence_inputs_amdgpu(config, batch_size::Int, sequence_length::Int = 512)
    inputs = []
    for _ in 1:sequence_length
        input_data = generate_realistic_input_amdgpu(config.input_dim, batch_size)
        push!(inputs, input_data)
    end
    return inputs
end

function create_synchronized_moe_pair_amdgpu(config::AMDGPUMoEConfig)
    cpu_config = MoEConfig(
        num_experts = config.num_experts,
        expert_type = config.expert_type,
        input_dim = config.input_dim,
        hidden_dim = config.hidden_dim,
        output_dim = config.output_dim,
        top_k = config.top_k
    )
    
    cpu_moe = MoELayer(cpu_config)
    amdgpu_moe = AMDGPUMoELayer(config)
    
    for i in 1:config.num_experts
        if config.expert_type == :gated
            cpu_expert = cpu_moe.experts[i]
            amdgpu_expert = amdgpu_moe.experts[i]
            
            amdgpu_expert.w1 .= ROCArray(cpu_expert.w1.weight)
            amdgpu_expert.w2 .= ROCArray(cpu_expert.w2.weight)
            amdgpu_expert.w3 .= ROCArray(cpu_expert.w3.weight)
        else
            cpu_expert = cpu_moe.experts[i]
            amdgpu_expert = amdgpu_moe.experts[i]
            
            amdgpu_expert.w1 .= ROCArray(cpu_expert.w1.weight)
            amdgpu_expert.w2 .= ROCArray(cpu_expert.w2.weight)
            
            if !isnothing(amdgpu_expert.b1) && !isnothing(cpu_expert.w1.bias)
                amdgpu_expert.b1 .= ROCArray(cpu_expert.w1.bias)
            end
            if !isnothing(amdgpu_expert.b2) && !isnothing(cpu_expert.w2.bias)
                amdgpu_expert.b2 .= ROCArray(cpu_expert.w2.bias)
            end
        end
    end
    
    amdgpu_moe.router.weight .= ROCArray(cpu_moe.router.weight)
    if !isnothing(amdgpu_moe.router.noise_weight) && !isnothing(cpu_moe.router.noise_weight)
        amdgpu_moe.router.noise_weight .= ROCArray(cpu_moe.router.noise_weight)
    end
    
    return cpu_moe, amdgpu_moe, true
end