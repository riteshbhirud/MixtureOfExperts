using CUDA
using Statistics

function create_cuda_moe(; kwargs...)
    config = CudaMoEConfig(; kwargs...)
    return CudaMoELayer(config)
end

function to_cuda(x::AbstractArray)
    return x |> gpu
end

function to_cpu(x::CuArray)
    return Array(x)
end

function get_cuda_expert_stats(expert_indices::CuArray{Int32, 2}, num_experts::Int)
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

function generate_realistic_input(input_dim::Int, batch_size::Int; 
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
    
    return input_data
end

function generate_sequence_inputs(config, batch_size::Int, sequence_length::Int = 512)
    inputs = []
    for _ in 1:sequence_length
        input_data = generate_realistic_input(config.input_dim, batch_size)
        push!(inputs, input_data)
    end
    return inputs
end