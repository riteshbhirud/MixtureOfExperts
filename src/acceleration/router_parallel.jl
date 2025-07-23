using CUDA
using LinearAlgebra

struct ParallelRouterConfig
    use_batched_computation::Bool
    parallel_topk::Bool
    fused_softmax_topk::Bool
    cache_computations::Bool
    use_mixed_precision::Bool
    enable_load_balancing::Bool
end

function ParallelRouterConfig(device::AbstractDevice)
    if device isa GPUDevice
        return ParallelRouterConfig(true, true, true, true, true, true)
    else
        return ParallelRouterConfig(true, false, false, false, false, false)
    end
end

struct RouterComputationPlan
    batch_size::Int
    num_experts::Int
    top_k::Int
    computation_order::Vector{Symbol}
    memory_layout::Symbol
    parallelization_strategy::Symbol
end

function create_router_plan(batch_size::Int, num_experts::Int, top_k::Int, 
                          config::ParallelRouterConfig, device::AbstractDevice)
    
    computation_order = [:logits, :probabilities, :topk, :gates]
    
    if config.fused_softmax_topk
        computation_order = [:logits, :fused_softmax_topk, :gates]
    end
    
    memory_layout = if batch_size > 64 && device isa GPUDevice
        :column_major
    else
        :row_major
    end
    
    parallelization_strategy = if device isa GPUDevice
        if batch_size > 32
            :batch_parallel
        else
            :expert_parallel
        end
    else
        :sequential
    end
    
    return RouterComputationPlan(
        batch_size, num_experts, top_k, computation_order, 
        memory_layout, parallelization_strategy
    )
end

function parallel_router_forward!(router_logits::AbstractArray, router_probs::AbstractArray,
                                expert_indices::AbstractArray, expert_gates::AbstractArray,
                                router_weight::AbstractArray, input::AbstractArray,
                                gate_type::GatingMechanism, config::ParallelRouterConfig,
                                device::AbstractDevice)
    
    plan = create_router_plan(size(input, 2), size(router_weight, 1), 
                            size(expert_indices, 1), config, device)
    
    if device isa GPUDevice
        parallel_router_forward_gpu!(router_logits, router_probs, expert_indices, 
                                   expert_gates, router_weight, input, gate_type, 
                                   plan, config)
    else
        parallel_router_forward_cpu!(router_logits, router_probs, expert_indices, 
                                   expert_gates, router_weight, input, gate_type, 
                                   plan, config)
    end
end

function parallel_router_forward_gpu!(router_logits::CuArray, router_probs::CuArray,
                                    expert_indices::CuArray, expert_gates::CuArray,
                                    router_weight::CuArray, input::CuArray,
                                    gate_type::GatingMechanism, plan::RouterComputationPlan,
                                    config::ParallelRouterConfig)
    
    for step in plan.computation_order
        if step == :logits
            compute_router_logits_gpu!(router_logits, router_weight, input, plan)
        elseif step == :probabilities
            compute_router_probabilities_gpu!(router_probs, router_logits, plan)
        elseif step == :topk
            compute_topk_gpu!(expert_indices, expert_gates, router_probs, plan)
        elseif step == :fused_softmax_topk
            compute_fused_softmax_topk_gpu!(expert_indices, expert_gates, router_logits, plan)
        elseif step == :gates
            normalize_gates_gpu!(expert_gates, plan)
        end
    end
    
    CUDA.synchronize()
end

function compute_router_logits_gpu!(router_logits::CuArray, router_weight::CuArray,
                                  input::CuArray, plan::RouterComputationPlan)
    
    if plan.parallelization_strategy == :batch_parallel
        launch_router_forward!(router_logits, router_weight, input)
    else
        CUDA.@sync CUDA.CUBLAS.gemm!('N', 'N', 1.0f0, router_weight, input, 0.0f0, router_logits)
    end
end

function compute_router_probabilities_gpu!(router_probs::CuArray, router_logits::CuArray,
                                         plan::RouterComputationPlan)
    
    if plan.parallelization_strategy == :batch_parallel
        launch_softmax!(router_probs, router_logits)
    else
        copyto!(router_probs, router_logits)
        memory_efficient_softmax!(router_probs)
    end
end

function compute_topk_gpu!(expert_indices::CuArray, expert_gates::CuArray,
                         router_probs::CuArray, plan::RouterComputationPlan)
    
    if plan.parallelization_strategy == :batch_parallel
        indices_temp = CuArray{Int32}(undef, size(expert_indices))
        launch_topk!(indices_temp, expert_gates, router_probs, plan.top_k)
        copyto!(expert_indices, indices_temp)
    else
        compute_topk_sequential_gpu!(expert_indices, expert_gates, router_probs, plan)
    end
end

function compute_topk_sequential_gpu!(expert_indices::CuArray, expert_gates::CuArray,
                                    router_probs::CuArray, plan::RouterComputationPlan)
    
    for batch_idx in 1:plan.batch_size
        probs_col = view(router_probs, :, batch_idx)
        indices_col = view(expert_indices, :, batch_idx)
        gates_col = view(expert_gates, :, batch_idx)
        
        cpu_probs = Array(probs_col)
        topk_indices = partialsortperm(cpu_probs, 1:plan.top_k, rev=true)
        
        copyto!(indices_col, CuArray(topk_indices))
        copyto!(gates_col, CuArray(cpu_probs[topk_indices]))
    end
end

function compute_fused_softmax_topk_gpu!(expert_indices::CuArray, expert_gates::CuArray,
                                       router_logits::CuArray, plan::RouterComputationPlan)
    
    threads = min(256, plan.batch_size)
    blocks = cld(plan.batch_size, threads)
    
    indices_temp = CuArray{Int32}(undef, size(expert_indices))
    
    CUDA.@cuda threads=threads blocks=blocks fused_softmax_topk_kernel!(
        indices_temp, expert_gates, router_logits, Int32(plan.top_k)
    )
    
    copyto!(expert_indices, indices_temp)
end

function fused_softmax_topk_kernel!(indices::CuDeviceArray{Int32, 2},
                                  values::CuDeviceArray{Float32, 2},
                                  logits::CuDeviceArray{Float32, 2},
                                  k::Int32)
    
    batch_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if batch_idx <= size(logits, 2)
        num_experts = size(logits, 1)
        
        max_logit = -Inf32
        for i in 1:num_experts
            max_logit = max(max_logit, logits[i, batch_idx])
        end
        
        sum_exp = 0.0f0
        for i in 1:num_experts
            sum_exp += exp(logits[i, batch_idx] - max_logit)
        end
        
        for pos in 1:k
            max_idx = 1
            max_prob = -Inf32
            
            for i in 1:num_experts
                prob = exp(logits[i, batch_idx] - max_logit) / sum_exp
                if prob > max_prob
                    skip_this = false
                    for prev_pos in 1:(pos-1)
                        if indices[prev_pos, batch_idx] == i
                            skip_this = true
                            break
                        end
                    end
                    
                    if !skip_this
                        max_prob = prob
                        max_idx = i
                    end
                end
            end
            
            indices[pos, batch_idx] = max_idx
            values[pos, batch_idx] = max_prob
        end
        
        gate_sum = 0.0f0
        for pos in 1:k
            gate_sum += values[pos, batch_idx]
        end
        
        if gate_sum > 0.0f0
            for pos in 1:k
                values[pos, batch_idx] /= gate_sum
            end
        end
    end
    
    return nothing
end

function normalize_gates_gpu!(expert_gates::CuArray, plan::RouterComputationPlan)
    threads = min(256, plan.batch_size)
    blocks = cld(plan.batch_size, threads)
    
    CUDA.@cuda threads=threads blocks=blocks normalize_gates_kernel!(expert_gates)
end

function normalize_gates_kernel!(gates::CuDeviceArray{Float32, 2})
    batch_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if batch_idx <= size(gates, 2)
        gate_sum = 0.0f0
        for k in 1:size(gates, 1)
            gate_sum += gates[k, batch_idx]
        end
        
        if gate_sum > 0.0f0
            for k in 1:size(gates, 1)
                gates[k, batch_idx] /= gate_sum
            end
        end
    end
    
    return nothing
end

function parallel_router_forward_cpu!(router_logits::AbstractArray, router_probs::AbstractArray,
                                    expert_indices::AbstractArray, expert_gates::AbstractArray,
                                    router_weight::AbstractArray, input::AbstractArray,
                                    gate_type::GatingMechanism, plan::RouterComputationPlan,
                                    config::ParallelRouterConfig)
    
    mul!(router_logits, router_weight, input)
    
    if config.use_batched_computation && plan.batch_size > 8
        parallel_softmax_cpu!(router_probs, router_logits)
        parallel_topk_cpu!(expert_indices, expert_gates, router_probs, plan.top_k)
    else
        sequential_softmax_topk_cpu!(expert_indices, expert_gates, router_logits, plan.top_k)
    end
    
    normalize_gates_cpu!(expert_gates)
end

function parallel_softmax_cpu!(router_probs::AbstractArray, router_logits::AbstractArray)
    Threads.@threads for col in 1:size(router_logits, 2)
        logits_col = view(router_logits, :, col)
        probs_col = view(router_probs, :, col)
        
        max_logit = maximum(logits_col)
        
        exp_sum = 0.0f0
        for i in eachindex(logits_col)
            prob = exp(logits_col[i] - max_logit)
            probs_col[i] = prob
            exp_sum += prob
        end
        
        probs_col ./= exp_sum
    end
end

function parallel_topk_cpu!(expert_indices::AbstractArray, expert_gates::AbstractArray,
                          router_probs::AbstractArray, k::Int)
    
    Threads.@threads for col in 1:size(router_probs, 2)
        probs_col = view(router_probs, :, col)
        indices_col = view(expert_indices, :, col)
        gates_col = view(expert_gates, :, col)
        
        topk_indices = partialsortperm(probs_col, 1:k, rev=true)
        
        indices_col .= topk_indices
        gates_col .= probs_col[topk_indices]
    end
end

function sequential_softmax_topk_cpu!(expert_indices::AbstractArray, expert_gates::AbstractArray,
                                    router_logits::AbstractArray, k::Int)
    
    for col in 1:size(router_logits, 2)
        logits_col = view(router_logits, :, col)
        indices_col = view(expert_indices, :, col)
        gates_col = view(expert_gates, :, col)
        
        max_logit = maximum(logits_col)
        exp_logits = exp.(logits_col .- max_logit)
        probs = exp_logits ./ sum(exp_logits)
        
        topk_indices = partialsortperm(probs, 1:k, rev=true)
        
        indices_col .= topk_indices
        gates_col .= probs[topk_indices]
    end
end

function normalize_gates_cpu!(expert_gates::AbstractArray)
    for col in 1:size(expert_gates, 2)
        gates_col = view(expert_gates, :, col)
        gate_sum = sum(gates_col)
        
        if gate_sum > 0.0f0
            gates_col ./= gate_sum
        end
    end
end

struct RouterCache
    device::AbstractDevice
    cached_logits::Dict{UInt64, AbstractArray}
    cached_probs::Dict{UInt64, AbstractArray}
    cached_indices::Dict{UInt64, AbstractArray}
    cache_hits::Int
    cache_misses::Int
    max_cache_size::Int
end

function RouterCache(device::AbstractDevice, max_size::Int = 100)
    return RouterCache(
        device,
        Dict{UInt64, AbstractArray}(),
        Dict{UInt64, AbstractArray}(),
        Dict{UInt64, AbstractArray}(),
        0, 0, max_size
    )
end

function compute_input_hash(input::AbstractArray)
    return hash(input, UInt64(0x1234567890abcdef))
end

function get_cached_computation(cache::RouterCache, input::AbstractArray, 
                              computation_type::Symbol)
    
    input_hash = compute_input_hash(input)
    
    cached_dict = if computation_type == :logits
        cache.cached_logits
    elseif computation_type == :probs
        cache.cached_probs
    elseif computation_type == :indices
        cache.cached_indices
    else
        return nothing
    end
    
    if haskey(cached_dict, input_hash)
        cache.cache_hits += 1
        return cached_dict[input_hash]
    else
        cache.cache_misses += 1
        return nothing
    end
end

function cache_computation!(cache::RouterCache, input::AbstractArray,
                          result::AbstractArray, computation_type::Symbol)
    
    if length(cache.cached_logits) >= cache.max_cache_size
        clear_oldest_cache_entries!(cache)
    end
    
    input_hash = compute_input_hash(input)
    
    cached_dict = if computation_type == :logits
        cache.cached_logits
    elseif computation_type == :probs
        cache.cached_probs
    elseif computation_type == :indices
        cache.cached_indices
    else
        return
    end
    
    cached_dict[input_hash] = copy(result)
end

function clear_oldest_cache_entries!(cache::RouterCache)
    num_to_remove = cache.max_cache_size ÷ 4
    
    for cached_dict in [cache.cached_logits, cache.cached_probs, cache.cached_indices]
        keys_to_remove = collect(keys(cached_dict))[1:min(num_to_remove, length(cached_dict))]
        for key in keys_to_remove
            delete!(cached_dict, key)
        end
    end
end

function router_cache_stats(cache::RouterCache)
    total_requests = cache.cache_hits + cache.cache_misses
    hit_rate = total_requests > 0 ? cache.cache_hits / total_requests : 0.0
    
    return (
        hit_rate = hit_rate,
        cache_hits = cache.cache_hits,
        cache_misses = cache.cache_misses,
        cached_entries = length(cache.cached_logits) + length(cache.cached_probs) + length(cache.cached_indices)
    )
end