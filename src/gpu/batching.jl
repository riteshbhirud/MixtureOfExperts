using CUDA

struct BatchingConfig
    max_batch_size::Int
    target_memory_usage::Float64
    dynamic_batching::Bool
    adaptive_sizing::Bool
    prefetch_factor::Int
end

function BatchingConfig(device::AbstractDevice)
    if device isa GPUDevice
        mem_info = device_memory_info()
        available_gb = mem_info.free / (1024^3)
        
        return BatchingConfig(
            min(128, max(8, floor(Int, available_gb * 16))),
            0.75,
            true,
            true,
            2
        )
    else
        return BatchingConfig(
            min(32, Threads.nthreads() * 4),
            0.8,
            false,
            false,
            1
        )
    end
end

struct BatchedInput{T}
    data::T
    batch_indices::Vector{Int}
    sequence_lengths::Vector{Int}
    attention_mask::Union{AbstractArray, Nothing}
    metadata::Dict{Symbol, Any}
end

function BatchedInput(inputs::Vector{T}) where T
    if isempty(inputs)
        throw(ArgumentError("Cannot create batch from empty input vector"))
    end
    
    batch_size = length(inputs)
    sequence_lengths = [size(input, 1) for input in inputs]
    max_seq_len = maximum(sequence_lengths)
    
    if T <: AbstractVector
        input_dim = length(inputs[1])
        batched_data = zeros(eltype(inputs[1]), input_dim, batch_size)
        
        for (i, input) in enumerate(inputs)
            batched_data[:, i] = input
        end
    elseif T <: AbstractMatrix
        input_dim = size(inputs[1], 1)
        batched_data = zeros(eltype(inputs[1]), input_dim, max_seq_len, batch_size)
        
        for (i, input) in enumerate(inputs)
            seq_len = size(input, 2)
            batched_data[:, 1:seq_len, i] = input
        end
    else
        batched_data = inputs
    end
    
    attention_mask = create_attention_mask(sequence_lengths, max_seq_len)
    batch_indices = collect(1:batch_size)
    
    metadata = Dict{Symbol, Any}(
        :max_seq_len => max_seq_len,
        :batch_size => batch_size,
        :original_types => [typeof(x) for x in inputs]
    )
    
    return BatchedInput(batched_data, batch_indices, sequence_lengths, 
                       attention_mask, metadata)
end

function create_attention_mask(sequence_lengths::Vector{Int}, max_seq_len::Int)
    batch_size = length(sequence_lengths)
    mask = zeros(Bool, max_seq_len, batch_size)
    
    for (i, seq_len) in enumerate(sequence_lengths)
        mask[1:seq_len, i] .= true
    end
    
    return mask
end

struct DynamicBatcher
    config::BatchingConfig
    device::AbstractDevice
    current_batch::Vector{Any}
    batch_ready_event::Condition
    processing_lock::ReentrantLock
    stats::Dict{Symbol, Any}
end

function DynamicBatcher(config::BatchingConfig, device::AbstractDevice)
    stats = Dict{Symbol, Any}(
        :total_batches => 0,
        :total_items => 0,
        :avg_batch_size => 0.0,
        :memory_usage => Float64[],
        :processing_times => Float64[]
    )
    
    return DynamicBatcher(
        config, device, Any[], Condition(), ReentrantLock(), stats
    )
end

function add_to_batch!(batcher::DynamicBatcher, item)
    lock(batcher.processing_lock) do
        push!(batcher.current_batch, item)
        
        if length(batcher.current_batch) >= batcher.config.max_batch_size
            notify(batcher.batch_ready_event)
        elseif batcher.config.adaptive_sizing
            estimated_memory = estimate_batch_memory(batcher.current_batch, batcher.device)
            if estimated_memory >= batcher.config.target_memory_usage
                notify(batcher.batch_ready_event)
            end
        end
    end
end

function get_batch!(batcher::DynamicBatcher, timeout::Float64 = 0.1)
    start_time = time()
    
    while time() - start_time < timeout
        lock(batcher.processing_lock) do
            if !isempty(batcher.current_batch)
                batch = copy(batcher.current_batch)
                empty!(batcher.current_batch)
                
                batcher.stats[:total_batches] += 1
                batcher.stats[:total_items] += length(batch)
                batcher.stats[:avg_batch_size] = batcher.stats[:total_items] / batcher.stats[:total_batches]
                
                return BatchedInput(batch)
            end
        end
        
        wait(batcher.batch_ready_event)
    end
    
    lock(batcher.processing_lock) do
        if !isempty(batcher.current_batch)
            batch = copy(batcher.current_batch)
            empty!(batcher.current_batch)
            return BatchedInput(batch)
        end
    end
    
    return nothing
end

function estimate_batch_memory(batch::Vector, device::AbstractDevice)
    if isempty(batch)
        return 0.0
    end
    
    total_elements = 0
    for item in batch
        if item isa AbstractArray
            total_elements += length(item)
        elseif hasmethod(length, (typeof(item),))
            total_elements += length(item)
        else
            total_elements += 100
        end
    end
    
    bytes_per_element = 4
    total_bytes = total_elements * bytes_per_element
    
    if device isa GPUDevice
        mem_info = device_memory_info()
        return total_bytes / mem_info.total
    else
        total_memory = Sys.total_memory()
        return total_bytes / total_memory
    end
end

struct BatchedExecution
    batch_size::Int
    sequence_length::Int
    expert_outputs::AbstractArray
    routing_decisions::AbstractArray
    attention_weights::Union{AbstractArray, Nothing}
    temporary_buffers::Dict{Symbol, AbstractArray}
end

function create_batched_execution(batch::BatchedInput, config, device::AbstractDevice)
    batch_size = length(batch.batch_indices)
    max_seq_len = batch.metadata[:max_seq_len]
    
    if device isa GPUDevice
        expert_outputs = CUDA.zeros(Float32, config.dim, batch_size, config.moe_num_experts)
        routing_decisions = CUDA.zeros(Int32, config.moe_top_k, batch_size)
        attention_weights = CUDA.zeros(Float32, max_seq_len, max_seq_len, config.n_heads, batch_size)
        
        temp_buffers = Dict{Symbol, AbstractArray}(
            :router_logits => CUDA.zeros(Float32, config.moe_num_experts, batch_size),
            :gate_weights => CUDA.zeros(Float32, config.moe_top_k, batch_size),
            :hidden_states => CUDA.zeros(Float32, config.dim, batch_size),
            :query_states => CUDA.zeros(Float32, config.dim, batch_size),
            :key_states => CUDA.zeros(Float32, config.dim, batch_size),
            :value_states => CUDA.zeros(Float32, config.dim, batch_size)
        )
    else
        expert_outputs = zeros(Float32, config.dim, batch_size, config.moe_num_experts)
        routing_decisions = zeros(Int32, config.moe_top_k, batch_size)
        attention_weights = zeros(Float32, max_seq_len, max_seq_len, config.n_heads, batch_size)
        
        temp_buffers = Dict{Symbol, AbstractArray}(
            :router_logits => zeros(Float32, config.moe_num_experts, batch_size),
            :gate_weights => zeros(Float32, config.moe_top_k, batch_size),
            :hidden_states => zeros(Float32, config.dim, batch_size),
            :query_states => zeros(Float32, config.dim, batch_size),
            :key_states => zeros(Float32, config.dim, batch_size),
            :value_states => zeros(Float32, config.dim, batch_size)
        )
    end
    
    return BatchedExecution(
        batch_size, max_seq_len, expert_outputs, routing_decisions,
        attention_weights, temp_buffers
    )
end

function process_batched_moe(batch::BatchedInput, execution::BatchedExecution, 
                            experts, router, config, device::AbstractDevice)
    
    batch_size = execution.batch_size
    input_data = ensure_device(batch.data, device)
    
    fill!(execution.temporary_buffers[:router_logits], 0)
    
    if device isa GPUDevice
        process_batched_moe_gpu(batch, execution, experts, router, config)
    else
        process_batched_moe_cpu(batch, execution, experts, router, config)
    end
    
    return execution.expert_outputs
end

function process_batched_moe_gpu(batch::BatchedInput, execution::BatchedExecution,
                                experts, router, config)
    
    launch_router_forward!(
        execution.temporary_buffers[:router_logits],
        router.weight.weight,
        batch.data
    )
    
    launch_topk!(
        execution.routing_decisions,
        execution.temporary_buffers[:gate_weights],
        execution.temporary_buffers[:router_logits],
        config.moe_top_k
    )
    
    launch_softmax!(
        execution.temporary_buffers[:gate_weights],
        execution.temporary_buffers[:gate_weights]
    )
    
    for expert_idx in 1:length(experts)
        expert = experts[expert_idx]
        expert_input = batch.data
        expert_output = view(execution.expert_outputs, :, :, expert_idx)
        
        temp1 = get_buffer(Float32, size(expert_input), get_device())
        temp2 = get_buffer(Float32, size(expert_input), get_device())
        
        try
            launch_expert_forward!(expert_output, expert.w1, expert.w2, expert.w3,
                                 expert_input, temp1, temp2)
        finally
            return_buffer(temp1, get_device())
            return_buffer(temp2, get_device())
        end
    end
    
    final_output = execution.temporary_buffers[:hidden_states]
    launch_weighted_sum!(
        final_output,
        execution.expert_outputs,
        execution.temporary_buffers[:gate_weights],
        execution.routing_decisions
    )
    
    CUDA.synchronize()
end

function process_batched_moe_cpu(batch::BatchedInput, execution::BatchedExecution,
                                experts, router, config)
    
    mul!(execution.temporary_buffers[:router_logits], router.weight.weight, batch.data)
    
    for col in 1:size(execution.temporary_buffers[:router_logits], 2)
        logits_col = view(execution.temporary_buffers[:router_logits], :, col)
        
        topk_indices = partialsortperm(logits_col, 1:config.moe_top_k, rev=true)
        execution.routing_decisions[:, col] = topk_indices
        
        selected_logits = logits_col[topk_indices]
        selected_logits .-= maximum(selected_logits)
        exp_logits = exp.(selected_logits)
        gate_weights = exp_logits ./ sum(exp_logits)
        execution.temporary_buffers[:gate_weights][:, col] = gate_weights
    end
    
    Threads.@threads for expert_idx in 1:length(experts)
        expert = experts[expert_idx]
        expert_input = batch.data
        expert_output = view(execution.expert_outputs, :, :, expert_idx)
        
        hidden_dim = size(expert.w1, 1)
        temp1 = zeros(Float32, hidden_dim, size(expert_input, 2))
        temp2 = zeros(Float32, hidden_dim, size(expert_input, 2))
        
        mul!(temp1, expert.w1, expert_input)
        mul!(temp2, expert.w3, expert_input)
        
        @inbounds for i in eachindex(temp1)
            gate_val = temp1[i]
            silu_val = gate_val * (1.0f0 / (1.0f0 + exp(-gate_val)))
            temp1[i] = silu_val * temp2[i]
        end
        
        mul!(expert_output, expert.w2, temp1)
    end
    
    fill!(execution.temporary_buffers[:hidden_states], 0)
    
    for col in 1:execution.batch_size
        for k in 1:config.moe_top_k
            expert_idx = execution.routing_decisions[k, col]
            weight = execution.temporary_buffers[:gate_weights][k, col]
            
            if expert_idx > 0
                @inbounds for dim in 1:size(execution.expert_outputs, 1)
                    execution.temporary_buffers[:hidden_states][dim, col] += 
                        weight * execution.expert_outputs[dim, col, expert_idx]
                end
            end
        end
    end
end

function unbatch_results(execution::BatchedExecution, original_batch::BatchedInput)
    results = []
    
    for i in 1:execution.batch_size
        if execution.temporary_buffers[:hidden_states] isa CuArray
            result = Array(execution.temporary_buffers[:hidden_states][:, i])
        else
            result = execution.temporary_buffers[:hidden_states][:, i]
        end
        push!(results, result)
    end
    
    return results
end

function adaptive_batch_size(current_size::Int, processing_time::Float64, 
                           target_time::Float64, device::AbstractDevice)
    
    if processing_time < target_time * 0.8
        new_size = min(current_size + 4, 128)
    elseif processing_time > target_time * 1.2
        new_size = max(current_size - 2, 1)
    else
        new_size = current_size
    end
    
    if device isa GPUDevice
        mem_info = device_memory_info()
        max_safe_size = floor(Int, mem_info.free * 0.7 / (1024^2))
        new_size = min(new_size, max_safe_size)
    end
    
    return new_size
end