using CUDA

struct ParallelConfig
    max_concurrent_experts::Int
    stream_pool_size::Int
    enable_async_execution::Bool
    memory_pool_size::Int
    batch_processing_threshold::Int
end

function ParallelConfig()
    device = get_device()
    if device isa GPUDevice
        return ParallelConfig(
            min(16, CUDA.attribute(device.device_id, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)),
            8,
            true,
            100,
            4
        )
    else
        return ParallelConfig(
            Threads.nthreads(),
            1,
            false,
            50,
            2
        )
    end
end

mutable struct StreamPool
    streams::Vector{CUDA.CuStream}
    available::Vector{Bool}
    lock::ReentrantLock
end

function StreamPool(size::Int)
    streams = [CUDA.CuStream() for _ in 1:size]
    available = fill(true, size)
    return StreamPool(streams, available, ReentrantLock())
end

function acquire_stream(pool::StreamPool)
    lock(pool.lock) do
        for i in 1:length(pool.available)
            if pool.available[i]
                pool.available[i] = false
                return pool.streams[i], i
            end
        end
        return nothing, 0
    end
end

function release_stream(pool::StreamPool, stream_id::Int)
    lock(pool.lock) do
        if stream_id > 0 && stream_id <= length(pool.available)
            pool.available[stream_id] = true
        end
    end
end

struct ExpertDispatcher
    config::ParallelConfig
    stream_pool::Union{StreamPool, Nothing}
    expert_queues::Vector{Channel{Any}}
    worker_tasks::Vector{Task}
end

function ExpertDispatcher(config::ParallelConfig)
    stream_pool = if config.enable_async_execution && gpu_available()
        StreamPool(config.stream_pool_size)
    else
        nothing
    end
    
    expert_queues = [Channel{Any}(100) for _ in 1:config.max_concurrent_experts]
    worker_tasks = Task[]
    
    dispatcher = ExpertDispatcher(config, stream_pool, expert_queues, worker_tasks)
    
    for i in 1:config.max_concurrent_experts
        task = @async expert_worker(dispatcher, i)
        push!(dispatcher.worker_tasks, task)
    end
    
    return dispatcher
end

function expert_worker(dispatcher::ExpertDispatcher, worker_id::Int)
    queue = dispatcher.expert_queues[worker_id]
    
    while true
        try
            job = take!(queue)
            if job === :shutdown
                break
            end
            
            execute_expert_job(dispatcher, job, worker_id)
        catch e
            @warn "Expert worker $worker_id error: $e"
        end
    end
end

function execute_expert_job(dispatcher::ExpertDispatcher, job, worker_id::Int)
    expert, input, output, weight = job
    
    if dispatcher.config.enable_async_execution && !isnothing(dispatcher.stream_pool)
        stream, stream_id = acquire_stream(dispatcher.stream_pool)
        if !isnothing(stream)
            try
                execute_expert_gpu_async(expert, input, output, weight, stream)
            finally
                release_stream(dispatcher.stream_pool, stream_id)
            end
        else
            execute_expert_gpu_sync(expert, input, output, weight)
        end
    else
        execute_expert_cpu(expert, input, output, weight)
    end
end

function execute_expert_gpu_async(expert, input, output, weight, stream::CUDA.CuStream)
    CUDA.stream!(stream) do
        if expert.expert_type == :gated
            temp1 = get_buffer(Float32, size(input), get_device())
            temp2 = get_buffer(Float32, size(input), get_device())
            
            try
                launch_expert_forward!(output, expert.w1, expert.w2, expert.w3, 
                                     input, temp1, temp2)
                
                if weight != 1.0f0
                    output .*= weight
                end
            finally
                return_buffer(temp1, get_device())
                return_buffer(temp2, get_device())
            end
        end
    end
end

function execute_expert_gpu_sync(expert, input, output, weight)
    if expert.expert_type == :gated
        temp1 = get_buffer(Float32, size(input), get_device())
        temp2 = get_buffer(Float32, size(input), get_device())
        
        try
            launch_expert_forward!(output, expert.w1, expert.w2, expert.w3, 
                                 input, temp1, temp2)
            
            if weight != 1.0f0
                output .*= weight
            end
            
            CUDA.synchronize()
        finally
            return_buffer(temp1, get_device())
            return_buffer(temp2, get_device())
        end
    end
end

function execute_expert_cpu(expert, input, output, weight)
    if expert.expert_type == :gated
        hidden_dim = size(expert.w1, 1)
        temp1 = zeros(Float32, hidden_dim, size(input, 2))
        temp2 = zeros(Float32, hidden_dim, size(input, 2))
        
        mul!(temp1, expert.w1, input)
        mul!(temp2, expert.w3, input)
        
        @inbounds for i in eachindex(temp1)
            gate_val = temp1[i]
            silu_val = gate_val * (1.0f0 / (1.0f0 + exp(-gate_val)))
            temp1[i] = silu_val * temp2[i]
        end
        
        mul!(output, expert.w2, temp1)
        
        if weight != 1.0f0
            output .*= weight
        end
    end
end

function dispatch_experts(dispatcher::ExpertDispatcher, experts, inputs, outputs, weights)
    num_experts = length(experts)
    jobs_per_worker = cld(num_experts, dispatcher.config.max_concurrent_experts)
    
    for worker_id in 1:dispatcher.config.max_concurrent_experts
        start_idx = (worker_id - 1) * jobs_per_worker + 1
        end_idx = min(worker_id * jobs_per_worker, num_experts)
        
        if start_idx <= num_experts
            for expert_idx in start_idx:end_idx
                job = (experts[expert_idx], inputs[expert_idx], 
                      outputs[expert_idx], weights[expert_idx])
                put!(dispatcher.expert_queues[worker_id], job)
            end
        end
    end
end

function shutdown_dispatcher(dispatcher::ExpertDispatcher)
    for queue in dispatcher.expert_queues
        put!(queue, :shutdown)
    end
    
    for task in dispatcher.worker_tasks
        wait(task)
    end
end

struct BatchProcessor
    max_batch_size::Int
    optimal_batch_size::Int
    device::AbstractDevice
    workspace::Union{GPUWorkspace, Nothing}
end

function BatchProcessor(device::AbstractDevice, model_config)
    max_batch_size = optimal_batch_size(
        model_config.dim * model_config.n_layers, 
        model_config.seq_len
    )
    
    optimal_batch = max(1, max_batch_size ÷ 2)
    
    workspace = if device isa GPUDevice
        GPUWorkspace(device, max_batch_size, model_config.seq_len,
                    model_config.dim, model_config.moe_num_experts, 
                    model_config.n_heads)
    else
        nothing
    end
    
    return BatchProcessor(max_batch_size, optimal_batch, device, workspace)
end

function process_batch(processor::BatchProcessor, inputs::Vector, process_fn::Function)
    batch_size = length(inputs)
    
    if batch_size <= processor.optimal_batch_size
        return process_single_batch(processor, inputs, process_fn)
    else
        return process_multi_batch(processor, inputs, process_fn)
    end
end

function process_single_batch(processor::BatchProcessor, inputs::Vector, process_fn::Function)
    device = processor.device
    results = []
    
    if device isa GPUDevice && !isnothing(processor.workspace)
        batch_inputs = stack_inputs_gpu(inputs, device)
        batch_outputs = process_fn(batch_inputs, processor.workspace)
        results = unstack_outputs_gpu(batch_outputs)
    else
        batch_inputs = stack_inputs_cpu(inputs)
        batch_outputs = process_fn(batch_inputs)
        results = unstack_outputs_cpu(batch_outputs)
    end
    
    return results
end

function process_multi_batch(processor::BatchProcessor, inputs::Vector, process_fn::Function)
    num_batches = cld(length(inputs), processor.optimal_batch_size)
    all_results = []
    
    for batch_idx in 1:num_batches
        start_idx = (batch_idx - 1) * processor.optimal_batch_size + 1
        end_idx = min(batch_idx * processor.optimal_batch_size, length(inputs))
        
        batch_inputs = inputs[start_idx:end_idx]
        batch_results = process_single_batch(processor, batch_inputs, process_fn)
        append!(all_results, batch_results)
    end
    
    return all_results
end

function stack_inputs_gpu(inputs::Vector, device::GPUDevice)
    return with_device(device) do
        if all(x -> x isa Array, inputs)
            return cu(reduce((a, b) -> cat(a, b, dims=2), inputs))
        else
            return map(x -> to_device(x, device), inputs)
        end
    end
end

function unstack_outputs_gpu(outputs)
    if outputs isa CuArray
        cpu_outputs = Array(outputs)
        return [cpu_outputs[:, i] for i in 1:size(cpu_outputs, 2)]
    else
        return [cpu(x) for x in outputs]
    end
end

function stack_inputs_cpu(inputs::Vector)
    if all(x -> x isa Array, inputs)
        return reduce((a, b) -> cat(a, b, dims=2), inputs)
    else
        return inputs
    end
end

function unstack_outputs_cpu(outputs)
    if outputs isa Array && ndims(outputs) > 1
        return [outputs[:, i] for i in 1:size(outputs, 2)]
    else
        return outputs
    end
end

function parallel_reduce(values::AbstractVector{T}, op::Function, 
                        device::CPUDevice) where T
    return reduce(op, values)
end

function parallel_reduce(values::CuArray{T}, op::Function, 
                        device::GPUDevice) where T
    return CUDA.reduce(op, values)
end

function parallel_map(f::Function, inputs, device::CPUDevice)
    if Threads.nthreads() > 1
        return map(f, inputs)
    else
        return map(f, inputs)
    end
end

function parallel_map(f::Function, inputs::CuArray, device::GPUDevice)
    return f.(inputs)
end