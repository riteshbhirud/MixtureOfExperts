using CUDA

mutable struct MemoryPool
    buffers::Dict{Tuple{DataType, Tuple}, Vector{Any}}
    max_cached::Int
    current_usage::Int
    peak_usage::Int
end

function MemoryPool(max_cached::Int = 100)
    return MemoryPool(
        Dict{Tuple{DataType, Tuple}, Vector{Any}}(),
        max_cached,
        0,
        0
    )
end

const MEMORY_POOLS = Dict{AbstractDevice, MemoryPool}()

function get_memory_pool(device::AbstractDevice)
    if !haskey(MEMORY_POOLS, device)
        MEMORY_POOLS[device] = MemoryPool()
    end
    return MEMORY_POOLS[device]
end

function allocate_buffer(::Type{T}, size::Tuple, device::CPUDevice) where T
    return zeros(T, size)
end

function allocate_buffer(::Type{T}, size::Tuple, device::GPUDevice) where T
    return with_device(device) do
        CUDA.zeros(T, size)
    end
end

function get_buffer(::Type{T}, size::Tuple, device::AbstractDevice) where T
    pool = get_memory_pool(device)
    key = (T, size)
    
    if haskey(pool.buffers, key) && !isempty(pool.buffers[key])
        buffer = pop!(pool.buffers[key])
        fill!(buffer, zero(T))
        return buffer
    else
        buffer = allocate_buffer(T, size, device)
        pool.current_usage += 1
        pool.peak_usage = max(pool.peak_usage, pool.current_usage)
        return buffer
    end
end

function return_buffer(buffer::AbstractArray, device::AbstractDevice)
    pool = get_memory_pool(device)
    key = (eltype(buffer), size(buffer))
    
    if !haskey(pool.buffers, key)
        pool.buffers[key] = Any[]
    end
    
    if length(pool.buffers[key]) < pool.max_cached
        push!(pool.buffers[key], buffer)
    else
        pool.current_usage -= 1
    end
end

function clear_memory_pool(device::AbstractDevice)
    pool = get_memory_pool(device)
    for buffers in values(pool.buffers)
        empty!(buffers)
    end
    pool.current_usage = 0
    
    if device isa GPUDevice
        with_device(device) do
            CUDA.reclaim()
        end
    end
end

function memory_stats(device::AbstractDevice)
    pool = get_memory_pool(device)
    return (
        current_usage = pool.current_usage,
        peak_usage = pool.peak_usage,
        cached_buffers = sum(length, values(pool.buffers))
    )
end

mutable struct BufferCache
    device::AbstractDevice
    buffers::Dict{Symbol, Any}
end

function BufferCache(device::AbstractDevice)
    return BufferCache(device, Dict{Symbol, Any}())
end

function get_cached_buffer(cache::BufferCache, key::Symbol, ::Type{T}, size::Tuple) where T
    if haskey(cache.buffers, key)
        buffer = cache.buffers[key]
        if eltype(buffer) == T && Base.size(buffer) == size
            fill!(buffer, zero(T))
            return buffer
        else
            return_buffer(buffer, cache.device)
        end
    end
    
    buffer = get_buffer(T, size, cache.device)
    cache.buffers[key] = buffer
    return buffer
end

function clear_buffer_cache(cache::BufferCache)
    for buffer in values(cache.buffers)
        return_buffer(buffer, cache.device)
    end
    empty!(cache.buffers)
end

struct GPUWorkspace
    device::GPUDevice
    expert_outputs::Vector{CuArray{Float32, 2}}
    router_logits::CuArray{Float32, 2}
    attention_weights::CuArray{Float32, 3}
    temp_buffers::Dict{Symbol, CuArray}
    streams::Vector{CUDA.CuStream}
end

function GPUWorkspace(device::GPUDevice, batch_size::Int, seq_len::Int, 
                     dim::Int, num_experts::Int, num_heads::Int)
    return with_device(device) do
        expert_outputs = [CUDA.zeros(Float32, dim, batch_size) for _ in 1:num_experts]
        router_logits = CUDA.zeros(Float32, num_experts, batch_size)
        attention_weights = CUDA.zeros(Float32, seq_len, num_heads, batch_size)
        
        temp_buffers = Dict{Symbol, CuArray}(
            :hidden => CUDA.zeros(Float32, dim, batch_size),
            :query => CUDA.zeros(Float32, dim, batch_size),
            :key => CUDA.zeros(Float32, dim, batch_size),
            :value => CUDA.zeros(Float32, dim, batch_size),
            :expert_input => CUDA.zeros(Float32, dim, batch_size),
            :gate_weights => CUDA.zeros(Float32, num_experts, batch_size)
        )
        
        num_streams = min(8, num_experts)
        streams = [CUDA.CuStream() for _ in 1:num_streams]
        
        GPUWorkspace(device, expert_outputs, router_logits, attention_weights, 
                    temp_buffers, streams)
    end
end

function resize_workspace!(workspace::GPUWorkspace, new_batch_size::Int, 
                          new_seq_len::Int, dim::Int, num_experts::Int, num_heads::Int)
    with_device(workspace.device) do
        for i in 1:length(workspace.expert_outputs)
            if size(workspace.expert_outputs[i], 2) < new_batch_size
                workspace.expert_outputs[i] = CUDA.zeros(Float32, dim, new_batch_size)
            end
        end
        
        if size(workspace.router_logits, 2) < new_batch_size
            workspace.router_logits = CUDA.zeros(Float32, num_experts, new_batch_size)
        end
        
        if size(workspace.attention_weights, 1) < new_seq_len || 
           size(workspace.attention_weights, 3) < new_batch_size
            workspace.attention_weights = CUDA.zeros(Float32, new_seq_len, num_heads, new_batch_size)
        end
        
        for (key, buffer) in workspace.temp_buffers
            if size(buffer, 2) < new_batch_size
                workspace.temp_buffers[key] = CUDA.zeros(Float32, size(buffer, 1), new_batch_size)
            end
        end
    end
end

function prefetch_to_gpu(data, device::GPUDevice)
    return with_device(device) do
        if data isa Array
            return CuArray(data)
        elseif data isa Tuple
            return map(x -> prefetch_to_gpu(x, device), data)
        elseif data isa NamedTuple
            return map(x -> prefetch_to_gpu(x, device), data)
        else
            return data
        end
    end
end

function pin_memory(x::Array)
    if CUDA.functional()
        return CUDA.pin(x)
    else
        return x
    end
end

function async_memcpy!(dst::CuArray, src::Array, stream::CUDA.CuStream)
    CUDA.unsafe_copyto!(dst, 1, src, 1, length(src); async=true, stream=stream)
end

function async_memcpy!(dst::Array, src::CuArray, stream::CUDA.CuStream)
    CUDA.unsafe_copyto!(dst, 1, src, 1, length(src); async=true, stream=stream)
end

function memory_efficient_softmax!(x::CuArray{Float32, 2})
    CUDA.@sync begin
        CUDA.@cuda threads=256 blocks=size(x, 2) softmax_kernel!(x)
    end
end

function softmax_kernel!(x::CuDeviceArray{Float32, 2})
    idx = (blockIdx().x - 1) + 1
    if idx <= size(x, 2)
        col = view(x, :, idx)
        
        max_val = -Inf32
        for i in 1:length(col)
            max_val = max(max_val, col[i])
        end
        
        sum_exp = 0.0f0
        for i in 1:length(col)
            col[i] = exp(col[i] - max_val)
            sum_exp += col[i]
        end
        
        for i in 1:length(col)
            col[i] /= sum_exp
        end
    end
    
    return nothing
end