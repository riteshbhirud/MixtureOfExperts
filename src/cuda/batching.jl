"""
GPU Batch Processing and Memory Management Utilities

High-performance GPU batch processing system with intelligent memory allocation,
workspace management, and dynamic batching strategies optimized for MoE workloads.
Provides efficient memory pooling and automatic kernel configuration selection.
"""

struct GPUBatchConfig{T<:AbstractFloat}
    preferred_batch_size::Int
    min_batch_size::Int
    max_batch_size::Int
    
    max_memory_usage_gb::Float64
    memory_alignment::Int
    enable_memory_pooling::Bool
    pool_preallocation_mb::Int
    
    auto_kernel_selection::Bool
    preferred_block_size::Int
    enable_stream_overlap::Bool
    max_concurrent_streams::Int
    
    expert_batch_threshold::Int       # Min tokens before batching experts
    parallel_expert_threshold::Int    # Min experts before parallel processing
    enable_expert_fusion::Bool        # Fuse multiple small experts
    
    # Load balancing
    dynamic_load_balancing::Bool
    load_balance_window::Int
    imbalance_threshold::T
    
    function GPUBatchConfig{T}(;
        preferred_batch_size::Int = 128,
        min_batch_size::Int = 16,
        max_batch_size::Int = 512,
        max_memory_usage_gb::Float64 = 8.0,
        memory_alignment::Int = 32,
        enable_memory_pooling::Bool = true,
        pool_preallocation_mb::Int = 256,
        auto_kernel_selection::Bool = true,
        preferred_block_size::Int = 256,
        enable_stream_overlap::Bool = true,
        max_concurrent_streams::Int = 4,
        expert_batch_threshold::Int = 8,
        parallel_expert_threshold::Int = 4,
        enable_expert_fusion::Bool = true,
        dynamic_load_balancing::Bool = true,
        load_balance_window::Int = 100,
        imbalance_threshold::T = T(0.2)
    ) where T<:AbstractFloat
        
        # Validate parameters
        if min_batch_size > preferred_batch_size || preferred_batch_size > max_batch_size
            throw(ArgumentError("Batch size parameters must satisfy: min ≤ preferred ≤ max"))
        end
        
        if max_memory_usage_gb <= 0
            throw(ArgumentError("Max memory usage must be positive"))
        end
        
        if !ispow2(memory_alignment) || memory_alignment <= 0
            throw(ArgumentError("Memory alignment must be a positive power of 2"))
        end
        
        return new{T}(
            preferred_batch_size, min_batch_size, max_batch_size,
            max_memory_usage_gb, memory_alignment, enable_memory_pooling, pool_preallocation_mb,
            auto_kernel_selection, preferred_block_size, enable_stream_overlap, max_concurrent_streams,
            expert_batch_threshold, parallel_expert_threshold, enable_expert_fusion,
            dynamic_load_balancing, load_balance_window, imbalance_threshold
        )
    end
end

# Convenience constructor
GPUBatchConfig(args...; kwargs...) = GPUBatchConfig{Float32}(args...; kwargs...)

# Dynamic memory pool for efficient GPU memory management
mutable struct GPUMemoryPool{T<:AbstractFloat}
    # Pre-allocated memory pools
    tensor_pool::Dict{Tuple{Int, Int}, Vector{CuMatrix{T}}}
    vector_pool::Dict{Int, Vector{CuVector{T}}}
    index_pool::Dict{Tuple{Int, Int}, Vector{CuMatrix{Int32}}}
    
    # Pool management
    pool_usage_stats::Dict{String, Int}
    total_allocated_bytes::Ref{Int64}
    peak_allocated_bytes::Ref{Int64}
    allocation_count::Ref{Int}
    
    # Configuration
    max_pool_size_mb::Int
    cleanup_threshold::Float64
    enable_auto_cleanup::Bool
    
    # Performance tracking
    cache_hits::Ref{Int}
    cache_misses::Ref{Int}
    allocation_time::Ref{Float64}
    
    function GPUMemoryPool{T}(;
        max_pool_size_mb::Int = 512,
        cleanup_threshold::Float64 = 0.8,
        enable_auto_cleanup::Bool = true
    ) where T<:AbstractFloat
        
        return new{T}(
            Dict{Tuple{Int, Int}, Vector{CuMatrix{T}}}(),
            Dict{Int, Vector{CuVector{T}}}(),
            Dict{Tuple{Int, Int}, Vector{CuMatrix{Int32}}}(),
            Dict{String, Int}(),
            Ref(Int64(0)), Ref(Int64(0)), Ref(0),
            max_pool_size_mb, cleanup_threshold, enable_auto_cleanup,
            Ref(0), Ref(0), Ref(0.0)
        )
    end
end

# Global memory pool instance
const _global_memory_pool = Ref{Union{Nothing, GPUMemoryPool{Float32}}}(nothing)

function get_memory_pool()
    if isnothing(_global_memory_pool[])
        _global_memory_pool[] = GPUMemoryPool{Float32}()
    end
    return _global_memory_pool[]
end

"""
    allocate_tensor_from_pool(rows, cols, [T])

Allocate matrix from memory pool with automatic fallback to new allocation.
"""
function allocate_tensor_from_pool(::Type{T}, rows::Int, cols::Int) where T<:AbstractFloat
    pool = get_memory_pool()
    start_time = time()
    
    try
        key = (rows, cols)
        
        # Try to get from pool
        if haskey(pool.tensor_pool, key) && !isempty(pool.tensor_pool[key])
            tensor = pop!(pool.tensor_pool[key])
            pool.cache_hits[] += 1
            
            # Clear tensor data
            fill!(tensor, T(0))
            
            return tensor
        end
        
        # Pool miss - allocate new tensor
        pool.cache_misses[] += 1
        tensor = gpu_zeros(T, rows, cols; aligned=true)
        
        # Update allocation tracking
        tensor_bytes = sizeof(tensor)
        pool.total_allocated_bytes[] += tensor_bytes
        pool.peak_allocated_bytes[] = max(pool.peak_allocated_bytes[], pool.total_allocated_bytes[])
        pool.allocation_count[] += 1
        
        # Auto-cleanup if needed
        if pool.enable_auto_cleanup
            check_pool_cleanup!(pool)
        end
        
        return tensor
        
    finally
        pool.allocation_time[] += time() - start_time
    end
end

function allocate_tensor_from_pool(rows::Int, cols::Int)
    return allocate_tensor_from_pool(Float32, rows, cols)
end

"""
    return_tensor_to_pool(tensor)

Return matrix to memory pool for reuse.
"""
function return_tensor_to_pool!(tensor::CuMatrix{T}) where T<:AbstractFloat
    pool = get_memory_pool()
    
    rows, cols = size(tensor)
    key = (rows, cols)
    
    # Check pool size limits
    current_size_mb = pool.total_allocated_bytes[] / (1024^2)
    if current_size_mb < pool.max_pool_size_mb
        
        # Add to appropriate pool
        if !haskey(pool.tensor_pool, key)
            pool.tensor_pool[key] = CuMatrix{T}[]
        end
        
        push!(pool.tensor_pool[key], tensor)
        
        # Update usage statistics
        pool_key = "tensor_$(rows)x$(cols)"
        pool.pool_usage_stats[pool_key] = get(pool.pool_usage_stats, pool_key, 0) + 1
        
    else
        # Pool is full - let GC handle this tensor
        pool.total_allocated_bytes[] -= sizeof(tensor)
    end
end

"""
    allocate_vector_from_pool(length, [T])

Allocate vector from memory pool.
"""
function allocate_vector_from_pool(::Type{T}, length::Int) where T<:AbstractFloat
    pool = get_memory_pool()
    
    # Try to get from pool
    if haskey(pool.vector_pool, length) && !isempty(pool.vector_pool[length])
        vector = pop!(pool.vector_pool[length])
        pool.cache_hits[] += 1
        
        # Clear vector data
        fill!(vector, T(0))
        
        return vector
    end
    
    # Pool miss - allocate new vector
    pool.cache_misses[] += 1
    vector = gpu_zeros(T, length; aligned=true)
    
    # Update allocation tracking
    vector_bytes = sizeof(vector)
    pool.total_allocated_bytes[] += vector_bytes
    pool.allocation_count[] += 1
    
    return vector
end

function allocate_vector_from_pool(length::Int)
    return allocate_vector_from_pool(Float32, length)
end

"""
    return_vector_to_pool!(vector)

Return vector to memory pool for reuse.
"""
function return_vector_to_pool!(vector::CuVector{T}) where T<:AbstractFloat
    pool = get_memory_pool()
    
    length = size(vector, 1)
    
    # Check pool size limits
    current_size_mb = pool.total_allocated_bytes[] / (1024^2)
    if current_size_mb < pool.max_pool_size_mb
        
        # Add to appropriate pool
        if !haskey(pool.vector_pool, length)
            pool.vector_pool[length] = CuVector{T}[]
        end
        
        push!(pool.vector_pool[length], vector)
        
        # Update usage statistics
        pool_key = "vector_$length"
        pool.pool_usage_stats[pool_key] = get(pool.pool_usage_stats, pool_key, 0) + 1
        
    else
        # Pool is full
        pool.total_allocated_bytes[] -= sizeof(vector)
    end
end

"""
    check_pool_cleanup!(pool)

Check if memory pool needs cleanup and perform if necessary.
"""
function check_pool_cleanup!(pool::GPUMemoryPool{T}) where T<:AbstractFloat
    current_usage = pool.total_allocated_bytes[] / (1024^3)  # GB
    max_usage = pool.max_pool_size_mb / 1024  # Convert MB to GB
    
    usage_ratio = current_usage / max_usage
    
    if usage_ratio > pool.cleanup_threshold
        @debug "Performing memory pool cleanup: usage ratio = $usage_ratio"
        
        # Clear least recently used pools
        cleanup_pools!(pool)
        
        # Force garbage collection
        GC.gc()
        CUDA.reclaim()
    end
end

"""
    cleanup_pools!(pool)

Clean up memory pools by removing unused allocations.
"""
function cleanup_pools!(pool::GPUMemoryPool{T}) where T<:AbstractFloat
    
    # Clean tensor pools - keep only recently used sizes
    for (key, tensors) in pool.tensor_pool
        if length(tensors) > 2  # Keep a small buffer
            # Remove half of the cached tensors
            num_to_remove = length(tensors) ÷ 2
            for _ in 1:num_to_remove
                tensor = pop!(tensors)
                pool.total_allocated_bytes[] -= sizeof(tensor)
            end
        end
    end
    
    # Clean vector pools
    for (key, vectors) in pool.vector_pool
        if length(vectors) > 2
            num_to_remove = length(vectors) ÷ 2
            for _ in 1:num_to_remove
                vector = pop!(vectors)
                pool.total_allocated_bytes[] -= sizeof(vector)
            end
        end
    end
    
    # Clean index pools
    for (key, indices) in pool.index_pool
        if length(indices) > 2
            num_to_remove = length(indices) ÷ 2
            for _ in 1:num_to_remove
                index_matrix = pop!(indices)
                pool.total_allocated_bytes[] -= sizeof(index_matrix)
            end
        end
    end
end

# Batch workspace management for MoE operations
mutable struct GPUBatchWorkspace{T<:AbstractFloat}
    # Input/output buffers
    batch_input::CuMatrix{T}              # input_dim × max_batch_size
    batch_output::CuMatrix{T}             # output_dim × max_batch_size
    
    # Expert processing buffers  
    expert_inputs::Vector{CuMatrix{T}}    # One per expert
    expert_outputs::Vector{CuMatrix{T}}   # One per expert
    
    # Gating workspace
    router_logits::CuMatrix{T}            # num_experts × max_batch_size
    router_probs::CuMatrix{T}             # num_experts × max_batch_size
    expert_indices::CuMatrix{Int32}       # top_k × max_batch_size
    expert_gates::CuMatrix{T}             # top_k × max_batch_size
    
    # Intermediate computation buffers
    temp_buffers::Vector{CuMatrix{T}}     # Reusable temporary buffers
    index_buffers::Vector{CuMatrix{Int32}} # Reusable index buffers
    
    # Memory management
    allocated_batch_size::Int
    max_tokens_per_expert::Int
    total_workspace_bytes::Int64
    last_used_time::Float64
    
    # Performance tracking
    usage_count::Ref{Int}
    total_usage_time::Ref{Float64}
    
    function GPUBatchWorkspace{T}(
        config::GPUMoEConfig{T},
        batch_config::GPUBatchConfig{T},
        max_batch_size::Int
    ) where T<:AbstractFloat
        
        start_time = time()
        
        input_dim = config.input_dim
        output_dim = config.output_dim
        hidden_dim = config.hidden_dim
        num_experts = config.num_experts
        top_k = 2  # Hardcoded for TopKGating(2)
        
        # Estimate max tokens per expert (conservative)
        max_tokens_per_expert = max_batch_size * top_k ÷ num_experts + 32  # Add buffer
        
        # Allocate main input/output buffers
        batch_input = allocate_tensor_from_pool(T, input_dim, max_batch_size)
        batch_output = allocate_tensor_from_pool(T, output_dim, max_batch_size)
        
        # Allocate expert buffers
        expert_inputs = CuMatrix{T}[]
        expert_outputs = CuMatrix{T}[]
        
        for _ in 1:num_experts
            expert_input = allocate_tensor_from_pool(T, input_dim, max_tokens_per_expert)
            expert_output = allocate_tensor_from_pool(T, output_dim, max_tokens_per_expert)
            push!(expert_inputs, expert_input)
            push!(expert_outputs, expert_output)
        end
        
        # Allocate gating workspace
        router_logits = allocate_tensor_from_pool(T, num_experts, max_batch_size)
        router_probs = allocate_tensor_from_pool(T, num_experts, max_batch_size)
        expert_indices = CUDA.zeros(Int32, top_k, max_batch_size)  # Special allocation for Int32
        expert_gates = allocate_tensor_from_pool(T, top_k, max_batch_size)
        
        # Allocate temporary buffers for intermediate computations
        temp_buffers = CuMatrix{T}[]
        push!(temp_buffers, allocate_tensor_from_pool(T, hidden_dim, max_batch_size))  # For gated values
        push!(temp_buffers, allocate_tensor_from_pool(T, hidden_dim, max_batch_size))  # For intermediate
        push!(temp_buffers, allocate_tensor_from_pool(T, output_dim, max_batch_size))  # For output staging
        
        # Allocate index buffers
        index_buffers = CuMatrix{Int32}[]
        push!(index_buffers, CUDA.zeros(Int32, num_experts, max_batch_size))  # Expert assignments
        push!(index_buffers, CUDA.zeros(Int32, top_k, max_batch_size))        
        # Calculate total workspace size
        total_bytes = (
            sizeof(batch_input) + sizeof(batch_output) +
            sum(sizeof(ei) + sizeof(eo) for (ei, eo) in zip(expert_inputs, expert_outputs)) +
            sizeof(router_logits) + sizeof(router_probs) + sizeof(expert_indices) + sizeof(expert_gates) +
            sum(sizeof(tb) for tb in temp_buffers) +
            sum(sizeof(ib) for ib in index_buffers)
        )
        
        allocation_time = time() - start_time
        
        workspace = new{T}(
            batch_input, batch_output,
            expert_inputs, expert_outputs,
            router_logits, router_probs, expert_indices, expert_gates,
            temp_buffers, index_buffers,
            max_batch_size, max_tokens_per_expert, total_bytes, time(),
            Ref(0), Ref(0.0)
        )
        
        @debug "Allocated GPU batch workspace: $(total_bytes ÷ (1024^2)) MB in $(allocation_time * 1000) ms"
        
        return workspace
    end
end

# Convenience constructor
GPUBatchWorkspace(config::GPUMoEConfig{T}, batch_config::GPUBatchConfig{T}, max_batch_size::Int) where T = 
    GPUBatchWorkspace{T}(config, batch_config, max_batch_size)

"""
    get_batch_workspace(config, batch_config, batch_size)

Get or create batch workspace for given configuration and batch size.
"""
function get_batch_workspace(
    config::GPUMoEConfig{T},
    batch_config::GPUBatchConfig{T}, 
    batch_size::Int
) where T<:AbstractFloat
    
    # Determine required workspace size
    required_size = min(max(batch_size, batch_config.min_batch_size), batch_config.max_batch_size)
    
    # For now, create new workspace each time
    # In production, this could implement workspace caching
    workspace = GPUBatchWorkspace{T}(config, batch_config, required_size)
    
    workspace.usage_count[] += 1
    workspace.last_used_time = time()
    
    return workspace
end

"""
    release_batch_workspace!(workspace)

Release batch workspace back to pool or cleanup.
"""
function release_batch_workspace!(workspace::GPUBatchWorkspace{T}) where T<:AbstractFloat
    
    # Return tensors to memory pool
    return_tensor_to_pool!(workspace.batch_input)
    return_tensor_to_pool!(workspace.batch_output)
    
    for expert_input in workspace.expert_inputs
        return_tensor_to_pool!(expert_input)
    end
    
    for expert_output in workspace.expert_outputs
        return_tensor_to_pool!(expert_output)
    end
    
    return_tensor_to_pool!(workspace.router_logits)
    return_tensor_to_pool!(workspace.router_probs)
    return_tensor_to_pool!(workspace.expert_gates)
    
    for temp_buffer in workspace.temp_buffers
        return_tensor_to_pool!(temp_buffer)
    end
    
    # Note: Int32 matrices (expert_indices, index_buffers) are not returned to pool
    # since we don't have a separate Int32 pool - they'll be garbage collected
    
    @debug "Released batch workspace: $(workspace.total_workspace_bytes ÷ (1024^2)) MB"
end

"""
    optimize_batch_size(config, target_batch_size)

Optimize batch size based on GPU memory and performance characteristics.
"""
function optimize_batch_size(
    config::GPUMoEConfig{T},
    batch_config::GPUBatchConfig{T},
    target_batch_size::Int
) where T<:AbstractFloat
    
    # Check memory constraints
    available_memory_gb = CUDA.available_memory() / (1024^3)
    max_allowed_memory_gb = min(available_memory_gb * 0.8, batch_config.max_memory_usage_gb)
    
    # Estimate memory requirements for target batch size
    estimated_memory_gb = estimate_batch_memory_usage(config, target_batch_size)
    
    if estimated_memory_gb > max_allowed_memory_gb
        # Reduce batch size to fit memory constraints
        memory_ratio = max_allowed_memory_gb / estimated_memory_gb
        adjusted_batch_size = Int(floor(target_batch_size * memory_ratio))
        adjusted_batch_size = max(adjusted_batch_size, batch_config.min_batch_size)
        
        @debug "Reduced batch size from $target_batch_size to $adjusted_batch_size due to memory constraints"
        return adjusted_batch_size
    end
    
    # Check if batch size is within configured limits
    optimized_batch_size = clamp(target_batch_size, batch_config.min_batch_size, batch_config.max_batch_size)
    
    # Round to efficient sizes (multiples of 32 for better GPU utilization)
    if batch_config.memory_alignment > 1
        optimized_batch_size = (optimized_batch_size ÷ batch_config.memory_alignment) * batch_config.memory_alignment
        optimized_batch_size = max(optimized_batch_size, batch_config.min_batch_size)
    end
    
    return optimized_batch_size
end

"""
    estimate_batch_memory_usage(config, batch_size)

Estimate GPU memory usage for given configuration and batch size.
"""
function estimate_batch_memory_usage(config::GPUMoEConfig{T}, batch_size::Int) where T<:AbstractFloat
    
    input_dim = config.input_dim
    output_dim = config.output_dim
    hidden_dim = config.hidden_dim
    num_experts = config.num_experts
    top_k = 2  # Hardcoded for TopKGating(2)
    
    element_size = sizeof(T)
    
    # Main input/output buffers
    main_buffers = (input_dim + output_dim) * batch_size * element_size
    
    # Expert buffers (conservative estimate - all experts get max tokens)
    max_tokens_per_expert = batch_size * top_k ÷ num_experts + 32
    expert_buffers = num_experts * (input_dim + output_dim) * max_tokens_per_expert * element_size
    
    # Gating workspace
    gating_buffers = (num_experts * 2 + top_k) * batch_size * element_size  # router_logits, router_probs, expert_gates
    gating_buffers += top_k * batch_size * sizeof(Int32)  # expert_indices
    
    # Intermediate computation buffers
    temp_buffers = hidden_dim * batch_size * 3 * element_size  # temp computation buffers
    
    # Index buffers
    index_buffers = (num_experts * batch_size + top_k * batch_size) * sizeof(Int32)
    
    # Expert weight storage (not per-batch, but included for completeness)
    expert_weights = num_experts * (input_dim * hidden_dim * 2 + hidden_dim * output_dim) * element_size
    
    total_bytes = main_buffers + expert_buffers + gating_buffers + temp_buffers + index_buffers + expert_weights
    
    return total_bytes / (1024^3)  # Convert to GB
end

"""
    create_batch_streams(config, num_streams)

Create CUDA streams for overlapped batch processing.
"""
function create_batch_streams(batch_config::GPUBatchConfig{T}, num_streams::Int = 0) where T<:AbstractFloat
    
    if !batch_config.enable_stream_overlap
        return [CUDA.stream()]  # Return default stream
    end
    
    # Use configured number of streams if not specified
    if num_streams == 0
        num_streams = batch_config.max_concurrent_streams
    end
    
    # Limit to reasonable number based on GPU capabilities
    device_info = GPUDeviceInfo()
    max_streams = min(num_streams, device_info.multiprocessor_count ÷ 2)
    
    streams = CUDA.CuStream[]
    for _ in 1:max_streams
        push!(streams, CUDA.stream())
    end
    
    @debug "Created $max_streams CUDA streams for batch processing"
    
    return streams
end

"""
    select_optimal_kernel_config(operation, batch_size, problem_size)

Select optimal CUDA kernel configuration for given operation and problem size.
"""
function select_optimal_kernel_config(
    operation::Symbol,
    batch_size::Int,
    problem_size::Tuple{Int, Int},
    config::GPUMoEConfig{T}
) where T<:AbstractFloat
    
    if !config.device_info.compute_capability >= v"6.0"
        # Fallback for older GPUs
        return GPUKernelConfig(batch_size; threads_per_block=128)
    end
    
    if operation == :expert_forward
        # Optimize for expert computation
        input_dim, hidden_dim = problem_size
        
        if batch_size <= 32 && hidden_dim <= 512
            # Small batch, use more threads per block
            return GPUKernelConfig(batch_size; threads_per_block=256)
        elseif batch_size >= 128
            # Large batch, optimize for throughput
            return GPUKernelConfig(batch_size; threads_per_block=512)
        else
            # Medium batch, balanced approach
            return GPUKernelConfig(batch_size; threads_per_block=256)
        end
        
    elseif operation == :gating_computation
        num_experts, _ = problem_size
        
        if num_experts <= 32
            return GPUKernelConfig(batch_size; threads_per_block=min(512, num_experts * 8))
        else
            return GPUKernelConfig(batch_size; threads_per_block=256)
        end
        
    elseif operation == :token_routing
        total_assignments = problem_size[1] * problem_size[2]
        return GPUKernelConfig(total_assignments; threads_per_block=256)
        
    else
        return GPUKernelConfig(batch_size; threads_per_block=256)
    end
end

"""
    get_memory_pool_statistics()

Get comprehensive statistics about memory pool usage.
"""
function get_memory_pool_statistics()
    pool = get_memory_pool()
    
    tensor_pool_sizes = Dict{String, Int}()
    for ((rows, cols), tensors) in pool.tensor_pool
        tensor_pool_sizes["$(rows)x$(cols)"] = length(tensors)
    end
    
    vector_pool_sizes = Dict{String, Int}()
    for (length, vectors) in pool.vector_pool
        vector_pool_sizes["$length"] = length(vectors)
    end
    
    cache_hit_rate = pool.cache_hits[] + pool.cache_misses[] > 0 ? 
                     pool.cache_hits[] / (pool.cache_hits[] + pool.cache_misses[]) : 0.0
    
    return Dict{String, Any}(
        "memory_usage" => Dict(
            "total_allocated_mb" => pool.total_allocated_bytes[] / (1024^2),
            "peak_allocated_mb" => pool.peak_allocated_bytes[] / (1024^2),
            "max_pool_size_mb" => pool.max_pool_size_mb
        ),
        "allocation_stats" => Dict(
            "total_allocations" => pool.allocation_count[],
            "cache_hits" => pool.cache_hits[],
            "cache_misses" => pool.cache_misses[],
            "cache_hit_rate" => cache_hit_rate,
            "allocation_time_ms" => pool.allocation_time[] * 1000
        ),
        "pool_contents" => Dict(
            "tensor_pools" => tensor_pool_sizes,
            "vector_pools" => vector_pool_sizes,
            "usage_stats" => copy(pool.pool_usage_stats)
        ),
        "configuration" => Dict(
            "cleanup_threshold" => pool.cleanup_threshold,
            "enable_auto_cleanup" => pool.enable_auto_cleanup
        )
    )
end

"""
    clear_memory_pools!()

Clear all memory pools and force garbage collection.
"""
function clear_memory_pools!()
    pool = get_memory_pool()
    
    empty!(pool.tensor_pool)
    empty!(pool.vector_pool)
    empty!(pool.index_pool)
    empty!(pool.pool_usage_stats)
    
    pool.total_allocated_bytes[] = 0
    pool.allocation_count[] = 0
    pool.cache_hits[] = 0
    pool.cache_misses[] = 0
    pool.allocation_time[] = 0.0
    
    GC.gc()
    CUDA.reclaim()
    
    @info "Cleared all GPU memory pools"
end