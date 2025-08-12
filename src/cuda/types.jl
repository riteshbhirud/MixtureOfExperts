"""
GPU Data Structures and Configurations for CUDA MoE

This file defines all GPU-compatible data structures, configurations, and memory layouts
optimized for CUDA computation with proper memory alignment and access patterns.
"""

# Device information structure
struct GPUDeviceInfo
    device_id::Int
    name::String
    compute_capability::VersionNumber
    total_memory::Int64
    multiprocessor_count::Int
    max_threads_per_block::Int
    max_threads_per_multiprocessor::Int
    max_shared_memory_per_block::Int
    warp_size::Int
    max_grid_size::Tuple{Int,Int,Int}
    max_block_size::Tuple{Int,Int,Int}
end

# Helper: collect all device attributes into a Dict
function _device_attributes(dev::CUDA.CuDevice)
    Dict(attr => CUDA.attribute(dev, attr) for attr in instances(CUDA.CUdevice_attribute))
end

function GPUDeviceInfo()
    if !CUDA.functional()
        throw(ErrorException("CUDA not functional"))
    end

    dev   = CUDA.device()
    attrs = _device_attributes(dev)

    # Use OPTIN shared memory if available (some GPUs expose a larger limit there)
    shmem = get(attrs, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                get(attrs, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, 0))

    # Build tuples (convert Int32 -> Int to match your field types)
    max_grid = (
        Int(attrs[CUDA.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X]),
        Int(attrs[CUDA.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y]),
        Int(attrs[CUDA.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z]),
    )

    max_block = (
        Int(attrs[CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X]),
        Int(attrs[CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y]),
        Int(attrs[CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z]),
    )

    return GPUDeviceInfo(
        Int(CUDA.deviceid(dev)),
        CUDA.name(dev),
        CUDA.capability(dev),                          # VersionNumber
        Int64(CUDA.totalmem(dev)),                     # bytes
        Int(attrs[CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT]),
        Int(attrs[CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK]),
        Int(attrs[CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR]),
        Int(shmem),
        Int(attrs[CUDA.CU_DEVICE_ATTRIBUTE_WARP_SIZE]),
        max_grid,
        max_block,
    )
end

# Memory information structure
struct GPUMemoryInfo
    total_bytes::Int64
    available_bytes::Int64
    used_bytes::Int64
    reserved_bytes::Int64
    active_bytes::Int64
    cached_bytes::Int64
    utilization_percentage::Float64
end

function GPUMemoryInfo()
    if !CUDA.functional()
        throw(ErrorException("CUDA not functional"))
    end
    dev = CUDA.device()

    
    total = CUDA.totalmem(dev)
    available = CUDA.available_memory()
    used = total - available
    
    # Get more detailed memory info from CUDA memory pool
    pool_info = CUDA.pool_status()
    reserved = pool_info.reserved_bytes
    active = pool_info.used_bytes
    cached = reserved - active
    
    utilization = (used / total) * 100.0
    
    return GPUMemoryInfo(
        total, available, used, reserved, active, cached, utilization
    )
end

# Kernel launch configuration
struct GPUKernelConfig
    threads_per_block::Int
    blocks_per_grid::Int
    shared_memory_bytes::Int
    stream::CUDA.CuStream
    
    function GPUKernelConfig(total_threads::Int; 
                            threads_per_block::Int = 256,
                            shared_memory_bytes::Int = 0,
                            stream::CUDA.CuStream = CUDA.stream())
        
        # Ensure threads_per_block is within device limits
        device_info = GPUDeviceInfo()
        max_threads = device_info.max_threads_per_block
        
        if threads_per_block > max_threads
            @warn "Requested threads_per_block ($threads_per_block) exceeds device limit ($max_threads), using $max_threads"
            threads_per_block = max_threads
        end
        
        # Calculate optimal blocks_per_grid
        blocks_per_grid = cld(total_threads, threads_per_block)
        
        # Ensure blocks_per_grid doesn't exceed device limits
        max_blocks = device_info.max_grid_size[1]
        if blocks_per_grid > max_blocks
            @warn "Calculated blocks_per_grid ($blocks_per_grid) exceeds device limit ($max_blocks), using $max_blocks"
            blocks_per_grid = max_blocks
        end
        
        return new(threads_per_block, blocks_per_grid, shared_memory_bytes, stream)
    end
end

# GPU MoE Configuration
struct GPUMoEConfig{T<:AbstractFloat}
    # Model dimensions
    input_dim::Int
    hidden_dim::Int
    output_dim::Int
    num_experts::Int
    top_k::Int
    
    # Memory layout preferences
    memory_alignment::Int
    use_half_precision::Bool
    use_tensor_cores::Bool
    
    # Performance optimization settings
    max_batch_size::Int
    preferred_block_size::Int
    use_shared_memory::Bool
    enable_kernel_fusion::Bool
    
    # Numerical stability
    epsilon::T
    inf_value::T
    log_epsilon::T
    
    # Device-specific optimizations
    device_info::GPUDeviceInfo
    compute_capability::VersionNumber
    
    function GPUMoEConfig{T}(
        input_dim::Int,
        hidden_dim::Int,
        output_dim::Int,
        num_experts::Int,
        top_k::Int;
        memory_alignment::Int = 32,
        use_half_precision::Bool = false,
        use_tensor_cores::Bool = true,
        max_batch_size::Int = 512,
        preferred_block_size::Int = 256,
        use_shared_memory::Bool = true,
        enable_kernel_fusion::Bool = true,
        epsilon::T = T(1e-6),
        inf_value::T = T(1e10),
        log_epsilon::T = T(1e-20)
    ) where T
        
        if !CUDA.functional()
            throw(ErrorException("CUDA not functional"))
        end
        
        # Validate dimensions
        if input_dim <= 0 || hidden_dim <= 0 || output_dim <= 0
            throw(ArgumentError("All dimensions must be positive"))
        end
        
        if num_experts <= 0 || top_k <= 0 || top_k > num_experts
            throw(ArgumentError("Invalid expert configuration"))
        end
        
        # Get device information
        device_info = GPUDeviceInfo()
        compute_capability = device_info.compute_capability
        
        # Adjust settings based on compute capability
        actual_use_tensor_cores = use_tensor_cores && compute_capability >= v"7.0"
        actual_preferred_block_size = min(preferred_block_size, device_info.max_threads_per_block)
        
        # Validate memory alignment
        if memory_alignment <= 0 || !ispow2(memory_alignment)
            throw(ArgumentError("Memory alignment must be a positive power of 2"))
        end
        
        return new{T}(
            input_dim, hidden_dim, output_dim, num_experts, top_k,
            memory_alignment, use_half_precision, actual_use_tensor_cores,
            max_batch_size, actual_preferred_block_size, use_shared_memory, enable_kernel_fusion,
            epsilon, inf_value, log_epsilon,
            device_info, compute_capability
        )
    end
end

# Convenience constructors
GPUMoEConfig(args...; kwargs...) = GPUMoEConfig{Float32}(args...; kwargs...)

# GPU Gated Expert Weights with optimized memory layout
struct GPUGatedExpertWeights{T<:AbstractFloat, A<:AbstractMatrix{T}}
    # Weight matrices with memory-aligned storage
    w1::A  # input_dim × hidden_dim - gate projection
    w2::A  # hidden_dim × output_dim - output projection  
    w3::A  # input_dim × hidden_dim - up projection
    
    # Bias vectors (optional)
    b1::Union{Nothing, CuVector{T}}  # hidden_dim bias for gate
    b2::Union{Nothing, CuVector{T}}  # output_dim bias for output
    b3::Union{Nothing, CuVector{T}}  # hidden_dim bias for up
    
    # Expert metadata
    expert_id::Int
    is_active::Bool
    
    # Memory layout information
    w1_stride::Int
    w2_stride::Int  
    w3_stride::Int
    memory_aligned::Bool
    
    function GPUGatedExpertWeights{T,A}(
        w1::A, w2::A, w3::A,
        expert_id::Int = 0;
        b1::Union{Nothing, CuVector{T}} = nothing,
        b2::Union{Nothing, CuVector{T}} = nothing,
        b3::Union{Nothing, CuVector{T}} = nothing,
        is_active::Bool = true
    ) where {T<:AbstractFloat, A<:AbstractMatrix{T}}
        
        # Validate matrix dimensions
        input_dim, hidden_dim1 = size(w1)
        hidden_dim2, output_dim = size(w2)
        input_dim2, hidden_dim3 = size(w3)
        
        if hidden_dim1 != hidden_dim2 || hidden_dim1 != hidden_dim3
            throw(DimensionMismatch("Hidden dimensions must match across weight matrices"))
        end
        
        if input_dim != input_dim2
            throw(DimensionMismatch("Input dimensions must match for w1 and w3"))
        end
        
        # Check bias dimensions if provided
        if !isnothing(b1) && length(b1) != hidden_dim1
            throw(DimensionMismatch("b1 dimension mismatch"))
        end
        if !isnothing(b2) && length(b2) != output_dim
            throw(DimensionMismatch("b2 dimension mismatch"))
        end
        if !isnothing(b3) && length(b3) != hidden_dim1
            throw(DimensionMismatch("b3 dimension mismatch"))
        end
        
        # Calculate memory strides for efficient access
        w1_stride = stride(w1, 2)
        w2_stride = stride(w2, 2)
        w3_stride = stride(w3, 2)
        
        # Check memory alignment (for optimal GPU access patterns)
        memory_aligned = (w1_stride % 32 == 0) && (w2_stride % 32 == 0) && (w3_stride % 32 == 0)
        
        return new{T,A}(
            w1, w2, w3, b1, b2, b3,
            expert_id, is_active,
            w1_stride, w2_stride, w3_stride, memory_aligned
        )
    end
end

# Convenience constructor
function GPUGatedExpertWeights(w1::CuMatrix{T}, w2::CuMatrix{T}, w3::CuMatrix{T}, 
                              expert_id::Int = 0; kwargs...) where T
    return GPUGatedExpertWeights{T, CuMatrix{T}}(w1, w2, w3, expert_id; kwargs...)
end

# GPU TopK Gating State for computation workspace
mutable struct GPUTopKGatingState{T<:AbstractFloat}
    # Input/output buffers
    router_logits::CuMatrix{T}      # num_experts × batch_size
    router_probs::CuMatrix{T}       # num_experts × batch_size
    expert_indices::CuMatrix{Int32} # top_k × batch_size
    expert_gates::CuMatrix{T}       # top_k × batch_size
    
    # Intermediate computation buffers
    softmax_workspace::CuVector{T}  # For numerically stable softmax
    topk_workspace::CuVector{T}     # For top-k selection algorithm
    reduction_workspace::CuVector{T} # For parallel reductions
    
    # Index and sorting buffers
    sort_indices::CuMatrix{Int32}   # For efficient sorting operations
    temp_values::CuMatrix{T}        # Temporary storage for sorting
    
    # Memory management
    allocated_batch_size::Int
    allocated_num_experts::Int
    allocated_top_k::Int
    workspace_bytes::Int64
    
    # Performance tracking
    last_allocation_time::Float64
    total_allocations::Int
    
    function GPUTopKGatingState{T}(
        num_experts::Int,
        top_k::Int,
        batch_size::Int
    ) where T<:AbstractFloat
        
        if num_experts <= 0 || top_k <= 0 || batch_size <= 0
            throw(ArgumentError("All sizes must be positive"))
        end
        
        if top_k > num_experts
            throw(ArgumentError("top_k cannot exceed num_experts"))
        end
        
        # Allocate main buffers
        router_logits = CUDA.zeros(T, num_experts, batch_size)
        router_probs = CUDA.zeros(T, num_experts, batch_size)
        expert_indices = CUDA.zeros(Int32, top_k, batch_size)
        expert_gates = CUDA.zeros(T, top_k, batch_size)
        
        # Allocate workspace buffers with padding for memory alignment
        workspace_size = max(num_experts, batch_size) * 2
        softmax_workspace = CUDA.zeros(T, workspace_size)
        topk_workspace = CUDA.zeros(T, workspace_size)
        reduction_workspace = CUDA.zeros(T, workspace_size)
        
        # Allocate sorting buffers
        sort_indices = CUDA.zeros(Int32, num_experts, batch_size)
        temp_values = CUDA.zeros(T, num_experts, batch_size)
        
        # Calculate total workspace memory usage
        workspace_bytes = (
            sizeof(router_logits) + sizeof(router_probs) +
            sizeof(expert_indices) + sizeof(expert_gates) +
            sizeof(softmax_workspace) + sizeof(topk_workspace) + 
            sizeof(reduction_workspace) + sizeof(sort_indices) + sizeof(temp_values)
        )
        
        allocation_time = time()
        
        return new{T}(
            router_logits, router_probs, expert_indices, expert_gates,
            softmax_workspace, topk_workspace, reduction_workspace,
            sort_indices, temp_values,
            batch_size, num_experts, top_k, workspace_bytes,
            allocation_time, 1
        )
    end
end

# Convenience constructor
GPUTopKGatingState(num_experts::Int, top_k::Int, batch_size::Int) = 
    GPUTopKGatingState{Float32}(num_experts, top_k, batch_size)

# Resize state for different batch sizes
function resize_gating_state!(state::GPUTopKGatingState{T}, new_batch_size::Int) where T
    if new_batch_size <= state.allocated_batch_size
        return state  # No need to resize
    end
    
    # Record reallocation
    state.total_allocations += 1
    state.last_allocation_time = time()
    
    # Reallocate buffers with new batch size
    num_experts = state.allocated_num_experts
    top_k = state.allocated_top_k
    
    state.router_logits = CUDA.zeros(T, num_experts, new_batch_size)
    state.router_probs = CUDA.zeros(T, num_experts, new_batch_size)
    state.expert_indices = CUDA.zeros(Int32, top_k, new_batch_size)
    state.expert_gates = CUDA.zeros(T, top_k, new_batch_size)
    state.sort_indices = CUDA.zeros(Int32, num_experts, new_batch_size)
    state.temp_values = CUDA.zeros(T, num_experts, new_batch_size)
    
    # Update workspace sizes if needed
    workspace_size = max(num_experts, new_batch_size) * 2
    if length(state.softmax_workspace) < workspace_size
        state.softmax_workspace = CUDA.zeros(T, workspace_size)
        state.topk_workspace = CUDA.zeros(T, workspace_size)
        state.reduction_workspace = CUDA.zeros(T, workspace_size)
    end
    
    state.allocated_batch_size = new_batch_size
    
    # Recalculate workspace memory usage
    state.workspace_bytes = (
        sizeof(state.router_logits) + sizeof(state.router_probs) +
        sizeof(state.expert_indices) + sizeof(state.expert_gates) +
        sizeof(state.softmax_workspace) + sizeof(state.topk_workspace) + 
        sizeof(state.reduction_workspace) + sizeof(state.sort_indices) + sizeof(state.temp_values)
    )
    
    return state
end

# GPU Switch Transformer Loss State
mutable struct GPUSwitchLossState{T<:AbstractFloat}
    # Expert assignment tracking
    expert_counts::CuVector{T}      # Fraction of tokens per expert
    expert_probs::CuVector{T}       # Average probability per expert
    
    # Intermediate computation buffers
    assignment_buffer::CuMatrix{Int32}  # Binary assignment matrix
    probability_sums::CuVector{T}       # Sums for probability computation
    count_workspace::CuVector{T}        # Workspace for counting operations
    
    # Loss computation workspace
    loss_terms::CuVector{T}         # Individual loss terms
    reduction_buffer::CuVector{T}   # For parallel reduction
    
    # Batch processing state
    current_batch_size::Int
    current_num_experts::Int
    total_processed_tokens::Int64
    
    # Performance metrics
    loss_computation_time::Float64
    reduction_time::Float64
    assignment_time::Float64
    
    function GPUSwitchLossState{T}(
        num_experts::Int,
        max_batch_size::Int
    ) where T<:AbstractFloat
        
        if num_experts <= 0 || max_batch_size <= 0
            throw(ArgumentError("All sizes must be positive"))
        end
        
        # Allocate expert-level buffers
        expert_counts = CUDA.zeros(T, num_experts)
        expert_probs = CUDA.zeros(T, num_experts)
        probability_sums = CUDA.zeros(T, num_experts)
        loss_terms = CUDA.zeros(T, num_experts)
        
        # Allocate batch-level buffers
        assignment_buffer = CUDA.zeros(Int32, num_experts, max_batch_size)
        count_workspace = CUDA.zeros(T, max_batch_size)
        reduction_buffer = CUDA.zeros(T, max_batch_size)
        
        return new{T}(
            expert_counts, expert_probs,
            assignment_buffer, probability_sums, count_workspace,
            loss_terms, reduction_buffer,
            0, num_experts, 0,
            0.0, 0.0, 0.0
        )
    end
end

# Convenience constructor
GPUSwitchLossState(num_experts::Int, max_batch_size::Int) = 
    GPUSwitchLossState{Float32}(num_experts, max_batch_size)

# GPU workspace allocation utilities
struct GPUWorkspace{T<:AbstractFloat}
    # Large pre-allocated buffers for temporary computation
    main_buffer::CuVector{T}
    index_buffer::CuVector{Int32}
    bool_buffer::CuVector{Bool}
    
    # Buffer size tracking
    main_buffer_size::Int
    index_buffer_size::Int  
    bool_buffer_size::Int
    
    # Usage tracking
    main_buffer_offset::Ref{Int}
    index_buffer_offset::Ref{Int}
    bool_buffer_offset::Ref{Int}
    
    allocation_count::Ref{Int}
    peak_usage_bytes::Ref{Int64}
    
    function GPUWorkspace{T}(
        main_buffer_mb::Int = 512,
        index_buffer_mb::Int = 128,
        bool_buffer_mb::Int = 64
    ) where T<:AbstractFloat
        
        # Convert MB to element counts
        main_size = (main_buffer_mb * 1024 * 1024) ÷ sizeof(T)
        index_size = (index_buffer_mb * 1024 * 1024) ÷ sizeof(Int32)
        bool_size = (bool_buffer_mb * 1024 * 1024) ÷ sizeof(Bool)
        
        # Allocate buffers
        main_buffer = CUDA.zeros(T, main_size)
        index_buffer = CUDA.zeros(Int32, index_size)
        bool_buffer = CUDA.zeros(Bool, bool_size)
        
        return new{T}(
            main_buffer, index_buffer, bool_buffer,
            main_size, index_size, bool_size,
            Ref(0), Ref(0), Ref(0),
            Ref(0), Ref(Int64(0))
        )
    end
end

# Global workspace instance (thread-local for safety)
const _gpu_workspace = Ref{Union{Nothing, GPUWorkspace{Float32}}}(nothing)

function get_gpu_workspace()
    if isnothing(_gpu_workspace[])
        _gpu_workspace[] = GPUWorkspace{Float32}()
    end
    return _gpu_workspace[]
end

# Workspace allocation functions
function allocate_gpu_temp_buffer(::Type{T}, size::Int) where T<:AbstractFloat
    workspace = get_gpu_workspace()
    
    if workspace.main_buffer_offset[] + size > workspace.main_buffer_size
        # Reset offset if we've run out of space (simple wraparound)
        workspace.main_buffer_offset[] = 0
        if size > workspace.main_buffer_size
            throw(OutOfMemoryError("Requested buffer size exceeds workspace capacity"))
        end
    end
    
    start_idx = workspace.main_buffer_offset[] + 1
    end_idx = workspace.main_buffer_offset[] + size
    
    workspace.main_buffer_offset[] += size
    workspace.allocation_count[] += 1
    
    # Track peak usage
    current_usage = workspace.main_buffer_offset[] * sizeof(T)
    if current_usage > workspace.peak_usage_bytes[]
        workspace.peak_usage_bytes[] = current_usage
    end
    
    return view(workspace.main_buffer, start_idx:end_idx)
end

function allocate_gpu_index_buffer(size::Int)
    workspace = get_gpu_workspace()
    
    if workspace.index_buffer_offset[] + size > workspace.index_buffer_size
        workspace.index_buffer_offset[] = 0
        if size > workspace.index_buffer_size
            throw(OutOfMemoryError("Requested index buffer size exceeds workspace capacity"))
        end
    end
    
    start_idx = workspace.index_buffer_offset[] + 1
    end_idx = workspace.index_buffer_offset[] + size
    
    workspace.index_buffer_offset[] += size
    
    return view(workspace.index_buffer, start_idx:end_idx)
end

# Reset workspace offsets (call between major operations)
function reset_gpu_workspace!()
    workspace = get_gpu_workspace()
    workspace.main_buffer_offset[] = 0
    workspace.index_buffer_offset[] = 0
    workspace.bool_buffer_offset[] = 0
end

# Memory alignment utilities
function align_dimension(dim::Int, alignment::Int = 32)
    return cld(dim, alignment) * alignment
end

function create_aligned_matrix(::Type{T}, rows::Int, cols::Int, alignment::Int = 32) where T
    aligned_rows = align_dimension(rows, alignment)
    matrix = CUDA.zeros(T, aligned_rows, cols)
    return view(matrix, 1:rows, :)
end

# Type conversion utilities for mixed precision
function to_gpu_precision(x::AbstractArray{T}, config::GPUMoEConfig) where T
    if config.use_half_precision && T != Float16
        return CuArray{Float16}(x)
    elseif !config.use_half_precision && T != Float32
        return CuArray{Float32}(x)
    else
        return CuArray(x)
    end
end

# Validation utilities
function validate_gpu_array_dimensions(arrays::AbstractArray...)
    if isempty(arrays)
        return true
    end
    
    for arr in arrays
        if !isa(arr, CuArray)
            throw(ArgumentError("All arrays must be CuArrays"))
        end
        
        if !all(isfinite, arr)
            throw(ArgumentError("Array contains non-finite values"))
        end
    end
    
    return true
end

function check_memory_requirements(required_bytes::Int64)
    available = CUDA.available_memory()
    if required_bytes > available
        throw(ErrorException("Operation requires $(required_bytes ÷ (1024^2)) MB but only $(available ÷ (1024^2)) MB available"))
    end
    return true
end