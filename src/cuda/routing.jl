"""
GPU Token Routing and Expert Assignment Utilities

Efficient GPU-based token routing for MoE layers that organizes tokens by expert
assignment for optimal parallel computation. Handles sparse routing patterns
and optimizes memory access patterns for maximum GPU throughput.
"""

# Main routing state structure for efficient token organization
mutable struct GPURoutingState{T<:AbstractFloat}
    # Expert assignment organization
    expert_token_counts::Vector{Int32}        # CPU: tokens assigned to each expert
    expert_token_offsets::Vector{Int32}       # CPU: start indices for each expert's tokens
    
    # GPU arrays for sorting and organization
    sorted_token_indices::CuVector{Int32}     # GPU: token indices sorted by expert assignment
    sorted_gating_weights::CuVector{T}        # GPU: corresponding gating weights
    sort_workspace::CuVector{Int32}           # GPU: workspace for sorting operations
    
    # Temporary arrays for efficient sorting
    temp_expert_ids::CuVector{Int32}          # GPU: temporary expert IDs for sorting
    temp_token_ids::CuVector{Int32}           # GPU: temporary token IDs for sorting
    temp_weights::CuVector{T}                 # GPU: temporary weights for sorting
    
    # Configuration
    num_experts::Int
    top_k::Int
    max_batch_size::Int
    allocated_batch_size::Int
    
    # Performance tracking
    routing_calls::Ref{Int}
    total_routing_time::Ref{Float64}
    sorting_time::Ref{Float64}
    organization_time::Ref{Float64}
    
    function GPURoutingState{T}(
        num_experts::Int,
        top_k::Int,
        max_batch_size::Int
    ) where T<:AbstractFloat
        
        # Validate parameters
        if num_experts <= 0 || top_k <= 0 || max_batch_size <= 0
            throw(ArgumentError("All parameters must be positive"))
        end
        
        if top_k > num_experts
            throw(ArgumentError("top_k cannot exceed num_experts"))
        end
        
        # Calculate maximum number of token assignments
        max_assignments = top_k * max_batch_size
        
        # Initialize CPU arrays for expert organization
        expert_token_counts = zeros(Int32, num_experts)
        expert_token_offsets = zeros(Int32, num_experts + 1)
        
        # Initialize GPU arrays
        sorted_token_indices = CUDA.zeros(Int32, max_assignments)
        sorted_gating_weights = CUDA.zeros(T, max_assignments)
        sort_workspace = CUDA.zeros(Int32, max_assignments)
        
        # Initialize temporary arrays
        temp_expert_ids = CUDA.zeros(Int32, max_assignments)
        temp_token_ids = CUDA.zeros(Int32, max_assignments)
        temp_weights = CUDA.zeros(T, max_assignments)
        
        # Initialize performance tracking
        routing_calls = Ref(0)
        total_routing_time = Ref(0.0)
        sorting_time = Ref(0.0)
        organization_time = Ref(0.0)
        
        return new{T}(
            expert_token_counts, expert_token_offsets,
            sorted_token_indices, sorted_gating_weights, sort_workspace,
            temp_expert_ids, temp_token_ids, temp_weights,
            num_experts, top_k, max_batch_size, max_batch_size,
            routing_calls, total_routing_time, sorting_time, organization_time
        )
    end
end

# Convenience constructor
function GPURoutingState(num_experts::Int, top_k::Int, max_batch_size::Int)
    return GPURoutingState{Float32}(num_experts, top_k, max_batch_size)
end

# Routing information structure returned by organization
struct GPURoutingInfo{T<:AbstractFloat}
    expert_token_counts::Vector{Int32}
    expert_token_offsets::Vector{Int32}
    sorted_token_indices::CuVector{Int32}
    sorted_gating_weights::CuVector{T}
    total_assignments::Int
    
    function GPURoutingInfo{T}(
        expert_token_counts::Vector{Int32},
        expert_token_offsets::Vector{Int32},
        sorted_token_indices::CuVector{Int32},
        sorted_gating_weights::CuVector{T},
        total_assignments::Int
    ) where T<:AbstractFloat
        return new{T}(
            expert_token_counts, expert_token_offsets,
            sorted_token_indices, sorted_gating_weights, total_assignments
        )
    end
end

# Main token routing organization function
function organize_token_routing!(
    routing_state::GPURoutingState{T},
    expert_indices::CuMatrix{Int32},      # top_k × batch_size
    expert_gates::CuMatrix{T},            # top_k × batch_size
    batch_size::Int
) where T<:AbstractFloat
    
    routing_state.routing_calls[] += 1
    start_time = time()
    
    try
        # Validate input dimensions
        top_k, input_batch_size = size(expert_indices)
        if input_batch_size != batch_size || size(expert_gates) != (top_k, batch_size)
            throw(DimensionMismatch("Input dimensions do not match expected batch size"))
        end
        
        if top_k != routing_state.top_k
            throw(DimensionMismatch("top_k mismatch"))
        end
        
        # Resize state if needed
        if batch_size > routing_state.allocated_batch_size
            resize_routing_state!(routing_state, batch_size)
        end
        
        # Step 1: Flatten and prepare data for sorting
        total_assignments = top_k * batch_size
        
        flatten_start = time()
        flatten_expert_assignments!(
            routing_state.temp_expert_ids,
            routing_state.temp_token_ids,
            routing_state.temp_weights,
            expert_indices,
            expert_gates,
            top_k,
            batch_size
        )
        flatten_time = time() - flatten_start
        
        # Step 2: Sort by expert ID for efficient grouping
        sort_start = time()
        sort_by_expert_id!(
            routing_state.sorted_token_indices,
            routing_state.sorted_gating_weights,
            routing_state.temp_expert_ids,
            routing_state.temp_token_ids,
            routing_state.temp_weights,
            total_assignments
        )
        routing_state.sorting_time[] += time() - sort_start
        
        # Step 3: Compute expert token counts and offsets
        organize_start = time()
        compute_expert_organization!(
            routing_state.expert_token_counts,
            routing_state.expert_token_offsets,
            routing_state.sorted_token_indices,
            routing_state.temp_expert_ids,
            routing_state.num_experts,
            total_assignments
        )
        routing_state.organization_time[] += time() - organize_start
        
        # Create routing info structure
        routing_info = GPURoutingInfo{T}(
            routing_state.expert_token_counts,
            routing_state.expert_token_offsets,
            view(routing_state.sorted_token_indices, 1:total_assignments),
            view(routing_state.sorted_gating_weights, 1:total_assignments),
            total_assignments
        )
        
        return routing_info
        
    catch e
        @error "Error in token routing organization" exception=e
        rethrow(e)
    finally
        elapsed_time = time() - start_time
        routing_state.total_routing_time[] += elapsed_time
    end
end

# Flatten expert assignments for sorting
function flatten_expert_assignments!(
    temp_expert_ids::CuVector{Int32},
    temp_token_ids::CuVector{Int32},
    temp_weights::CuVector{T},
    expert_indices::CuMatrix{Int32},
    expert_gates::CuMatrix{T},
    top_k::Int,
    batch_size::Int
) where T<:AbstractFloat
    
    total_assignments = top_k * batch_size
    kernel_config = GPUKernelConfig(total_assignments)
    
    @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid flatten_assignments_kernel!(
        temp_expert_ids, temp_token_ids, temp_weights,
        expert_indices, expert_gates,
        top_k, batch_size, total_assignments
    )
    
    CUDA.synchronize()
    return nothing
end

# CUDA kernel for flattening assignments
function flatten_assignments_kernel!(
    temp_expert_ids::CuDeviceVector{Int32},
    temp_token_ids::CuDeviceVector{Int32},
    temp_weights::CuDeviceVector{T},
    expert_indices::CuDeviceMatrix{Int32},
    expert_gates::CuDeviceMatrix{T},
    top_k::Int,
    batch_size::Int,
    total_assignments::Int
) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= total_assignments
        # Convert linear index to (k, batch) coordinates
        k_idx = ((idx - 1) % top_k) + 1
        batch_idx = ((idx - 1) ÷ top_k) + 1
        
        if k_idx <= top_k && batch_idx <= batch_size
            # Extract assignment information
            expert_id = expert_indices[k_idx, batch_idx]
            weight = expert_gates[k_idx, batch_idx]
            
            # Store in flattened arrays
            temp_expert_ids[idx] = expert_id
            temp_token_ids[idx] = Int32(batch_idx)
            temp_weights[idx] = weight
        end
    end
    
    return nothing
end

# Sort assignments by expert ID for efficient grouping
function sort_by_expert_id!(
    sorted_token_indices::CuVector{Int32},
    sorted_gating_weights::CuVector{T},
    temp_expert_ids::CuVector{Int32},
    temp_token_ids::CuVector{Int32},
    temp_weights::CuVector{T},
    total_assignments::Int
) where T<:AbstractFloat
    
    # Use GPU sorting with expert ID as key
    # We'll implement a simple but efficient GPU sorting algorithm
    
    if total_assignments <= 1
        if total_assignments == 1
            sorted_token_indices[1] = temp_token_ids[1]
            sorted_gating_weights[1] = temp_weights[1]
        end
        return nothing
    end
    
    # Use bitonic sort for power-of-2 sizes, radix sort otherwise
    if ispow2(total_assignments) && total_assignments <= 2048
        bitonic_sort_by_expert!(
            sorted_token_indices, sorted_gating_weights,
            temp_expert_ids, temp_token_ids, temp_weights,
            total_assignments
        )
    else
        # For larger or non-power-of-2 sizes, use a simpler approach
        # Copy and sort (this could be optimized further with a full GPU radix sort)
        simple_gpu_sort_by_expert!(
            sorted_token_indices, sorted_gating_weights,
            temp_expert_ids, temp_token_ids, temp_weights,
            total_assignments
        )
    end
    
    return nothing
end

# Bitonic sort implementation for sorting by expert ID
function bitonic_sort_by_expert!(
    sorted_token_indices::CuVector{Int32},
    sorted_gating_weights::CuVector{T},
    temp_expert_ids::CuVector{Int32},
    temp_token_ids::CuVector{Int32},
    temp_weights::CuVector{T},
    total_assignments::Int
) where T<:AbstractFloat
    
    # Copy initial data
    copyto!(sorted_token_indices, view(temp_token_ids, 1:total_assignments))
    copyto!(sorted_gating_weights, view(temp_weights, 1:total_assignments))
    expert_keys = view(temp_expert_ids, 1:total_assignments)
    
    # Bitonic sort implementation
    step = 2
    while step <= total_assignments
        substep = step ÷ 2
        while substep > 0
            
            kernel_config = GPUKernelConfig(total_assignments ÷ 2)
            
            @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid bitonic_sort_step_kernel!(
                expert_keys, sorted_token_indices, sorted_gating_weights,
                total_assignments, step, substep
            )
            
            CUDA.synchronize()
            substep ÷= 2
        end
        step *= 2
    end
    
    return nothing
end

# CUDA kernel for bitonic sort step
function bitonic_sort_step_kernel!(
    expert_keys::CuDeviceVector{Int32},
    token_indices::CuDeviceVector{Int32},
    weights::CuDeviceVector{T},
    n::Int,
    step::Int,
    substep::Int
) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= n ÷ 2
        i = (idx - 1) * 2 + 1
        j = i + substep
        
        if j <= n
            # Determine sort direction based on bitonic sequence
            ascending = ((i - 1) ÷ step) % 2 == 0
            
            key_i = expert_keys[i]
            key_j = expert_keys[j]
            
            should_swap = ascending ? (key_i > key_j) : (key_i < key_j)
            
            if should_swap
                # Swap all associated values
                expert_keys[i], expert_keys[j] = key_j, key_i
                token_indices[i], token_indices[j] = token_indices[j], token_indices[i]
                weights[i], weights[j] = weights[j], weights[i]
            end
        end
    end
    
    return nothing
end

# Simple GPU sort for non-power-of-2 cases
function simple_gpu_sort_by_expert!(
    sorted_token_indices::CuVector{Int32},
    sorted_gating_weights::CuVector{T},
    temp_expert_ids::CuVector{Int32},
    temp_token_ids::CuVector{Int32},
    temp_weights::CuVector{T},
    total_assignments::Int
) where T<:AbstractFloat
    
    # For simplicity, use a basic parallel selection sort
    # This is not the most efficient but works for moderate sizes
    
    # Copy initial data
    copyto!(sorted_token_indices, view(temp_token_ids, 1:total_assignments))
    copyto!(sorted_gating_weights, view(temp_weights, 1:total_assignments))
    expert_keys = view(temp_expert_ids, 1:total_assignments)
    
    # Parallel selection sort
    for i in 1:total_assignments
        kernel_config = GPUKernelConfig(total_assignments - i + 1)
        
        @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid simple_sort_kernel!(
            expert_keys, sorted_token_indices, sorted_gating_weights,
            i, total_assignments
        )
        
        CUDA.synchronize()
    end
    
    return nothing
end

# CUDA kernel for simple sorting
function simple_sort_kernel!(
    expert_keys::CuDeviceVector{Int32},
    token_indices::CuDeviceVector{Int32},
    weights::CuDeviceVector{T},
    start_pos::Int,
    total_assignments::Int
) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x + start_pos - 1
    
    if idx <= total_assignments && idx >= start_pos
        min_key = expert_keys[idx]
        min_idx = idx
        
        # Find minimum expert key in remaining elements
        for j in (idx + 1):total_assignments
            if expert_keys[j] < min_key
                min_key = expert_keys[j]
                min_idx = j
            end
        end
        
        # Swap if needed
        if min_idx != idx
            # Swap expert keys
            temp_key = expert_keys[idx]
            expert_keys[idx] = expert_keys[min_idx]
            expert_keys[min_idx] = temp_key
            
            # Swap token indices
            temp_token = token_indices[idx]
            token_indices[idx] = token_indices[min_idx]
            token_indices[min_idx] = temp_token
            
            # Swap weights
            temp_weight = weights[idx]
            weights[idx] = weights[min_idx]
            weights[min_idx] = temp_weight
        end
    end
    
    return nothing
end

# Compute expert organization (counts and offsets)
function compute_expert_organization!(
    expert_token_counts::Vector{Int32},
    expert_token_offsets::Vector{Int32},
    sorted_token_indices::CuVector{Int32},
    temp_expert_ids::CuVector{Int32},
    num_experts::Int,
    total_assignments::Int
) where T<:AbstractFloat
    
    # Initialize counts
    fill!(expert_token_counts, Int32(0))
    fill!(expert_token_offsets, Int32(0))
    
    if total_assignments == 0
        return nothing
    end
    
    # Copy expert IDs to CPU for counting (more efficient than GPU reduction for this)
    expert_ids_cpu = Array(view(temp_expert_ids, 1:total_assignments))
    
    # Count tokens per expert
    for expert_id in expert_ids_cpu
        if expert_id >= 1 && expert_id <= num_experts
            expert_token_counts[expert_id] += 1
        end
    end
    
    # Compute cumulative offsets
    expert_token_offsets[1] = 0
    for i in 2:(num_experts + 1)
        expert_token_offsets[i] = expert_token_offsets[i - 1] + expert_token_counts[i - 1]
    end
    
    return nothing
end

# Resize routing state for larger batch sizes
function resize_routing_state!(routing_state::GPURoutingState{T}, new_batch_size::Int) where T
    if new_batch_size <= routing_state.allocated_batch_size
        return routing_state
    end
    
    # Calculate new assignment capacity
    new_max_assignments = routing_state.top_k * new_batch_size
    current_capacity = length(routing_state.sorted_token_indices)
    
    if new_max_assignments > current_capacity
        # Reallocate GPU arrays
        routing_state.sorted_token_indices = CUDA.zeros(Int32, new_max_assignments)
        routing_state.sorted_gating_weights = CUDA.zeros(T, new_max_assignments)
        routing_state.sort_workspace = CUDA.zeros(Int32, new_max_assignments)
        
        routing_state.temp_expert_ids = CUDA.zeros(Int32, new_max_assignments)
        routing_state.temp_token_ids = CUDA.zeros(Int32, new_max_assignments)
        routing_state.temp_weights = CUDA.zeros(T, new_max_assignments)
    end
    
    routing_state.allocated_batch_size = new_batch_size
    
    return routing_state
end

# Performance analysis functions
function get_routing_performance_stats(routing_state::GPURoutingState{T}) where T
    routing_calls = routing_state.routing_calls[]
    total_time = routing_state.total_routing_time[]
    
    stats = Dict{String, Any}(
        "routing_calls" => routing_calls,
        "total_routing_time_ms" => total_time * 1000,
        "avg_routing_time_ms" => routing_calls > 0 ? (total_time / routing_calls) * 1000 : 0.0,
        "sorting_time_ms" => routing_state.sorting_time[] * 1000,
        "organization_time_ms" => routing_state.organization_time[] * 1000,
        "avg_sorting_time_ms" => routing_calls > 0 ? (routing_state.sorting_time[] / routing_calls) * 1000 : 0.0,
        "avg_organization_time_ms" => routing_calls > 0 ? (routing_state.organization_time[] / routing_calls) * 1000 : 0.0,
        "num_experts" => routing_state.num_experts,
        "top_k" => routing_state.top_k,
        "max_batch_size" => routing_state.max_batch_size,
        "allocated_batch_size" => routing_state.allocated_batch_size
    )
    
    # Add efficiency metrics
    if total_time > 0
        sorting_percentage = (routing_state.sorting_time[] / total_time) * 100
        organization_percentage = (routing_state.organization_time[] / total_time) * 100
        
        stats["timing_breakdown"] = Dict(
            "sorting_percentage" => sorting_percentage,
            "organization_percentage" => organization_percentage,
            "other_percentage" => 100.0 - sorting_percentage - organization_percentage
        )
    end
    
    return stats
end

function reset_routing_performance_stats!(routing_state::GPURoutingState)
    routing_state.routing_calls[] = 0
    routing_state.total_routing_time[] = 0.0
    routing_state.sorting_time[] = 0.0
    routing_state.organization_time[] = 0.0
end

# Utility functions for routing analysis
function analyze_routing_efficiency(routing_info::GPURoutingInfo{T}) where T
    analysis = Dict{String, Any}()
    
    # Expert utilization analysis
    total_assignments = routing_info.total_assignments
    num_experts = length(routing_info.expert_token_counts)
    
    if total_assignments > 0
        active_experts = count(c -> c > 0, routing_info.expert_token_counts)
        expert_utilization = active_experts / num_experts
        
        # Load balancing metrics
        if active_experts > 0
            active_counts = filter(c -> c > 0, routing_info.expert_token_counts)
            mean_load = sum(active_counts) / active_experts
            load_variance = sum((c - mean_load)^2 for c in active_counts) / active_experts
            load_balance_score = 1.0 - sqrt(load_variance) / mean_load
        else
            load_balance_score = 0.0
        end
        
        analysis["total_assignments"] = total_assignments
        analysis["active_experts"] = active_experts
        analysis["expert_utilization"] = expert_utilization
        analysis["load_balance_score"] = load_balance_score
        analysis["expert_token_counts"] = copy(routing_info.expert_token_counts)
        
        # Distribution statistics
        max_load = maximum(routing_info.expert_token_counts)
        min_load = minimum(routing_info.expert_token_counts)
        
        analysis["load_distribution"] = Dict(
            "max_load" => max_load,
            "min_load" => min_load,
            "load_imbalance" => max_load - min_load,
            "ideal_load" => total_assignments / num_experts
        )
    else
        analysis["status"] = "no_assignments"
    end
    
    return analysis
end

# Expert assignment validation
function validate_routing_info(routing_info::GPURoutingInfo{T}) where T
    # Check consistency of counts and offsets
    calculated_total = sum(routing_info.expert_token_counts)
    if calculated_total != routing_info.total_assignments
        @warn "Routing info inconsistency: calculated total ($calculated_total) != stored total ($(routing_info.total_assignments))"
        return false
    end
    
    # Check offset consistency
    for i in 1:length(routing_info.expert_token_counts)
        expected_offset = i == 1 ? 0 : routing_info.expert_token_offsets[i-1] + routing_info.expert_token_counts[i-1]
        if routing_info.expert_token_offsets[i] != expected_offset
            @warn "Routing offset inconsistency at expert $i"
            return false
        end
    end
    
    # Check that token indices are valid
    if routing_info.total_assignments > 0
        min_token = CUDA.reduce(min, view(routing_info.sorted_token_indices, 1:routing_info.total_assignments))
        max_token = CUDA.reduce(max, view(routing_info.sorted_token_indices, 1:routing_info.total_assignments))
        
        if min_token < 1
            @warn "Invalid token indices: minimum token index is $min_token (should be >= 1)"
            return false
        end
    end
    
    return true
end

# Memory usage estimation
function estimate_routing_memory_usage(routing_state::GPURoutingState{T}) where T
    element_size_int32 = sizeof(Int32)
    element_size_T = sizeof(T)
    
    # GPU arrays
    max_assignments = routing_state.top_k * routing_state.allocated_batch_size
    
    gpu_memory = (
        length(routing_state.sorted_token_indices) * element_size_int32 +
        length(routing_state.sorted_gating_weights) * element_size_T +
        length(routing_state.sort_workspace) * element_size_int32 +
        length(routing_state.temp_expert_ids) * element_size_int32 +
        length(routing_state.temp_token_ids) * element_size_int32 +
        length(routing_state.temp_weights) * element_size_T
    )
    
    # CPU arrays
    cpu_memory = (
        length(routing_state.expert_token_counts) * element_size_int32 +
        length(routing_state.expert_token_offsets) * element_size_int32
    )
    
    return Dict(
        "gpu_memory_bytes" => gpu_memory,
        "cpu_memory_bytes" => cpu_memory,
        "total_memory_bytes" => gpu_memory + cpu_memory,
        "gpu_memory_mb" => gpu_memory / (1024^2),
        "cpu_memory_mb" => cpu_memory / (1024^2),
        "total_memory_mb" => (gpu_memory + cpu_memory) / (1024^2)
    )
end