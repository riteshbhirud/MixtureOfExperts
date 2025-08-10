"""
GPU Token Routing and Expert Assignment Utilities

High-performance GPU utilities for routing tokens to experts based on gating decisions,
creating efficient batching patterns for parallel expert computation, and managing
expert assignment mappings with optimized memory access patterns.
"""

# Token assignment mapping structure for efficient GPU routing
struct GPUTokenAssignment{T<:AbstractFloat}
    # Expert assignment mapping
    expert_token_counts::CuVector{Int32}     # num_experts - tokens per expert
    expert_token_offsets::CuVector{Int32}    # num_experts - cumulative offsets
    token_to_expert_map::CuMatrix{Int32}     # top_k × batch_size - expert assignments
    token_expert_weights::CuMatrix{T}        # top_k × batch_size - gating weights
    
    # Sorted token indices for efficient expert processing
    sorted_token_indices::CuVector{Int32}    # total_assignments - global token order
    sorted_expert_ids::CuVector{Int32}       # total_assignments - corresponding experts
    sorted_weights::CuVector{T}              # total_assignments - corresponding weights
    
    # Reverse mapping for output combination
    output_token_indices::CuVector{Int32}    # total_assignments - output positions
    output_weights::CuVector{T}              # total_assignments - combination weights
    
    # Metadata
    batch_size::Int
    top_k::Int
    num_experts::Int
    total_assignments::Int
    max_tokens_per_expert::Int
    
    # Workspace allocation tracking
    workspace_bytes::Int64
    allocation_time::Float64
end

"""
    create_token_assignment(expert_indices, expert_gates, config)

Create efficient token assignment mapping from gating decisions.
"""
function create_token_assignment(
    expert_indices::CuMatrix{Int32},      # top_k × batch_size
    expert_gates::CuMatrix{T},            # top_k × batch_size  
    config::GPUMoEConfig{T}
) where T<:AbstractFloat
    
    start_time = time()
    
    top_k, batch_size = size(expert_indices)
    num_experts = config.num_experts
    total_assignments = top_k * batch_size
    
    # Allocate assignment tracking arrays
    expert_token_counts = CUDA.zeros(Int32, num_experts)
    expert_token_offsets = CUDA.zeros(Int32, num_experts + 1)  # +1 for easier indexing
    
    # Copy input data for processing
    token_to_expert_map = copy(expert_indices)
    token_expert_weights = copy(expert_gates)
    
    # Phase 1: Count tokens per expert
    @gpu_time "count_tokens_per_expert" count_tokens_per_expert_kernel!(
        expert_token_counts, expert_indices, top_k, batch_size, num_experts
    )
    
    # Phase 2: Compute cumulative offsets for expert batching
    @gpu_time "compute_expert_offsets" compute_expert_offsets!(
        expert_token_offsets, expert_token_counts, num_experts
    )
    
    # Get max tokens per expert for workspace allocation
    max_tokens_per_expert = Int(CUDA.reduce(max, expert_token_counts))
    
    # Phase 3: Create sorted assignment arrays for efficient processing
    sorted_token_indices = CUDA.zeros(Int32, total_assignments)
    sorted_expert_ids = CUDA.zeros(Int32, total_assignments)
    sorted_weights = CUDA.zeros(T, total_assignments)
    output_token_indices = CUDA.zeros(Int32, total_assignments)
    output_weights = CUDA.zeros(T, total_assignments)
    
    @gpu_time "create_sorted_assignments" create_sorted_assignments_kernel!(
        sorted_token_indices, sorted_expert_ids, sorted_weights,
        output_token_indices, output_weights,
        expert_indices, expert_gates, expert_token_offsets,
        top_k, batch_size, num_experts
    )
    
    # Calculate workspace memory usage
    workspace_bytes = (
        sizeof(expert_token_counts) + sizeof(expert_token_offsets) +
        sizeof(token_to_expert_map) + sizeof(token_expert_weights) +
        sizeof(sorted_token_indices) + sizeof(sorted_expert_ids) + 
        sizeof(sorted_weights) + sizeof(output_token_indices) + sizeof(output_weights)
    )
    
    allocation_time = time() - start_time
    
    return GPUTokenAssignment{T}(
        expert_token_counts, expert_token_offsets,
        token_to_expert_map, token_expert_weights,
        sorted_token_indices, sorted_expert_ids, sorted_weights,
        output_token_indices, output_weights,
        batch_size, top_k, num_experts, total_assignments, max_tokens_per_expert,
        workspace_bytes, allocation_time
    )
end

# CUDA kernel to count tokens assigned to each expert
function count_tokens_per_expert_kernel!(
    expert_counts::CuDeviceVector{Int32},
    expert_indices::CuDeviceMatrix{Int32},
    top_k::Int,
    batch_size::Int,
    num_experts::Int
)
    expert_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if expert_id <= num_experts
        count = Int32(0)
        
        # Count assignments for this expert across all tokens and k positions
        for batch_idx in 1:batch_size
            for k in 1:top_k
                if expert_indices[k, batch_idx] == expert_id
                    count += Int32(1)
                end
            end
        end
        
        expert_counts[expert_id] = count
    end
    
    return nothing
end

# Compute cumulative offsets for expert-wise token batching
function compute_expert_offsets!(
    expert_offsets::CuVector{Int32},
    expert_counts::CuVector{Int32},
    num_experts::Int
)
    # Use prefix sum to compute cumulative offsets
    # This is done on CPU for simplicity, but could be GPU-accelerated
    counts_cpu = Array(expert_counts)
    offsets_cpu = zeros(Int32, num_experts + 1)
    
    cumsum = Int32(0)
    for i in 1:num_experts
        offsets_cpu[i] = cumsum
        cumsum += counts_cpu[i]
    end
    offsets_cpu[num_experts + 1] = cumsum
    
    copyto!(expert_offsets, offsets_cpu)
    CUDA.synchronize()
end

# CUDA kernel to create sorted assignment arrays
function create_sorted_assignments_kernel!(
    sorted_token_indices::CuDeviceVector{Int32},
    sorted_expert_ids::CuDeviceVector{Int32},
    sorted_weights::CuDeviceVector{T},
    output_token_indices::CuDeviceVector{Int32},
    output_weights::CuDeviceVector{T},
    expert_indices::CuDeviceMatrix{Int32},
    expert_gates::CuDeviceMatrix{T},
    expert_offsets::CuDeviceVector{Int32},
    top_k::Int,
    batch_size::Int,
    num_experts::Int
) where T<:AbstractFloat
    
    thread_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total_assignments = top_k * batch_size
    
    if thread_id <= total_assignments
        # Convert linear thread ID to (k, batch) coordinates
        k_idx = ((thread_id - 1) % top_k) + 1
        batch_idx = ((thread_id - 1) ÷ top_k) + 1
        
        expert_id = expert_indices[k_idx, batch_idx]
        weight = expert_gates[k_idx, batch_idx]
        
        if expert_id >= 1 && expert_id <= num_experts
            # Calculate global token index
            global_token_idx = (batch_idx - 1) * top_k + k_idx
            
            # Find position in sorted arrays using atomic operations
            expert_offset = expert_offsets[expert_id]
            
            # Use atomic increment to get next available slot for this expert
            slot_idx = CUDA.atomic_add!(pointer(expert_offsets, expert_id), Int32(1)) + Int32(1)
            
            if slot_idx <= length(sorted_token_indices)
                # Store assignment information
                sorted_token_indices[slot_idx] = global_token_idx
                sorted_expert_ids[slot_idx] = expert_id
                sorted_weights[slot_idx] = weight
                
                # Store reverse mapping for output combination
                output_token_indices[slot_idx] = global_token_idx
                output_weights[slot_idx] = weight
            end
        end
    end
    
    return nothing
end

"""
    route_tokens_to_experts!(expert_inputs, input, assignment)

Route input tokens to expert-specific input buffers based on assignment mapping.
"""
function route_tokens_to_experts!(
    expert_inputs::Vector{CuMatrix{T}},    # One input buffer per expert
    input::CuMatrix{T},                    # input_dim × batch_size
    assignment::GPUTokenAssignment{T}
) where T<:AbstractFloat
    
    input_dim, batch_size = size(input)
    num_experts = length(expert_inputs)
    
    # Validate expert input buffer sizes
    for expert_id in 1:num_experts
        expected_tokens = Int(assignment.expert_token_counts[expert_id])
        if expected_tokens > 0 && size(expert_inputs[expert_id], 2) < expected_tokens
            throw(DimensionMismatch("Expert $expert_id input buffer too small: has $(size(expert_inputs[expert_id], 2)), needs $expected_tokens"))
        end
    end
    
    # Launch routing kernel for each expert that has assigned tokens
    for expert_id in 1:num_experts
        expert_token_count = Int(assignment.expert_token_counts[expert_id])
        
        if expert_token_count > 0
            expert_offset = Int(assignment.expert_token_offsets[expert_id])
            expert_input_buffer = view(expert_inputs[expert_id], :, 1:expert_token_count)
            
            @gpu_time "route_tokens_expert_$expert_id" route_tokens_kernel!(
                expert_input_buffer, input, assignment.sorted_token_indices,
                expert_offset, expert_token_count, input_dim, batch_size, assignment.top_k
            )
        end
    end
    
    CUDA.synchronize()
end

# CUDA kernel to route tokens to specific expert
function route_tokens_kernel!(
    expert_input::CuDeviceMatrix{T},
    global_input::CuDeviceMatrix{T},
    sorted_token_indices::CuDeviceVector{Int32},
    expert_offset::Int,
    expert_token_count::Int,
    input_dim::Int,
    batch_size::Int,
    top_k::Int
) where T<:AbstractFloat
    
    dim_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    token_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if dim_idx <= input_dim && token_idx <= expert_token_count
        # Get global token index for this expert token
        global_idx = sorted_token_indices[expert_offset + token_idx]
        
        # Convert global index back to (k, batch) coordinates
        k_pos = ((global_idx - 1) % top_k) + 1
        batch_pos = ((global_idx - 1) ÷ top_k) + 1
        
        if batch_pos <= batch_size
            # Copy input data for this token
            expert_input[dim_idx, token_idx] = global_input[dim_idx, batch_pos]
        end
    end
    
    return nothing
end

"""
    combine_expert_outputs!(output, expert_outputs, assignment)

Combine expert outputs back to original token order using gating weights.
"""
function combine_expert_outputs!(
    output::CuMatrix{T},                   # output_dim × batch_size
    expert_outputs::Vector{CuMatrix{T}},   # Vector of expert outputs
    assignment::GPUTokenAssignment{T}
) where T<:AbstractFloat
    
    output_dim, batch_size = size(output)
    num_experts = length(expert_outputs)
    
    # Initialize output to zero
    fill!(output, T(0))
    
    # Combine outputs from each expert
    for expert_id in 1:num_experts
        expert_token_count = Int(assignment.expert_token_counts[expert_id])
        
        if expert_token_count > 0
            expert_offset = Int(assignment.expert_token_offsets[expert_id])
            expert_output_buffer = view(expert_outputs[expert_id], :, 1:expert_token_count)
            
            @gpu_time "combine_expert_$expert_id" combine_expert_output_kernel!(
                output, expert_output_buffer, assignment.sorted_token_indices,
                assignment.sorted_weights, expert_offset, expert_token_count,
                output_dim, batch_size, assignment.top_k
            )
        end
    end
    
    CUDA.synchronize()
end

# CUDA kernel to combine expert output with gating weights
function combine_expert_output_kernel!(
    global_output::CuDeviceMatrix{T},
    expert_output::CuDeviceMatrix{T},
    sorted_token_indices::CuDeviceVector{Int32},
    sorted_weights::CuDeviceVector{T},
    expert_offset::Int,
    expert_token_count::Int,
    output_dim::Int,
    batch_size::Int,
    top_k::Int
) where T<:AbstractFloat
    
    dim_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    token_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if dim_idx <= output_dim && token_idx <= expert_token_count
        # Get global token index and weight
        global_idx = sorted_token_indices[expert_offset + token_idx]
        weight = sorted_weights[expert_offset + token_idx]
        
        # Convert global index back to batch position
        k_pos = ((global_idx - 1) % top_k) + 1
        batch_pos = ((global_idx - 1) ÷ top_k) + 1
        
        if batch_pos <= batch_size
            # Get expert output value and apply gating weight
            expert_value = expert_output[dim_idx, token_idx]
            weighted_value = expert_value * weight
            
            # Atomically add to global output (multiple experts may contribute to same token)
            CUDA.atomic_add!(pointer(global_output, dim_idx, batch_pos), weighted_value)
        end
    end
    
    return nothing
end

"""
    validate_token_assignment(assignment)

Validate token assignment mapping for correctness and efficiency.
"""
function validate_token_assignment(assignment::GPUTokenAssignment{T}) where T<:AbstractFloat
    
    # Check that total assignments match expected
    total_counted = CUDA.reduce(+, assignment.expert_token_counts)
    if total_counted != assignment.total_assignments
        @warn "Assignment count mismatch: counted $total_counted, expected $(assignment.total_assignments)"
        return false
    end
    
    # Check expert offsets are monotonic
    offsets_cpu = Array(assignment.expert_token_offsets)
    for i in 2:length(offsets_cpu)
        if offsets_cpu[i] < offsets_cpu[i-1]
            @warn "Non-monotonic expert offsets detected at position $i"
            return false
        end
    end
    
    # Check weights are finite and non-negative
    if !gpu_check_finite(assignment.sorted_weights)
        @warn "Non-finite weights detected in assignment"
        return false
    end
    
    min_weight = CUDA.reduce(min, assignment.sorted_weights)
    if min_weight < T(0)
        @warn "Negative weights detected: minimum = $min_weight"
        return false
    end
    
    # Check token indices are in valid range
    max_token_idx = CUDA.reduce(max, assignment.sorted_token_indices)
    if max_token_idx > assignment.total_assignments
        @warn "Token index out of range: maximum = $max_token_idx, expected ≤ $(assignment.total_assignments)"
        return false
    end
    
    return true
end

"""
    get_assignment_statistics(assignment)

Get detailed statistics about token assignment for analysis.
"""
function get_assignment_statistics(assignment::GPUTokenAssignment{T}) where T<:AbstractFloat
    
    counts_cpu = Array(assignment.expert_token_counts)
    weights_cpu = Array(assignment.sorted_weights)
    
    # Expert usage statistics
    active_experts = count(x -> x > 0, counts_cpu)
    min_usage = minimum(counts_cpu)
    max_usage = maximum(counts_cpu)
    mean_usage = sum(counts_cpu) / length(counts_cpu)
    usage_variance = sum((counts_cpu .- mean_usage).^2) / length(counts_cpu)
    
    # Weight statistics
    mean_weight = mean(weights_cpu)
    weight_variance = var(weights_cpu)
    min_weight = minimum(weights_cpu)
    max_weight = maximum(weights_cpu)
    
    # Load balance metrics
    ideal_usage = assignment.total_assignments / assignment.num_experts
    max_deviation = maximum(abs.(counts_cpu .- ideal_usage))
    balance_score = 1.0 - (max_deviation / max(ideal_usage, 1.0))
    
    return Dict{String, Any}(
        "assignment_info" => Dict(
            "batch_size" => assignment.batch_size,
            "top_k" => assignment.top_k,
            "num_experts" => assignment.num_experts,
            "total_assignments" => assignment.total_assignments,
            "max_tokens_per_expert" => assignment.max_tokens_per_expert
        ),
        "expert_usage" => Dict(
            "active_experts" => active_experts,
            "min_usage" => min_usage,
            "max_usage" => max_usage,
            "mean_usage" => mean_usage,
            "usage_variance" => usage_variance,
            "expert_counts" => counts_cpu
        ),
        "weight_statistics" => Dict(
            "mean_weight" => mean_weight,
            "weight_variance" => weight_variance,
            "min_weight" => min_weight,
            "max_weight" => max_weight
        ),
        "load_balance" => Dict(
            "ideal_usage" => ideal_usage,
            "max_deviation" => max_deviation,
            "balance_score" => balance_score,
            "is_balanced" => balance_score >= 0.8
        ),
        "memory_usage" => Dict(
            "workspace_bytes" => assignment.workspace_bytes,
            "workspace_mb" => assignment.workspace_bytes / (1024^2),
            "allocation_time_ms" => assignment.allocation_time * 1000
        )
    )
end

"""
    optimize_assignment_memory!(assignment)

Optimize memory usage patterns for the assignment mapping.
"""
function optimize_assignment_memory!(assignment::GPUTokenAssignment{T}) where T<:AbstractFloat
    
    counts_cpu = Array(assignment.expert_token_counts)
    
    unused_experts = findall(x -> x == 0, counts_cpu)
    
    if !isempty(unused_experts)
        @debug "Found $(length(unused_experts)) unused experts in assignment"
        

    end
    
    max_usage = maximum(counts_cpu)
    min_usage = minimum(counts_cpu[counts_cpu .> 0]) 
    if length(counts_cpu[counts_cpu .> 0]) > 0 && max_usage / min_usage > 10
        @debug "Highly imbalanced expert usage detected: max=$max_usage, min=$min_usage"
    end
    
    return assignment
end