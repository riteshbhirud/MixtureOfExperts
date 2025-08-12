"""
CUDA Kernels for Switch Transformer Loss Computation

High-performance CUDA kernels for computing load balancing losses including
Switch Transformer auxiliary loss with expert assignment tracking and 
parallel reduction operations.
"""

# Kernel to compute expert assignment fractions from routing decisions
function gpu_compute_expert_fractions_kernel!(
    expert_fractions::CuDeviceVector{T},    # num_experts
    expert_assignments::CuDeviceMatrix{Int32}, # top_k × batch_size
    assignment_counts::CuDeviceVector{Int32},   # num_experts (workspace)
    num_experts::Int,
    top_k::Int,
    batch_size::Int,
    total_assignments::Int
) where T<:AbstractFloat
    
    expert_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if expert_idx <= num_experts
        
        # Count assignments for this expert
        assignment_count = 0
        
        for batch_idx in 1:batch_size
            for k in 1:top_k
                if expert_assignments[k, batch_idx] == expert_idx
                    assignment_count += 1
                end
            end
        end
        
        # Store count and compute fraction
        assignment_counts[expert_idx] = assignment_count
        expert_fractions[expert_idx] = T(assignment_count) / T(total_assignments)
    end
    
    return nothing
end

# Optimized kernel using shared memory for small batch sizes
function gpu_compute_expert_fractions_shared_kernel!(
    expert_fractions::CuDeviceVector{T},
    expert_assignments::CuDeviceMatrix{Int32},
    num_experts::Int,
    top_k::Int,
    batch_size::Int,
    total_assignments::Int
) where T<:AbstractFloat
    
    # Shared memory for assignment counts
    shared_counts = CuDynamicSharedArray(Int32, num_experts)
    
    expert_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_id = threadIdx().x
    
    # Initialize shared memory
    if thread_id <= num_experts
        shared_counts[thread_id] = 0
    end
    sync_threads()
    
    # Each thread processes multiple batch elements
    elements_per_thread = cld(batch_size * top_k, blockDim().x)
    start_idx = (thread_id - 1) * elements_per_thread + 1
    end_idx = min(start_idx + elements_per_thread - 1, batch_size * top_k)
    
    for linear_idx in start_idx:end_idx
        # Convert linear index to (k, batch) coordinates
        k = ((linear_idx - 1) % top_k) + 1
        batch_idx = ((linear_idx - 1) ÷ top_k) + 1
        
        if batch_idx <= batch_size
            assigned_expert = expert_assignments[k, batch_idx]
            if assigned_expert >= 1 && assigned_expert <= num_experts
                CUDA.atomic_add!(pointer(shared_counts, assigned_expert), Int32(1))
            end
        end
    end
    
    sync_threads()
    
    # Write results to global memory
    if thread_id <= num_experts
        count = shared_counts[thread_id]
        expert_fractions[thread_id] = T(count) / T(total_assignments)
    end
    
    return nothing
end

# Kernel to compute average router probabilities per expert
function gpu_compute_expert_probabilities_kernel!(
    expert_probabilities::CuDeviceVector{T}, # num_experts
    router_probs::CuDeviceMatrix{T},         # num_experts × batch_size
    num_experts::Int,
    batch_size::Int
) where T<:AbstractFloat
    
    expert_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if expert_idx <= num_experts
        
        # Compute average probability for this expert across all tokens
        prob_sum = T(0)
        
        # Vectorized accumulation with loop unrolling
        batch_idx = 1
        while batch_idx + 3 <= batch_size
            prob_sum += router_probs[expert_idx, batch_idx] +
                       router_probs[expert_idx, batch_idx + 1] +
                       router_probs[expert_idx, batch_idx + 2] +
                       router_probs[expert_idx, batch_idx + 3]
            batch_idx += 4
        end
        
        # Handle remaining elements
        while batch_idx <= batch_size
            prob_sum += router_probs[expert_idx, batch_idx]
            batch_idx += 1
        end
        
        expert_probabilities[expert_idx] = prob_sum / T(batch_size)
    end
    
    return nothing
end

# Parallel reduction kernel for computing expert probabilities (more efficient for large batches)
function gpu_reduce_expert_probabilities_kernel!(
    expert_probabilities::CuDeviceVector{T},
    router_probs::CuDeviceMatrix{T},
    temp_sums::CuDeviceMatrix{T},           # num_experts × num_blocks (workspace)
    num_experts::Int,
    batch_size::Int,
    elements_per_block::Int
) where T<:AbstractFloat
    
    expert_idx = blockIdx().x
    thread_id = threadIdx().x
    block_id = blockIdx().y
    
    if expert_idx <= num_experts
        
        # Each block handles elements_per_block batch elements
        start_batch = (block_id - 1) * elements_per_block + 1
        end_batch = min(start_batch + elements_per_block - 1, batch_size)
        
        # Parallel reduction within block
        local_sum = T(0)
        
        # Each thread accumulates a subset of the range
        threads_per_block = blockDim().x
        elements_per_thread = cld(end_batch - start_batch + 1, threads_per_block)
        
        thread_start = start_batch + (thread_id - 1) * elements_per_thread
        thread_end = min(thread_start + elements_per_thread - 1, end_batch)
        
        for batch_idx in thread_start:thread_end
            if batch_idx <= batch_size
                local_sum += router_probs[expert_idx, batch_idx]
            end
        end
        
        # Shared memory for block-level reduction
        shared_sums = CuDynamicSharedArray(T, threads_per_block)
        shared_sums[thread_id] = local_sum
        sync_threads()
        
        # Tree reduction
        stride = threads_per_block ÷ 2
        while stride > 0
            if thread_id <= stride
                shared_sums[thread_id] += shared_sums[thread_id + stride]
            end
            sync_threads()
            stride ÷= 2
        end
        
        # Store block result
        if thread_id == 1
            temp_sums[expert_idx, block_id] = shared_sums[1]
        end
    end
    
    return nothing
end

# Final reduction kernel to combine block results
function gpu_finalize_expert_probabilities_kernel!(
    expert_probabilities::CuDeviceVector{T},
    temp_sums::CuDeviceMatrix{T},
    num_experts::Int,
    num_blocks::Int,
    batch_size::Int
) where T<:AbstractFloat
    
    expert_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if expert_idx <= num_experts
        
        # Sum across all blocks for this expert
        total_sum = T(0)
        for block_id in 1:num_blocks
            total_sum += temp_sums[expert_idx, block_id]
        end
        
        # Compute average probability
        expert_probabilities[expert_idx] = total_sum / T(batch_size)
    end
    
    return nothing
end

# Switch Transformer auxiliary loss computation kernel
function gpu_switch_transformer_loss_kernel!(
    loss_terms::CuDeviceVector{T},          # num_experts
    expert_fractions::CuDeviceVector{T},    # num_experts
    expert_probabilities::CuDeviceVector{T}, # num_experts
    alpha::T,
    num_experts::Int
) where T<:AbstractFloat
    
    expert_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if expert_idx <= num_experts
        
        # Compute loss term for this expert: f_i * P_i
        fraction = expert_fractions[expert_idx]
        probability = expert_probabilities[expert_idx]
        
        loss_terms[expert_idx] = fraction * probability
    end
    
    return nothing
end

# Parallel reduction to sum loss terms
function gpu_reduce_loss_terms_kernel!(
    partial_sums::CuDeviceVector{T},        # num_blocks
    loss_terms::CuDeviceVector{T},          # num_experts
    num_experts::Int,
    elements_per_block::Int
) where T<:AbstractFloat
    
    block_id = blockIdx().x
    thread_id = threadIdx().x
    
    # Calculate range for this block
    start_idx = (block_id - 1) * elements_per_block + 1
    end_idx = min(start_idx + elements_per_block - 1, num_experts)
    
    # Each thread accumulates a subset
    threads_per_block = blockDim().x
    elements_per_thread = cld(end_idx - start_idx + 1, threads_per_block)
    
    thread_start = start_idx + (thread_id - 1) * elements_per_thread
    thread_end = min(thread_start + elements_per_thread - 1, end_idx)
    
    local_sum = T(0)
    for idx in thread_start:thread_end
        if idx <= num_experts
            local_sum += loss_terms[idx]
        end
    end
    
    # Shared memory reduction
    shared_sums = CuDynamicSharedArray(T, threads_per_block)
    shared_sums[thread_id] = local_sum
    sync_threads()
    
    # Tree reduction
    stride = threads_per_block ÷ 2
    while stride > 0
        if thread_id <= stride
            shared_sums[thread_id] += shared_sums[thread_id + stride]
        end
        sync_threads()
        stride ÷= 2
    end
    
    # Store block result
    if thread_id == 1
        partial_sums[block_id] = shared_sums[1]
    end
    
    return nothing
end

# Load balancing score computation kernel
# Fixed GPU-compatible version of the load balance score kernel
function gpu_load_balance_score_kernel!(
    balance_scores::CuDeviceVector{Float32},
    expert_assignments::CuDeviceMatrix{Int32},
    num_experts::Int64,
    top_k::Int64,
    batch_size::Int64
)
    
    # Use first thread to compute the score
    if blockIdx().x == 1 && threadIdx().x == 1
        
        # Use a simple approach without dynamic arrays
        # Since we can't use MVector in GPU kernels, we'll compute directly
        
        # Count actual assignments per expert (up to 256 experts)
        total_assignments = 0
        sum_squared_deviations = Float32(0)
        
        # First pass: count total assignments
        for batch_idx in 1:batch_size
            for k in 1:top_k
                expert_id = expert_assignments[k, batch_idx]
                if expert_id >= 1 && expert_id <= num_experts
                    total_assignments += 1
                end
            end
        end
        
        if total_assignments > 0
            # Compute ideal assignment count
            ideal_count = Float32(total_assignments) / Float32(num_experts)
            
            # Second pass: compute variance by counting each expert's assignments
            for expert_idx in 1:num_experts
                expert_count = 0
                
                # Count assignments for this specific expert
                for batch_idx in 1:batch_size
                    for k in 1:top_k
                        expert_id = expert_assignments[k, batch_idx]
                        if expert_id == expert_idx
                            expert_count += 1
                        end
                    end
                end
                
                # Accumulate squared deviation
                diff = Float32(expert_count) - ideal_count
                sum_squared_deviations += diff * diff
            end
            
            # Compute variance
            variance = sum_squared_deviations / Float32(num_experts)
            
            # Compute balance score (1.0 = perfect balance)
            if ideal_count > Float32(0)
                balance_scores[1] = Float32(1) - sqrt(variance) / ideal_count
            else
                balance_scores[1] = Float32(1)
            end
        else
            balance_scores[1] = Float32(1)
        end
    end
    
    return nothing
end
# Expert usage entropy computation kernel
function gpu_expert_entropy_kernel!(
    entropy_values::CuDeviceVector{T},      # 1 element  
    expert_probabilities::CuDeviceVector{T}, # num_experts
    num_experts::Int,
    epsilon::T
) where T<:AbstractFloat
    
    if blockIdx().x == 1 && threadIdx().x == 1
        
        entropy = T(0)
        
        for expert_idx in 1:num_experts
            prob = expert_probabilities[expert_idx]
            if prob > epsilon
                entropy -= prob * log(prob)
            end
        end
        
        entropy_values[1] = entropy
    end
    
    return nothing
end

# High-level kernel launcher functions
function launch_expert_fractions_kernel!(
    expert_fractions::CuVector{T},
    expert_assignments::CuMatrix{Int32};
    use_shared_memory::Bool = false
) where T<:AbstractFloat
    
    num_experts = length(expert_fractions)
    top_k, batch_size = size(expert_assignments)
    total_assignments = top_k * batch_size
    
    if use_shared_memory && num_experts <= 256 && batch_size * top_k <= 4096
        # Use shared memory optimization
        threads = min(256, num_experts * 4)
        blocks = 1
        shared_mem_size = sizeof(Int32) * num_experts
        
        @cuda threads=threads blocks=blocks shmem=shared_mem_size gpu_compute_expert_fractions_shared_kernel!(
            expert_fractions, expert_assignments,
            num_experts, top_k, batch_size, total_assignments
        )
        
    else
        # Use general kernel
        assignment_counts = CUDA.zeros(Int32, num_experts)
        
        threads = 256
        blocks = cld(num_experts, 256)
        
        @cuda threads=threads blocks=blocks gpu_compute_expert_fractions_kernel!(
            expert_fractions, expert_assignments, assignment_counts,
            num_experts, top_k, batch_size, total_assignments
        )
    end
    
    CUDA.synchronize()
    return expert_fractions
end

function launch_expert_probabilities_kernel!(
    expert_probabilities::CuVector{T},
    router_probs::CuMatrix{T};
    use_parallel_reduction::Bool = false
) where T<:AbstractFloat
    
    num_experts, batch_size = size(router_probs)
    
    if use_parallel_reduction && batch_size >= 1024
        # Use parallel reduction for large batches
        elements_per_block = 256
        num_blocks = cld(batch_size, elements_per_block)
        
        temp_sums = CUDA.zeros(T, num_experts, num_blocks)
        
        # Phase 1: Parallel reduction within blocks
        threads = 256
        blocks = (num_experts, num_blocks)
        shared_mem_size = sizeof(T) * threads
        
        @cuda threads=threads blocks=blocks shmem=shared_mem_size gpu_reduce_expert_probabilities_kernel!(
            expert_probabilities, router_probs, temp_sums,
            num_experts, batch_size, elements_per_block
        )
        
        # Phase 2: Final reduction across blocks
        threads = 256
        blocks = cld(num_experts, 256)
        
        @cuda threads=threads blocks=blocks gpu_finalize_expert_probabilities_kernel!(
            expert_probabilities, temp_sums, num_experts, num_blocks, batch_size
        )
        
    else
        # Use simple kernel for small to medium batches
        threads = 256
        blocks = cld(num_experts, 256)
        
        @cuda threads=threads blocks=blocks gpu_compute_expert_probabilities_kernel!(
            expert_probabilities, router_probs, num_experts, batch_size
        )
    end
    
    CUDA.synchronize()
    return expert_probabilities
end

function launch_switch_loss_kernel!(
    expert_fractions::CuVector{T},
    expert_probabilities::CuVector{T},
    alpha::T
) where T<:AbstractFloat
    
    num_experts = length(expert_fractions)
    
    # Compute individual loss terms
    loss_terms = CUDA.zeros(T, num_experts)
    
    threads = 256
    blocks = cld(num_experts, 256)
    
    @cuda threads=threads blocks=blocks gpu_switch_transformer_loss_kernel!(
        loss_terms, expert_fractions, expert_probabilities, alpha, num_experts
    )
    
    # Reduce to final loss value
    elements_per_block = 256
    num_reduction_blocks = cld(num_experts, elements_per_block)
    partial_sums = CUDA.zeros(T, num_reduction_blocks)
    
    threads = 256
    blocks = num_reduction_blocks
    shared_mem_size = sizeof(T) * threads
    
    @cuda threads=threads blocks=blocks shmem=shared_mem_size gpu_reduce_loss_terms_kernel!(
        partial_sums, loss_terms, num_experts, elements_per_block
    )
    
    # Final sum on CPU (small array)
    partial_sums_cpu = Array(partial_sums)
    total_loss = sum(partial_sums_cpu)
    
    # Apply scaling factor: α * N * Σ(f_i * P_i)
    final_loss = alpha * T(num_experts) * total_loss
    
    CUDA.synchronize()
    return final_loss
end

function launch_load_balance_score_kernel!(
    expert_assignments::CuMatrix{Int32}
)  # Remove the "where T<:AbstractFloat"
    
    num_experts = Int64(maximum(Array(expert_assignments)))   # Assuming 1-indexed
    top_k, batch_size = size(expert_assignments)
    
    balance_scores = CUDA.zeros(Float32, 1)  # Use concrete Float32 instead of T
    
    threads = 1
    blocks = 1
    
    @cuda threads=threads blocks=blocks gpu_load_balance_score_kernel!(
        balance_scores, expert_assignments, num_experts, top_k, batch_size
    )
    
    CUDA.synchronize()
    return Array(balance_scores)[1]
end

function launch_expert_entropy_kernel!(
    expert_probabilities::CuVector{T};
    epsilon::T = T(1e-8)
) where T<:AbstractFloat
    
    num_experts = length(expert_probabilities)
    entropy_values = CUDA.zeros(T, 1)
    
    threads = 1
    blocks = 1
    
    @cuda threads=threads blocks=blocks gpu_expert_entropy_kernel!(
        entropy_values, expert_probabilities, num_experts, epsilon
    )
    
    CUDA.synchronize()
    return Array(entropy_values)[1]
end