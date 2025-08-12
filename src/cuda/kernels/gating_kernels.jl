"""
CUDA Kernels for TopK Gating and Routing

High-performance CUDA kernels for computing router logits, softmax normalization,
top-k expert selection, and gate weight computation with optimizations for
different batch sizes and expert counts.
"""

# Router logits computation kernel with optimized memory access
function gpu_router_logits_kernel!(
    router_logits::CuDeviceMatrix{T},    # num_experts × batch_size
    input::CuDeviceMatrix{T},            # input_dim × batch_size
    router_weights::CuDeviceMatrix{T},   # input_dim × num_experts
    router_bias::CuDeviceVector{T},      # num_experts (optional)
    input_dim::Int,
    num_experts::Int,
    batch_size::Int,
    use_bias::Bool
) where T<:AbstractFloat
    
    expert_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    batch_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if expert_idx <= num_experts && batch_idx <= batch_size
        
        # Compute dot product for this expert-batch pair
        logit_val = T(0)
        
        # Loop unrolling for better performance
        i = 1
        while i + 3 <= input_dim
            logit_val += router_weights[i, expert_idx] * input[i, batch_idx] +
                        router_weights[i+1, expert_idx] * input[i+1, batch_idx] +
                        router_weights[i+2, expert_idx] * input[i+2, batch_idx] +
                        router_weights[i+3, expert_idx] * input[i+3, batch_idx]
            i += 4
        end
        
        # Handle remaining elements
        while i <= input_dim
            logit_val += router_weights[i, expert_idx] * input[i, batch_idx]
            i += 1
        end
        
        # Add bias if present
        if use_bias
            logit_val += router_bias[expert_idx]
        end
        
        router_logits[expert_idx, batch_idx] = logit_val
    end
    
    return nothing
end

# Numerically stable softmax kernel for router probabilities
function gpu_router_softmax_kernel!(
    router_probs::CuDeviceMatrix{T},     # num_experts × batch_size
    router_logits::CuDeviceMatrix{T},    # num_experts × batch_size
    max_logits::CuDeviceVector{T},       # batch_size (workspace)
    sum_exp::CuDeviceVector{T},          # batch_size (workspace)
    num_experts::Int,
    batch_size::Int,
    epsilon::T
) where T<:AbstractFloat
    
    batch_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if batch_idx <= batch_size
        
        # Phase 1: Find maximum logit for numerical stability
        max_val = router_logits[1, batch_idx]
        for expert_idx in 2:num_experts
            val = router_logits[expert_idx, batch_idx]
            if val > max_val
                max_val = val
            end
        end
        max_logits[batch_idx] = max_val
        
        # Phase 2: Compute sum of exponentials
        exp_sum = T(0)
        for expert_idx in 1:num_experts
            exp_val = exp(router_logits[expert_idx, batch_idx] - max_val)
            exp_sum += exp_val
        end
        
        # Ensure numerical stability
        exp_sum = max(exp_sum, epsilon)
        sum_exp[batch_idx] = exp_sum
        
        # Phase 3: Compute softmax probabilities
        for expert_idx in 1:num_experts
            exp_val = exp(router_logits[expert_idx, batch_idx] - max_val)
            router_probs[expert_idx, batch_idx] = exp_val / exp_sum
        end
    end
    
    return nothing
end

# Optimized softmax kernel using shared memory for small numbers of experts
function gpu_router_softmax_shared_kernel!(
    router_probs::CuDeviceMatrix{T},
    router_logits::CuDeviceMatrix{T},
    num_experts::Int,
    batch_size::Int,
    epsilon::T
) where T<:AbstractFloat
    
    # Shared memory for this block's logits and probabilities
    shared_logits = CuDynamicSharedArray(T, num_experts)
    
    batch_idx = blockIdx().x
    tid = threadIdx().x
    
    if batch_idx <= batch_size
        
        # Load logits into shared memory
        if tid <= num_experts
            shared_logits[tid] = router_logits[tid, batch_idx]
        end
        sync_threads()
        
        # Find maximum using parallel reduction
        if tid == 1
            max_val = shared_logits[1]
            for i in 2:num_experts
                if shared_logits[i] > max_val
                    max_val = shared_logits[i]
                end
            end
            
            # Compute exponentials and sum
            exp_sum = T(0)
            for i in 1:num_experts
                exp_val = exp(shared_logits[i] - max_val)
                shared_logits[i] = exp_val  # Reuse shared memory
                exp_sum += exp_val
            end
            
            # Ensure numerical stability
            exp_sum = max(exp_sum, epsilon)
            
            # Normalize and write to global memory
            for i in 1:num_experts
                router_probs[i, batch_idx] = shared_logits[i] / exp_sum
            end
        end
    end
    
    return nothing
end

function gpu_topk_selection_simple_kernel!(
    expert_indices::CuDeviceMatrix{Int32}, # top_k × batch_size
    expert_gates::CuDeviceMatrix{T},       # top_k × batch_size
    router_probs::CuDeviceMatrix{T},       # num_experts × batch_size
    num_experts::Int,
    top_k::Int,
    batch_size::Int
) where T<:AbstractFloat
    
    batch_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if batch_idx <= batch_size
        # Simple selection sort for top-k (efficient for small k)
        for k_pos in 1:top_k
            max_val = T(-Inf)
            max_idx = Int32(1)
            
            # Find the k_pos-th largest element
            for expert_idx in 1:num_experts
                val = router_probs[expert_idx, batch_idx]
                
                # Check if this value is larger than current max
                # and hasn't been selected before
                is_larger = val > max_val
                already_selected = false
                
                # Check if this expert was already selected
                for prev_k in 1:(k_pos-1)
                    if expert_indices[prev_k, batch_idx] == expert_idx
                        already_selected = true
                        break
                    end
                end
                
                if is_larger && !already_selected
                    max_val = val
                    max_idx = Int32(expert_idx)
                end
            end
            
            expert_indices[k_pos, batch_idx] = max_idx
            expert_gates[k_pos, batch_idx] = max_val
        end
        
        # Renormalize gates for this batch
        gate_sum = T(0)
        for k in 1:top_k
            gate_sum += expert_gates[k, batch_idx]
        end
        
        if gate_sum > T(1e-8)
            for k in 1:top_k
                expert_gates[k, batch_idx] /= gate_sum
            end
        else
            # Uniform distribution if all gates are zero
            uniform_gate = T(1) / T(top_k)
            for k in 1:top_k
                expert_gates[k, batch_idx] = uniform_gate
            end
        end
    end
    
    return nothing
end

# Top-K selection kernel using heap-based algorithm for larger k
function gpu_topk_selection_heap_kernel!(
    expert_indices::CuDeviceMatrix{Int32},
    expert_gates::CuDeviceMatrix{T},
    router_probs::CuDeviceMatrix{T},
    temp_workspace::CuDeviceMatrix{T},     # num_experts × batch_size workspace
    num_experts::Int,
    top_k::Int,
    batch_size::Int
) where T<:AbstractFloat
    
    batch_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if batch_idx <= batch_size
        
        # Use workspace for this batch
        values_col = @view temp_workspace[:, batch_idx]
        
        # Copy probabilities to workspace with indices
        for i in 1:num_experts
            values_col[i] = router_probs[i, batch_idx]
        end
        
        # Partial sort to get top-k elements using selection algorithm
        for k in 1:top_k
            # Find k-th largest element
            max_idx = k
            max_val = values_col[k]
            
            for i in (k + 1):num_experts
                if values_col[i] > max_val
                    max_val = values_col[i]
                    max_idx = i
                end
            end
            
            # Swap with position k
            if max_idx != k
                temp_val = values_col[k]
                values_col[k] = values_col[max_idx] 
                values_col[max_idx] = temp_val
            end
            
            expert_indices[k, batch_idx] = Int32(k)
            expert_gates[k, batch_idx] = values_col[k]
        end
        
        # Renormalize gates
        gate_sum = T(0)
        for k in 1:top_k
            gate_sum += expert_gates[k, batch_idx]
        end
        
        if gate_sum > T(0)
            for k in 1:top_k
                expert_gates[k, batch_idx] /= gate_sum
            end
        end
    end
    
    return nothing
end

# Advanced top-k selection using parallel merge for very large expert counts
function gpu_topk_selection_parallel_kernel!(
    expert_indices::CuDeviceMatrix{Int32},
    expert_gates::CuDeviceMatrix{T},
    router_probs::CuDeviceMatrix{T},
    sorted_indices::CuDeviceMatrix{Int32}, # num_experts × batch_size workspace
    sorted_values::CuDeviceMatrix{T},      # num_experts × batch_size workspace
    num_experts::Int,
    top_k::Int,
    batch_size::Int
) where T<:AbstractFloat
    
    batch_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    thread_id = threadIdx().x
    block_size = blockDim().x
    
    if batch_idx <= batch_size
        
        # Initialize index array
        for i in 1:num_experts
            sorted_indices[i, batch_idx] = Int32(i)
            sorted_values[i, batch_idx] = router_probs[i, batch_idx]
        end
        
        # Bitonic sort for power-of-2 sizes, selection sort otherwise
        if ispow2(num_experts) && num_experts <= 1024
            # Bitonic sort implementation
            step = 2
            while step <= num_experts
                substep = step ÷ 2
                while substep > 0
                    for i in 1:num_experts
                        j = i ⊻ substep  # XOR for bitonic pattern
                        if j > i && j <= num_experts
                            val_i = sorted_values[i, batch_idx]
                            val_j = sorted_values[j, batch_idx]
                            
                            # Determine sort direction
                            ascending = ((i - 1) ÷ step) % 2 == 0
                            should_swap = ascending ? (val_i < val_j) : (val_i > val_j)
                            
                            if should_swap
                                # Swap values
                                sorted_values[i, batch_idx] = val_j
                                sorted_values[j, batch_idx] = val_i
                                
                                # Swap indices
                                idx_i = sorted_indices[i, batch_idx]
                                sorted_indices[i, batch_idx] = sorted_indices[j, batch_idx]
                                sorted_indices[j, batch_idx] = idx_i
                            end
                        end
                    end
                    substep ÷= 2
                end
                step *= 2
            end
            
        else
            # Selection sort for non-power-of-2 or large arrays
            for k in 1:min(top_k, num_experts)
                max_idx = k
                max_val = sorted_values[k, batch_idx]
                
                # Find maximum in remaining elements
                for i in (k + 1):num_experts
                    if sorted_values[i, batch_idx] > max_val
                        max_val = sorted_values[i, batch_idx]
                        max_idx = i
                    end
                end
                
                # Swap with position k
                if max_idx != k
                    # Swap values
                    temp_val = sorted_values[k, batch_idx]
                    sorted_values[k, batch_idx] = sorted_values[max_idx, batch_idx]
                    sorted_values[max_idx, batch_idx] = temp_val
                    
                    # Swap indices
                    temp_idx = sorted_indices[k, batch_idx]
                    sorted_indices[k, batch_idx] = sorted_indices[max_idx, batch_idx]
                    sorted_indices[max_idx, batch_idx] = temp_idx
                end
            end
        end
        
        # Extract top-k results and renormalize
        gate_sum = T(0)
        for k in 1:top_k
            expert_gates[k, batch_idx] = sorted_values[k, batch_idx]
            expert_indices[k, batch_idx] = sorted_indices[k, batch_idx]
            gate_sum += expert_gates[k, batch_idx]
        end
        
        # Renormalize gates
        if gate_sum > T(0)
            for k in 1:top_k
                expert_gates[k, batch_idx] /= gate_sum
            end
        end
    end
    
    return nothing
end

# Gate renormalization kernel (ensures top-k gates sum to 1)
function gpu_renormalize_gates_kernel!(
    expert_gates::CuDeviceMatrix{T},
    top_k::Int,
    batch_size::Int,
    epsilon::T
) where T<:AbstractFloat
    
    batch_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if batch_idx <= batch_size
        
        # Compute sum of gates for this batch
        gate_sum = T(0)
        for k in 1:top_k
            gate_sum += expert_gates[k, batch_idx]
        end
        
        # Renormalize if sum is non-zero
        if gate_sum > epsilon
            for k in 1:top_k
                expert_gates[k, batch_idx] /= gate_sum
            end
        else
            # Uniform distribution if all gates are zero
            uniform_gate = T(1) / T(top_k)
            for k in 1:top_k
                expert_gates[k, batch_idx] = uniform_gate
            end
        end
    end
    
    return nothing
end

# Noise injection kernel for training (adds Gaussian noise to router logits)
function gpu_add_router_noise_kernel!(
    router_logits::CuDeviceMatrix{T},
    noise_scale::T,
    num_experts::Int,
    batch_size::Int,
    seed::UInt64
) where T<:AbstractFloat
    
    expert_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    batch_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if expert_idx <= num_experts && batch_idx <= batch_size
        
        # Generate deterministic but pseudo-random noise
        # Using simple LCG for GPU-friendly random number generation
        rng_state = seed + expert_idx * 1664525 + batch_idx * 1013904223
        rng_state = rng_state * 1664525 + 1013904223
        
        # Convert to uniform [0,1]
        uniform_val = T(rng_state % 1000000) / T(1000000)
        
        # Box-Muller transform for Gaussian noise
        if expert_idx % 2 == 1 && expert_idx < num_experts
            # Process pairs for Box-Muller
            rng_state2 = seed + (expert_idx + 1) * 1664525 + batch_idx * 1013904223
            rng_state2 = rng_state2 * 1664525 + 1013904223
            uniform_val2 = T(rng_state2 % 1000000) / T(1000000)
            
            # Box-Muller transformation
            magnitude = sqrt(-T(2) * log(uniform_val + T(1e-8)))
            angle = T(2) * T(π) * uniform_val2
            
            noise1 = magnitude * cos(angle) * noise_scale
            noise2 = magnitude * sin(angle) * noise_scale
            
            router_logits[expert_idx, batch_idx] += noise1
            if expert_idx + 1 <= num_experts
                router_logits[expert_idx + 1, batch_idx] += noise2
            end
        end
    end
    
    return nothing
end

# High-level kernel launcher functions
function launch_router_computation_kernel!(
    router_logits::CuMatrix{T},
    input::CuMatrix{T},
    router_weights::CuMatrix{T},
    router_bias::Union{Nothing, CuVector{T}} = nothing
) where T<:AbstractFloat
    
    input_dim, batch_size = size(input)
    num_experts = size(router_weights, 2)
    
    use_bias = !isnothing(router_bias)
    
    # Configure kernel launch parameters
    threads = (16, 16)
    blocks = (cld(num_experts, 16), cld(batch_size, 16))
    
    if use_bias
        # Call kernel with bias
        @cuda threads=threads blocks=blocks gpu_router_logits_kernel!(
            router_logits, input, router_weights, router_bias,
            input_dim, num_experts, batch_size, use_bias
        )
    else
        # Call kernel without bias (create dummy bias vector)
        dummy_bias = CUDA.zeros(T, num_experts)
        @cuda threads=threads blocks=blocks gpu_router_logits_kernel!(
            router_logits, input, router_weights, dummy_bias,
            input_dim, num_experts, batch_size, use_bias
        )
    end
    
    CUDA.synchronize()
    return router_logits
end

function launch_router_softmax_kernel!(
    router_probs::CuMatrix{T},
    router_logits::CuMatrix{T};
    epsilon::T = T(1e-8),
    use_shared_memory::Bool = false
) where T<:AbstractFloat
    
    num_experts, batch_size = size(router_logits)
    
    if use_shared_memory && num_experts <= 256
        # Use shared memory optimization for small expert counts
        threads = min(num_experts, 256)
        blocks = batch_size
        shared_mem_size = sizeof(T) * num_experts
        
        @cuda threads=threads blocks=blocks shmem=shared_mem_size gpu_router_softmax_shared_kernel!(
            router_probs, router_logits, num_experts, batch_size, epsilon
        )
        
    else
        # Use general softmax kernel
        max_logits = CUDA.zeros(T, batch_size)
        sum_exp = CUDA.zeros(T, batch_size)
        
        threads = 256
        blocks = cld(batch_size, 256)
        
        @cuda threads=threads blocks=blocks gpu_router_softmax_kernel!(
            router_probs, router_logits, max_logits, sum_exp,
            num_experts, batch_size, epsilon
        )
    end
    
    CUDA.synchronize()
    return router_probs
end

function launch_topk_selection_kernel!(
    expert_indices::CuMatrix{Int32},
    expert_gates::CuMatrix{T},
    router_probs::CuMatrix{T};
    algorithm::Symbol = :auto
) where T<:AbstractFloat
    
    num_experts, batch_size = size(router_probs)
    top_k = size(expert_indices, 1)
    
    # Use the simple, reliable kernel
    threads = 256
    blocks = cld(batch_size, 256)
    
    @cuda threads=threads blocks=blocks gpu_topk_selection_simple_kernel!(
        expert_indices, expert_gates, router_probs,
        num_experts, top_k, batch_size
    )
    
    CUDA.synchronize()
    return expert_indices, expert_gates
end

function launch_gate_renormalization_kernel!(
    expert_gates::CuMatrix{T};
    epsilon::T = T(1e-8)
) where T<:AbstractFloat
    
    top_k, batch_size = size(expert_gates)
    
    threads = 256
    blocks = cld(batch_size, 256)
    
    @cuda threads=threads blocks=blocks gpu_renormalize_gates_kernel!(
        expert_gates, top_k, batch_size, epsilon
    )
    
    CUDA.synchronize()
    return expert_gates
end

function launch_add_router_noise_kernel!(
    router_logits::CuMatrix{T},
    noise_scale::T;
    seed::UInt64 = rand(UInt64)
) where T<:AbstractFloat
    
    num_experts, batch_size = size(router_logits)
    
    threads = (16, 16)
    blocks = (cld(num_experts, 16), cld(batch_size, 16))
    
    @cuda threads=threads blocks=blocks gpu_add_router_noise_kernel!(
        router_logits, noise_scale, num_experts, batch_size, seed
    )
    
    CUDA.synchronize()
    return router_logits
end