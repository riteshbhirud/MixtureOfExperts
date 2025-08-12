"""
CUDA Kernels for Gated Expert Computation

High-performance CUDA kernels for gated FFN computation optimized for
different batch sizes and expert dimensions. Implements the gated expert
forward pass: output = w2(silu(w1(x)) * w3(x))
"""

# Forward pass kernel for gated expert with fused operations
function gpu_gated_expert_forward_kernel!(
    output::CuDeviceMatrix{T},           # hidden_dim × batch_size
    input::CuDeviceMatrix{T},            # input_dim × batch_size  
    w1::CuDeviceMatrix{T},               # input_dim × hidden_dim
    w2::CuDeviceMatrix{T},               # hidden_dim × output_dim
    w3::CuDeviceMatrix{T},               # input_dim × hidden_dim
    b1::CuDeviceVector{T},               # hidden_dim (optional bias)
    b3::CuDeviceVector{T},               # hidden_dim (optional bias)
    temp_gate::CuDeviceMatrix{T},        # hidden_dim × batch_size (workspace)
    temp_up::CuDeviceMatrix{T},          # hidden_dim × batch_size (workspace)
    input_dim::Int,
    hidden_dim::Int,
    output_dim::Int,
    batch_size::Int,
    use_bias::Bool
) where T<:AbstractFloat
    
    # Thread indices
    row_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    # Check bounds
    if row_idx <= hidden_dim && col_idx <= batch_size
        
        # Compute gate projection: w1 * input + b1
        gate_val = T(0)
        for i in 1:input_dim
            gate_val += w1[i, row_idx] * input[i, col_idx]
        end
        
        if use_bias
            gate_val += b1[row_idx]
        end
        
        # Compute up projection: w3 * input + b3  
        up_val = T(0)
        for i in 1:input_dim
            up_val += w3[i, row_idx] * input[i, col_idx]
        end
        
        if use_bias
            up_val += b3[row_idx]
        end
        
        # Apply SiLU activation to gate: x * sigmoid(x)
        sigmoid_gate = T(1) / (T(1) + exp(-gate_val))
        activated_gate = gate_val * sigmoid_gate
        
        # Element-wise multiplication: gate * up
        gated_value = activated_gate * up_val
        
        # Store intermediate result
        temp_gate[row_idx, col_idx] = gated_value
    end
    
    # Synchronize before second phase
    sync_threads()
    
    # Second phase: output projection w2 * gated_values
    if row_idx <= output_dim && col_idx <= batch_size
        output_val = T(0)
        for i in 1:hidden_dim
            output_val += w2[i, row_idx] * temp_gate[i, col_idx]
        end
        
        output[row_idx, col_idx] = output_val
    end
    
    return nothing
end

# Optimized kernel for small batch sizes using shared memory
function gpu_gated_expert_forward_small_batch_kernel!(
    output::CuDeviceMatrix{T},
    input::CuDeviceMatrix{T},
    w1::CuDeviceMatrix{T},
    w2::CuDeviceMatrix{T},
    w3::CuDeviceMatrix{T},
    b1::CuDeviceVector{T},
    b3::CuDeviceVector{T},
    input_dim::Int,
    hidden_dim::Int,
    output_dim::Int,
    batch_size::Int,
    use_bias::Bool
) where T<:AbstractFloat
    
    # Shared memory for input vectors (assuming small batch size)
    shared_input = CuDynamicSharedArray(T, (input_dim, batch_size))
    shared_intermediate = CuDynamicSharedArray(T, (hidden_dim, batch_size), sizeof(T) * input_dim * batch_size)
    
    # Thread indices
    tid_x = threadIdx().x
    tid_y = threadIdx().y
    block_size_x = blockDim().x
    block_size_y = blockDim().y
    
    # Load input into shared memory
    if tid_x <= input_dim && tid_y <= batch_size
        shared_input[tid_x, tid_y] = input[tid_x, tid_y]
    end
    
    sync_threads()
    
    # Compute gated transformation
    row_idx = (blockIdx().x - 1) * block_size_x + tid_x
    col_idx = tid_y
    
    if row_idx <= hidden_dim && col_idx <= batch_size
        # Gate projection
        gate_val = T(0)
        for i in 1:input_dim
            gate_val += w1[i, row_idx] * shared_input[i, col_idx]
        end
        if use_bias
            gate_val += b1[row_idx]
        end
        
        # Up projection
        up_val = T(0)
        for i in 1:input_dim
            up_val += w3[i, row_idx] * shared_input[i, col_idx]
        end
        if use_bias
            up_val += b3[row_idx]
        end
        
        # SiLU activation and gating
        sigmoid_gate = T(1) / (T(1) + exp(-gate_val))
        activated_gate = gate_val * sigmoid_gate
        gated_value = activated_gate * up_val
        
        shared_intermediate[row_idx, col_idx] = gated_value
    end
    
    sync_threads()
    
    # Output projection
    out_row_idx = (blockIdx().y - 1) * block_size_x + tid_x
    if out_row_idx <= output_dim && col_idx <= batch_size
        output_val = T(0)
        for i in 1:hidden_dim
            output_val += w2[i, out_row_idx] * shared_intermediate[i, col_idx]
        end
        output[out_row_idx, col_idx] = output_val
    end
    
    return nothing
end

# Kernel optimized for large batch sizes with memory coalescing
function gpu_gated_expert_forward_large_batch_kernel!(
    output::CuDeviceMatrix{T},
    input::CuDeviceMatrix{T},
    w1::CuDeviceMatrix{T},
    w2::CuDeviceMatrix{T}, 
    w3::CuDeviceMatrix{T},
    b1::CuDeviceVector{T},
    b3::CuDeviceVector{T},
    temp_gate::CuDeviceMatrix{T},
    temp_up::CuDeviceMatrix{T},
    input_dim::Int,
    hidden_dim::Int,
    output_dim::Int,
    batch_size::Int,
    use_bias::Bool
) where T<:AbstractFloat
    
    # Use larger blocks for better occupancy with large batches
    batch_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    hidden_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    # Phase 1: Compute gate and up projections
    if batch_idx <= batch_size && hidden_idx <= hidden_dim
        
        # Gate projection with loop unrolling for better performance
        gate_val = T(0)
        up_val = T(0)
        
        # Manual loop unrolling for common input dimensions
        i = 1
        while i + 3 <= input_dim
            # Unroll 4 iterations
            gate_val += w1[i, hidden_idx] * input[i, batch_idx] +
                       w1[i+1, hidden_idx] * input[i+1, batch_idx] +
                       w1[i+2, hidden_idx] * input[i+2, batch_idx] +
                       w1[i+3, hidden_idx] * input[i+3, batch_idx]
                       
            up_val += w3[i, hidden_idx] * input[i, batch_idx] +
                     w3[i+1, hidden_idx] * input[i+1, batch_idx] +
                     w3[i+2, hidden_idx] * input[i+2, batch_idx] +
                     w3[i+3, hidden_idx] * input[i+3, batch_idx]
                     
            i += 4
        end
        
        # Handle remaining elements
        while i <= input_dim
            gate_val += w1[i, hidden_idx] * input[i, batch_idx]
            up_val += w3[i, hidden_idx] * input[i, batch_idx]
            i += 1
        end
        
        # Add bias terms
        if use_bias
            gate_val += b1[hidden_idx]
            up_val += b3[hidden_idx]
        end
        
        # SiLU activation: x * sigmoid(x)
        sigmoid_gate = T(1) / (T(1) + exp(-gate_val))
        activated_gate = gate_val * sigmoid_gate
        
        # Store intermediate results
        temp_gate[hidden_idx, batch_idx] = activated_gate
        temp_up[hidden_idx, batch_idx] = up_val
    end
    
    return nothing
end

# Separate kernel for element-wise gating operation
function gpu_elementwise_gating_kernel!(
    output::CuDeviceMatrix{T},
    gate_values::CuDeviceMatrix{T},
    up_values::CuDeviceMatrix{T},
    total_elements::Int
) where T<:AbstractFloat
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= total_elements
        output[idx] = gate_values[idx] * up_values[idx]
    end
    
    return nothing
end

# Final output projection kernel  
function gpu_output_projection_kernel!(
    output::CuDeviceMatrix{T},
    gated_values::CuDeviceMatrix{T},
    w2::CuDeviceMatrix{T},
    b2::CuDeviceVector{T},
    hidden_dim::Int,
    output_dim::Int,
    batch_size::Int,
    use_bias::Bool
) where T<:AbstractFloat
    
    batch_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    output_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if batch_idx <= batch_size && output_idx <= output_dim
        
        output_val = T(0)
        
        # Vectorized accumulation with loop unrolling
        i = 1
        while i + 3 <= hidden_dim
            output_val += w2[i, output_idx] * gated_values[i, batch_idx] +
                         w2[i+1, output_idx] * gated_values[i+1, batch_idx] +
                         w2[i+2, output_idx] * gated_values[i+2, batch_idx] +
                         w2[i+3, output_idx] * gated_values[i+3, batch_idx]
            i += 4
        end
        
        # Handle remaining elements
        while i <= hidden_dim
            output_val += w2[i, output_idx] * gated_values[i, batch_idx]
            i += 1
        end
        
        # Add bias if present
        if use_bias
            output_val += b2[output_idx]
        end
        
        output[output_idx, batch_idx] = output_val
    end
    
    return nothing
end

# Backward pass kernels for gradient computation
function gpu_gated_expert_backward_output_kernel!(
    grad_w2::CuDeviceMatrix{T},
    grad_b2::CuDeviceVector{T},
    grad_gated::CuDeviceMatrix{T},
    grad_output::CuDeviceMatrix{T},
    gated_values::CuDeviceMatrix{T},
    w2::CuDeviceMatrix{T},
    hidden_dim::Int,
    output_dim::Int,
    batch_size::Int,
    use_bias::Bool
) where T<:AbstractFloat
    
    # Gradient computation for output layer
    row_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    # Gradient w.r.t. w2
    if row_idx <= hidden_dim && col_idx <= output_dim
        grad_val = T(0)
        for b in 1:batch_size
            grad_val += gated_values[row_idx, b] * grad_output[col_idx, b]
        end
        grad_w2[row_idx, col_idx] = grad_val
    end
    
    # Gradient w.r.t. b2 (if using bias)
    if use_bias && row_idx <= output_dim && col_idx == 1
        grad_val = T(0)
        for b in 1:batch_size
            grad_val += grad_output[row_idx, b]
        end
        grad_b2[row_idx] = grad_val
    end
    
    # Gradient w.r.t. gated values
    if row_idx <= hidden_dim && col_idx <= batch_size
        grad_val = T(0)
        for o in 1:output_dim
            grad_val += w2[row_idx, o] * grad_output[o, col_idx]
        end
        grad_gated[row_idx, col_idx] = grad_val
    end
    
    return nothing
end

function gpu_gated_expert_backward_gating_kernel!(
    grad_gate::CuDeviceMatrix{T},
    grad_up::CuDeviceMatrix{T},
    grad_gated::CuDeviceMatrix{T},
    gate_values::CuDeviceMatrix{T},
    up_values::CuDeviceMatrix{T},
    hidden_dim::Int,
    batch_size::Int
) where T<:AbstractFloat
    
    row_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if row_idx <= hidden_dim && col_idx <= batch_size
        grad_gated_val = grad_gated[row_idx, col_idx]
        gate_val = gate_values[row_idx, col_idx]
        up_val = up_values[row_idx, col_idx]
        
        # Gradient w.r.t. up values
        grad_up[row_idx, col_idx] = grad_gated_val * gate_val
        
        # Gradient w.r.t. gate values (before activation)
        # d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        sigmoid_val = T(1) / (T(1) + exp(-gate_val))
        silu_grad = sigmoid_val + gate_val * sigmoid_val * (T(1) - sigmoid_val)
        grad_gate[row_idx, col_idx] = grad_gated_val * up_val * silu_grad
    end
    
    return nothing
end

function gpu_gated_expert_backward_input_kernel!(
    grad_w1::CuDeviceMatrix{T},
    grad_w3::CuDeviceMatrix{T},
    grad_b1::CuDeviceVector{T},
    grad_b3::CuDeviceVector{T},
    grad_input::CuDeviceMatrix{T},
    grad_gate::CuDeviceMatrix{T},
    grad_up::CuDeviceMatrix{T},
    input::CuDeviceMatrix{T},
    w1::CuDeviceMatrix{T},
    w3::CuDeviceMatrix{T},
    input_dim::Int,
    hidden_dim::Int,
    batch_size::Int,
    use_bias::Bool
) where T<:AbstractFloat
    
    row_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    # Gradient w.r.t. w1
    if row_idx <= input_dim && col_idx <= hidden_dim
        grad_val = T(0)
        for b in 1:batch_size
            grad_val += input[row_idx, b] * grad_gate[col_idx, b]
        end
        grad_w1[row_idx, col_idx] = grad_val
    end
    
    # Gradient w.r.t. w3
    if row_idx <= input_dim && col_idx <= hidden_dim
        grad_val = T(0)
        for b in 1:batch_size
            grad_val += input[row_idx, b] * grad_up[col_idx, b]
        end
        grad_w3[row_idx, col_idx] = grad_val
    end
    
    # Gradient w.r.t. bias terms
    if use_bias
        if row_idx <= hidden_dim && col_idx == 1
            # grad_b1
            grad_val = T(0)
            for b in 1:batch_size
                grad_val += grad_gate[row_idx, b]
            end
            grad_b1[row_idx] = grad_val
            
            # grad_b3
            grad_val = T(0)
            for b in 1:batch_size
                grad_val += grad_up[row_idx, b]
            end
            grad_b3[row_idx] = grad_val
        end
    end
    
    # Gradient w.r.t. input
    if row_idx <= input_dim && col_idx <= batch_size
        grad_val = T(0)
        for h in 1:hidden_dim
            grad_val += w1[row_idx, h] * grad_gate[h, col_idx] +
                       w3[row_idx, h] * grad_up[h, col_idx]
        end
        grad_input[row_idx, col_idx] = grad_val
    end
    
    return nothing
end

function launch_gated_expert_forward_kernel!(
    output::AbstractMatrix{T},      
    input::AbstractMatrix{T},       
    weights::GPUGatedExpertWeights{T},
    workspace::Dict{Symbol, Any};   
    use_small_batch_optimization::Bool = false,
    shared_memory_bytes::Int = 0
) where T<:AbstractFloat
    
    # Convert SubArrays to concrete CuArrays for CUDA kernels
    concrete_input = if isa(input, SubArray)
        CuArray{T}(input)  
    else
        input
    end
    
    concrete_output = if isa(output, SubArray)
        CUDA.zeros(T, size(output)...)  
    else
        output
    end
    
    input_dim, batch_size = size(concrete_input)
    hidden_dim = size(weights.w1, 2)
    output_dim = size(weights.w2, 2)
    
    use_bias = !isnothing(weights.b1) && !isnothing(weights.b3)
    
    # Calculate required shared memory for small batch optimization
    required_shared_memory = sizeof(T) * (input_dim * batch_size + hidden_dim * batch_size)
    
    # Use hardcoded limit for RTX 4060 (99KB opt-in limit from your debug output)
    max_shared_memory = 101376  # 99KB in bytes
    safe_shared_memory_limit = (max_shared_memory * 7) ÷ 10  # 70KB safe limit
    
    # BE VERY CONSERVATIVE: Only use shared memory for tiny inputs
    can_use_shared_memory = (use_small_batch_optimization && 
                            batch_size <= 8 &&  
                            input_dim <= 512 &&  
                            hidden_dim <= 1024 && 
                            required_shared_memory <= safe_shared_memory_limit)
    
    if can_use_shared_memory
        println("Using shared memory optimization: $(required_shared_memory ÷ 1024)KB / $(safe_shared_memory_limit ÷ 1024)KB")
    end
    
    # Get workspace arrays with safe type conversion
    temp_gate = if haskey(workspace, :temp_gate)
        workspace_array = workspace[:temp_gate]
        if isa(workspace_array, SubArray)
            CuArray{T}(workspace_array)
        else
            workspace_array
        end
    else
        CUDA.zeros(T, hidden_dim, batch_size)
    end

    temp_up = if haskey(workspace, :temp_up)
        workspace_array = workspace[:temp_up]
        if isa(workspace_array, SubArray)
            CuArray{T}(workspace_array)
        else
            workspace_array
        end
    else
        CUDA.zeros(T, hidden_dim, batch_size)
    end
    
    if can_use_shared_memory
        # Use shared memory optimization for very small inputs only
        threads = (8, min(batch_size, 4))  
        blocks = (cld(hidden_dim, 8), cld(output_dim, 8))
        
        @cuda threads=threads blocks=blocks shmem=required_shared_memory gpu_gated_expert_forward_small_batch_kernel!(
            concrete_output, concrete_input, weights.w1, weights.w2, weights.w3,
            weights.b1, weights.b3,
            input_dim, hidden_dim, output_dim, batch_size, use_bias
        )
        
    elseif batch_size >= 32  
        # Use multi-phase approach for larger batches
        println("Using multi-phase kernel for batch_size=$batch_size")
        
        # Phase 1: Gate and up projections
        threads = (16, 16)
        blocks = (cld(batch_size, 16), cld(hidden_dim, 16))
        
        @cuda threads=threads blocks=blocks gpu_gated_expert_forward_large_batch_kernel!(
            concrete_output, concrete_input, weights.w1, weights.w2, weights.w3,
            weights.b1, weights.b3, temp_gate, temp_up,
            input_dim, hidden_dim, output_dim, batch_size, use_bias
        )
        
        # Phase 2: Element-wise gating
        total_elements = hidden_dim * batch_size
        threads = 256
        blocks = cld(total_elements, 256)
        
        @cuda threads=threads blocks=blocks gpu_elementwise_gating_kernel!(
            temp_gate, temp_gate, temp_up, total_elements
        )
        
        # Phase 3: Output projection
        threads = (16, 16)
        blocks = (cld(batch_size, 16), cld(output_dim, 16))
        
        @cuda threads=threads blocks=blocks gpu_output_projection_kernel!(
            concrete_output, temp_gate, weights.w2, weights.b2,
            hidden_dim, output_dim, batch_size, !isnothing(weights.b2)
        )
        
    else
        # Use simple single-phase kernel (no shared memory) - SAFEST OPTION
        println("Using simple kernel for batch_size=$batch_size")
        
        threads = (16, 16)
        blocks = (cld(hidden_dim, 16), cld(batch_size, 16))
        
        @cuda threads=threads blocks=blocks gpu_gated_expert_forward_kernel!(
            concrete_output, concrete_input, weights.w1, weights.w2, weights.w3,
            weights.b1, weights.b3, temp_gate, temp_up,
            input_dim, hidden_dim, output_dim, batch_size, use_bias
        )
    end
    
    # Copy result back to original output if it was a SubArray
    if isa(output, SubArray)
        copyto!(output, concrete_output)
    end
    
    CUDA.synchronize()
    return output
end
# Kernel launcher for backward pass
# Kernel launcher for backward pass
function launch_gated_expert_backward_kernel!(
    grad_weights::GPUGatedExpertWeights{T},
    grad_input::AbstractMatrix{T},     # Accept SubArray
    grad_output::AbstractMatrix{T},    # Accept SubArray
    input::AbstractMatrix{T},          # Accept SubArray
    weights::GPUGatedExpertWeights{T},
    forward_workspace::Dict{Symbol, Any}  # Match workspace type
) where T<:AbstractFloat
    
    # Convert SubArrays to concrete CuArrays for CUDA kernels
    concrete_input = if isa(input, SubArray)
        CuArray{T}(input)
    else
        input
    end
    
    concrete_grad_output = if isa(grad_output, SubArray)
        CuArray{T}(grad_output)
    else
        grad_output
    end
    
    concrete_grad_input = if isa(grad_input, SubArray)
        CUDA.zeros(T, size(grad_input)...)  # Create temporary for output
    else
        grad_input
    end
    
    input_dim, batch_size = size(concrete_input)
    hidden_dim = size(weights.w1, 2)
    output_dim = size(weights.w2, 2)
    
    use_bias = !isnothing(weights.b1) && !isnothing(weights.b3)
    
    # Validate that required workspace arrays exist
    if !haskey(forward_workspace, :temp_gate) || !haskey(forward_workspace, :temp_up) || !haskey(forward_workspace, :gated_values)
        error("Required forward pass intermediate values not found in workspace. Make sure forward pass was called with training=true or return_intermediates=true")
    end
    
    # Get intermediate values from forward pass (handle SubArrays)
    gate_values = forward_workspace[:temp_gate]
    up_values = forward_workspace[:temp_up] 
    gated_values = forward_workspace[:gated_values]
    
    # Convert workspace arrays to concrete if needed
    gate_values = isa(gate_values, SubArray) ? CuArray{T}(gate_values) : gate_values
    up_values = isa(up_values, SubArray) ? CuArray{T}(up_values) : up_values
    gated_values = isa(gated_values, SubArray) ? CuArray{T}(gated_values) : gated_values
    
    # Allocate gradient workspace
    grad_gated = CUDA.zeros(T, hidden_dim, batch_size)
    grad_gate = CUDA.zeros(T, hidden_dim, batch_size)
    grad_up = CUDA.zeros(T, hidden_dim, batch_size)
    
    # Phase 1: Backward through output layer
    threads = (16, 16)
    blocks = (cld(hidden_dim, 16), cld(output_dim, 16))
    
    @cuda threads=threads blocks=blocks gpu_gated_expert_backward_output_kernel!(
        grad_weights.w2, grad_weights.b2, grad_gated, concrete_grad_output,
        gated_values, weights.w2, hidden_dim, output_dim, batch_size, use_bias
    )
    
    # Phase 2: Backward through gating operation
    blocks = (cld(hidden_dim, 16), cld(batch_size, 16))
    
    @cuda threads=threads blocks=blocks gpu_gated_expert_backward_gating_kernel!(
        grad_gate, grad_up, grad_gated, gate_values, up_values,
        hidden_dim, batch_size
    )
    
    # Phase 3: Backward through input projections
    blocks = (cld(input_dim, 16), cld(hidden_dim, 16))
    
    @cuda threads=threads blocks=blocks gpu_gated_expert_backward_input_kernel!(
        grad_weights.w1, grad_weights.w3, grad_weights.b1, grad_weights.b3,
        concrete_grad_input, grad_gate, grad_up, concrete_input, weights.w1, weights.w3,
        input_dim, hidden_dim, batch_size, use_bias
    )
    
    # Copy result back to original grad_input if it was a SubArray
    if isa(grad_input, SubArray)
        copyto!(grad_input, concrete_grad_input)
    end
    
    CUDA.synchronize()
    return grad_weights, grad_input
end