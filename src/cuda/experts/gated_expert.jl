"""
GPU Gated Expert Implementation

High-performance GPU implementation of gated FFN experts using custom CUDA kernels
for optimal memory usage and computational efficiency. Supports both forward and
backward passes with automatic kernel selection based on problem size.
"""

# Main GPU Gated Expert structure
mutable struct GPUGatedExpert{T<:AbstractFloat}
    weights::GPUGatedExpertWeights{T}
    config::GPUMoEConfig{T}
    
    # Workspace management
    workspace::Dict{Symbol, Any}
    workspace_allocated::Ref{Bool}
    workspace_size_bytes::Ref{Int64}
    
    # Performance optimization settings
    use_small_batch_optimization::Bool
    use_kernel_fusion::Bool
    enable_mixed_precision::Bool
    
    # Statistics tracking
    forward_calls::Ref{Int}
    backward_calls::Ref{Int}
    total_forward_time::Ref{Float64}
    total_backward_time::Ref{Float64}
    
    function GPUGatedExpert{T}(
        weights::GPUGatedExpertWeights{T},
        config::GPUMoEConfig{T};
        use_small_batch_optimization::Bool = true,
        use_kernel_fusion::Bool = true,
        enable_mixed_precision::Bool = false
    ) where T<:AbstractFloat
        
        # Validate weight dimensions match config
        input_dim, hidden_dim = size(weights.w1)
        hidden_dim2, output_dim = size(weights.w2)
        
        if input_dim != config.input_dim || 
           hidden_dim != config.hidden_dim || 
           hidden_dim != hidden_dim2 ||
           output_dim != config.output_dim
            throw(DimensionMismatch("Weight dimensions do not match configuration"))
        end
        
        # Initialize workspace management
        workspace = Dict{Symbol, Any}()
        workspace_allocated = Ref(false)
        workspace_size_bytes = Ref(Int64(0))
        
        # Initialize performance counters
        forward_calls = Ref(0)
        backward_calls = Ref(0)
        total_forward_time = Ref(0.0)
        total_backward_time = Ref(0.0)
        
        expert = new{T}(
            weights, config,
            workspace, workspace_allocated, workspace_size_bytes,
            use_small_batch_optimization, use_kernel_fusion, enable_mixed_precision,
            forward_calls, backward_calls, total_forward_time, total_backward_time
        )
        
        return expert
    end
end

# Convenience constructor
function GPUGatedExpert(weights::GPUGatedExpertWeights{T}, config::GPUMoEConfig{T}; kwargs...) where T
    return GPUGatedExpert{T}(weights, config; kwargs...)
end

# Create GPUGatedExpert from CPU weights
function GPUGatedExpert(
    w1::AbstractMatrix{T}, w2::AbstractMatrix{T}, w3::AbstractMatrix{T},
    config::GPUMoEConfig{T};
    expert_id::Int = 0,
    b1::Union{Nothing, AbstractVector{T}} = nothing,
    b2::Union{Nothing, AbstractVector{T}} = nothing,
    b3::Union{Nothing, AbstractVector{T}} = nothing,
    kwargs...
) where T<:AbstractFloat
    
    # Convert weights to GPU with proper alignment
    gpu_w1 = to_gpu_precision(w1, config)
    gpu_w2 = to_gpu_precision(w2, config)
    gpu_w3 = to_gpu_precision(w3, config)
    
    # Convert biases if present
    gpu_b1 = isnothing(b1) ? nothing : CuVector{T}(b1)
    gpu_b2 = isnothing(b2) ? nothing : CuVector{T}(b2)
    gpu_b3 = isnothing(b3) ? nothing : CuVector{T}(b3)
    
    # Create GPU weights structure
    gpu_weights = GPUGatedExpertWeights(
        gpu_w1, gpu_w2, gpu_w3, expert_id;
        b1=gpu_b1, b2=gpu_b2, b3=gpu_b3
    )
    
    return GPUGatedExpert{T}(gpu_weights, config; kwargs...)
end

# Workspace allocation and management
function allocate_workspace!(expert::GPUGatedExpert{T}, batch_size::Int) where T
    
    if expert.workspace_allocated[] && 
       haskey(expert.workspace, :allocated_batch_size) &&
       expert.workspace[:allocated_batch_size] >= batch_size
        return expert.workspace  # Already allocated with sufficient size
    end
    
    config = expert.config
    
    # Calculate workspace requirements
    hidden_dim = config.hidden_dim
    input_dim = config.input_dim
    output_dim = config.output_dim
    
    # Allocate workspace arrays
    workspace = Dict{Symbol, Any}()
    
    # Intermediate computation buffers
    workspace[:temp_gate] = gpu_zeros(T, hidden_dim, batch_size; aligned=true)
    workspace[:temp_up] = gpu_zeros(T, hidden_dim, batch_size; aligned=true)
    workspace[:gated_values] = gpu_zeros(T, hidden_dim, batch_size; aligned=true)
    workspace[:intermediate_output] = gpu_zeros(T, output_dim, batch_size; aligned=true)
    
    # Gradient workspace for backward pass
    workspace[:grad_gate] = gpu_zeros(T, hidden_dim, batch_size; aligned=true)
    workspace[:grad_up] = gpu_zeros(T, hidden_dim, batch_size; aligned=true)
    workspace[:grad_gated] = gpu_zeros(T, hidden_dim, batch_size; aligned=true)
    workspace[:grad_intermediate] = gpu_zeros(T, hidden_dim, batch_size; aligned=true)
    
    # Memory optimization: reuse buffers where possible
    if expert.use_kernel_fusion
        # When using kernel fusion, we need fewer intermediate buffers
        workspace[:fused_temp1] = gpu_zeros(T, max(hidden_dim, output_dim), batch_size; aligned=true)
        workspace[:fused_temp2] = gpu_zeros(T, max(hidden_dim, output_dim), batch_size; aligned=true)
    end
    
    # Store allocation metadata
    workspace[:allocated_batch_size] = batch_size
    workspace[:allocation_time] = time()
    
    # Calculate total workspace memory usage
    total_bytes = Int64(0)
    for (key, array) in workspace
        if isa(array, CuArray)
            total_bytes += sizeof(array)
        end
    end
    
    expert.workspace = workspace
    expert.workspace_allocated[] = true
    expert.workspace_size_bytes[] = total_bytes
    
    @debug "Allocated GPU expert workspace: $(total_bytes ÷ (1024^2)) MB for batch size $batch_size"
    
    return workspace
end

function free_workspace!(expert::GPUGatedExpert)
    if expert.workspace_allocated[]
        # Clear workspace references (GC will handle cleanup)
        empty!(expert.workspace)
        expert.workspace_allocated[] = false
        expert.workspace_size_bytes[] = 0
        
        # Force garbage collection to free GPU memory
        GC.gc()
        CUDA.reclaim()
    end
end

# Forward pass implementation
function gpu_gated_expert_forward!(
    output::AbstractMatrix{T},    # Accept SubArray
    input::AbstractMatrix{T},     # Accept SubArray
    expert::GPUGatedExpert{T};
    training::Bool = false,
    return_intermediates::Bool = false
) where T<:AbstractFloat
    
    expert.forward_calls[] += 1
    start_time = time()
    
    try
        # Validate input dimensions
        input_dim, batch_size = size(input)
        expected_output_size = (expert.config.output_dim, batch_size)
        
        if size(output) != expected_output_size
            throw(DimensionMismatch("Output size $(size(output)) does not match expected $expected_output_size"))
        end
        
        if input_dim != expert.config.input_dim
            throw(DimensionMismatch("Input dimension $input_dim does not match config $(expert.config.input_dim)"))
        end
        
        # Allocate workspace if needed
        workspace = allocate_workspace!(expert, batch_size)
        
        # Choose kernel strategy based on batch size and configuration
        if expert.use_kernel_fusion && expert.config.enable_kernel_fusion
            # Use fused kernel for maximum efficiency
            @gpu_time "gated_expert_fused_forward" launch_gated_expert_forward_kernel!(
                output, input, expert.weights, workspace;
                use_small_batch_optimization = expert.use_small_batch_optimization && batch_size <= 32
            )
        else
            # Use multi-phase approach for better memory usage
            gpu_gated_expert_forward_phases!(output, input, expert, workspace)
        end
        
        # Store intermediate values if requested (for backward pass)
        if return_intermediates || training
            # Save intermediate results in workspace for potential backward pass
            workspace[:last_input] = copy(input)
            workspace[:last_output] = copy(output)
            workspace[:last_batch_size] = batch_size
        end
        
        # Validate output for numerical stability
        if !gpu_check_finite(output)
            @warn "Non-finite values detected in expert output"
            gpu_clamp!(output, output, T(-1e6), T(1e6))
        end
        
    catch e
        @error "Error in GPU gated expert forward pass" exception=e
        rethrow(e)
    finally
        elapsed_time = time() - start_time
        expert.total_forward_time[] += elapsed_time
    end
    
    return output
end

# Multi-phase forward pass for better memory control
function gpu_gated_expert_forward_phases!(
    output::AbstractMatrix{T},   # Accept SubArray
    input::AbstractMatrix{T},    # Accept SubArray  
    expert::GPUGatedExpert{T},
    workspace::Dict{Symbol, Any}
) where T<:AbstractFloat
    
    weights = expert.weights
    config = expert.config
    
    # Phase 1: Gate and up projections
    @gpu_time "expert_phase1_projections" begin
        # Compute w1 * input and w3 * input
        CUBLAS.gemm!('N', 'N', T(1), weights.w1, input, T(0), workspace[:temp_gate])
        CUBLAS.gemm!('N', 'N', T(1), weights.w3, input, T(0), workspace[:temp_up])
        
        # Add biases if present
        if !isnothing(weights.b1)
            # Broadcast bias addition
            gpu_broadcast_add_bias!(workspace[:temp_gate], weights.b1)
        end
        
        if !isnothing(weights.b3)
            gpu_broadcast_add_bias!(workspace[:temp_up], weights.b3)
        end
    end
    
    # Phase 2: Activation and gating
    @gpu_time "expert_phase2_activation" begin
        # Apply SiLU activation to gate values in-place
        gpu_silu!(workspace[:temp_gate], workspace[:temp_gate])
        
        # Element-wise multiplication: gate * up
        gpu_elementwise_multiply!(workspace[:gated_values], workspace[:temp_gate], workspace[:temp_up])
    end
    
    # Phase 3: Output projection
    @gpu_time "expert_phase3_output" begin
        # Compute w2 * gated_values
        CUBLAS.gemm!('N', 'N', T(1), weights.w2, workspace[:gated_values], T(0), output)
        
        # Add output bias if present
        if !isnothing(weights.b2)
            gpu_broadcast_add_bias!(output, weights.b2)
        end
    end
end

# Bias addition kernel
function gpu_broadcast_add_bias!(matrix::CuMatrix{T}, bias::CuVector{T}) where T
    if size(matrix, 1) != length(bias)
        throw(DimensionMismatch("Matrix rows must match bias length"))
    end
    
    rows, cols = size(matrix)
    total_elements = rows * cols
    
    kernel_config = GPUKernelConfig(total_elements)
    
    @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid gpu_add_bias_kernel!(
        matrix, bias, rows, cols
    )
    
    CUDA.synchronize()
    return matrix
end

function gpu_add_bias_kernel!(matrix::CuDeviceMatrix{T}, bias::CuDeviceVector{T}, 
                             rows::Int, cols::Int) where T
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= rows * cols
        row = ((idx - 1) % rows) + 1
        col = ((idx - 1) ÷ rows) + 1
        matrix[row, col] += bias[row]
    end
    
    return nothing
end

# Backward pass implementation
function gpu_gated_expert_backward!(
    grad_input::CuMatrix{T},
    grad_output::CuMatrix{T},
    expert::GPUGatedExpert{T};
    compute_input_grad::Bool = true,
    compute_weight_grad::Bool = true
) where T<:AbstractFloat
    
    expert.backward_calls[] += 1
    start_time = time()
    
    try
        # Validate that forward pass was called with training=true
        workspace = expert.workspace
        if !haskey(workspace, :last_input) || !haskey(workspace, :last_output)
            throw(ErrorException("Backward pass requires forward pass to be called with training=true or return_intermediates=true"))
        end
        
        # Get intermediate values from forward pass
        input = workspace[:last_input]
        batch_size = workspace[:last_batch_size]
        
        # Validate gradient dimensions
        if size(grad_output) != size(workspace[:last_output])
            throw(DimensionMismatch("Gradient output size does not match forward output size"))
        end
        
        if compute_input_grad && size(grad_input) != size(input)
            throw(DimensionMismatch("Gradient input size does not match forward input size"))
        end
        
        # Allocate gradient workspace if needed
        if !haskey(workspace, :grad_weights_allocated)
            allocate_gradient_workspace!(expert, workspace)
        end
        
        # Create gradient weights structure
        grad_weights = create_gradient_weights_structure(expert, workspace)
        
        # Launch backward kernel
        @gpu_time "gated_expert_backward" launch_gated_expert_backward_kernel!(
            grad_weights, grad_input, grad_output, input, expert.weights, workspace
        )
        
        # Validate gradients for numerical stability
        if compute_weight_grad
            validate_gradients!(grad_weights)
        end
        
        if compute_input_grad
            if !gpu_check_finite(grad_input)
                @warn "Non-finite values detected in input gradients"
                gpu_clamp!(grad_input, grad_input, T(-1e6), T(1e6))
            end
        end
        
    catch e
        @error "Error in GPU gated expert backward pass" exception=e
        rethrow(e)
    finally
        elapsed_time = time() - start_time
        expert.total_backward_time[] += elapsed_time
    end
    
    return grad_input
end

function allocate_gradient_workspace!(expert::GPUGatedExpert{T}, workspace::Dict{Symbol, Any}) where T
    config = expert.config
    weights = expert.weights
    
    # Allocate gradient arrays for weights
    workspace[:grad_w1] = gpu_zeros(T, size(weights.w1)...)
    workspace[:grad_w2] = gpu_zeros(T, size(weights.w2)...)
    workspace[:grad_w3] = gpu_zeros(T, size(weights.w3)...)
    
    # Allocate gradient arrays for biases if present
    if !isnothing(weights.b1)
        workspace[:grad_b1] = gpu_zeros(T, length(weights.b1))
    end
    if !isnothing(weights.b2)
        workspace[:grad_b2] = gpu_zeros(T, length(weights.b2))
    end
    if !isnothing(weights.b3)
        workspace[:grad_b3] = gpu_zeros(T, length(weights.b3))
    end
    
    workspace[:grad_weights_allocated] = true
end

function create_gradient_weights_structure(expert::GPUGatedExpert{T}, workspace::Dict{Symbol, Any}) where T
    weights = expert.weights
    
    grad_b1 = haskey(workspace, :grad_b1) ? workspace[:grad_b1] : nothing
    grad_b2 = haskey(workspace, :grad_b2) ? workspace[:grad_b2] : nothing
    grad_b3 = haskey(workspace, :grad_b3) ? workspace[:grad_b3] : nothing
    
    return GPUGatedExpertWeights(
        workspace[:grad_w1], workspace[:grad_w2], workspace[:grad_w3],
        weights.expert_id;
        b1=grad_b1, b2=grad_b2, b3=grad_b3
    )
end

function validate_gradients!(grad_weights::GPUGatedExpertWeights{T}) where T
    # Check for numerical issues in gradients
    arrays_to_check = [grad_weights.w1, grad_weights.w2, grad_weights.w3]
    
    if !isnothing(grad_weights.b1)
        push!(arrays_to_check, grad_weights.b1)
    end
    if !isnothing(grad_weights.b2)
        push!(arrays_to_check, grad_weights.b2)
    end
    if !isnothing(grad_weights.b3)
        push!(arrays_to_check, grad_weights.b3)
    end
    
    for (i, array) in enumerate(arrays_to_check)
        if !gpu_check_finite(array)
            @warn "Non-finite values detected in gradient array $i"
            gpu_clamp!(array, array, T(-1e6), T(1e6))
        end
    end
end

# Performance analysis and optimization
function get_expert_performance_stats(expert::GPUGatedExpert{T}) where T
    forward_calls = expert.forward_calls[]
    backward_calls = expert.backward_calls[]
    total_forward_time = expert.total_forward_time[]
    total_backward_time = expert.total_backward_time[]
    
    stats = Dict{String, Any}(
        "forward_calls" => forward_calls,
        "backward_calls" => backward_calls,
        "total_forward_time_ms" => total_forward_time * 1000,
        "total_backward_time_ms" => total_backward_time * 1000,
        "avg_forward_time_ms" => forward_calls > 0 ? (total_forward_time / forward_calls) * 1000 : 0.0,
        "avg_backward_time_ms" => backward_calls > 0 ? (total_backward_time / backward_calls) * 1000 : 0.0,
        "workspace_size_mb" => expert.workspace_size_bytes[] / (1024^2),
        "workspace_allocated" => expert.workspace_allocated[]
    )
    
    return stats
end

function reset_expert_performance_stats!(expert::GPUGatedExpert)
    expert.forward_calls[] = 0
    expert.backward_calls[] = 0
    expert.total_forward_time[] = 0.0
    expert.total_backward_time[] = 0.0
end

# Expert optimization utilities
function optimize_expert_for_batch_size!(expert::GPUGatedExpert{T}, target_batch_size::Int) where T
    
    # Adjust optimization settings based on batch size
    if target_batch_size <= 32
        expert.use_small_batch_optimization = true
        # Pre-allocate workspace for this batch size
        allocate_workspace!(expert, target_batch_size)
    elseif target_batch_size >= 128
        expert.use_small_batch_optimization = false
        expert.use_kernel_fusion = true
    end
    
    @info "Optimized expert for batch size $target_batch_size"
end

function benchmark_expert_performance(expert::GPUGatedExpert{T}, batch_sizes::Vector{Int}; 
                                    num_warmup::Int = 3, num_benchmark::Int = 10) where T
    
    results = Dict{Int, Dict{String, Float64}}()
    
    for batch_size in batch_sizes
        @info "Benchmarking expert with batch size $batch_size"
        
        # Create test data
        input = CUDA.randn(T, expert.config.input_dim, batch_size)
        output = gpu_zeros(T, expert.config.output_dim, batch_size)
        
        # Warmup
        for _ in 1:num_warmup
            gpu_gated_expert_forward!(output, input, expert)
            CUDA.synchronize()
        end
        
        # Benchmark forward pass
        forward_times = Float64[]
        for _ in 1:num_benchmark
            start_time = time_ns()
            gpu_gated_expert_forward!(output, input, expert)
            CUDA.synchronize()
            end_time = time_ns()
            push!(forward_times, (end_time - start_time) / 1e6)  # Convert to ms
        end
        
        results[batch_size] = Dict(
            "mean_forward_time_ms" => Statistics.mean(forward_times),
            "std_forward_time_ms" => Statistics.std(forward_times),
            "min_forward_time_ms" => minimum(forward_times),
            "max_forward_time_ms" => maximum(forward_times)
        )
    end
    
    return results
end

# Expert factory functions
function create_random_gpu_expert(config::GPUMoEConfig{T}; 
                                 expert_id::Int = 0,
                                 initialization_scale::T = T(0.02),
                                 use_bias::Bool = true) where T<:AbstractFloat
    
    # Initialize weights with proper scaling
    w1 = CUDA.randn(T, config.input_dim, config.hidden_dim) .* initialization_scale
    w2 = CUDA.randn(T, config.hidden_dim, config.output_dim) .* initialization_scale
    w3 = CUDA.randn(T, config.input_dim, config.hidden_dim) .* initialization_scale
    
    # Initialize biases
    b1 = use_bias ? CUDA.zeros(T, config.hidden_dim) : nothing
    b2 = use_bias ? CUDA.zeros(T, config.output_dim) : nothing
    b3 = use_bias ? CUDA.zeros(T, config.hidden_dim) : nothing
    
    # Create weights structure
    weights = GPUGatedExpertWeights(w1, w2, w3, expert_id; b1=b1, b2=b2, b3=b3)
    
    return GPUGatedExpert{T}(weights, config)
end

function copy_expert_to_gpu(cpu_expert::Any, config::GPUMoEConfig{T}) where T<:AbstractFloat
    # Convert CPU expert weights to GPU
    # This would integrate with the main MoE library's expert types
    
    # Extract weights from CPU expert (implementation depends on CPU expert structure)
    w1 = CuArray{T}(cpu_expert.w1)
    w2 = CuArray{T}(cpu_expert.w2) 
    w3 = CuArray{T}(cpu_expert.w3)
    
    # Convert biases if present
    b1 = hasproperty(cpu_expert, :b1) && !isnothing(cpu_expert.b1) ? CuArray{T}(cpu_expert.b1) : nothing
    b2 = hasproperty(cpu_expert, :b2) && !isnothing(cpu_expert.b2) ? CuArray{T}(cpu_expert.b2) : nothing
    b3 = hasproperty(cpu_expert, :b3) && !isnothing(cpu_expert.b3) ? CuArray{T}(cpu_expert.b3) : nothing
    
    expert_id = hasproperty(cpu_expert, :expert_id) ? cpu_expert.expert_id : 0
    
    # Create GPU weights
    gpu_weights = GPUGatedExpertWeights(w1, w2, w3, expert_id; b1=b1, b2=b2, b3=b3)
    
    return GPUGatedExpert{T}(gpu_weights, config)
end

# Add this function to gated_expert.jl for Part 2 integration
# This provides a clean interface for batch expert processing

"""
    batch_expert_forward!(outputs, inputs, experts, expert_assignments, expert_gates)

Process multiple experts in batch for MoE layer integration.
This is the main interface Part 2 will use.
"""
function batch_expert_forward!(
    outputs::Vector{CuMatrix{T}},           # One output per expert
    inputs::CuMatrix{T},                    # input_dim × batch_size
    experts::Vector{GPUGatedExpert{T}},     # Vector of experts
    expert_assignments::CuMatrix{Int32},    # top_k × batch_size
    expert_gates::CuMatrix{T}               # top_k × batch_size
) where T<:AbstractFloat
    
    batch_size = size(inputs, 2)
    top_k = size(expert_assignments, 1)
    
    # Process each expert that has assigned tokens
    for expert_id in 1:length(experts)
        expert = experts[expert_id]
        
        # Find tokens assigned to this expert
        assigned_mask = (expert_assignments .== expert_id)
        assigned_positions = findall(assigned_mask)
        
        if !isempty(assigned_positions)
            # Extract assigned tokens
            assigned_tokens = Int32[]
            assigned_weights = Float32[]
            
            for pos in assigned_positions
                k_idx, batch_idx = Tuple(pos)
                push!(assigned_tokens, batch_idx)
                push!(assigned_weights, expert_gates[k_idx, batch_idx])
            end
            
            if !isempty(assigned_tokens)
                # Get input for this expert
                expert_input = inputs[:, assigned_tokens]
                expert_output = outputs[expert_id][:, assigned_tokens]
                
                # Run expert forward pass
                gpu_gated_expert_forward!(expert_output, expert_input, expert)
                
                # Apply gating weights
                for (i, weight) in enumerate(assigned_weights)
                    expert_output[:, i] .*= weight
                end
            end
        end
    end
    
    return outputs
end