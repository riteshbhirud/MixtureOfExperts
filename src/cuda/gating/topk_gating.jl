"""
GPU TopK Gating Implementation

High-performance GPU implementation of TopK gating mechanism using custom CUDA kernels
for router computation, softmax normalization, and top-k expert selection with
automatic optimization based on problem size and hardware capabilities.
"""

# Main GPU TopK Gating structure
struct GPUTopKGating{T<:AbstractFloat}
    k::Int
    config::GPUMoEConfig{T}
    
    # Router network weights
    router_weights::CuMatrix{T}           # input_dim × num_experts
    router_bias::Union{Nothing, CuVector{T}}  # num_experts (optional)
    
    # Computational state management
    state::GPUTopKGatingState{T}
    state_allocated::Ref{Bool}
    
    # Algorithm selection parameters
    softmax_algorithm::Symbol             # :standard, :shared, :fused
    topk_algorithm::Symbol               # :bitonic, :heap, :parallel, :auto
    use_noise_injection::Bool
    noise_scale::T
    
    # Performance optimization settings
    use_mixed_precision::Bool
    enable_kernel_fusion::Bool
    prefer_shared_memory::Bool
    
    # Performance tracking
    forward_calls::Ref{Int}
    total_forward_time::Ref{Float64}
    router_compute_time::Ref{Float64}
    softmax_compute_time::Ref{Float64}
    topk_compute_time::Ref{Float64}
    
    function GPUTopKGating{T}(
        k::Int,
        config::GPUMoEConfig{T};
        router_weights::Union{Nothing, AbstractMatrix{T}} = nothing,
        router_bias::Union{Nothing, AbstractVector{T}} = nothing,
        softmax_algorithm::Symbol = :auto,
        topk_algorithm::Symbol = :auto,
        use_noise_injection::Bool = false,
        noise_scale::T = T(0.01),
        use_mixed_precision::Bool = false,
        enable_kernel_fusion::Bool = true,
        prefer_shared_memory::Bool = true
    ) where T<:AbstractFloat
        
        # Validate parameters
        if k <= 0 || k > config.num_experts
            throw(ArgumentError("k must be positive and not exceed num_experts"))
        end
        
        # Initialize or create router weights
        if isnothing(router_weights)
            # Xavier/Glorot initialization
            scale = sqrt(T(2) / (config.input_dim + config.num_experts))
            gpu_router_weights = CUDA.randn(T, config.input_dim, config.num_experts) .* scale
        else
            gpu_router_weights = CuArray{T}(router_weights)
        end
        
        # Initialize router bias
        gpu_router_bias = isnothing(router_bias) ? nothing : CuArray{T}(router_bias)
        
        # Validate router dimensions
        if size(gpu_router_weights) != (config.input_dim, config.num_experts)
            throw(DimensionMismatch("Router weights must be input_dim × num_experts"))
        end
        
        if !isnothing(gpu_router_bias) && length(gpu_router_bias) != config.num_experts
            throw(DimensionMismatch("Router bias must have num_experts elements"))
        end
        
        # Select algorithms based on problem size if auto
        actual_softmax_algorithm = select_softmax_algorithm(softmax_algorithm, config)
        actual_topk_algorithm = select_topk_algorithm(topk_algorithm, k, config)
        
        # Initialize computational state
        initial_batch_size = min(config.max_batch_size, 64)  # Start with modest allocation
        state = GPUTopKGatingState{T}(config.num_experts, k, initial_batch_size)
        state_allocated = Ref(true)
        
        # Initialize performance counters
        forward_calls = Ref(0)
        total_forward_time = Ref(0.0)
        router_compute_time = Ref(0.0)
        softmax_compute_time = Ref(0.0)
        topk_compute_time = Ref(0.0)
        
        gating = new{T}(
            k, config,
            gpu_router_weights, gpu_router_bias,
            state, state_allocated,
            actual_softmax_algorithm, actual_topk_algorithm, use_noise_injection, noise_scale,
            use_mixed_precision, enable_kernel_fusion, prefer_shared_memory,
            forward_calls, total_forward_time, router_compute_time, 
            softmax_compute_time, topk_compute_time
        )
        
        return gating
    end
end

# Convenience constructor
function GPUTopKGating(k::Int, config::GPUMoEConfig{T}; kwargs...) where T
    return GPUTopKGating{T}(k, config; kwargs...)
end

# Algorithm selection functions
function select_softmax_algorithm(algorithm::Symbol, config::GPUMoEConfig{T}) where T
    if algorithm != :auto
        return algorithm
    end
    
    # Auto-select based on problem characteristics
    if config.num_experts <= 256 && config.device_info.max_shared_memory_per_block >= 48 * 1024
        return :shared  # Use shared memory for small expert counts
    elseif config.enable_kernel_fusion && config.compute_capability >= v"7.0"
        return :fused   # Use fused kernels on newer hardware
    else
        return :standard  # Default to standard implementation
    end
end

function select_topk_algorithm(algorithm::Symbol, k::Int, config::GPUMoEConfig{T}) where T
    if algorithm != :auto
        return algorithm
    end
    
    # Auto-select based on k and num_experts
    if k <= 32 && config.num_experts <= 256
        return :bitonic  # Efficient for small k and expert counts
    elseif config.num_experts <= 1024
        return :heap     # Good middle ground
    else
        return :parallel # Best for large expert counts
    end
end

# Forward pass implementation
function gpu_topk_gating_forward!(
    expert_indices::CuMatrix{Int32},      # top_k × batch_size (output)
    expert_gates::CuMatrix{T},            # top_k × batch_size (output)
    input::CuMatrix{T},                   # input_dim × batch_size
    gating::GPUTopKGating{T};
    training::Bool = false,
    return_router_probs::Bool = false
) where T<:AbstractFloat
    
    gating.forward_calls[] += 1
    start_time = time()
    
    try
        # Validate input dimensions
        input_dim, batch_size = size(input)
        if input_dim != gating.config.input_dim
            throw(DimensionMismatch("Input dimension mismatch"))
        end
        
        # Validate output dimensions
        if size(expert_indices) != (gating.k, batch_size) || size(expert_gates) != (gating.k, batch_size)
            throw(DimensionMismatch("Output arrays have incorrect dimensions"))
        end
        
        # Resize state if needed for larger batch size
        if batch_size > gating.state.allocated_batch_size
            resize_gating_state!(gating.state, batch_size)
        end
        
        # Get views for current batch size
        router_logits = view(gating.state.router_logits, :, 1:batch_size)
        router_probs = view(gating.state.router_probs, :, 1:batch_size)
        
        # Phase 1: Compute router logits
        router_start = time()
        @gpu_time "router_computation" launch_router_computation_kernel!(
            router_logits, input, gating.router_weights, gating.router_bias
        )
        gating.router_compute_time[] += time() - router_start
        
        # Add noise during training if enabled
        if training && gating.use_noise_injection && gating.noise_scale > 0
            @gpu_time "router_noise_injection" launch_add_router_noise_kernel!(
                router_logits, gating.noise_scale
            )
        end
        
        # Phase 2: Compute softmax probabilities
        softmax_start = time()
        @gpu_time "router_softmax" launch_router_softmax_kernel!(
            router_probs, router_logits;
            epsilon = gating.config.epsilon,
            use_shared_memory = (gating.softmax_algorithm == :shared)
        )
        gating.softmax_compute_time[] += time() - softmax_start
        
        # Phase 3: Top-k expert selection
        topk_start = time()
        @gpu_time "topk_selection" launch_topk_selection_kernel!(
            expert_indices, expert_gates, router_probs;
            algorithm = gating.topk_algorithm
        )
        gating.topk_compute_time[] += time() - topk_start
        
        # Phase 4: Renormalize gates (ensure they sum to 1)
        @gpu_time "gate_renormalization" launch_gate_renormalization_kernel!(
            expert_gates; epsilon = gating.config.epsilon
        )
        
        # Validate outputs
        validate_gating_outputs!(expert_indices, expert_gates, gating)
        
        # Return router probabilities if requested
        if return_router_probs
            return expert_indices, expert_gates, router_probs
        else
            return expert_indices, expert_gates
        end
        
    catch e
        @error "Error in GPU TopK gating forward pass" exception=e
        rethrow(e)
    finally
        elapsed_time = time() - start_time
        gating.total_forward_time[] += elapsed_time
    end
end

# Validation function for gating outputs
function validate_gating_outputs!(expert_indices::CuMatrix{Int32}, expert_gates::CuMatrix{T}, 
                                 gating::GPUTopKGating{T}) where T
    
    # Check expert indices are within valid range
    min_index = CUDA.reduce(min, expert_indices)
    max_index = CUDA.reduce(max, expert_indices)
    
    if min_index < 1 || max_index > gating.config.num_experts
        @warn "Expert indices out of range: [$min_index, $max_index], expected [1, $(gating.config.num_experts)]"
    end
    
    # Check gates are finite and non-negative
    if !gpu_check_finite(expert_gates)
        @warn "Non-finite values detected in expert gates"
        gpu_clamp!(expert_gates, expert_gates, T(0), T(1))
    end
    
    # Check gate normalization (should sum to ~1 for each token)
    gate_sums = gpu_reduce_sum(expert_gates; dims=1)
    min_sum = CUDA.reduce(min, gate_sums)
    max_sum = CUDA.reduce(max, gate_sums)
    
    tolerance = T(0.01)
    if abs(min_sum - 1) > tolerance || abs(max_sum - 1) > tolerance
        @warn "Gate normalization issue: sums range from $min_sum to $max_sum"
    end
end

# Batch computation interface
function gpu_topk_gating_forward_batch!(
    expert_indices::CuMatrix{Int32},
    expert_gates::CuMatrix{T},
    inputs::Vector{CuMatrix{T}},
    gating::GPUTopKGating{T};
    kwargs...
) where T<:AbstractFloat
    
    total_batch_size = sum(size(input, 2) for input in inputs)
    
    if size(expert_indices, 2) != total_batch_size || size(expert_gates, 2) != total_batch_size
        throw(DimensionMismatch("Output arrays must accommodate total batch size $total_batch_size"))
    end
    
    # Concatenate inputs efficiently on GPU
    concatenated_input = gpu_hcat(inputs...)
    
    # Process entire batch at once
    return gpu_topk_gating_forward!(
        expert_indices, expert_gates, concatenated_input, gating; kwargs...
    )
end

# Efficient GPU horizontal concatenation
function gpu_hcat(arrays::CuMatrix{T}...) where T
    if isempty(arrays)
        throw(ArgumentError("Cannot concatenate empty array list"))
    end
    
    # Check dimension consistency
    input_dim = size(arrays[1], 1)
    for arr in arrays
        if size(arr, 1) != input_dim
            throw(DimensionMismatch("All arrays must have the same number of rows"))
        end
    end
    
    # Calculate total columns
    total_cols = sum(size(arr, 2) for arr in arrays)
    
    # Allocate result
    result = gpu_zeros(T, input_dim, total_cols)
    
    # Copy arrays
    col_offset = 0
    for arr in arrays
        cols = size(arr, 2)
        result[:, (col_offset + 1):(col_offset + cols)] .= arr
        col_offset += cols
    end
    
    return result
end

# Router weight management
function update_router_weights!(gating::GPUTopKGating{T}, new_weights::AbstractMatrix{T}) where T
    if size(new_weights) != size(gating.router_weights)
        throw(DimensionMismatch("New weights must match existing dimensions"))
    end
    
    # Update weights on GPU
    copyto!(gating.router_weights, new_weights)
    CUDA.synchronize()
end

function update_router_bias!(gating::GPUTopKGating{T}, new_bias::Union{Nothing, AbstractVector{T}}) where T
    if isnothing(new_bias)
        gating.router_bias = nothing
    else
        if length(new_bias) != gating.config.num_experts
            throw(DimensionMismatch("Bias must have num_experts elements"))
        end
        
        if isnothing(gating.router_bias)
            gating.router_bias = CuVector{T}(new_bias)
        else
            copyto!(gating.router_bias, new_bias)
        end
    end
    
    CUDA.synchronize()
end

# Performance analysis
function get_gating_performance_stats(gating::GPUTopKGating{T}) where T
    forward_calls = gating.forward_calls[]
    total_time = gating.total_forward_time[]
    
    stats = Dict{String, Any}(
        "forward_calls" => forward_calls,
        "total_forward_time_ms" => total_time * 1000,
        "avg_forward_time_ms" => forward_calls > 0 ? (total_time / forward_calls) * 1000 : 0.0,
        "router_compute_time_ms" => gating.router_compute_time[] * 1000,
        "softmax_compute_time_ms" => gating.softmax_compute_time[] * 1000,
        "topk_compute_time_ms" => gating.topk_compute_time[] * 1000,
        "avg_router_time_ms" => forward_calls > 0 ? (gating.router_compute_time[] / forward_calls) * 1000 : 0.0,
        "avg_softmax_time_ms" => forward_calls > 0 ? (gating.softmax_compute_time[] / forward_calls) * 1000 : 0.0,
        "avg_topk_time_ms" => forward_calls > 0 ? (gating.topk_compute_time[] / forward_calls) * 1000 : 0.0,
        "workspace_size_mb" => gating.state.workspace_bytes / (1024^2),
        "k" => gating.k,
        "num_experts" => gating.config.num_experts,
        "softmax_algorithm" => gating.softmax_algorithm,
        "topk_algorithm" => gating.topk_algorithm
    )
    
    return stats
end

function reset_gating_performance_stats!(gating::GPUTopKGating)
    gating.forward_calls[] = 0
    gating.total_forward_time[] = 0.0
    gating.router_compute_time[] = 0.0
    gating.softmax_compute_time[] = 0.0
    gating.topk_compute_time[] = 0.0
end

# Benchmark different algorithms
function benchmark_gating_algorithms(
    config::GPUMoEConfig{T},
    k::Int,
    batch_sizes::Vector{Int};
    num_warmup::Int = 3,
    num_benchmark::Int = 10
) where T<:AbstractFloat
    
    algorithms = [
        (:standard, :bitonic),
        (:standard, :heap),
        (:standard, :parallel),
        (:shared, :bitonic),
        (:shared, :heap),
        (:fused, :parallel)
    ]
    
    results = Dict{Tuple{Symbol, Symbol}, Dict{Int, Dict{String, Float64}}}()
    
    for (softmax_alg, topk_alg) in algorithms
        @info "Benchmarking algorithms: softmax=$softmax_alg, topk=$topk_alg"
        
        # Skip invalid combinations
        if softmax_alg == :shared && config.num_experts > 256
            continue
        end
        
        algorithm_results = Dict{Int, Dict{String, Float64}}()
        
        for batch_size in batch_sizes
            try
                # Create gating instance
                gating = GPUTopKGating{T}(k, config; 
                    softmax_algorithm=softmax_alg, 
                    topk_algorithm=topk_alg
                )
                
                # Create test data
                input = CUDA.randn(T, config.input_dim, batch_size)
                expert_indices = CUDA.zeros(Int32, k, batch_size)
                expert_gates = gpu_zeros(T, k, batch_size)
                
                # Warmup
                for _ in 1:num_warmup
                    gpu_topk_gating_forward!(expert_indices, expert_gates, input, gating)
                    CUDA.synchronize()
                end
                
                # Benchmark
                times = Float64[]
                for _ in 1:num_benchmark
                    start_time = time_ns()
                    gpu_topk_gating_forward!(expert_indices, expert_gates, input, gating)
                    CUDA.synchronize()
                    end_time = time_ns()
                    push!(times, (end_time - start_time) / 1e6)  # Convert to ms
                end
                
                algorithm_results[batch_size] = Dict(
                    "mean_time_ms" => Statistics.mean(times),
                    "std_time_ms" => Statistics.std(times),
                    "min_time_ms" => minimum(times),
                    "max_time_ms" => maximum(times)
                )
                
            catch e
                @warn "Benchmark failed for algorithms ($softmax_alg, $topk_alg) with batch size $batch_size" exception=e
            end
        end
        
        results[(softmax_alg, topk_alg)] = algorithm_results
    end
    
    return results
end

# Adaptive algorithm selection based on runtime performance
mutable struct AdaptiveGating{T<:AbstractFloat}
    base_gating::GPUTopKGating{T}
    algorithm_performance::Dict{Tuple{Symbol, Symbol}, Float64}
    adaptation_window::Int
    calls_since_adaptation::Ref{Int}
    
    function AdaptiveGating{T}(base_gating::GPUTopKGating{T}; adaptation_window::Int = 100) where T
        return new{T}(
            base_gating,
            Dict{Tuple{Symbol, Symbol}, Float64}(),
            adaptation_window,
            Ref(0)
        )
    end
end

function adaptive_gating_forward!(
    expert_indices::CuMatrix{Int32},
    expert_gates::CuMatrix{T},
    input::CuMatrix{T},
    adaptive_gating::AdaptiveGating{T};
    kwargs...
) where T<:AbstractFloat
    
    adaptive_gating.calls_since_adaptation[] += 1
    
    # Use current algorithm
    result = gpu_topk_gating_forward!(
        expert_indices, expert_gates, input, adaptive_gating.base_gating; kwargs...
    )
    
    # Periodically evaluate different algorithms
    if adaptive_gating.calls_since_adaptation[] >= adaptive_gating.adaptation_window
        @debug "Evaluating algorithm performance for adaptation"
        
        batch_size = size(input, 2)
        current_performance = benchmark_current_setup(adaptive_gating.base_gating, input)
        
        current_algorithms = (
            adaptive_gating.base_gating.softmax_algorithm,
            adaptive_gating.base_gating.topk_algorithm
        )
        
        adaptive_gating.algorithm_performance[current_algorithms] = current_performance
        
        # Try alternative algorithms and compare
        best_algorithms = find_best_algorithms(adaptive_gating, input)
        
        if best_algorithms != current_algorithms
            @info "Adapting gating algorithms from $current_algorithms to $best_algorithms"
            adaptive_gating.base_gating.softmax_algorithm = best_algorithms[1]
            adaptive_gating.base_gating.topk_algorithm = best_algorithms[2]
        end
        
        adaptive_gating.calls_since_adaptation[] = 0
    end
    
    return result
end

function benchmark_current_setup(gating::GPUTopKGating{T}, input::CuMatrix{T}) where T
    batch_size = size(input, 2)
    k = gating.k
    
    # Create output arrays
    expert_indices = CUDA.zeros(Int32, k, batch_size)
    expert_gates = gpu_zeros(T, k, batch_size)
    
    # Benchmark current setup
    num_runs = 5
    times = Float64[]
    
    for _ in 1:num_runs
        start_time = time_ns()
        gpu_topk_gating_forward!(expert_indices, expert_gates, input, gating)
        CUDA.synchronize()
        end_time = time_ns()
        push!(times, (end_time - start_time) / 1e6)
    end
    
    return Statistics.mean(times)
end

function find_best_algorithms(adaptive_gating::AdaptiveGating{T}, input::CuMatrix{T}) where T
    # This would implement a more sophisticated algorithm selection strategy
    # For now, return current algorithms
    return (
        adaptive_gating.base_gating.softmax_algorithm,
        adaptive_gating.base_gating.topk_algorithm
    )
end

# Factory functions
function create_random_gpu_gating(config::GPUMoEConfig{T}, k::Int; 
                                 initialization_scale::T = T(0.02)) where T<:AbstractFloat
    
    # Random initialization with proper scaling
    router_weights = CUDA.randn(T, config.input_dim, config.num_experts) .* initialization_scale
    
    return GPUTopKGating{T}(k, config; router_weights=Array(router_weights))
end

function copy_gating_to_gpu(cpu_gating::Any, config::GPUMoEConfig{T}) where T<:AbstractFloat
    # Extract parameters from CPU gating (implementation depends on CPU gating structure)
    k = cpu_gating.k
    
    # Convert weights to GPU
    router_weights = hasproperty(cpu_gating, :router_weights) ? Array(cpu_gating.router_weights) : nothing
    router_bias = hasproperty(cpu_gating, :router_bias) ? Array(cpu_gating.router_bias) : nothing
    
    return GPUTopKGating{T}(k, config; router_weights=router_weights, router_bias=router_bias)
end