"""
GPU MoE Layer - Main Orchestrating Component

Complete GPU-accelerated Mixture of Experts layer that orchestrates gating, expert computation,
and load balancing with optimized parallel processing, memory management, and batch efficiency.
Provides functionally equivalent API to CPU implementation with GPU acceleration.
"""

# Main GPU MoE Layer configuration
struct GPUMoELayerConfig{T<:AbstractFloat}
    # Core MoE parameters
    input_dim::Int
    hidden_dim::Int
    output_dim::Int
    num_experts::Int
    top_k::Int
    
    # Expert configuration
    expert_type::Symbol                    # Only :gated supported for now
    expert_dropout::Float32
    expert_bias::Bool
    
    # Gating configuration  
    gate_type::Symbol                      # Only :topk supported for now
    router_bias::Bool
    noise_scale::Float32
    
    # Load balancing
    balance_loss_type::Symbol              # Only :switch supported for now
    balance_alpha::T
    z_loss_weight::T
    
    # GPU-specific optimizations
    use_mixed_precision::Bool
    enable_kernel_fusion::Bool
    memory_efficient::Bool
    
    # Batch processing
    batch_config::GPUBatchConfig{T}
    
    function GPUMoELayerConfig{T}(;
        input_dim::Int,
        hidden_dim::Int,
        output_dim::Int,
        num_experts::Int = 8,
        top_k::Int = 2,
        expert_type::Symbol = :gated,
        expert_dropout::Float32 = 0.0f0,
        expert_bias::Bool = false,
        gate_type::Symbol = :topk,
        router_bias::Bool = false,
        noise_scale::Float32 = 0.0f0,
        balance_loss_type::Symbol = :switch,
        balance_alpha::T = T(0.01),
        z_loss_weight::T = T(0.001),
        use_mixed_precision::Bool = false,
        enable_kernel_fusion::Bool = true,
        memory_efficient::Bool = true,
        batch_config::Union{Nothing, GPUBatchConfig{T}} = nothing
    ) where T<:AbstractFloat
        
        # Validate core parameters
        if input_dim <= 0 || hidden_dim <= 0 || output_dim <= 0
            throw(ArgumentError("All dimensions must be positive"))
        end
        
        if num_experts <= 0 || top_k <= 0 || top_k > num_experts
            throw(ArgumentError("Invalid expert configuration"))
        end
        
        # Validate supported configurations (scope limitation)
        if expert_type != :gated
            throw(ArgumentError("Only :gated expert type is supported in current implementation"))
        end
        
        if gate_type != :topk
            throw(ArgumentError("Only :topk gate type is supported in current implementation"))
        end
        
        if balance_loss_type != :switch
            throw(ArgumentError("Only :switch balance loss type is supported in current implementation"))
        end
        
        if top_k != 2
            throw(ArgumentError("Only top_k=2 is supported in current implementation"))
        end
        
        # Create default batch config if not provided
        if isnothing(batch_config)
            batch_config = GPUBatchConfig{T}()
        end
        
        return new{T}(
            input_dim, hidden_dim, output_dim, num_experts, top_k,
            expert_type, expert_dropout, expert_bias,
            gate_type, router_bias, noise_scale,
            balance_loss_type, balance_alpha, z_loss_weight,
            use_mixed_precision, enable_kernel_fusion, memory_efficient,
            batch_config
        )
    end
end

# Convenience constructor
GPUMoELayerConfig(args...; kwargs...) = GPUMoELayerConfig{Float32}(args...; kwargs...)

# Main GPU MoE Layer
struct GPUMoELayer{T<:AbstractFloat}
    # Configuration
    config::GPUMoELayerConfig{T}
    gpu_config::GPUMoEConfig{T}
    
    # Core components
    experts::Vector{GPUGatedExpert{T}}
    gating::GPUTopKGating{T}
    loss_function::GPUSwitchTransformerLoss{T}
    
    # Workspace management
    current_workspace::Ref{Union{Nothing, GPUBatchWorkspace{T}}}
    workspace_allocated::Ref{Bool}
    
    # CUDA streams for parallel processing
    streams::Vector{CUDA.CuStream}
    main_stream::CUDA.CuStream
    
    # Performance tracking
    forward_calls::Ref{Int}
    total_forward_time::Ref{Float64}
    gating_time::Ref{Float64}
    expert_time::Ref{Float64}
    combination_time::Ref{Float64}
    loss_time::Ref{Float64}
    
    # Training state
    is_training::Ref{Bool}
    last_loss_value::Ref{T}
    
    function GPUMoELayer{T}(config::GPUMoELayerConfig{T}) where T<:AbstractFloat
        
        # Create GPU MoE configuration
        gpu_config = GPUMoEConfig{T}(
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            config.num_experts,
            config.top_k;
            use_mixed_precision = config.use_mixed_precision,
            enable_kernel_fusion = config.enable_kernel_fusion,
            max_batch_size = config.batch_config.max_batch_size
        )
        
        # Create experts
        experts = GPUGatedExpert{T}[]
        for expert_id in 1:config.num_experts
            expert = create_random_gpu_expert(
                gpu_config;
                expert_id = expert_id,
                use_bias = config.expert_bias
            )
            push!(experts, expert)
        end
        
        # Create gating mechanism (TopKGating only)
        gating = create_random_gpu_gating(gpu_config, config.top_k)
        
        # Create loss function (SwitchTransformerLoss only)
        loss_function = create_gpu_switch_loss(gpu_config, config.balance_alpha)
        
        # Initialize workspace management
        current_workspace = Ref{Union{Nothing, GPUBatchWorkspace{T}}}(nothing)
        workspace_allocated = Ref(false)
        
        # Create CUDA streams
        streams = create_batch_streams(config.batch_config)
        main_stream = streams[1]  # Use first stream as main stream
        
        # Initialize performance tracking
        forward_calls = Ref(0)
        total_forward_time = Ref(0.0)
        gating_time = Ref(0.0)
        expert_time = Ref(0.0)
        combination_time = Ref(0.0)
        loss_time = Ref(0.0)
        
        # Initialize training state
        is_training = Ref(false)
        last_loss_value = Ref(T(0))
        
        layer = new{T}(
            config, gpu_config,
            experts, gating, loss_function,
            current_workspace, workspace_allocated,
            streams, main_stream,
            forward_calls, total_forward_time, gating_time, expert_time, combination_time, loss_time,
            is_training, last_loss_value
        )
        
        @info "Created GPU MoE Layer with $(config.num_experts) experts, top_k=$(config.top_k)"
        
        return layer
    end
end

# Convenience constructor
GPUMoELayer(config::GPUMoELayerConfig{T}) where T = GPUMoELayer{T}(config)

"""
    gpu_moe_forward!(output, input, moe_layer; training=false, return_loss=false)

Main forward pass through GPU MoE layer with full orchestration.
"""
function gpu_moe_forward!(
    output::CuMatrix{T},                   # output_dim × batch_size
    input::CuMatrix{T},                    # input_dim × batch_size
    moe_layer::GPUMoELayer{T};
    training::Bool = false,
    return_loss::Bool = false,
    return_stats::Bool = false
) where T<:AbstractFloat
    
    moe_layer.forward_calls[] += 1
    moe_layer.is_training[] = training
    start_time = time()
    
    try
        # Validate input dimensions
        input_dim, batch_size = size(input)
        output_dim_expected, batch_size_output = size(output)
        
        if input_dim != moe_layer.config.input_dim
            throw(DimensionMismatch("Input dimension mismatch: got $input_dim, expected $(moe_layer.config.input_dim)"))
        end
        
        if output_dim_expected != moe_layer.config.output_dim
            throw(DimensionMismatch("Output dimension mismatch: got $output_dim_expected, expected $(moe_layer.config.output_dim)"))
        end
        
        if batch_size != batch_size_output
            throw(DimensionMismatch("Batch size mismatch between input ($batch_size) and output ($batch_size_output)"))
        end
        
        # Optimize batch size if needed
        optimized_batch_size = optimize_batch_size(moe_layer.gpu_config, moe_layer.config.batch_config, batch_size)
        if optimized_batch_size != batch_size
            @warn "Batch size optimization reduced batch from $batch_size to $optimized_batch_size"
        end
        
        # Get or create workspace
        workspace = get_batch_workspace(moe_layer.gpu_config, moe_layer.config.batch_config, batch_size)
        moe_layer.current_workspace[] = workspace
        moe_layer.workspace_allocated[] = true
        
        # Phase 1: Gating computation
        gating_start = time()
        @gpu_time "moe_gating_phase" begin
            expert_indices = view(workspace.expert_indices, :, 1:batch_size)
            expert_gates = view(workspace.expert_gates, :, 1:batch_size)
            
            # Compute gating decisions
            expert_indices_result, expert_gates_result, router_probs = gpu_topk_gating_forward!(
                expert_indices, expert_gates, input, moe_layer.gating;
                training = training,
                return_router_probs = true
            )
        end
        moe_layer.gating_time[] += time() - gating_start
        
        # Phase 2: Token routing and assignment
        routing_start = time()
        @gpu_time "moe_routing_phase" begin
            # Create token assignment mapping
            assignment = create_token_assignment(expert_indices_result, expert_gates_result, moe_layer.gpu_config)
            
            # Validate assignment
            if !validate_token_assignment(assignment)
                @warn "Token assignment validation failed"
            end
            
            # Route tokens to expert-specific buffers
            expert_inputs = [view(workspace.expert_inputs[i], :, 1:Int(assignment.expert_token_counts[i])) 
                           for i in 1:moe_layer.config.num_experts if assignment.expert_token_counts[i] > 0]
            active_expert_ids = [i for i in 1:moe_layer.config.num_experts if assignment.expert_token_counts[i] > 0]
            
            if !isempty(active_expert_ids)
                route_tokens_to_experts!(workspace.expert_inputs, input, assignment)
            end
        end
        routing_time = time() - routing_start
        
        # Phase 3: Parallel expert computation
        expert_start = time()
        @gpu_time "moe_expert_phase" begin
            # Process active experts in parallel
            if !isempty(active_expert_ids)
                process_experts_parallel!(workspace, moe_layer.experts, assignment, active_expert_ids, training)
            end
        end
        moe_layer.expert_time[] += time() - expert_start
        
        # Phase 4: Output combination
        combination_start = time()
        @gpu_time "moe_combination_phase" begin
            # Combine expert outputs back to original token order
            if !isempty(active_expert_ids)
                combine_expert_outputs!(output, workspace.expert_outputs, assignment)
            else
                # No experts activated - zero output
                fill!(output, T(0))
            end
        end
        moe_layer.combination_time[] += time() - combination_start
        
        # Phase 5: Load balancing loss computation
        loss_value = T(0)
        if training && return_loss
            loss_start = time()
            @gpu_time "moe_loss_phase" begin
                loss_value = gpu_switch_loss_forward!(expert_indices_result, router_probs, moe_layer.loss_function)
                moe_layer.last_loss_value[] = loss_value
            end
            moe_layer.loss_time[] += time() - loss_start
        end
        
        # Clean up workspace
        release_batch_workspace!(workspace)
        moe_layer.current_workspace[] = nothing
        moe_layer.workspace_allocated[] = false
        
        # Prepare return values
        result = if return_loss && return_stats
            stats = get_moe_layer_stats(moe_layer, assignment)
            (output, loss_value, stats)
        elseif return_loss
            (output, loss_value)
        elseif return_stats
            stats = get_moe_layer_stats(moe_layer, assignment)
            (output, stats)
        else
            output
        end
        
        return result
        
    catch e
        # Clean up on error
        if moe_layer.workspace_allocated[]
            workspace = moe_layer.current_workspace[]
            if !isnothing(workspace)
                release_batch_workspace!(workspace)
            end
            moe_layer.current_workspace[] = nothing
            moe_layer.workspace_allocated[] = false
        end
        
        @error "Error in GPU MoE forward pass" exception=e
        rethrow(e)
        
    finally
        elapsed_time = time() - start_time
        moe_layer.total_forward_time[] += elapsed_time
    end
end

"""
    process_experts_parallel!(workspace, experts, assignment, active_expert_ids, training)

Process multiple experts in parallel on GPU.
"""
function process_experts_parallel!(
    workspace::GPUBatchWorkspace{T},
    experts::Vector{GPUGatedExpert{T}},
    assignment::GPUTokenAssignment{T},
    active_expert_ids::Vector{Int},
    training::Bool
) where T<:AbstractFloat
    
    # Process each active expert
    for expert_id in active_expert_ids
        expert = experts[expert_id]
        expert_token_count = Int(assignment.expert_token_counts[expert_id])
        
        if expert_token_count > 0
            # Get input and output buffers for this expert
            expert_input = view(workspace.expert_inputs[expert_id], :, 1:expert_token_count)
            expert_output = view(workspace.expert_outputs[expert_id], :, 1:expert_token_count)
            
            # Run expert forward pass
            @gpu_time "expert_$expert_id" gpu_gated_expert_forward!(
                expert_output, expert_input, expert;
                training = training
            )
        end
    end
    
    # Synchronize all expert computations
    CUDA.synchronize()
end

"""
    get_moe_layer_stats(moe_layer, assignment)

Get comprehensive statistics about MoE layer execution.
"""
function get_moe_layer_stats(moe_layer::GPUMoELayer{T}, assignment::GPUTokenAssignment{T}) where T<:AbstractFloat
    
    # Get assignment statistics
    assignment_stats = get_assignment_statistics(assignment)
    
    # Get performance statistics
    forward_calls = moe_layer.forward_calls[]
    total_time = moe_layer.total_forward_time[]
    
    performance_stats = Dict{String, Any}(
        "forward_calls" => forward_calls,
        "total_forward_time_ms" => total_time * 1000,
        "avg_forward_time_ms" => forward_calls > 0 ? (total_time / forward_calls) * 1000 : 0.0,
        "gating_time_ms" => moe_layer.gating_time[] * 1000,
        "expert_time_ms" => moe_layer.expert_time[] * 1000,
        "combination_time_ms" => moe_layer.combination_time[] * 1000,
        "loss_time_ms" => moe_layer.loss_time[] * 1000,
        "last_loss_value" => moe_layer.last_loss_value[]
    )
    
    # Get component performance stats
    gating_stats = get_gating_performance_stats(moe_layer.gating)
    loss_stats = get_loss_performance_stats(moe_layer.loss_function)
    
    expert_stats = Dict{String, Any}()
    for (i, expert) in enumerate(moe_layer.experts)
        expert_stats["expert_$i"] = get_expert_performance_stats(expert)
    end
    
    # Get memory pool statistics
    memory_stats = get_memory_pool_statistics()
    
    return Dict{String, Any}(
        "assignment" => assignment_stats,
        "performance" => performance_stats,
        "gating" => gating_stats,
        "loss" => loss_stats,
        "experts" => expert_stats,
        "memory" => memory_stats,
        "configuration" => Dict(
            "num_experts" => moe_layer.config.num_experts,
            "top_k" => moe_layer.config.top_k,
            "input_dim" => moe_layer.config.input_dim,
            "hidden_dim" => moe_layer.config.hidden_dim,
            "output_dim" => moe_layer.config.output_dim
        )
    )
end

"""
    reset_moe_layer_stats!(moe_layer)

Reset all performance statistics for the MoE layer.
"""
function reset_moe_layer_stats!(moe_layer::GPUMoELayer{T}) where T<:AbstractFloat
    
    # Reset layer performance stats
    moe_layer.forward_calls[] = 0
    moe_layer.total_forward_time[] = 0.0
    moe_layer.gating_time[] = 0.0
    moe_layer.expert_time[] = 0.0
    moe_layer.combination_time[] = 0.0
    moe_layer.loss_time[] = 0.0
    moe_layer.last_loss_value[] = T(0)
    
    # Reset component stats
    reset_gating_performance_stats!(moe_layer.gating)
    reset_loss_performance_stats!(moe_layer.loss_function)
    
    for expert in moe_layer.experts
        reset_expert_performance_stats!(expert)
    end
    
    @info "Reset all MoE layer performance statistics"
end

"""
    optimize_moe_layer!(moe_layer, target_batch_size)

Optimize MoE layer for specific batch size and usage patterns.
"""
function optimize_moe_layer!(moe_layer::GPUMoELayer{T}, target_batch_size::Int) where T<:AbstractFloat
    
    @info "Optimizing MoE layer for batch size $target_batch_size"
    
    # Optimize individual experts
    for expert in moe_layer.experts
        optimize_expert_for_batch_size!(expert, target_batch_size ÷ moe_layer.config.num_experts + 32)
    end
    
    # Optimize batch configuration
    optimized_batch_size = optimize_batch_size(moe_layer.gpu_config, moe_layer.config.batch_config, target_batch_size)
    
    if optimized_batch_size != target_batch_size
        @info "Optimized batch size: $target_batch_size → $optimized_batch_size"
    end
    
    # Pre-warm GPU kernels with target batch size
    @info "Pre-warming GPU kernels..."
    warmup_input = CUDA.randn(T, moe_layer.config.input_dim, optimized_batch_size)
    warmup_output = gpu_zeros(T, moe_layer.config.output_dim, optimized_batch_size)
    
    # Run a few warmup iterations
    for _ in 1:3
        gpu_moe_forward!(warmup_output, warmup_input, moe_layer; training=false)
        CUDA.synchronize()
    end
    
    @info "MoE layer optimization completed"
end

"""
    validate_moe_layer(moe_layer)

Comprehensive validation of MoE layer configuration and state.
"""
function validate_moe_layer(moe_layer::GPUMoELayer{T}) where T<:AbstractFloat
    
    validation_results = Dict{String, Any}()
    all_valid = true
    
    # Validate configuration consistency
    config_valid = true
    if length(moe_layer.experts) != moe_layer.config.num_experts
        @error "Expert count mismatch: config=$(moe_layer.config.num_experts), actual=$(length(moe_layer.experts))"
        config_valid = false
    end
    
    if moe_layer.gating.k != moe_layer.config.top_k
        @error "Top-k mismatch: config=$(moe_layer.config.top_k), gating=$(moe_layer.gating.k)"
        config_valid = false
    end
    
    validation_results["configuration"] = config_valid
    all_valid &= config_valid
    
    # Validate expert dimensions
    expert_dims_valid = true
    for (i, expert) in enumerate(moe_layer.experts)
        weights = expert.weights
        
        if size(weights.w1) != (moe_layer.config.input_dim, moe_layer.config.hidden_dim)
            @error "Expert $i w1 dimension mismatch: expected $(moe_layer.config.input_dim)×$(moe_layer.config.hidden_dim), got $(size(weights.w1))"
            expert_dims_valid = false
        end
        
        if size(weights.w2) != (moe_layer.config.hidden_dim, moe_layer.config.output_dim)
            @error "Expert $i w2 dimension mismatch: expected $(moe_layer.config.hidden_dim)×$(moe_layer.config.output_dim), got $(size(weights.w2))"
            expert_dims_valid = false
        end
        
        if size(weights.w3) != (moe_layer.config.input_dim, moe_layer.config.hidden_dim)
            @error "Expert $i w3 dimension mismatch: expected $(moe_layer.config.input_dim)×$(moe_layer.config.hidden_dim), got $(size(weights.w3))"
            expert_dims_valid = false
        end
    end
    
    validation_results["expert_dimensions"] = expert_dims_valid
    all_valid &= expert_dims_valid
    
    # Validate gating dimensions
    gating_dims_valid = true
    router_size = size(moe_layer.gating.router_weights)
    expected_router_size = (moe_layer.config.input_dim, moe_layer.config.num_experts)
    
    if router_size != expected_router_size
        @error "Router dimension mismatch: expected $expected_router_size, got $router_size"
        gating_dims_valid = false
    end
    
    validation_results["gating_dimensions"] = gating_dims_valid
    all_valid &= gating_dims_valid
    
    # Validate numerical stability
    numerical_valid = true
    
    # Check expert weights for NaN/Inf
    for (i, expert) in enumerate(moe_layer.experts)
        weights = expert.weights
        if !gpu_check_finite(weights.w1) || !gpu_check_finite(weights.w2) || !gpu_check_finite(weights.w3)
            @error "Expert $i contains non-finite weights"
            numerical_valid = false
        end
    end
    
    # Check router weights
    if !gpu_check_finite(moe_layer.gating.router_weights)
        @error "Router weights contain non-finite values"
        numerical_valid = false
    end
    
    validation_results["numerical_stability"] = numerical_valid
    all_valid &= numerical_valid
    
    # Validate GPU memory state
    memory_valid = true
    try
        available_memory = CUDA.available_memory()
        if available_memory < 100 * 1024 * 1024  # Less than 100MB
            @warn "Low GPU memory available: $(available_memory ÷ (1024^2)) MB"
        end
    catch e
        @error "Error checking GPU memory" exception=e
        memory_valid = false
    end
    
    validation_results["memory_state"] = memory_valid
    all_valid &= memory_valid
    
    validation_results["overall_valid"] = all_valid
    
    if all_valid
        @info "MoE layer validation passed"
    else
        @error "MoE layer validation failed"
    end
    
    return validation_results
end

"""
    create_gpu_moe_layer(config)

Factory function to create a complete GPU MoE layer.
"""
function create_gpu_moe_layer(config::GPUMoELayerConfig{T}) where T<:AbstractFloat
    
    @info "Creating GPU MoE layer..."
    
    # Validate CUDA availability
    if !CUDA.functional()
        throw(ErrorException("CUDA not functional - cannot create GPU MoE layer"))
    end
    
    # Create the layer
    moe_layer = GPUMoELayer{T}(config)
    
    # Validate the created layer
    validation_results = validate_moe_layer(moe_layer)
    if !validation_results["overall_valid"]
        throw(ErrorException("Created MoE layer failed validation"))
    end
    
    @info "GPU MoE layer created successfully"
    return moe_layer
end

"""
    benchmark_moe_layer(moe_layer, batch_sizes; num_warmup=3, num_benchmark=10)

Benchmark MoE layer performance across different batch sizes.
"""
function benchmark_moe_layer(
    moe_layer::GPUMoELayer{T},
    batch_sizes::Vector{Int};
    num_warmup::Int = 3,
    num_benchmark::Int = 10,
    include_loss::Bool = true
) where T<:AbstractFloat
    
    results = Dict{Int, Dict{String, Float64}}()
    
    for batch_size in batch_sizes
        @info "Benchmarking MoE layer with batch size $batch_size"
        
        # Create test data
        input = CUDA.randn(T, moe_layer.config.input_dim, batch_size)
        output = gpu_zeros(T, moe_layer.config.output_dim, batch_size)
        
        # Warmup
        for _ in 1:num_warmup
            gpu_moe_forward!(output, input, moe_layer; training=include_loss, return_loss=include_loss)
            CUDA.synchronize()
        end
        
        # Benchmark
        times = Float64[]
        for _ in 1:num_benchmark
            start_time = time_ns()
            result = gpu_moe_forward!(output, input, moe_layer; training=include_loss, return_loss=include_loss)
            CUDA.synchronize()
            end_time = time_ns()
            push!(times, (end_time - start_time) / 1e6)  # Convert to ms
        end
        
        results[batch_size] = Dict(
            "mean_time_ms" => Statistics.mean(times),
            "std_time_ms" => Statistics.std(times),
            "min_time_ms" => minimum(times),
            "max_time_ms" => maximum(times),
            "throughput_tokens_per_sec" => batch_size / (Statistics.mean(times) / 1000)
        )
    end
    
    return results
end