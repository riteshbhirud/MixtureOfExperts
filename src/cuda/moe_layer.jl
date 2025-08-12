"""
GPU MoE Layer Implementation

Complete GPU-accelerated Mixture of Experts layer that orchestrates TopK gating,
gated experts, and Switch Transformer loss for optimal GPU performance.
This is the main orchestrating component that ties all GPU MoE components together.
"""

# Main GPU MoE Layer structure
struct GPUMoELayer{T<:AbstractFloat}
    # Core components
    experts::Vector{GPUGatedExpert{T}}
    gating::GPUTopKGating{T}
    load_balance_loss::GPUSwitchTransformerLoss{T}
    
    # Configuration
    config::GPUMoEConfig{T}
    
    # Workspace management
    workspace::Dict{Symbol, CuArray}
    workspace_allocated::Ref{Bool}
    workspace_size_bytes::Ref{Int64}
    
    # Routing state
    routing_state::GPURoutingState{T}
    
    # Performance optimization settings
    enable_expert_parallelism::Bool
    use_dynamic_batching::Bool
    enable_memory_optimization::Bool
    
    # Statistics tracking
    training_stats::Dict{Symbol, Any}
    performance_stats::Dict{Symbol, Any}
    
    # Performance counters
    forward_calls::Ref{Int}
    total_forward_time::Ref{Float64}
    routing_time::Ref{Float64}
    expert_compute_time::Ref{Float64}
    combination_time::Ref{Float64}
    loss_compute_time::Ref{Float64}
    
    function GPUMoELayer{T}(
        config::GPUMoEConfig{T};
        enable_expert_parallelism::Bool = true,
        use_dynamic_batching::Bool = true,
        enable_memory_optimization::Bool = true,
        initialization_scale::T = T(0.02)
    ) where T<:AbstractFloat
        
        # Validate configuration
        validate_gpu_moe_config(config)
        
        # Create experts
        experts = create_gpu_experts(config, initialization_scale)
        
        # Create gating mechanism
        gating = create_gpu_gating(config)
        
        # Create load balancing loss
        load_balance_loss = create_gpu_switch_loss(config)
        
        # Initialize workspace management
        workspace = Dict{Symbol, CuArray}()
        workspace_allocated = Ref(false)
        workspace_size_bytes = Ref(Int64(0))
        
        # Create routing state
        routing_state = GPURoutingState{T}(config.num_experts, config.top_k, config.max_batch_size)
        
        # Initialize statistics tracking
        training_stats = Dict{Symbol, Any}(
            :tokens_per_expert => zeros(Int, config.num_experts),
            :routing_entropy => Float32[],
            :capacity_overflow => 0,
            :expert_utilization => zeros(Float32, config.num_experts),
            :load_balance_scores => Float32[]
        )
        
        performance_stats = Dict{Symbol, Any}(
            :forward_calls => 0,
            :total_time_ms => 0.0,
            :avg_time_ms => 0.0,
            :throughput_tokens_per_sec => 0.0,
            :gpu_utilization => 0.0
        )
        
        # Initialize performance counters
        forward_calls = Ref(0)
        total_forward_time = Ref(0.0)
        routing_time = Ref(0.0)
        expert_compute_time = Ref(0.0)
        combination_time = Ref(0.0)
        loss_compute_time = Ref(0.0)
        
        moe_layer = new{T}(
            experts, gating, load_balance_loss,
            config,
            workspace, workspace_allocated, workspace_size_bytes,
            routing_state,
            enable_expert_parallelism, use_dynamic_batching, enable_memory_optimization,
            training_stats, performance_stats,
            forward_calls, total_forward_time, routing_time, 
            expert_compute_time, combination_time, loss_compute_time
        )
        
        return moe_layer
    end
end

# Convenience constructor
function GPUMoELayer(config::GPUMoEConfig{T}; kwargs...) where T
    return GPUMoELayer{T}(config; kwargs...)
end

# Main forward pass - this is the core orchestration function
function (moe_layer::GPUMoELayer{T})(
    input::CuMatrix{T};
    training::Bool = false,
    return_stats::Bool = false
) where T<:AbstractFloat
    
    moe_layer.forward_calls[] += 1
    start_time = time()
    
    try
        # Validate input dimensions
        input_dim, batch_size = size(input)
        if input_dim != moe_layer.config.input_dim
            throw(DimensionMismatch("Input dimension $input_dim does not match config $(moe_layer.config.input_dim)"))
        end
        
        # Allocate workspace if needed
        workspace = allocate_moe_workspace!(moe_layer, batch_size)
        
        # Prepare output tensor
        output = workspace[:main_output]
        if size(output, 2) < batch_size
            workspace[:main_output] = gpu_zeros(T, moe_layer.config.output_dim, batch_size)
            output = workspace[:main_output]
        end
        output_view = view(output, :, 1:batch_size)
        fill!(output_view, T(0))  # Initialize to zero
        
        # Phase 1: TopK Gating - Route tokens to experts
        routing_start = time()
        expert_indices, expert_gates, router_probs = gpu_topk_gating_forward!(
            workspace[:expert_indices][:, 1:batch_size],
            workspace[:expert_gates][:, 1:batch_size],
            input,
            moe_layer.gating;
            training = training,
            return_router_probs = true
        )
        moe_layer.routing_time[] += time() - routing_start
        
        # Phase 2: Token Routing - Organize tokens by expert assignment
        routing_organize_start = time()
        routing_info = organize_token_routing!(
            moe_layer.routing_state,
            expert_indices,
            expert_gates,
            batch_size
        )
        moe_layer.routing_time[] += time() - routing_organize_start
        
        # Phase 3: Parallel Expert Computation
        expert_start = time()
        if moe_layer.enable_expert_parallelism
            compute_experts_parallel!(
                output_view,
                input,
                moe_layer.experts,
                routing_info,
                workspace
            )
        else
            compute_experts_sequential!(
                output_view,
                input,
                moe_layer.experts,
                routing_info,
                workspace
            )
        end
        moe_layer.expert_compute_time[] += time() - expert_start
        
        # Phase 4: Load Balancing Loss Computation (if training)
        balance_loss = T(0)
        if training
            loss_start = time()
            balance_loss = gpu_switch_loss_forward!(
                expert_indices,
                router_probs,
                moe_layer.load_balance_loss
            )
            moe_layer.loss_compute_time[] += time() - loss_start
        end
        
        # Phase 5: Update Statistics
        if training || return_stats
            update_moe_statistics!(
                moe_layer,
                expert_indices,
                expert_gates,
                router_probs,
                balance_loss,
                batch_size
            )
        end
        
        # Return results
        if return_stats
            current_stats = get_current_moe_statistics(moe_layer)
            if training
                return output_view, balance_loss, current_stats
            else
                return output_view, current_stats
            end
        else
            if training
                return output_view, balance_loss
            else
                return output_view
            end
        end
        
    catch e
        @error "Error in GPU MoE layer forward pass" exception=e
        rethrow(e)
    finally
        elapsed_time = time() - start_time
        moe_layer.total_forward_time[] += elapsed_time
        
        # Update performance statistics
        update_performance_stats!(moe_layer, elapsed_time, size(input, 2))
    end
end

# Workspace allocation and management
function allocate_moe_workspace!(moe_layer::GPUMoELayer{T}, batch_size::Int) where T
    
    if moe_layer.workspace_allocated[] && 
       haskey(moe_layer.workspace, :allocated_batch_size) &&
       moe_layer.workspace[:allocated_batch_size] >= batch_size
        return moe_layer.workspace  # Already allocated with sufficient size
    end
    
    config = moe_layer.config
    workspace = moe_layer.workspace
    
    # Calculate workspace requirements
    top_k = config.top_k
    num_experts = config.num_experts
    input_dim = config.input_dim
    hidden_dim = config.hidden_dim
    output_dim = config.output_dim
    
    # Core MoE computation buffers
    workspace[:main_output] = gpu_zeros(T, output_dim, batch_size; aligned=true)
    workspace[:expert_indices] = CUDA.zeros(Int32, top_k, batch_size)
    workspace[:expert_gates] = gpu_zeros(T, top_k, batch_size; aligned=true)
    
    # Expert computation workspace
    max_tokens_per_expert = batch_size  # Worst case: all tokens go to one expert
    workspace[:expert_inputs] = gpu_zeros(T, input_dim, max_tokens_per_expert; aligned=true)
    workspace[:expert_outputs] = gpu_zeros(T, output_dim, max_tokens_per_expert; aligned=true)
    
    # Token routing workspace
    workspace[:token_assignments] = CUDA.zeros(Int32, top_k * batch_size)
    workspace[:expert_token_counts] = CUDA.zeros(Int32, num_experts)
    workspace[:expert_token_offsets] = CUDA.zeros(Int32, num_experts + 1)
    workspace[:sorted_tokens] = CUDA.zeros(Int32, top_k * batch_size)
    workspace[:gating_weights] = gpu_zeros(T, top_k * batch_size; aligned=true)
    
    # Temporary computation buffers for experts
    workspace[:temp_hidden] = gpu_zeros(T, hidden_dim, max_tokens_per_expert; aligned=true)
    workspace[:temp_gate] = gpu_zeros(T, hidden_dim, max_tokens_per_expert; aligned=true)
    workspace[:temp_up] = gpu_zeros(T, hidden_dim, max_tokens_per_expert; aligned=true)
    
    # Memory optimization: reuse buffers where possible
    if moe_layer.enable_memory_optimization
        # Use views of larger buffers for smaller operations
        workspace[:small_buffer_1] = gpu_zeros(T, max(hidden_dim, output_dim), batch_size; aligned=true)
        workspace[:small_buffer_2] = gpu_zeros(T, max(hidden_dim, output_dim), batch_size; aligned=true)
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
    
    moe_layer.workspace_allocated[] = true
    moe_layer.workspace_size_bytes[] = total_bytes
    
    @debug "Allocated GPU MoE workspace: $(total_bytes ÷ (1024^2)) MB for batch size $batch_size"
    
    return workspace
end

# Parallel expert computation - core GPU acceleration
function compute_experts_parallel!(
    output::CuMatrix{T},
    input::CuMatrix{T},
    experts::Vector{GPUGatedExpert{T}},
    routing_info::GPURoutingInfo{T},
    workspace::Dict{Symbol, CuArray}
) where T<:AbstractFloat
    
    # Process all experts that have assigned tokens
    active_experts = findall(count -> count > 0, routing_info.expert_token_counts)
    
    if isempty(active_experts)
        @warn "No experts have assigned tokens"
        return output
    end
    
    # Launch expert computations in parallel (GPU handles the parallelism)
    for expert_id in active_experts
        token_count = routing_info.expert_token_counts[expert_id]
        if token_count == 0
            continue
        end
        
        # Get token indices for this expert
        start_idx = routing_info.expert_token_offsets[expert_id] + 1
        end_idx = routing_info.expert_token_offsets[expert_id + 1]
        
        if start_idx > end_idx
            continue
        end
        
        # Get the tokens assigned to this expert
        assigned_token_indices = routing_info.sorted_token_indices[start_idx:end_idx]
        assigned_weights = routing_info.sorted_gating_weights[start_idx:end_idx]
        
        # Extract input for this expert
        expert_input = workspace[:expert_inputs][:, 1:token_count]
        extract_tokens_for_expert!(expert_input, input, assigned_token_indices)
        
        # Run expert forward pass
        expert_output = workspace[:expert_outputs][:, 1:token_count]
        gpu_gated_expert_forward!(expert_output, expert_input, experts[expert_id])
        
        # Apply gating weights and accumulate to output
        apply_gating_weights_and_accumulate!(
            output,
            expert_output,
            assigned_token_indices,
            assigned_weights
        )
    end
    
    return output
end

# Sequential expert computation (fallback)
function compute_experts_sequential!(
    output::CuMatrix{T},
    input::CuMatrix{T},
    experts::Vector{GPUGatedExpert{T}},
    routing_info::GPURoutingInfo{T},
    workspace::Dict{Symbol, CuArray}
) where T<:AbstractFloat
    
    # This is essentially the same as parallel but processes experts one by one
    # GPU will still parallelize within each expert computation
    compute_experts_parallel!(output, input, experts, routing_info, workspace)
    
    return output
end

# Token extraction for expert computation
function extract_tokens_for_expert!(
    expert_input::CuMatrix{T},
    full_input::CuMatrix{T},
    token_indices::CuVector{Int32}
) where T<:AbstractFloat
    
    input_dim, expert_batch_size = size(expert_input)
    total_input_dim, total_batch_size = size(full_input)
    
    if input_dim != total_input_dim
        throw(DimensionMismatch("Input dimensions must match"))
    end
    
    if expert_batch_size != length(token_indices)
        throw(DimensionMismatch("Expert batch size must match number of token indices"))
    end
    
    # Launch CUDA kernel to extract tokens
    total_elements = input_dim * expert_batch_size
    kernel_config = GPUKernelConfig(total_elements)
    
    @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid extract_tokens_kernel!(
        expert_input, full_input, token_indices, input_dim, expert_batch_size
    )
    
    CUDA.synchronize()
    return expert_input
end

# CUDA kernel for token extraction
function extract_tokens_kernel!(
    expert_input::CuDeviceMatrix{T},
    full_input::CuDeviceMatrix{T},
    token_indices::CuDeviceVector{Int32},
    input_dim::Int,
    expert_batch_size::Int
) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= input_dim * expert_batch_size
        # Convert linear index to 2D coordinates
        feature_idx = ((idx - 1) % input_dim) + 1
        expert_token_idx = ((idx - 1) ÷ input_dim) + 1
        
        if expert_token_idx <= expert_batch_size
            # Get the original token index
            original_token_idx = token_indices[expert_token_idx]
            
            # Copy the feature
            expert_input[feature_idx, expert_token_idx] = full_input[feature_idx, original_token_idx]
        end
    end
    
    return nothing
end

# Apply gating weights and accumulate to output
function apply_gating_weights_and_accumulate!(
    output::CuMatrix{T},
    expert_output::CuMatrix{T},
    token_indices::CuVector{Int32},
    gating_weights::CuVector{T}
) where T<:AbstractFloat
    
    output_dim, expert_batch_size = size(expert_output)
    total_output_dim, total_batch_size = size(output)
    
    if output_dim != total_output_dim
        throw(DimensionMismatch("Output dimensions must match"))
    end
    
    if expert_batch_size != length(token_indices) || expert_batch_size != length(gating_weights)
        throw(DimensionMismatch("Batch sizes must match"))
    end
    
    # Launch CUDA kernel to apply weights and accumulate
    total_elements = output_dim * expert_batch_size
    kernel_config = GPUKernelConfig(total_elements)
    
    @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid accumulate_weighted_outputs_kernel!(
        output, expert_output, token_indices, gating_weights, output_dim, expert_batch_size
    )
    
    CUDA.synchronize()
    return output
end

# CUDA kernel for weighted accumulation
function accumulate_weighted_outputs_kernel!(
    output::CuDeviceMatrix{T},
    expert_output::CuDeviceMatrix{T},
    token_indices::CuDeviceVector{Int32},
    gating_weights::CuDeviceVector{T},
    output_dim::Int,
    expert_batch_size::Int
) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= output_dim * expert_batch_size
        # Convert linear index to 2D coordinates
        feature_idx = ((idx - 1) % output_dim) + 1
        expert_token_idx = ((idx - 1) ÷ output_dim) + 1
        
        if expert_token_idx <= expert_batch_size
            # Get the original token index and weight
            original_token_idx = token_indices[expert_token_idx]
            weight = gating_weights[expert_token_idx]
            
            # Apply weight and accumulate using atomic operation
            weighted_value = expert_output[feature_idx, expert_token_idx] * weight
            CUDA.atomic_add!(pointer(output, feature_idx + (original_token_idx - 1) * output_dim), weighted_value)
        end
    end
    
    return nothing
end

# Statistics and monitoring
function update_moe_statistics!(
    moe_layer::GPUMoELayer{T},
    expert_indices::CuMatrix{Int32},
    expert_gates::CuMatrix{T},
    router_probs::CuMatrix{T},
    balance_loss::T,
    batch_size::Int
) where T<:AbstractFloat
    
    stats = moe_layer.training_stats
    
    # Update expert token counts
    expert_indices_cpu = Array(expert_indices)
    for idx in expert_indices_cpu
        if idx > 0 && idx <= length(stats[:tokens_per_expert])
            stats[:tokens_per_expert][idx] += 1
        end
    end
    
    # Compute and store routing entropy
    router_probs_cpu = Array(router_probs)
    batch_entropy = -sum(router_probs_cpu .* log.(router_probs_cpu .+ 1e-8), dims=1)
    push!(stats[:routing_entropy], mean(batch_entropy))
    
    # Compute expert utilization
    expert_usage = zeros(Float32, moe_layer.config.num_experts)
    for i in 1:moe_layer.config.num_experts
        expert_usage[i] = sum(expert_indices_cpu .== i) / length(expert_indices_cpu)
    end
    stats[:expert_utilization] = expert_usage
    
    # Store load balance score
    if balance_loss > 0
        # Compute load balance score from expert usage
        ideal_usage = 1.0f0 / moe_layer.config.num_experts
        variance = sum((expert_usage .- ideal_usage).^2) / moe_layer.config.num_experts
        balance_score = 1.0f0 - sqrt(variance) / ideal_usage
        push!(stats[:load_balance_scores], balance_score)
    end
end

function update_performance_stats!(moe_layer::GPUMoELayer{T}, elapsed_time::Float64, batch_size::Int) where T
    stats = moe_layer.performance_stats
    forward_calls = moe_layer.forward_calls[]
    
    # Update basic timing stats
    stats[:forward_calls] = forward_calls
    stats[:total_time_ms] = moe_layer.total_forward_time[] * 1000
    stats[:avg_time_ms] = stats[:total_time_ms] / forward_calls
    
    # Compute throughput
    total_tokens_processed = batch_size * forward_calls
    stats[:throughput_tokens_per_sec] = total_tokens_processed / moe_layer.total_forward_time[]
    
    # GPU utilization (simplified metric)
    total_gpu_time = (moe_layer.routing_time[] + moe_layer.expert_compute_time[] + 
                     moe_layer.combination_time[] + moe_layer.loss_compute_time[])
    stats[:gpu_utilization] = min(1.0, total_gpu_time / moe_layer.total_forward_time[])
end

function get_current_moe_statistics(moe_layer::GPUMoELayer{T}) where T
    stats = Dict{String, Any}()
    
    # Copy training statistics
    for (key, value) in moe_layer.training_stats
        stats[String(key)] = copy(value)
    end
    
    # Add performance statistics
    for (key, value) in moe_layer.performance_stats
        stats[String(key)] = value
    end
    
    # Add detailed timing breakdown
    total_time = moe_layer.total_forward_time[]
    if total_time > 0
        stats["timing_breakdown"] = Dict(
            "routing_percentage" => (moe_layer.routing_time[] / total_time) * 100,
            "expert_compute_percentage" => (moe_layer.expert_compute_time[] / total_time) * 100,
            "combination_percentage" => (moe_layer.combination_time[] / total_time) * 100,
            "loss_compute_percentage" => (moe_layer.loss_compute_time[] / total_time) * 100
        )
    end
    
    # Add memory usage
    stats["workspace_size_mb"] = moe_layer.workspace_size_bytes[] / (1024^2)
    
    return stats
end

# Configuration validation
function validate_gpu_moe_config(config::GPUMoEConfig{T}) where T
    # Validate basic parameters
    if config.num_experts <= 0
        throw(ArgumentError("num_experts must be positive"))
    end
    
    if config.top_k <= 0 || config.top_k > config.num_experts
        throw(ArgumentError("top_k must be positive and not exceed num_experts"))
    end
    
    if config.input_dim <= 0 || config.hidden_dim <= 0 || config.output_dim <= 0
        throw(ArgumentError("All dimensions must be positive"))
    end
    
    if config.max_batch_size <= 0
        throw(ArgumentError("max_batch_size must be positive"))
    end
    
    # Check GPU memory requirements
    estimated_memory_mb = estimate_memory_requirements(config)
    available_memory_mb = CUDA.available_memory() ÷ (1024^2)
    
    if estimated_memory_mb > available_memory_mb * 0.8  # Leave 20% buffer
        @warn "Estimated memory usage ($estimated_memory_mb MB) may exceed available GPU memory ($available_memory_mb MB)"
    end
    
    return true
end

function estimate_memory_requirements(config::GPUMoEConfig{T}) where T
    # Rough estimation of memory requirements
    element_size = sizeof(T)
    
    # Expert weights
    expert_memory = config.num_experts * (
        config.input_dim * config.hidden_dim +  # w1
        config.hidden_dim * config.output_dim + # w2
        config.input_dim * config.hidden_dim    # w3
    ) * element_size
    
    # Router weights
    router_memory = config.input_dim * config.num_experts * element_size
    
    # Workspace memory (rough estimate)
    workspace_memory = config.max_batch_size * (
        config.input_dim + config.hidden_dim + config.output_dim
    ) * element_size * 10  # Factor for various intermediate buffers
    
    total_memory_bytes = expert_memory + router_memory + workspace_memory
    return total_memory_bytes ÷ (1024^2)  # Convert to MB
end

# Factory functions for creating experts, gating, and loss
function create_gpu_experts(config::GPUMoEConfig{T}, initialization_scale::T) where T
    experts = Vector{GPUGatedExpert{T}}()
    
    for i in 1:config.num_experts
        expert = create_random_gpu_expert(config; expert_id=i, initialization_scale=initialization_scale)
        push!(experts, expert)
    end
    
    return experts
end

function create_gpu_gating(config::GPUMoEConfig{T}) where T
    return GPUTopKGating{T}(config.top_k, config)
end

function create_gpu_switch_loss(config::GPUMoEConfig{T}) where T
    return GPUSwitchTransformerLoss{T}(T(0.01), config)  # Default alpha = 0.01
end

# Cleanup and resource management
function free_moe_workspace!(moe_layer::GPUMoELayer)
    if moe_layer.workspace_allocated[]
        # Clear workspace references
        empty!(moe_layer.workspace)
        moe_layer.workspace_allocated[] = false
        moe_layer.workspace_size_bytes[] = 0
        
        # Free individual component workspaces
        for expert in moe_layer.experts
            free_workspace!(expert)
        end
        
        # Force garbage collection
        GC.gc()
        CUDA.reclaim()
    end
end

# Performance monitoring
function reset_moe_performance_stats!(moe_layer::GPUMoELayer)
    moe_layer.forward_calls[] = 0
    moe_layer.total_forward_time[] = 0.0
    moe_layer.routing_time[] = 0.0
    moe_layer.expert_compute_time[] = 0.0
    moe_layer.combination_time[] = 0.0
    moe_layer.loss_compute_time[] = 0.0
    
    # Reset component performance stats
    for expert in moe_layer.experts
        reset_expert_performance_stats!(expert)
    end
    
    reset_gating_performance_stats!(moe_layer.gating)
    reset_loss_performance_stats!(moe_layer.load_balance_loss)
    
    # Clear training statistics
    fill!(moe_layer.training_stats[:tokens_per_expert], 0)
    empty!(moe_layer.training_stats[:routing_entropy])
    fill!(moe_layer.training_stats[:expert_utilization], 0.0f0)
    empty!(moe_layer.training_stats[:load_balance_scores])
    moe_layer.training_stats[:capacity_overflow] = 0
end

function get_moe_performance_report(moe_layer::GPUMoELayer{T}) where T
    report = Dict{String, Any}()
    
    # Overall performance
    report["overall"] = get_current_moe_statistics(moe_layer)
    
    # Component performance
    report["gating"] = get_gating_performance_stats(moe_layer.gating)
    report["load_balance_loss"] = get_loss_performance_stats(moe_layer.load_balance_loss)
    
    # Expert performance
    expert_stats = []
    for (i, expert) in enumerate(moe_layer.experts)
        expert_stat = get_expert_performance_stats(expert)
        expert_stat["expert_id"] = i
        push!(expert_stats, expert_stat)
    end
    report["experts"] = expert_stats
    
    return report
end