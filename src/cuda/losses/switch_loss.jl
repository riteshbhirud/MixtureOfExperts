"""
GPU Switch Transformer Loss Implementation

High-performance GPU implementation of Switch Transformer auxiliary loss for load balancing
across experts using custom CUDA kernels for expert assignment tracking, probability
computation, and parallel reduction operations.
"""

# Main GPU Switch Transformer Loss structure
struct GPUSwitchTransformerLoss{T<:AbstractFloat}
    alpha::T                              # Loss scaling factor
    config::GPUMoEConfig{T}
    
    # Computational state management
    state::GPUSwitchLossState{T}
    state_allocated::Ref{Bool}
    
    # Algorithm configuration
    use_parallel_reduction::Bool
    use_shared_memory::Bool
    enable_gradient_clipping::Bool
    gradient_clip_value::T
    
    # Loss computation tracking
    loss_history::Vector{T}
    max_history_length::Int
    
    # Performance tracking
    forward_calls::Ref{Int}
    total_forward_time::Ref{Float64}
    expert_fraction_time::Ref{Float64}
    probability_time::Ref{Float64}
    loss_computation_time::Ref{Float64}
    
    # Statistical monitoring
    last_expert_fractions::Vector{T}
    last_expert_probabilities::Vector{T}
    last_balance_score::Ref{T}
    last_entropy::Ref{T}
    
    function GPUSwitchTransformerLoss{T}(
        alpha::T,
        config::GPUMoEConfig{T};
        use_parallel_reduction::Bool = true,
        use_shared_memory::Bool = true,
        enable_gradient_clipping::Bool = true,
        gradient_clip_value::T = T(10.0),
        max_history_length::Int = 1000
    ) where T<:AbstractFloat
        
        # Validate parameters
        if alpha < 0
            throw(ArgumentError("Alpha must be non-negative"))
        end
        
        # Initialize computational state
        initial_batch_size = min(config.max_batch_size, 64)
        state = GPUSwitchLossState{T}(config.num_experts, initial_batch_size)
        state_allocated = Ref(true)
        
        # Initialize performance tracking
        forward_calls = Ref(0)
        total_forward_time = Ref(0.0)
        expert_fraction_time = Ref(0.0)
        probability_time = Ref(0.0)
        loss_computation_time = Ref(0.0)
        
        # Initialize statistical monitoring
        last_expert_fractions = zeros(T, config.num_experts)
        last_expert_probabilities = zeros(T, config.num_experts)
        last_balance_score = Ref(T(1))
        last_entropy = Ref(T(0))
        
        # Initialize loss history
        loss_history = T[]
        
        loss = new{T}(
            alpha, config,
            state, state_allocated,
            use_parallel_reduction, use_shared_memory, enable_gradient_clipping, gradient_clip_value,
            loss_history, max_history_length,
            forward_calls, total_forward_time, expert_fraction_time, 
            probability_time, loss_computation_time,
            last_expert_fractions, last_expert_probabilities, last_balance_score, last_entropy
        )
        
        return loss
    end
end

# Convenience constructor
function GPUSwitchTransformerLoss(alpha::T, config::GPUMoEConfig{T}; kwargs...) where T
    return GPUSwitchTransformerLoss{T}(alpha, config; kwargs...)
end

# Forward loss computation
function gpu_switch_loss_forward!(
    expert_assignments::CuMatrix{Int32},  # top_k × batch_size
    router_probs::CuMatrix{T},           # num_experts × batch_size
    loss::GPUSwitchTransformerLoss{T};
    return_statistics::Bool = false,
    update_history::Bool = true
) where T<:AbstractFloat
    
    loss.forward_calls[] += 1
    start_time = time()
    
    try
        # Validate input dimensions
        top_k, batch_size = size(expert_assignments)
        num_experts, batch_size2 = size(router_probs)
        
        if batch_size != batch_size2
            throw(DimensionMismatch("Expert assignments and router probabilities must have same batch size"))
        end
        
        if num_experts != loss.config.num_experts
            throw(DimensionMismatch("Router probabilities must match configured number of experts"))
        end
        
        # Resize state if needed for larger batch size
        if batch_size > loss.state.current_batch_size
            resize_loss_state!(loss.state, batch_size)
        end
        
        # Update current state dimensions
        loss.state.current_batch_size = batch_size
        loss.state.total_processed_tokens += Int64(batch_size)
        
        # Phase 1: Compute expert assignment fractions
        fraction_start = time()
        @gpu_time "expert_fractions" expert_fractions = compute_expert_fractions!(
            expert_assignments, loss
        )
        loss.expert_fraction_time[] += time() - fraction_start
        
        # Phase 2: Compute expert probabilities
        prob_start = time()
        @gpu_time "expert_probabilities" expert_probabilities = compute_expert_probabilities!(
            router_probs, loss
        )
        loss.probability_time[] += time() - prob_start
        
        # Phase 3: Compute Switch Transformer loss
        loss_start = time()
        @gpu_time "switch_loss_computation" switch_loss_value = compute_switch_loss!(
            expert_fractions, expert_probabilities, loss
        )
        loss.loss_computation_time[] += time() - loss_start
        
        # Update monitoring statistics
        if return_statistics || update_history
            update_loss_statistics!(loss, expert_fractions, expert_probabilities, 
                                   expert_assignments, switch_loss_value)
        end
        
        # Update loss history
        if update_history
            push!(loss.loss_history, switch_loss_value)
            if length(loss.loss_history) > loss.max_history_length
                popfirst!(loss.loss_history)
            end
        end
        
        # Apply gradient clipping if enabled
        if loss.enable_gradient_clipping
            switch_loss_value = clamp(switch_loss_value, -loss.gradient_clip_value, loss.gradient_clip_value)
        end
        
        # Return results
        if return_statistics
            statistics = get_current_loss_statistics(loss)
            return switch_loss_value, statistics
        else
            return switch_loss_value
        end
        
    catch e
        @error "Error in GPU Switch Transformer loss computation" exception=e
        rethrow(e)
    finally
        elapsed_time = time() - start_time
        loss.total_forward_time[] += elapsed_time
    end
end

# Compute expert assignment fractions
function compute_expert_fractions!(
    expert_assignments::CuMatrix{Int32},
    loss::GPUSwitchTransformerLoss{T}
) where T<:AbstractFloat
    
    num_experts = loss.config.num_experts
    expert_fractions = view(loss.state.expert_counts, 1:num_experts)
    
    # Launch kernel to compute fractions
    launch_expert_fractions_kernel!(
        expert_fractions, expert_assignments;
        use_shared_memory = loss.use_shared_memory
    )
    
    # Copy results to monitoring arrays
    copyto!(loss.last_expert_fractions, Array(expert_fractions))
    
    return expert_fractions
end

# Compute expert probabilities
function compute_expert_probabilities!(
    router_probs::CuMatrix{T},
    loss::GPUSwitchTransformerLoss{T}
) where T<:AbstractFloat
    
    num_experts = loss.config.num_experts
    expert_probabilities = view(loss.state.expert_probs, 1:num_experts)
    
    # Launch kernel to compute probabilities
    launch_expert_probabilities_kernel!(
        expert_probabilities, router_probs;
        use_parallel_reduction = loss.use_parallel_reduction
    )
    
    # Copy results to monitoring arrays
    copyto!(loss.last_expert_probabilities, Array(expert_probabilities))
    
    return expert_probabilities
end

# Compute Switch Transformer loss
function compute_switch_loss!(
    expert_fractions::CuVector{T},
    expert_probabilities::CuVector{T},
    loss::GPUSwitchTransformerLoss{T}
) where T<:AbstractFloat
    
    # Launch kernel to compute loss
    switch_loss_value = launch_switch_loss_kernel!(
        expert_fractions, expert_probabilities, loss.alpha
    )
    
    return switch_loss_value
end

# Update comprehensive loss statistics
function update_loss_statistics!(
    loss::GPUSwitchTransformerLoss{T},
    expert_fractions::CuVector{T},
    expert_probabilities::CuVector{T},
    expert_assignments::CuMatrix{Int32},
    switch_loss_value::T
) where T<:AbstractFloat
    
    # Compute load balance score
    loss.last_balance_score[] = launch_load_balance_score_kernel!(expert_assignments)
    
    # Compute expert usage entropy
    loss.last_entropy[] = launch_expert_entropy_kernel!(expert_probabilities)
    
    # Update computational state performance metrics
    loss.state.loss_computation_time += loss.loss_computation_time[]
    loss.state.assignment_time += loss.expert_fraction_time[]
    loss.state.reduction_time += loss.probability_time[]
end

# Resize loss state for larger batch sizes
function resize_loss_state!(state::GPUSwitchLossState{T}, new_batch_size::Int) where T
    if new_batch_size <= state.current_batch_size
        return state  # No need to resize
    end
    
    # Reallocate batch-dependent buffers
    num_experts = state.current_num_experts
    
    state.assignment_buffer = CUDA.zeros(Int32, num_experts, new_batch_size)
    state.count_workspace = CUDA.zeros(T, new_batch_size)
    state.reduction_buffer = CUDA.zeros(T, new_batch_size)
    
    state.current_batch_size = new_batch_size
    
    return state
end

# Get current statistics
function get_current_loss_statistics(loss::GPUSwitchTransformerLoss{T}) where T
    statistics = Dict{String, Any}(
        "switch_loss_value" => length(loss.loss_history) > 0 ? loss.loss_history[end] : T(0),
        "load_balance_score" => loss.last_balance_score[],
        "expert_entropy" => loss.last_entropy[],
        "expert_fractions" => copy(loss.last_expert_fractions),
        "expert_probabilities" => copy(loss.last_expert_probabilities),
        "num_experts" => loss.config.num_experts,
        "alpha" => loss.alpha,
        "total_processed_tokens" => loss.state.total_processed_tokens
    )
    
    # Add expert usage analysis
    if !isempty(loss.last_expert_fractions)
        max_usage = maximum(loss.last_expert_fractions)
        min_usage = minimum(loss.last_expert_fractions)
        mean_usage = sum(loss.last_expert_fractions) / length(loss.last_expert_fractions)
        usage_variance = sum((f - mean_usage)^2 for f in loss.last_expert_fractions) / length(loss.last_expert_fractions)
        
        statistics["expert_usage_analysis"] = Dict(
            "max_usage" => max_usage,
            "min_usage" => min_usage,
            "mean_usage" => mean_usage,
            "usage_variance" => usage_variance,
            "usage_imbalance" => max_usage - min_usage
        )
    end
    
    return statistics
end

# Loss history analysis
function analyze_loss_history(loss::GPUSwitchTransformerLoss{T}; window_size::Int = 50) where T
    if length(loss.loss_history) < window_size
        return Dict{String, Any}("status" => "insufficient_data", "samples" => length(loss.loss_history))
    end
    
    recent_losses = loss.loss_history[end-window_size+1:end]
    
    analysis = Dict{String, Any}(
        "window_size" => window_size,
        "mean_loss" => Statistics.mean(recent_losses),
        "std_loss" => Statistics.std(recent_losses),
        "min_loss" => minimum(recent_losses),
        "max_loss" => maximum(recent_losses),
        "current_loss" => recent_losses[end],
        "total_samples" => length(loss.loss_history)
    )
    
    # Trend analysis
    if window_size >= 10
        mid_point = window_size ÷ 2
        first_half_mean = Statistics.mean(recent_losses[1:mid_point])
        second_half_mean = Statistics.mean(recent_losses[mid_point+1:end])
        
        analysis["trend"] = Dict(
            "first_half_mean" => first_half_mean,
            "second_half_mean" => second_half_mean,
            "trend_direction" => second_half_mean > first_half_mean ? "increasing" : "decreasing",
            "trend_magnitude" => abs(second_half_mean - first_half_mean)
        )
    end
    
    # Stability analysis
    if length(recent_losses) >= 20
        recent_variance = Statistics.var(recent_losses[end-19:end])
        earlier_variance = Statistics.var(recent_losses[1:20])
        
        analysis["stability"] = Dict(
            "recent_variance" => recent_variance,
            "earlier_variance" => earlier_variance,
            "stability_ratio" => recent_variance / max(earlier_variance, T(1e-8)),
            "is_stabilizing" => recent_variance < earlier_variance
        )
    end
    
    return analysis
end

# Expert balance monitoring
function monitor_expert_balance(loss::GPUSwitchTransformerLoss{T}; 
                               imbalance_threshold::T = T(0.1)) where T
    
    if isempty(loss.last_expert_fractions)
        return Dict{String, Any}("status" => "no_data")
    end
    
    fractions = loss.last_expert_fractions
    num_experts = length(fractions)
    ideal_fraction = T(1) / T(num_experts)
    
    # Compute imbalance metrics
    max_deviation = maximum(abs.(fractions .- ideal_fraction))
    relative_imbalance = max_deviation / ideal_fraction
    
    # Identify problematic experts
    overused_experts = findall(f -> f > ideal_fraction + imbalance_threshold, fractions)
    underused_experts = findall(f -> f < ideal_fraction - imbalance_threshold, fractions)
    
    balance_report = Dict{String, Any}(
        "balance_score" => loss.last_balance_score[],
        "max_deviation" => max_deviation,
        "relative_imbalance" => relative_imbalance,
        "ideal_fraction" => ideal_fraction,
        "imbalance_threshold" => imbalance_threshold,
        "is_balanced" => relative_imbalance <= imbalance_threshold,
        "overused_experts" => overused_experts,
        "underused_experts" => underused_experts,
        "num_overused" => length(overused_experts),
        "num_underused" => length(underused_experts)
    )
    
    # Add expert-specific details
    if !isempty(overused_experts) || !isempty(underused_experts)
        expert_details = Dict{String, Any}()
        
        for expert_id in overused_experts
            expert_details["expert_$expert_id"] = Dict(
                "usage_fraction" => fractions[expert_id],
                "excess_usage" => fractions[expert_id] - ideal_fraction,
                "status" => "overused"
            )
        end
        
        for expert_id in underused_experts
            expert_details["expert_$expert_id"] = Dict(
                "usage_fraction" => fractions[expert_id],
                "deficit_usage" => ideal_fraction - fractions[expert_id],
                "status" => "underused"
            )
        end
        
        balance_report["expert_details"] = expert_details
    end
    
    return balance_report
end

# Performance analysis
function get_loss_performance_stats(loss::GPUSwitchTransformerLoss{T}) where T
    forward_calls = loss.forward_calls[]
    total_time = loss.total_forward_time[]
    
    stats = Dict{String, Any}(
        "forward_calls" => forward_calls,
        "total_forward_time_ms" => total_time * 1000,
        "avg_forward_time_ms" => forward_calls > 0 ? (total_time / forward_calls) * 1000 : 0.0,
        "expert_fraction_time_ms" => loss.expert_fraction_time[] * 1000,
        "probability_time_ms" => loss.probability_time[] * 1000,
        "loss_computation_time_ms" => loss.loss_computation_time[] * 1000,
        "avg_fraction_time_ms" => forward_calls > 0 ? (loss.expert_fraction_time[] / forward_calls) * 1000 : 0.0,
        "avg_probability_time_ms" => forward_calls > 0 ? (loss.probability_time[] / forward_calls) * 1000 : 0.0,
        "avg_loss_time_ms" => forward_calls > 0 ? (loss.loss_computation_time[] / forward_calls) * 1000 : 0.0,
        "alpha" => loss.alpha,
        "use_parallel_reduction" => loss.use_parallel_reduction,
        "use_shared_memory" => loss.use_shared_memory,
        "state_workspace_mb" => 0.0  # State doesn't track workspace size directly
    )
    
    # Add state-specific performance metrics
    state = loss.state
    if state.loss_computation_time > 0
        stats["state_loss_time_ms"] = state.loss_computation_time * 1000
        stats["state_assignment_time_ms"] = state.assignment_time * 1000
        stats["state_reduction_time_ms"] = state.reduction_time * 1000
    end
    
    return stats
end

function reset_loss_performance_stats!(loss::GPUSwitchTransformerLoss)
    loss.forward_calls[] = 0
    loss.total_forward_time[] = 0.0
    loss.expert_fraction_time[] = 0.0
    loss.probability_time[] = 0.0
    loss.loss_computation_time[] = 0.0
    
    # Reset state performance metrics
    loss.state.loss_computation_time = 0.0
    loss.state.assignment_time = 0.0
    loss.state.reduction_time = 0.0
end

# Benchmark loss computation with different configurations
function benchmark_loss_computation(
    config::GPUMoEConfig{T},
    alpha::T,
    batch_sizes::Vector{Int},
    top_k_values::Vector{Int};
    num_warmup::Int = 3,
    num_benchmark::Int = 10
) where T<:AbstractFloat
    
    results = Dict{Tuple{Int, Int}, Dict{String, Float64}}()
    
    for batch_size in batch_sizes
        for top_k in top_k_values
            @info "Benchmarking loss computation: batch_size=$batch_size, top_k=$top_k"
            
            try
                # Create loss instance
                loss = GPUSwitchTransformerLoss{T}(alpha, config)
                
                # Create test data
                expert_assignments = CUDA.rand(1:config.num_experts, top_k, batch_size)
                router_probs = gpu_softmax(CUDA.randn(T, config.num_experts, batch_size))
                
                # Warmup
                for _ in 1:num_warmup
                    gpu_switch_loss_forward!(expert_assignments, router_probs, loss)
                    CUDA.synchronize()
                end
                
                # Benchmark
                times = Float64[]
                for _ in 1:num_benchmark
                    start_time = time_ns()
                    loss_value = gpu_switch_loss_forward!(expert_assignments, router_probs, loss)
                    CUDA.synchronize()
                    end_time = time_ns()
                    push!(times, (end_time - start_time) / 1e6)  # Convert to ms
                end
                
                results[(batch_size, top_k)] = Dict(
                    "mean_time_ms" => Statistics.mean(times),
                    "std_time_ms" => Statistics.std(times),
                    "min_time_ms" => minimum(times),
                    "max_time_ms" => maximum(times)
                )
                
            catch e
                @warn "Benchmark failed for batch_size=$batch_size, top_k=$top_k" exception=e
            end
        end
    end
    
    return results
end

# Factory functions
function create_gpu_switch_loss(config::GPUMoEConfig{T}, alpha::T = T(0.01); kwargs...) where T<:AbstractFloat
    return GPUSwitchTransformerLoss{T}(alpha, config; kwargs...)
end

function copy_loss_to_gpu(cpu_loss::Any, config::GPUMoEConfig{T}) where T<:AbstractFloat
    # Extract parameters from CPU loss (implementation depends on CPU loss structure)
    alpha = T(cpu_loss.alpha)
    
    # Copy any relevant settings
    kwargs = Dict{Symbol, Any}()
    if hasproperty(cpu_loss, :use_parallel_reduction)
        kwargs[:use_parallel_reduction] = cpu_loss.use_parallel_reduction
    end
    if hasproperty(cpu_loss, :enable_gradient_clipping)
        kwargs[:enable_gradient_clipping] = cpu_loss.enable_gradient_clipping
        kwargs[:gradient_clip_value] = T(cpu_loss.gradient_clip_value)
    end
    
    return GPUSwitchTransformerLoss{T}(alpha, config; kwargs...)
end

# Integration utilities for loss scaling and scheduling
mutable struct AdaptiveLossScaling{T<:AbstractFloat}
    base_loss::GPUSwitchTransformerLoss{T}
    initial_alpha::T
    current_alpha::T
    target_balance_score::T
    adaptation_rate::T
    min_alpha::T
    max_alpha::T
    adaptation_window::Int
    calls_since_adaptation::Ref{Int}
    
    function AdaptiveLossScaling{T}(
        base_loss::GPUSwitchTransformerLoss{T};
        target_balance_score::T = T(0.9),
        adaptation_rate::T = T(0.1),
        min_alpha::T = T(0.001),
        max_alpha::T = T(0.1),
        adaptation_window::Int = 100
    ) where T
        
        initial_alpha = base_loss.alpha
        
        return new{T}(
            base_loss, initial_alpha, initial_alpha,
            target_balance_score, adaptation_rate, min_alpha, max_alpha,
            adaptation_window, Ref(0)
        )
    end
end

function adaptive_loss_forward!(
    expert_assignments::CuMatrix{Int32},
    router_probs::CuMatrix{T},
    adaptive_loss::AdaptiveLossScaling{T};
    kwargs...
) where T<:AbstractFloat
    
    adaptive_loss.calls_since_adaptation[] += 1
    
    # Compute loss with current alpha
    result = gpu_switch_loss_forward!(
        expert_assignments, router_probs, adaptive_loss.base_loss; kwargs...
    )
    
    # Periodically adapt alpha based on balance score
    if adaptive_loss.calls_since_adaptation[] >= adaptive_loss.adaptation_window
        current_balance = adaptive_loss.base_loss.last_balance_score[]
        
        if current_balance < adaptive_loss.target_balance_score
            # Increase alpha to encourage better balance
            new_alpha = min(
                adaptive_loss.current_alpha * (1 + adaptive_loss.adaptation_rate),
                adaptive_loss.max_alpha
            )
        else
            # Decrease alpha to avoid over-regularization
            new_alpha = max(
                adaptive_loss.current_alpha * (1 - adaptive_loss.adaptation_rate * 0.5),
                adaptive_loss.min_alpha
            )
        end
        
        if abs(new_alpha - adaptive_loss.current_alpha) > T(0.001)
            @debug "Adapting loss alpha from $(adaptive_loss.current_alpha) to $new_alpha (balance: $current_balance)"
            adaptive_loss.current_alpha = new_alpha
            adaptive_loss.base_loss.alpha = new_alpha
        end
        
        adaptive_loss.calls_since_adaptation[] = 0
    end
    
    return result
end