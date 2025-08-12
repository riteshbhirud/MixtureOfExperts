"""
GPU MoE Integration Interface

Factory functions, configuration builders, and integration utilities for seamless
GPU MoE usage. Provides clean API that matches CPU implementation while enabling
GPU acceleration. Handles CPU-to-GPU conversion and configuration optimization.
"""

# GPU MoE Configuration Builder (matches CPU API)
function create_gpu_moe_config(;
    num_experts::Int = 8,
    input_dim::Int = 768,
    hidden_dim::Int = 3072,
    output_dim::Int = 768,
    top_k::Int = 2,
    max_batch_size::Int = 512,
    use_half_precision::Bool = false,
    use_tensor_cores::Bool = true,
    enable_kernel_fusion::Bool = true,
    memory_alignment::Int = 32,
    preferred_block_size::Int = 256,
    epsilon::Float32 = Float32(1e-6),
    device_id::Union{Int, Nothing} = nothing
)
    # Set GPU device if specified
    if !isnothing(device_id)
        CUDA.device!(device_id)
    end
    
    # Validate GPU availability
    if !CUDA.functional()
        throw(ErrorException("CUDA not functional - cannot create GPU MoE configuration"))
    end
    
    # Determine precision type
    T = use_half_precision ? Float16 : Float32
    
    # Create GPU configuration
    config = GPUMoEConfig{T}(
        input_dim, hidden_dim, output_dim, num_experts, top_k;
        memory_alignment = memory_alignment,
        use_half_precision = use_half_precision,
        use_tensor_cores = use_tensor_cores,
        max_batch_size = max_batch_size,
        preferred_block_size = preferred_block_size,
        enable_kernel_fusion = enable_kernel_fusion,
        epsilon = T(epsilon)
    )
    
    return config
end

# GPU MoE Layer Factory (matches CPU API)
function create_gpu_moe_layer(
    input_dim::Int,
    hidden_dim::Int,
    output_dim::Int;
    num_experts::Int = 8,
    top_k::Int = 2,
    max_batch_size::Int = 512,
    initialization_scale::Float32 = Float32(0.02),
    alpha::Float32 = Float32(0.01),
    use_half_precision::Bool = false,
    enable_expert_parallelism::Bool = true,
    use_dynamic_batching::Bool = true,
    enable_memory_optimization::Bool = true,
    kwargs...
)
    # Create configuration
    config = create_gpu_moe_config(;
        num_experts = num_experts,
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        output_dim = output_dim,
        top_k = top_k,
        max_batch_size = max_batch_size,
        use_half_precision = use_half_precision,
        kwargs...
    )
    
    # Create MoE layer
    T = use_half_precision ? Float16 : Float32
    moe_layer = GPUMoELayer{T}(
        config;
        enable_expert_parallelism = enable_expert_parallelism,
        use_dynamic_batching = use_dynamic_batching,
        enable_memory_optimization = enable_memory_optimization,
        initialization_scale = T(initialization_scale)
    )
    
    # Set custom alpha for load balancing loss
    moe_layer.load_balance_loss.alpha = T(alpha)
    
    return moe_layer
end

# Convenient constructor that matches CPU API exactly
function create_gpu_moe_layer(config_dict::Dict{Symbol, Any})
    return create_gpu_moe_layer(
        config_dict[:input_dim],
        config_dict[:hidden_dim],
        config_dict[:output_dim];
        num_experts = get(config_dict, :num_experts, 8),
        top_k = get(config_dict, :top_k, 2),
        max_batch_size = get(config_dict, :max_batch_size, 512),
        initialization_scale = get(config_dict, :initialization_scale, Float32(0.02)),
        alpha = get(config_dict, :alpha, Float32(0.01)),
        use_half_precision = get(config_dict, :use_half_precision, false),
        enable_expert_parallelism = get(config_dict, :enable_expert_parallelism, true),
        use_dynamic_batching = get(config_dict, :use_dynamic_batching, true),
        enable_memory_optimization = get(config_dict, :enable_memory_optimization, true)
    )
end

# CPU-to-GPU conversion utilities (for future Llama2 integration)
function convert_cpu_moe_to_gpu(
    cpu_config::Any,  # CPU MoEConfig
    cpu_experts::Vector{Any},  # Vector of CPU experts
    cpu_router_weights::AbstractMatrix,
    cpu_router_bias::Union{Nothing, AbstractVector} = nothing;
    use_half_precision::Bool = false,
    max_batch_size::Int = 512,
    enable_expert_parallelism::Bool = true,
    kwargs...
)
    # Validate CPU configuration
    if !hasproperty(cpu_config, :num_experts) || !hasproperty(cpu_config, :input_dim)
        throw(ArgumentError("Invalid CPU configuration - missing required properties"))
    end
    
    # Determine precision
    T = use_half_precision ? Float16 : Float32
    
    # Create GPU configuration
    gpu_config = create_gpu_moe_config(;
        num_experts = cpu_config.num_experts,
        input_dim = cpu_config.input_dim,
        hidden_dim = cpu_config.hidden_dim,
        output_dim = cpu_config.output_dim,
        top_k = get(cpu_config, :top_k, 2),
        max_batch_size = max_batch_size,
        use_half_precision = use_half_precision,
        kwargs...
    )
    
    # Convert experts to GPU
    gpu_experts = Vector{GPUGatedExpert{T}}()
    for (i, cpu_expert) in enumerate(cpu_experts)
        gpu_expert = convert_cpu_expert_to_gpu(cpu_expert, gpu_config, i)
        push!(gpu_experts, gpu_expert)
    end
    
    # Convert router to GPU
    gpu_gating = convert_cpu_router_to_gpu(cpu_router_weights, cpu_router_bias, gpu_config)
    
    # Create load balancing loss
    gpu_loss = create_gpu_switch_loss(gpu_config)
    
    # Create MoE layer manually (bypassing factory function)
    moe_layer = create_gpu_moe_layer_from_components(
        gpu_experts, gpu_gating, gpu_loss, gpu_config;
        enable_expert_parallelism = enable_expert_parallelism
    )
    
    return moe_layer
end

# Convert individual CPU expert to GPU
function convert_cpu_expert_to_gpu(
    cpu_expert::Any,
    gpu_config::GPUMoEConfig{T},
    expert_id::Int
) where T<:AbstractFloat
    
    # Extract weights from CPU expert (assuming gated expert structure)
    if !hasproperty(cpu_expert, :w1) || !hasproperty(cpu_expert, :w2) || !hasproperty(cpu_expert, :w3)
        throw(ArgumentError("CPU expert must have w1, w2, w3 weight matrices (gated expert structure)"))
    end
    
    # Convert weights to GPU with proper precision
    w1 = CuArray{T}(cpu_expert.w1)
    w2 = CuArray{T}(cpu_expert.w2)
    w3 = CuArray{T}(cpu_expert.w3)
    
    # Convert biases if present
    b1 = hasproperty(cpu_expert, :b1) && !isnothing(cpu_expert.b1) ? CuArray{T}(cpu_expert.b1) : nothing
    b2 = hasproperty(cpu_expert, :b2) && !isnothing(cpu_expert.b2) ? CuArray{T}(cpu_expert.b2) : nothing
    b3 = hasproperty(cpu_expert, :b3) && !isnothing(cpu_expert.b3) ? CuArray{T}(cpu_expert.b3) : nothing
    
    # Create GPU expert weights
    gpu_weights = GPUGatedExpertWeights(w1, w2, w3, expert_id; b1=b1, b2=b2, b3=b3)
    
    # Create GPU expert
    return GPUGatedExpert{T}(gpu_weights, gpu_config)
end

# Convert CPU router to GPU
function convert_cpu_router_to_gpu(
    cpu_router_weights::AbstractMatrix{T1},
    cpu_router_bias::Union{Nothing, AbstractVector{T2}},
    gpu_config::GPUMoEConfig{T}
) where {T1, T2, T<:AbstractFloat}
    
    # Convert router weights
    gpu_router_weights = CuArray{T}(cpu_router_weights)
    
    # Convert router bias
    gpu_router_bias = isnothing(cpu_router_bias) ? nothing : CuArray{T}(cpu_router_bias)
    
    # Create GPU gating
    return GPUTopKGating{T}(
        gpu_config.top_k, gpu_config;
        router_weights = Array(gpu_router_weights),
        router_bias = isnothing(gpu_router_bias) ? nothing : Array(gpu_router_bias)
    )
end

# Create MoE layer from individual components
function create_gpu_moe_layer_from_components(
    experts::Vector{GPUGatedExpert{T}},
    gating::GPUTopKGating{T},
    load_balance_loss::GPUSwitchTransformerLoss{T},
    config::GPUMoEConfig{T};
    enable_expert_parallelism::Bool = true,
    use_dynamic_batching::Bool = true,
    enable_memory_optimization::Bool = true
) where T<:AbstractFloat
    
    # Validate components
    if length(experts) != config.num_experts
        throw(ArgumentError("Number of experts must match configuration"))
    end
    
    # Create routing state
    routing_state = GPURoutingState{T}(config.num_experts, config.top_k, config.max_batch_size)
    
    # Initialize workspace and statistics
    workspace = Dict{Symbol, CuArray}()
    workspace_allocated = Ref(false)
    workspace_size_bytes = Ref(Int64(0))
    
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
    
    # Create MoE layer structure manually
    moe_layer = GPUMoELayer{T}(
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

# GPU-to-CPU conversion utilities (for checkpointing/saving)
function convert_gpu_moe_to_cpu(gpu_moe_layer::GPUMoELayer{T}) where T<:AbstractFloat
    
    config = gpu_moe_layer.config
    
    # Convert experts to CPU
    cpu_experts = []
    for gpu_expert in gpu_moe_layer.experts
        cpu_expert = convert_gpu_expert_to_cpu(gpu_expert)
        push!(cpu_experts, cpu_expert)
    end
    
    # Convert router to CPU
    cpu_router_weights = Array(gpu_moe_layer.gating.router_weights)
    cpu_router_bias = isnothing(gpu_moe_layer.gating.router_bias) ? nothing : Array(gpu_moe_layer.gating.router_bias)
    
    # Extract configuration for CPU
    cpu_config = Dict(
        :num_experts => config.num_experts,
        :input_dim => config.input_dim,
        :hidden_dim => config.hidden_dim,
        :output_dim => config.output_dim,
        :top_k => config.top_k,
        :alpha => Float32(gpu_moe_layer.load_balance_loss.alpha)
    )
    
    return cpu_config, cpu_experts, cpu_router_weights, cpu_router_bias
end

# Convert individual GPU expert to CPU
function convert_gpu_expert_to_cpu(gpu_expert::GPUGatedExpert{T}) where T<:AbstractFloat
    weights = gpu_expert.weights
    
    # Convert weights to CPU
    cpu_w1 = Array(weights.w1)
    cpu_w2 = Array(weights.w2)
    cpu_w3 = Array(weights.w3)
    
    # Convert biases if present
    cpu_b1 = isnothing(weights.b1) ? nothing : Array(weights.b1)
    cpu_b2 = isnothing(weights.b2) ? nothing : Array(weights.b2)
    cpu_b3 = isnothing(weights.b3) ? nothing : Array(weights.b3)
    
    # Create CPU expert structure (this would depend on your CPU implementation)
    cpu_expert = Dict(
        :w1 => cpu_w1,
        :w2 => cpu_w2,
        :w3 => cpu_w3,
        :b1 => cpu_b1,
        :b2 => cpu_b2,
        :b3 => cpu_b3,
        :expert_id => weights.expert_id
    )
    
    return cpu_expert
end

# Input validation and preprocessing
function prepare_gpu_input(
    input::AbstractMatrix{T1},
    moe_layer::GPUMoELayer{T2}
) where {T1, T2<:AbstractFloat}
    
    # Validate dimensions
    input_dim, batch_size = size(input)
    if input_dim != moe_layer.config.input_dim
        throw(DimensionMismatch("Input dimension $input_dim does not match MoE layer input dimension $(moe_layer.config.input_dim)"))
    end
    
    # Convert to GPU with correct precision
    gpu_input = if isa(input, CuArray) && eltype(input) == T2
        input  # Already on GPU with correct type
    else
        CuArray{T2}(input)  # Convert to GPU and/or correct type
    end
    
    # Validate batch size
    if batch_size > moe_layer.config.max_batch_size
        @warn "Batch size ($batch_size) exceeds configured maximum ($(moe_layer.config.max_batch_size)). Performance may be suboptimal."
    end
    
    return gpu_input
end

# Output postprocessing
function process_gpu_output(
    gpu_output::CuMatrix{T},
    return_cpu::Bool = false
) where T<:AbstractFloat
    
    # Validate output for numerical stability
    if !gpu_check_finite(gpu_output)
        @warn "Non-finite values detected in GPU MoE output"
        gpu_clamp!(gpu_output, gpu_output, T(-1e6), T(1e6))
    end
    
    # Return CPU version if requested
    if return_cpu
        return Array(gpu_output)
    else
        return gpu_output
    end
end

# High-level inference interface (matches CPU API)
function gpu_moe_forward(
    moe_layer::GPUMoELayer{T},
    input::AbstractMatrix;
    training::Bool = false,
    return_stats::Bool = false,
    return_cpu::Bool = false
) where T<:AbstractFloat
    
    # Prepare input
    gpu_input = prepare_gpu_input(input, moe_layer)
    
    # Forward pass
    if training
        if return_stats
            gpu_output, balance_loss, stats = moe_layer(gpu_input; training=true, return_stats=true)
            output = process_gpu_output(gpu_output, return_cpu)
            return output, balance_loss, stats
        else
            gpu_output, balance_loss = moe_layer(gpu_input; training=true, return_stats=false)
            output = process_gpu_output(gpu_output, return_cpu)
            return output, balance_loss
        end
    else
        if return_stats
            gpu_output, stats = moe_layer(gpu_input; training=false, return_stats=true)
            output = process_gpu_output(gpu_output, return_cpu)
            return output, stats
        else
            gpu_output = moe_layer(gpu_input; training=false, return_stats=false)
            output = process_gpu_output(gpu_output, return_cpu)
            return output
        end
    end
end

# Batch processing utilities
function gpu_moe_forward_batch(
    moe_layer::GPUMoELayer{T},
    inputs::Vector{<:AbstractMatrix};
    training::Bool = false,
    return_stats::Bool = false,
    return_cpu::Bool = false
) where T<:AbstractFloat
    
    # Validate all inputs have same dimensions
    input_dim = size(inputs[1], 1)
    if !all(size(inp, 1) == input_dim for inp in inputs)
        throw(DimensionMismatch("All inputs must have the same input dimension"))
    end
    
    # Check if we can process as single batch
    total_batch_size = sum(size(inp, 2) for inp in inputs)
    
    if total_batch_size <= moe_layer.config.max_batch_size
        # Process as single concatenated batch
        concatenated_input = hcat(inputs...)
        return gpu_moe_forward(
            moe_layer, concatenated_input;
            training=training, return_stats=return_stats, return_cpu=return_cpu
        )
    else
        # Process individually and concatenate results
        outputs = []
        all_stats = []
        total_balance_loss = T(0)
        
        for input in inputs
            if training
                if return_stats
                    output, balance_loss, stats = gpu_moe_forward(
                        moe_layer, input; training=true, return_stats=true, return_cpu=return_cpu
                    )
                    push!(outputs, output)
                    push!(all_stats, stats)
                    total_balance_loss += balance_loss
                else
                    output, balance_loss = gpu_moe_forward(
                        moe_layer, input; training=true, return_stats=false, return_cpu=return_cpu
                    )
                    push!(outputs, output)
                    total_balance_loss += balance_loss
                end
            else
                if return_stats
                    output, stats = gpu_moe_forward(
                        moe_layer, input; training=false, return_stats=true, return_cpu=return_cpu
                    )
                    push!(outputs, output)
                    push!(all_stats, stats)
                else
                    output = gpu_moe_forward(
                        moe_layer, input; training=false, return_stats=false, return_cpu=return_cpu
                    )
                    push!(outputs, output)
                end
            end
        end
        
        # Concatenate outputs
        final_output = hcat(outputs...)
        
        # Return appropriate results
        if training
            if return_stats
                return final_output, total_balance_loss, all_stats
            else
                return final_output, total_balance_loss
            end
        else
            if return_stats
                return final_output, all_stats
            else
                return final_output
            end
        end
    end
end

# Configuration optimization utilities
function optimize_gpu_config_for_hardware(
    base_config::GPUMoEConfig{T};
    target_batch_size::Int = 256,
    memory_limit_gb::Float64 = 8.0
) where T<:AbstractFloat
    
    # Get device information
    device_info = gpu_device_info()
    available_memory_gb = CUDA.available_memory() / (1024^3)
    
    # Adjust configuration based on hardware
    optimized_config = base_config
    
    # Adjust block size based on device capabilities
    optimal_block_size = min(
        base_config.preferred_block_size,
        device_info.max_threads_per_block,
        256  # Sweet spot for most operations
    )
    
    # Adjust batch size based on memory constraints
    estimated_memory_gb = estimate_memory_requirements(base_config) / 1024
    if estimated_memory_gb > min(available_memory_gb * 0.8, memory_limit_gb)
        # Reduce batch size to fit memory
        reduction_factor = min(available_memory_gb * 0.8, memory_limit_gb) / estimated_memory_gb
        suggested_batch_size = max(32, Int(floor(target_batch_size * reduction_factor)))
        
        @warn "Reducing max_batch_size from $(base_config.max_batch_size) to $suggested_batch_size due to memory constraints"
        
        # Create new config with adjusted parameters
        optimized_config = GPUMoEConfig{T}(
            base_config.input_dim, base_config.hidden_dim, base_config.output_dim,
            base_config.num_experts, base_config.top_k;
            memory_alignment = base_config.memory_alignment,
            use_half_precision = base_config.use_half_precision,
            use_tensor_cores = base_config.use_tensor_cores,
            max_batch_size = suggested_batch_size,
            preferred_block_size = optimal_block_size,
            use_shared_memory = base_config.use_shared_memory,
            enable_kernel_fusion = base_config.enable_kernel_fusion,
            epsilon = base_config.epsilon
        )
    end
    
    return optimized_config
end

# Device management utilities
function set_gpu_device_for_moe(device_id::Int)
    if device_id < 0 || device_id >= CUDA.ndevices()
        throw(ArgumentError("Invalid device ID: $device_id. Available devices: 0-$(CUDA.ndevices()-1)"))
    end
    
    CUDA.device!(device_id)
    
    # Verify device is functional
    if !CUDA.functional()
        throw(ErrorException("GPU device $device_id is not functional"))
    end
    
    # Warm up device
    test_array = CUDA.zeros(Float32, 10, 10)
    CUDA.synchronize()
    
    @info "Set GPU device $device_id for MoE computation"
    
    return device_id
end

function get_optimal_gpu_device()
    if !CUDA.functional()
        throw(ErrorException("CUDA not functional"))
    end
    
    num_devices = CUDA.ndevices()
    if num_devices == 0
        throw(ErrorException("No CUDA devices available"))
    end
    
    if num_devices == 1
        return 0
    end
    
    # Select device with most available memory
    best_device = 0
    max_memory = 0
    
    for device_id in 0:(num_devices-1)
        CUDA.device!(device_id)
        available_memory = CUDA.available_memory()
        
        if available_memory > max_memory
            max_memory = available_memory
            best_device = device_id
        end
    end
    
    CUDA.device!(best_device)
    return best_device
end

# Diagnostic and debugging utilities
function diagnose_gpu_moe_setup(moe_layer::GPUMoELayer{T}) where T<:AbstractFloat
    diagnosis = Dict{String, Any}()
    
    # Hardware information
    diagnosis["device_info"] = gpu_device_info()
    diagnosis["memory_info"] = gpu_memory_info()
    
    # Configuration analysis
    diagnosis["config"] = Dict(
        "num_experts" => moe_layer.config.num_experts,
        "top_k" => moe_layer.config.top_k,
        "dimensions" => (moe_layer.config.input_dim, moe_layer.config.hidden_dim, moe_layer.config.output_dim),
        "max_batch_size" => moe_layer.config.max_batch_size,
        "use_half_precision" => moe_layer.config.use_half_precision,
        "estimated_memory_mb" => estimate_memory_requirements(moe_layer.config)
    )
    
    # Component status
    diagnosis["components"] = Dict(
        "experts_allocated" => length(moe_layer.experts),
        "gating_allocated" => !isnothing(moe_layer.gating),
        "loss_allocated" => !isnothing(moe_layer.load_balance_loss),
        "workspace_allocated" => moe_layer.workspace_allocated[]
    )
    
    # Performance settings
    diagnosis["performance_settings"] = Dict(
        "enable_expert_parallelism" => moe_layer.enable_expert_parallelism,
        "use_dynamic_batching" => moe_layer.use_dynamic_batching,
        "enable_memory_optimization" => moe_layer.enable_memory_optimization
    )
    
    # Memory usage
    if moe_layer.workspace_allocated[]
        diagnosis["memory_usage"] = Dict(
            "workspace_size_mb" => moe_layer.workspace_size_bytes[] / (1024^2),
            "available_memory_mb" => CUDA.available_memory() / (1024^2)
        )
    end
    
    return diagnosis
end

# Performance testing utilities
function test_gpu_moe_performance(
    moe_layer::GPUMoELayer{T},
    batch_sizes::Vector{Int} = [32, 64, 128, 256, 512];
    num_warmup::Int = 3,
    num_benchmark::Int = 10
) where T<:AbstractFloat
    
    results = Dict{Int, Dict{String, Float64}}()
    
    for batch_size in batch_sizes
        if batch_size > moe_layer.config.max_batch_size
            @warn "Skipping batch size $batch_size (exceeds max_batch_size $(moe_layer.config.max_batch_size))"
            continue
        end
        
        @info "Testing GPU MoE performance with batch size $batch_size"
        
        try
            # Create test input
            input = CUDA.randn(T, moe_layer.config.input_dim, batch_size)
            
            # Warmup
            for _ in 1:num_warmup
                output = moe_layer(input)
                CUDA.synchronize()
            end
            
            # Reset performance stats
            reset_moe_performance_stats!(moe_layer)
            
            # Benchmark
            times = Float64[]
            for _ in 1:num_benchmark
                start_time = time_ns()
                output = moe_layer(input)
                CUDA.synchronize()
                end_time = time_ns()
                push!(times, (end_time - start_time) / 1e6)  # Convert to ms
            end
            
            # Calculate throughput
            tokens_per_sec = (batch_size * num_benchmark) / (sum(times) / 1000)
            
            results[batch_size] = Dict(
                "mean_time_ms" => Statistics.mean(times),
                "std_time_ms" => Statistics.std(times),
                "min_time_ms" => minimum(times),
                "max_time_ms" => maximum(times),
                "tokens_per_sec" => tokens_per_sec,
                "memory_usage_mb" => moe_layer.workspace_size_bytes[] / (1024^2)
            )
            
        catch e
            @warn "Performance test failed for batch size $batch_size" exception=e
        end
    end
    
    return results
end