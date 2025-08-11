"""
GPU MoE Integration Interface and Factory Functions

High-level interface for creating, configuring, and using GPU MoE layers with
seamless integration to existing CPU implementations. Provides factory functions,
conversion utilities, and standardized APIs for research and production use.
"""

"""
    GPUMoE

Main GPU MoE interface that matches the CPU MoE API for seamless integration.
Provides the same functionality as CPU MoELayer but with GPU acceleration.
"""
struct GPUMoE{T<:AbstractFloat}
    layer::GPUMoELayer{T}
    
    original_config::Dict{Symbol, Any}
    conversion_metadata::Dict{Symbol, Any}
    
    is_optimized::Ref{Bool}
    target_batch_size::Ref{Int}
    
    training_mode::Ref{Bool}
    step_counter::Ref{Int}
    
    function GPUMoE{T}(layer::GPUMoELayer{T}; 
                      original_config::Dict{Symbol, Any} = Dict{Symbol, Any}(),
                      conversion_metadata::Dict{Symbol, Any} = Dict{Symbol, Any}()) where T<:AbstractFloat
        
        return new{T}(
            layer,
            original_config,
            conversion_metadata,
            Ref(false), Ref(128), 
            Ref(false), Ref(0)   
        )
    end
end

GPUMoE(layer::GPUMoELayer{T}; kwargs...) where T = GPUMoE{T}(layer; kwargs...)

"""
    (moe::GPUMoE)(input; training=false, return_loss=false, return_stats=false)

Forward pass through GPU MoE with CPU-compatible interface.
"""
function (moe::GPUMoE{T})(input::AbstractMatrix{T}; 
                         training::Bool = false,
                         return_loss::Bool = false,
                         return_stats::Bool = false) where T<:AbstractFloat
    
    moe.training_mode[] = training
    moe.step_counter[] += 1
    
    gpu_input = isa(input, CuMatrix) ? input : CuArray{T}(input)
    
    batch_size = size(gpu_input, 2)
    gpu_output = gpu_zeros(T, moe.layer.config.output_dim, batch_size)
    
    result = gpu_moe_forward!(
        gpu_output, gpu_input, moe.layer;
        training = training,
        return_loss = return_loss,
        return_stats = return_stats
    )
    
    if !isa(input, CuMatrix)
        if return_loss && return_stats
            output, loss, stats = result
            return Array(output), loss, stats
        elseif return_loss
            output, loss = result
            return Array(output), loss
        elseif return_stats
            output, stats = result
            return Array(output), stats
        else
            return Array(result)
        end
    else
        return result
    end
end

"""
    create_gpu_moe(; kwargs...)

Factory function to create GPU MoE with sensible defaults.
"""
function create_gpu_moe(;
    input_dim::Int,
    hidden_dim::Int,
    output_dim::Int,
    num_experts::Int = 8,
    top_k::Int = 2,
    expert_type::Symbol = :gated,
    balance_alpha::Float32 = 0.01f0,
    use_mixed_precision::Bool = false,
    memory_efficient::Bool = true,
    max_batch_size::Int = 512,
    kwargs...)
    
    @info "Creating GPU MoE: $num_experts experts, top_k=$top_k, dims=($input_dim,$hidden_dim,$output_dim)"
    
    if !CUDA.functional()
        throw(ErrorException("CUDA not functional - cannot create GPU MoE"))
    end
    
    batch_config = GPUBatchConfig{Float32}(;
        max_batch_size = max_batch_size,
        kwargs...
    )
    
    layer_config = GPUMoELayerConfig{Float32}(;
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        output_dim = output_dim,
        num_experts = num_experts,
        top_k = top_k,
        expert_type = expert_type,
        balance_alpha = balance_alpha,
        use_mixed_precision = use_mixed_precision,
        memory_efficient = memory_efficient,
        batch_config = batch_config,
        kwargs...
    )
    
    gpu_layer = create_gpu_moe_layer(layer_config)
    
    original_config = Dict{Symbol, Any}(
        :input_dim => input_dim,
        :hidden_dim => hidden_dim,
        :output_dim => output_dim,
        :num_experts => num_experts,
        :top_k => top_k,
        :expert_type => expert_type,
        :balance_alpha => balance_alpha
    )
    
    # Fixed version detection
    conversion_metadata = Dict{Symbol, Any}(
        :creation_time => time(),
        :cuda_runtime_version => CUDA.runtime_version(),
        :cuda_driver_version => CUDA.driver_version(),
        :device_name => CUDA.name(CUDA.device()),
        :library_version => "1.0.0"
    )
    
    gpu_moe = GPUMoE{Float32}(gpu_layer; 
                             original_config = original_config,
                             conversion_metadata = conversion_metadata)
    
    @info "GPU MoE created successfully"
    return gpu_moe
end

"""
    convert_cpu_moe_to_gpu(cpu_moe_layer; kwargs...)

Convert CPU MoE layer to GPU implementation.
"""
function convert_cpu_moe_to_gpu(cpu_moe_layer; 
                               use_mixed_precision::Bool = false,
                               memory_efficient::Bool = true,
                               preserve_weights::Bool = true,
                               kwargs...)
    
    @info "Converting CPU MoE layer to GPU..."
    
    # Extract configuration from CPU layer
    # Note: This assumes CPU layer has similar structure to GPU implementation
    # Actual implementation would depend on CPU MoE layer structure
    
    if !hasfield(typeof(cpu_moe_layer), :config)
        throw(ArgumentError("CPU MoE layer must have config field for conversion"))
    end
    
    cpu_config = cpu_moe_layer.config
    
    batch_config = GPUBatchConfig{Float32}(; kwargs...)
    
    gpu_layer_config = GPUMoELayerConfig{Float32}(;
        input_dim = cpu_config.input_dim,
        hidden_dim = cpu_config.hidden_dim,
        output_dim = cpu_config.output_dim,
        num_experts = cpu_config.num_experts,
        top_k = cpu_config.top_k,
        expert_type = :gated,  
        balance_alpha = Float32(get(cpu_config, :balance_alpha, 0.01)),
        use_mixed_precision = use_mixed_precision,
        memory_efficient = memory_efficient,
        batch_config = batch_config
    )
    
    gpu_layer = create_gpu_moe_layer(gpu_layer_config)
    
    if preserve_weights && hasfield(typeof(cpu_moe_layer), :experts)
        @info "Transferring weights from CPU to GPU..."
        transfer_weights_cpu_to_gpu!(gpu_layer, cpu_moe_layer)
    end
    
    original_config = Dict{Symbol, Any}(
        :source => "cpu_conversion",
        :cpu_type => string(typeof(cpu_moe_layer)),
        :input_dim => cpu_config.input_dim,
        :hidden_dim => cpu_config.hidden_dim,
        :output_dim => cpu_config.output_dim,
        :num_experts => cpu_config.num_experts,
        :top_k => cpu_config.top_k
    )
    
    conversion_metadata = Dict{Symbol, Any}(
        :conversion_time => time(),
        :preserve_weights => preserve_weights,
        :source_type => "cpu_moe_layer",
        :target_type => "gpu_moe_layer"
    )
    
    gpu_moe = GPUMoE{Float32}(gpu_layer;
                             original_config = original_config,
                             conversion_metadata = conversion_metadata)
    
    @info "CPU to GPU conversion completed"
    return gpu_moe
end

"""
    transfer_weights_cpu_to_gpu!(gpu_layer, cpu_layer)

Transfer weights from CPU MoE layer to GPU MoE layer.
"""
function transfer_weights_cpu_to_gpu!(gpu_layer::GPUMoELayer{T}, cpu_layer) where T<:AbstractFloat
    
    @info "Transferring expert weights..."
    
    if hasfield(typeof(cpu_layer), :experts) && length(cpu_layer.experts) == length(gpu_layer.experts)
        for (i, (gpu_expert, cpu_expert)) in enumerate(zip(gpu_layer.experts, cpu_layer.experts))
            
            if hasfield(typeof(cpu_expert), :w1)
                copyto!(gpu_expert.weights.w1, T.(cpu_expert.w1))
            end
            if hasfield(typeof(cpu_expert), :w2)
                copyto!(gpu_expert.weights.w2, T.(cpu_expert.w2))
            end
            if hasfield(typeof(cpu_expert), :w3)
                copyto!(gpu_expert.weights.w3, T.(cpu_expert.w3))
            end
            
            if hasfield(typeof(cpu_expert), :b1) && !isnothing(cpu_expert.b1) && !isnothing(gpu_expert.weights.b1)
                copyto!(gpu_expert.weights.b1, T.(cpu_expert.b1))
            end
            if hasfield(typeof(cpu_expert), :b2) && !isnothing(cpu_expert.b2) && !isnothing(gpu_expert.weights.b2)
                copyto!(gpu_expert.weights.b2, T.(cpu_expert.b2))
            end
            if hasfield(typeof(cpu_expert), :b3) && !isnothing(cpu_expert.b3) && !isnothing(gpu_expert.weights.b3)
                copyto!(gpu_expert.weights.b3, T.(cpu_expert.b3))
            end
            
            @debug "Transferred weights for expert $i"
        end
    end
    
    if hasfield(typeof(cpu_layer), :router) && hasfield(typeof(cpu_layer.router), :weight)
        router_weights = cpu_layer.router.weight
        if hasfield(typeof(router_weights), :weight)
            copyto!(gpu_layer.gating.router_weights, T.(router_weights.weight))
        else
            copyto!(gpu_layer.gating.router_weights, T.(router_weights))
        end
        @debug "Transferred router weights"
    end
    
    CUDA.synchronize()
    @info "Weight transfer completed"
end

"""
    optimize_gpu_moe!(moe, target_batch_size)

Optimize GPU MoE for specific usage patterns.
"""
function optimize_gpu_moe!(moe::GPUMoE{T}, target_batch_size::Int) where T<:AbstractFloat
    
    @info "Optimizing GPU MoE for batch size $target_batch_size"
    
    optimize_moe_layer!(moe.layer, target_batch_size)
    
    moe.is_optimized[] = true
    moe.target_batch_size[] = target_batch_size
    
    @info "GPU MoE optimization completed"
end

"""
    benchmark_gpu_moe(moe, batch_sizes; kwargs...)

Benchmark GPU MoE performance across different configurations.
"""
function benchmark_gpu_moe(moe::GPUMoE{T}, batch_sizes::Vector{Int}; 
                          compare_with_cpu::Bool = false,
                          include_memory_stats::Bool = true,
                          kwargs...) where T<:AbstractFloat
    
    @info "Benchmarking GPU MoE across $(length(batch_sizes)) batch sizes"
    
    gpu_results = benchmark_moe_layer(moe.layer, batch_sizes; kwargs...)
    
    benchmark_summary = Dict{String, Any}(
        "gpu_results" => gpu_results,
        "device_info" => Dict(
            "device_name" => CUDA.name(CUDA.device()),
            "compute_capability" => string(CUDA.capability(CUDA.device())),
            "total_memory_gb" => CUDA.totalmem(CUDA.device()) / (1024^3)
        ),
        "configuration" => Dict(
            "num_experts" => moe.layer.config.num_experts,
            "top_k" => moe.layer.config.top_k,
            "input_dim" => moe.layer.config.input_dim,
            "hidden_dim" => moe.layer.config.hidden_dim,
            "output_dim" => moe.layer.config.output_dim
        )
    )
    
    if include_memory_stats
        benchmark_summary["memory_stats"] = get_memory_pool_statistics()
    end
    
    best_batch_size = 0
    best_throughput = 0.0
    
    for (batch_size, stats) in gpu_results
        throughput = stats["throughput_tokens_per_sec"]
        if throughput > best_throughput
            best_throughput = throughput
            best_batch_size = batch_size
        end
    end
    
    benchmark_summary["performance_analysis"] = Dict(
        "best_batch_size" => best_batch_size,
        "best_throughput_tokens_per_sec" => best_throughput,
        "throughput_range" => [minimum(r["throughput_tokens_per_sec"] for r in values(gpu_results)),
                              maximum(r["throughput_tokens_per_sec"] for r in values(gpu_results))]
    )
    
    @info "Benchmark completed - Best performance: $best_throughput tokens/sec at batch size $best_batch_size"
    
    return benchmark_summary
end

"""
    get_gpu_moe_info(moe)

Get comprehensive information about GPU MoE configuration and state.
"""
function get_gpu_moe_info(moe::GPUMoE{T}) where T<:AbstractFloat
    
    info = Dict{String, Any}(
        "type" => "GPUMoE",
        "precision" => string(T),
        "is_optimized" => moe.is_optimized[],
        "target_batch_size" => moe.target_batch_size[],
        "training_mode" => moe.training_mode[],
        "step_counter" => moe.step_counter[],
        "original_config" => copy(moe.original_config),
        "conversion_metadata" => copy(moe.conversion_metadata)
    )
    
    layer_config = moe.layer.config
    info["layer_config"] = Dict(
        "input_dim" => layer_config.input_dim,
        "hidden_dim" => layer_config.hidden_dim,
        "output_dim" => layer_config.output_dim,
        "num_experts" => layer_config.num_experts,
        "top_k" => layer_config.top_k,
        "expert_type" => layer_config.expert_type,
        "balance_alpha" => layer_config.balance_alpha,
        "use_mixed_precision" => layer_config.use_mixed_precision,
        "enable_kernel_fusion" => layer_config.enable_kernel_fusion,
        "memory_efficient" => layer_config.memory_efficient
    )
    
    info["performance_stats"] = Dict(
        "forward_calls" => moe.layer.forward_calls[],
        "total_forward_time_sec" => moe.layer.total_forward_time[],
        "avg_forward_time_ms" => moe.layer.forward_calls[] > 0 ? 
                                (moe.layer.total_forward_time[] / moe.layer.forward_calls[]) * 1000 : 0.0
    )
    
    info["device_info"] = Dict(
        "device_name" => CUDA.name(CUDA.device()),
        "device_id" => CUDA.deviceid(CUDA.device()),
        "compute_capability" => string(CUDA.capability(CUDA.device())),
        "total_memory_gb" => CUDA.totalmem(CUDA.device()) / (1024^3),
        "available_memory_gb" => CUDA.available_memory() / (1024^3)
    )
    
    return info
end

"""
    validate_gpu_moe(moe)

Comprehensive validation of GPU MoE state and functionality.
"""
function validate_gpu_moe(moe::GPUMoE{T}) where T<:AbstractFloat
    
    @info "Validating GPU MoE..."
    
    layer_validation = validate_moe_layer(moe.layer)
    
    interface_validation = Dict{String, Any}()
    
    try
        test_input = CUDA.randn(T, moe.layer.config.input_dim, 16)
        test_output = moe(test_input)
        
        if size(test_output) == (moe.layer.config.output_dim, 16)
            interface_validation["forward_pass"] = true
        else
            interface_validation["forward_pass"] = false
            @error "Forward pass output size mismatch"
        end
    catch e
        interface_validation["forward_pass"] = false
        @error "Forward pass failed" exception=e
    end
    
    try
        test_input = CUDA.randn(T, moe.layer.config.input_dim, 8)
        test_output, test_loss = moe(test_input; training=true, return_loss=true)
        
        if isa(test_loss, T) && isfinite(test_loss)
            interface_validation["training_mode"] = true
        else
            interface_validation["training_mode"] = false
            @error "Training mode loss computation failed"
        end
    catch e
        interface_validation["training_mode"] = false
        @error "Training mode failed" exception=e
    end
    
    try
        cpu_input = randn(T, moe.layer.config.input_dim, 4)
        cpu_output = moe(cpu_input)
        
        if isa(cpu_output, Array) && size(cpu_output) == (moe.layer.config.output_dim, 4)
            interface_validation["cpu_gpu_conversion"] = true
        else
            interface_validation["cpu_gpu_conversion"] = false
            @error "CPU/GPU conversion failed"
        end
    catch e
        interface_validation["cpu_gpu_conversion"] = false
        @error "CPU/GPU conversion failed" exception=e
    end
    
    overall_valid = layer_validation["overall_valid"] && 
                   all(values(interface_validation))
    
    validation_result = Dict{String, Any}(
        "overall_valid" => overall_valid,
        "layer_validation" => layer_validation,
        "interface_validation" => interface_validation
    )
    
    if overall_valid
        @info "GPU MoE validation passed"
    else
        @error "GPU MoE validation failed"
    end
    
    return validation_result
end

"""
    save_gpu_moe(moe, filename)

Save GPU MoE to disk with comprehensive metadata.
"""
function save_gpu_moe(moe::GPUMoE{T}, filename::String) where T<:AbstractFloat
    
    @info "Saving GPU MoE to $filename"
    
    save_data = Dict{String, Any}(
        "type" => "GPUMoE",
        "version" => "1.0.0",
        "precision" => string(T),
        "save_timestamp" => string(now()),
        "original_config" => moe.original_config,
        "conversion_metadata" => moe.conversion_metadata
    )
    
    layer_config = moe.layer.config
    save_data["layer_config"] = Dict(
        "input_dim" => layer_config.input_dim,
        "hidden_dim" => layer_config.hidden_dim,
        "output_dim" => layer_config.output_dim,
        "num_experts" => layer_config.num_experts,
        "top_k" => layer_config.top_k,
        "expert_type" => layer_config.expert_type,
        "balance_alpha" => layer_config.balance_alpha,
        "use_mixed_precision" => layer_config.use_mixed_precision,
        "enable_kernel_fusion" => layer_config.enable_kernel_fusion,
        "memory_efficient" => layer_config.memory_efficient
    )
    
    expert_weights = []
    for (i, expert) in enumerate(moe.layer.experts)
        expert_data = Dict(
            "expert_id" => i,
            "w1" => Array(expert.weights.w1),
            "w2" => Array(expert.weights.w2),
            "w3" => Array(expert.weights.w3)
        )
        
        if !isnothing(expert.weights.b1)
            expert_data["b1"] = Array(expert.weights.b1)
        end
        if !isnothing(expert.weights.b2)
            expert_data["b2"] = Array(expert.weights.b2)
        end
        if !isnothing(expert.weights.b3)
            expert_data["b3"] = Array(expert.weights.b3)
        end
        
        push!(expert_weights, expert_data)
    end
    save_data["expert_weights"] = expert_weights
    
    save_data["router_weights"] = Array(moe.layer.gating.router_weights)
    if !isnothing(moe.layer.gating.router_bias)
        save_data["router_bias"] = Array(moe.layer.gating.router_bias)
    end
    
    save_data["performance_stats"] = Dict(
        "forward_calls" => moe.layer.forward_calls[],
        "total_forward_time" => moe.layer.total_forward_time[],
        "is_optimized" => moe.is_optimized[],
        "target_batch_size" => moe.target_batch_size[]
    )
    
    open(filename, "w") do f
        serialize(f, save_data)
    end
    
    @info "GPU MoE saved successfully"
end

"""
    load_gpu_moe(filename)

Load GPU MoE from disk.
"""
function load_gpu_moe(filename::String)
    
    @info "Loading GPU MoE from $filename"
    
    if !isfile(filename)
        throw(ArgumentError("File not found: $filename"))
    end
    
    save_data = open(filename, "r") do f
        deserialize(f)
    end
    
    if get(save_data, "type", "") != "GPUMoE"
        throw(ArgumentError("Invalid save file: not a GPUMoE"))
    end
    
    layer_config_data = save_data["layer_config"]
    precision_str = get(save_data, "precision", "Float32")
    T = precision_str == "Float32" ? Float32 : Float64
    
    batch_config = GPUBatchConfig{T}()
    
    layer_config = GPUMoELayerConfig{T}(;
        input_dim = layer_config_data["input_dim"],
        hidden_dim = layer_config_data["hidden_dim"],
        output_dim = layer_config_data["output_dim"],
        num_experts = layer_config_data["num_experts"],
        top_k = layer_config_data["top_k"],
        expert_type = Symbol(layer_config_data["expert_type"]),
        balance_alpha = T(layer_config_data["balance_alpha"]),
        use_mixed_precision = layer_config_data["use_mixed_precision"],
        enable_kernel_fusion = layer_config_data["enable_kernel_fusion"],
        memory_efficient = layer_config_data["memory_efficient"],
        batch_config = batch_config
    )
    
    gpu_layer = create_gpu_moe_layer(layer_config)
    
    expert_weights_data = save_data["expert_weights"]
    for (i, expert_data) in enumerate(expert_weights_data)
        expert = gpu_layer.experts[i]
        
        copyto!(expert.weights.w1, T.(expert_data["w1"]))
        copyto!(expert.weights.w2, T.(expert_data["w2"]))
        copyto!(expert.weights.w3, T.(expert_data["w3"]))
        
        if haskey(expert_data, "b1") && !isnothing(expert.weights.b1)
            copyto!(expert.weights.b1, T.(expert_data["b1"]))
        end
        if haskey(expert_data, "b2") && !isnothing(expert.weights.b2)
            copyto!(expert.weights.b2, T.(expert_data["b2"]))
        end
        if haskey(expert_data, "b3") && !isnothing(expert.weights.b3)
            copyto!(expert.weights.b3, T.(expert_data["b3"]))
        end
    end
    
    copyto!(gpu_layer.gating.router_weights, T.(save_data["router_weights"]))
    if haskey(save_data, "router_bias") && !isnothing(gpu_layer.gating.router_bias)
        copyto!(gpu_layer.gating.router_bias, T.(save_data["router_bias"]))
    end
    
    CUDA.synchronize()
    
    original_config = get(save_data, "original_config", Dict{Symbol, Any}())
    conversion_metadata = get(save_data, "conversion_metadata", Dict{Symbol, Any}())
    
    gpu_moe = GPUMoE{T}(gpu_layer;
                       original_config = original_config,
                       conversion_metadata = conversion_metadata)
    
    if haskey(save_data, "performance_stats")
        perf_stats = save_data["performance_stats"]
        gpu_moe.is_optimized[] = get(perf_stats, "is_optimized", false)
        gpu_moe.target_batch_size[] = get(perf_stats, "target_batch_size", 128)
    end
    
    @info "GPU MoE loaded successfully"
    return gpu_moe
end