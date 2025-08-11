using CUDA
using LinearAlgebra
using Printf

include("src/cuda/CUDAMoE.jl")
using .CUDAMoE

include("src/MixtureOfExperts.jl")
using .MixtureOfExperts

"""
Comprehensive MoE CPU vs GPU Performance Benchmark Suite

This benchmark suite compares CPU and GPU MoE implementations under realistic 
large language model conditions with actual model dimensions and workloads.

IMPORTANT: This version uses explicit module namespacing:
- MixtureOfExperts.XXX for CPU implementation 
- CUDAMoE.XXX for GPU implementation
"""

using Statistics
using Printf
using CUDA
using LinearAlgebra
using Random
using JSON
using Dates

# Realistic model configurations based on actual LLMs
const MODEL_CONFIGS = Dict(
    "Llama2-7B-style" => Dict(
        :input_dim => 4096,
        :hidden_dim => 11008,
        :output_dim => 4096,
        :description => "Llama2-7B FFN dimensions"
    ),
    "Llama2-13B-style" => Dict(
        :input_dim => 5120,
        :hidden_dim => 13824,
        :output_dim => 5120,
        :description => "Llama2-13B FFN dimensions"
    ),
    "Llama2-70B-style" => Dict(
        :input_dim => 8192,
        :hidden_dim => 28672,
        :output_dim => 8192,
        :description => "Llama2-70B FFN dimensions"
    ),
    "GPT-3-style" => Dict(
        :input_dim => 12288,
        :hidden_dim => 49152,
        :output_dim => 12288,
        :description => "GPT-3 175B FFN dimensions"
    )
)

# Realistic test configurations
const TEST_CONFIGS = [
    # Format: (batch_size, seq_length, description)
    (1, 512, "Single sequence, moderate length"),
    (1, 1024, "Single sequence, long context"),
    (1, 2048, "Single sequence, very long context"),
    (8, 512, "Small batch, moderate length"),
    (8, 1024, "Small batch, long context"),
    (16, 512, "Medium batch, moderate length"),
    (16, 1024, "Medium batch, long context"),
    (32, 512, "Large batch, moderate length"),
    (64, 256, "Very large batch, short sequences"),
    (64, 512, "Very large batch, moderate length")
]

# Expert configurations to test
const EXPERT_CONFIGS = [
    (8, 2, "Standard 8 experts, top-2"),
    (16, 2, "Medium 16 experts, top-2"), 
    (32, 2, "Large 32 experts, top-2"),
    (64, 2, "Very large 64 experts, top-2")
]

# Benchmark results structure
mutable struct BenchmarkResults
    cpu_results::Dict{String, Any}
    gpu_results::Dict{String, Any}
    comparison_metrics::Dict{String, Any}
    system_info::Dict{String, Any}
    test_metadata::Dict{String, Any}
end

"""
Create realistic input data that mimics actual language model activations
"""
function create_realistic_input(input_dim::Int, batch_size::Int, seq_length::Int; 
                               activation_pattern::Symbol = :transformer)
    total_tokens = batch_size * seq_length
    
    if activation_pattern == :transformer
        # Mimic transformer activations: mostly small values with occasional large ones
        # Based on analysis of actual transformer intermediate activations
        base_activations = randn(Float32, input_dim, total_tokens) * 0.5f0
        
        # Add some structured patterns common in transformers
        for i in 1:input_dim
            # Add periodic patterns (positional encoding effects)
            phase = 2π * i / input_dim
            for j in 1:total_tokens
                pos_effect = 0.1f0 * sin(phase + 0.01f0 * j)
                base_activations[i, j] += pos_effect
            end
        end
        
        # Add sparse high-magnitude activations (attention effects)
        high_activation_prob = 0.05f0  # 5% of values are high magnitude
        for i in 1:input_dim, j in 1:total_tokens
            if rand() < high_activation_prob
                base_activations[i, j] += rand([-3.0f0, 3.0f0])  # High magnitude
            end
        end
        
        # Ensure numerical stability
        clamp!(base_activations, -10.0f0, 10.0f0)
        
    else
        # Simple random for comparison
        base_activations = randn(Float32, input_dim, total_tokens)
    end
    
    return base_activations
end

"""
Setup CPU MoE layer with realistic configuration using MixtureOfExperts module
"""
function setup_cpu_moe(model_config::Dict, num_experts::Int, top_k::Int)
    println("        🖥️  Creating CPU MoE using MixtureOfExperts module...")
    
    # Use explicit CPU module namespacing
    config = MixtureOfExperts.create_moe_config(
        num_experts = num_experts,
        expert_type = :gated,  # Match GPU implementation
        input_dim = model_config[:input_dim],
        hidden_dim = model_config[:hidden_dim],
        output_dim = model_config[:output_dim],
        top_k = top_k,
        gate_type = MixtureOfExperts.TopKGating(top_k),
        balance_loss = MixtureOfExperts.SwitchTransformerLoss(0.01f0),
        noise_scale = 0.0f0,  # Disable for fair comparison
        use_fp32_router = true
    )
    
    cpu_moe = MixtureOfExperts.MoELayer(config)
    println("        ✅ CPU MoE created successfully")
    return cpu_moe
end

"""
Setup GPU MoE layer with matching configuration using CUDAMoE module
"""
function setup_gpu_moe(model_config::Dict, num_experts::Int, top_k::Int)
    println("        🚀 Creating GPU MoE using CUDAMoE module...")
    
    # Check CUDA availability
    if !CUDA.functional()
        error("CUDA not available for GPU benchmarking")
    end
    
    # Use the fixed create_gpu_moe function from integration.jl
    gpu_moe = CUDAMoE.create_gpu_moe(
        input_dim = model_config[:input_dim],
        hidden_dim = model_config[:hidden_dim], 
        output_dim = model_config[:output_dim],
        num_experts = num_experts,
        top_k = top_k,
        expert_type = :gated,
        balance_alpha = 0.01f0,
        use_mixed_precision = false,
        memory_efficient = true,
        max_batch_size = 1024
    )
    
    println("        ✅ GPU MoE created successfully")
    return gpu_moe
end

"""
Benchmark CPU MoE performance using MixtureOfExperts implementation
"""
function benchmark_cpu_moe(cpu_moe, input_data, num_warmup::Int, num_benchmark::Int; 
                          training::Bool = false)
    println("        📊 Running CPU benchmark (MixtureOfExperts)...")
    
    batch_size, seq_length = size(input_data, 2), 1  # Flatten tokens
    
    # Warmup using CPU implementation
    for i in 1:num_warmup
        println("           Warmup $i/$num_warmup...")
        output, loss = cpu_moe(input_data; training=training)
        GC.gc()  # Force garbage collection between runs
    end
    
    # Benchmark
    timings = Float64[]
    memory_usage = Int64[]
    losses = Float32[]
    
    println("        🏃 Running $num_benchmark benchmark iterations...")
    for i in 1:num_benchmark
        # Memory before
        GC.gc()
        mem_before = Base.gc_live_bytes()
        
        # Timing
        start_time = time_ns()
        output, loss = cpu_moe(input_data; training=training)
        end_time = time_ns()
        
        # Memory after  
        mem_after = Base.gc_live_bytes()
        
        # Record metrics
        push!(timings, (end_time - start_time) / 1e6)  # Convert to ms
        push!(memory_usage, mem_after - mem_before)
        push!(losses, loss)
        
        # Clean up
        output = nothing
        GC.gc()
        
        if i % 5 == 0
            println("           Completed $i/$num_benchmark iterations...")
        end
    end
    
    println("        ✅ CPU benchmark completed")
    
    return Dict(
        "implementation" => "MixtureOfExperts (CPU)",
        "timings_ms" => timings,
        "memory_usage_bytes" => memory_usage,
        "losses" => losses,
        "mean_time_ms" => mean(timings),
        "std_time_ms" => std(timings),
        "min_time_ms" => minimum(timings),
        "max_time_ms" => maximum(timings),
        "mean_memory_mb" => mean(memory_usage) / (1024^2),
        "peak_memory_mb" => maximum(memory_usage) / (1024^2),
        "mean_loss" => mean(losses),
        "tokens_per_second" => (batch_size * seq_length) / (mean(timings) / 1000),
        "throughput_tokens_per_ms" => (batch_size * seq_length) / mean(timings)
    )
end

"""
Benchmark GPU MoE performance using CUDAMoE implementation
"""
function benchmark_gpu_moe(gpu_moe, input_data, num_warmup::Int, num_benchmark::Int;
                          training::Bool = false)
    println("        📊 Running GPU benchmark (CUDAMoE)...")
    
    batch_size, seq_length = size(input_data, 2), 1  # Flatten tokens
    
    # Convert input to GPU
    gpu_input = CuArray{Float32}(input_data)
    println("        📤 Input data transferred to GPU")
    
    # Warmup using GPU implementation
    for i in 1:num_warmup
        println("           Warmup $i/$num_warmup...")
        if training
            result = gpu_moe(gpu_input; training=true, return_loss=true)
        else
            result = gpu_moe(gpu_input; training=false)
        end
        CUDA.synchronize()
        GC.gc()
        CUDA.reclaim()
    end
    
    # Benchmark
    timings = Float64[]
    memory_usage = Int64[]
    losses = Float32[]
    
    println("        🏃 Running $num_benchmark benchmark iterations...")
    for i in 1:num_benchmark
        # GPU memory before
        CUDA.reclaim()
        mem_before = CUDA.used_memory()
        
        # Timing
        CUDA.synchronize()
        start_time = time_ns()
        
        if training
            result = gpu_moe(gpu_input; training=true, return_loss=true)
            if isa(result, Tuple)
                output, loss = result
            else
                output = result
                loss = 0.0f0
            end
        else
            output = gpu_moe(gpu_input; training=false)
            loss = 0.0f0
        end
        
        CUDA.synchronize()
        end_time = time_ns()
        
        # GPU memory after
        mem_after = CUDA.used_memory()
        
        # Record metrics
        push!(timings, (end_time - start_time) / 1e6)  # Convert to ms
        push!(memory_usage, max(0, mem_after - mem_before))  # Ensure non-negative
        push!(losses, loss)
        
        # Clean up
        output = nothing
        GC.gc()
        CUDA.reclaim()
        
        if i % 5 == 0
            println("           Completed $i/$num_benchmark iterations...")
        end
    end
    
    println("        ✅ GPU benchmark completed")
    
    return Dict(
        "implementation" => "CUDAMoE (GPU)",
        "timings_ms" => timings,
        "memory_usage_bytes" => memory_usage,
        "losses" => losses,
        "mean_time_ms" => mean(timings),
        "std_time_ms" => std(timings),
        "min_time_ms" => minimum(timings),
        "max_time_ms" => maximum(timings),
        "mean_memory_mb" => mean(memory_usage) / (1024^2),
        "peak_memory_mb" => maximum(memory_usage) / (1024^2),
        "mean_loss" => mean(losses),
        "tokens_per_second" => (batch_size * seq_length) / (mean(timings) / 1000),
        "throughput_tokens_per_ms" => (batch_size * seq_length) / mean(timings)
    )
end

"""
Verify numerical accuracy between CPU (MixtureOfExperts) and GPU (CUDAMoE) implementations
"""
function verify_numerical_accuracy(cpu_moe, gpu_moe, test_input; rtol::Float32 = 1f-3, atol::Float32 = 1f-4)
    println("        🔍 Verifying numerical accuracy between implementations...")
    
    # CPU forward pass using MixtureOfExperts
    println("           Running CPU forward pass (MixtureOfExperts)...")
    cpu_output, cpu_loss = cpu_moe(test_input; training=true)
    
    # GPU forward pass using CUDAMoE
    println("           Running GPU forward pass (CUDAMoE)...")
    gpu_input = CuArray{Float32}(test_input)
    
    # Run GPU forward pass using the callable interface
    result = gpu_moe(gpu_input; training=true, return_loss=true)
    
    # Extract output and loss from result
    if isa(result, Tuple)
        gpu_output, gpu_loss = result
    else
        gpu_output = result
        gpu_loss = 0.0f0
    end
    
    # Convert GPU output to CPU for comparison
    gpu_output_cpu = Array(gpu_output)
    
    # Compare outputs
    output_close = isapprox(cpu_output, gpu_output_cpu; rtol=rtol, atol=atol)
    loss_close = isapprox(cpu_loss, gpu_loss; rtol=rtol, atol=atol)
    
    # Compute differences
    output_max_diff = maximum(abs.(cpu_output - gpu_output_cpu))
    output_mean_diff = mean(abs.(cpu_output - gpu_output_cpu))
    output_rel_diff = output_max_diff / (maximum(abs.(cpu_output)) + 1e-8)
    
    loss_abs_diff = abs(cpu_loss - gpu_loss)
    loss_rel_diff = loss_abs_diff / (abs(cpu_loss) + 1e-8)
    
    println("           Output numerically close: $output_close")
    println("           Loss numerically close: $loss_close")
    println("           Max output difference: $(output_max_diff:.2e)")
    println("           Max relative difference: $(output_rel_diff:.2e)")
    
    return Dict(
        "output_numerically_close" => output_close,
        "loss_numerically_close" => loss_close,
        "output_max_abs_diff" => output_max_diff,
        "output_mean_abs_diff" => output_mean_diff,
        "output_max_rel_diff" => output_rel_diff,
        "loss_abs_diff" => loss_abs_diff,
        "loss_rel_diff" => loss_rel_diff,
        "cpu_output_stats" => Dict(
            "mean" => mean(cpu_output),
            "std" => std(cpu_output),
            "min" => minimum(cpu_output),
            "max" => maximum(cpu_output)
        ),
        "gpu_output_stats" => Dict(
            "mean" => mean(gpu_output_cpu),
            "std" => std(gpu_output_cpu),
            "min" => minimum(gpu_output_cpu),
            "max" => maximum(gpu_output_cpu)
        ),
        "cpu_loss" => cpu_loss,
        "gpu_loss" => gpu_loss,
        "implementations" => Dict(
            "cpu" => "MixtureOfExperts",
            "gpu" => "CUDAMoE"
        )
    )
end

"""
Analyze expert usage patterns for comparison (CPU implementation)
"""
function analyze_expert_usage(moe_layer, input_data, description::String)
    println("        📈 Analyzing expert usage patterns for $description...")
    
    # Get expert statistics from CPU implementation
    output, loss, stats = moe_layer(input_data; training=true, return_stats=true)
    
    if haskey(stats, :tokens_per_expert)
        expert_counts = stats[:tokens_per_expert]
        total_tokens = sum(expert_counts)
        
        # Calculate load balance metrics
        if total_tokens > 0
            expert_fractions = expert_counts ./ total_tokens
            num_experts = length(expert_fractions)
            ideal_fraction = 1.0 / num_experts
            
            # Load balance score
            max_deviation = maximum(abs.(expert_fractions .- ideal_fraction))
            balance_score = 1.0 - (max_deviation / ideal_fraction)
            
            # Entropy (expert usage diversity)
            entropy = -sum(f * log(f + 1e-8) for f in expert_fractions if f > 0)
            max_entropy = log(num_experts)
            normalized_entropy = entropy / max_entropy
            
            println("           Load balance score: $(balance_score:.3f)")
            println("           Normalized entropy: $(normalized_entropy:.3f)")
            println("           Active experts: $(count(x -> x > 0, expert_counts))/$num_experts")
            
            return Dict(
                "expert_counts" => expert_counts,
                "expert_fractions" => expert_fractions,
                "load_balance_score" => balance_score,
                "usage_entropy" => entropy,
                "normalized_entropy" => normalized_entropy,
                "active_experts" => count(x -> x > 0, expert_counts),
                "max_expert_usage" => maximum(expert_fractions),
                "min_expert_usage" => minimum(expert_fractions),
                "usage_variance" => var(expert_fractions),
                "implementation" => "MixtureOfExperts"
            )
        end
    end
    
    return Dict("error" => "Could not extract expert usage statistics", "implementation" => "MixtureOfExperts")
end

"""
Collect comprehensive system information
"""
function collect_system_info()
    system_info = Dict{String, Any}()
    
    # CPU information
    system_info["cpu"] = Dict(
        "model" => Sys.cpu_info()[1].model,
        "cores" => Sys.CPU_THREADS,
        "memory_gb" => Sys.total_memory() / (1024^3)
    )
    
    # GPU information
    if CUDA.functional()
        system_info["gpu"] = Dict(
            "available" => true,
            "device_name" => CUDA.name(CUDA.device()),
            "compute_capability" => string(CUDA.capability(CUDA.device())),
            "memory_gb" => CUDA.totalmem(CUDA.device()) / (1024^3),
            "multiprocessors" => CUDA.attribute(CUDA.device(), CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        )
    else
        system_info["gpu"] = Dict("available" => false)
    end
    
    # Julia and package versions
    system_info["julia"] = Dict(
        "version" => string(VERSION),
        "threads" => Threads.nthreads()
    )
    
    # Implementation versions
    system_info["implementations"] = Dict(
        "cpu_module" => "MixtureOfExperts",
        "gpu_module" => "CUDAMoE",
        "benchmark_version" => "1.0.0"
    )
    
    # CUDA version if available
    if CUDA.functional()
        try
            system_info["cuda_runtime_version"] = CUDA.runtime_version()
            system_info["cuda_driver_version"] = CUDA.driver_version()
        catch
            system_info["cuda_version_note"] = "Version info not available"
        end
    end
    
    return system_info
end

"""
Run comprehensive benchmark comparing CPU (MixtureOfExperts) and GPU (CUDAMoE) implementations
"""
function run_comprehensive_benchmark(; 
    model_configs = ["Llama2-7B-style", "Llama2-13B-style"],
    expert_configs = [(8, 2), (16, 2)],
    test_configs = [(1, 512), (8, 512), (16, 512), (32, 512)],
    num_warmup::Int = 3,
    num_benchmark::Int = 10,
    test_training::Bool = true,
    verify_accuracy::Bool = true,
    output_file::String = "moe_benchmark_results_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).json"
)
    
    println("🚀 Starting Comprehensive MoE CPU vs GPU Benchmark")
    println("   CPU Implementation: MixtureOfExperts module")
    println("   GPU Implementation: CUDAMoE module")
    println("=" ^ 60)
    
    # Collect system information
    system_info = collect_system_info()
    println("📊 System Information:")
    println("   CPU: $(system_info["cpu"]["model"]) ($(system_info["cpu"]["cores"]) threads)")
    if haskey(system_info, "gpu") && system_info["gpu"]["available"] != false
        println("   GPU: $(system_info["gpu"]["device_name"]) ($(round(system_info["gpu"]["memory_gb"], digits=1)) GB)")
    end
    println("   Julia: $(system_info["julia"]["version"]) ($(system_info["julia"]["threads"]) threads)")
    println()
    
    # Initialize results storage
    all_results = Dict{String, Any}()
    all_results["system_info"] = system_info
    all_results["benchmark_metadata"] = Dict(
        "timestamp" => string(now()),
        "num_warmup" => num_warmup,
        "num_benchmark" => num_benchmark,
        "test_training" => test_training,
        "verify_accuracy" => verify_accuracy,
        "cpu_implementation" => "MixtureOfExperts",
        "gpu_implementation" => "CUDAMoE"
    )
    all_results["detailed_results"] = Dict{String, Any}()
    
    total_tests = length(model_configs) * length(expert_configs) * length(test_configs)
    current_test = 0
    
    for model_name in model_configs
        model_config = MODEL_CONFIGS[model_name]
        println("🏗️  Testing Model: $model_name ($(model_config[:description]))")
        
        for (num_experts, top_k, expert_desc) in expert_configs
            println("   🧠 Expert Config: $expert_desc")
            
            # Setup models with explicit module references
            cpu_moe = setup_cpu_moe(model_config, num_experts, top_k)
            
            if CUDA.functional()
                gpu_moe = setup_gpu_moe(model_config, num_experts, top_k)
            else
                println("      ⚠️  GPU not available, skipping GPU tests")
                continue
            end
            
            for (batch_size, seq_length, test_desc) in test_configs
                current_test += 1
                total_tokens = batch_size * seq_length
                
                test_key = "$(model_name)_$(num_experts)experts_$(top_k)topk_batch$(batch_size)_seq$(seq_length)"
                
                println("      🧪 Test $current_test/$total_tests: $test_desc")
                println("         Tokens: $total_tokens, Dimensions: $(model_config[:input_dim])→$(model_config[:hidden_dim])→$(model_config[:output_dim])")
                
                # Create realistic test data
                input_data = create_realistic_input(
                    model_config[:input_dim], 
                    batch_size, 
                    seq_length;
                    activation_pattern = :transformer
                )
                
                test_results = Dict{String, Any}()
                test_results["config"] = Dict(
                    "model" => model_name,
                    "num_experts" => num_experts,
                    "top_k" => top_k,
                    "batch_size" => batch_size,
                    "seq_length" => seq_length,
                    "total_tokens" => total_tokens,
                    "input_dim" => model_config[:input_dim],
                    "hidden_dim" => model_config[:hidden_dim],
                    "output_dim" => model_config[:output_dim],
                    "cpu_implementation" => "MixtureOfExperts",
                    "gpu_implementation" => "CUDAMoE"
                )
                
                # Numerical accuracy verification
                if verify_accuracy
                    println("         🔍 Verifying numerical accuracy...")
                    accuracy_results = verify_numerical_accuracy(cpu_moe, gpu_moe, input_data)
                    test_results["accuracy"] = accuracy_results
                    
                    if !accuracy_results["output_numerically_close"]
                        println("         ⚠️  Warning: Numerical differences detected!")
                        println("            Max relative diff: $(accuracy_results["output_max_rel_diff"]:.2e)")
                    end
                end
                
                # CPU Benchmarking (MixtureOfExperts)
                println("         ⏱️  Benchmarking CPU (MixtureOfExperts - inference)...")
                cpu_inference = benchmark_cpu_moe(cpu_moe, input_data, num_warmup, num_benchmark; training=false)
                
                if test_training
                    println("         ⏱️  Benchmarking CPU (MixtureOfExperts - training)...")
                    cpu_training = benchmark_cpu_moe(cpu_moe, input_data, num_warmup, num_benchmark; training=true)
                end
                
                # GPU Benchmarking (CUDAMoE)
                println("         ⚡ Benchmarking GPU (CUDAMoE - inference)...")
                gpu_inference = benchmark_gpu_moe(gpu_moe, input_data, num_warmup, num_benchmark; training=false)
                
                if test_training
                    println("         ⚡ Benchmarking GPU (CUDAMoE - training)...")
                    gpu_training = benchmark_gpu_moe(gpu_moe, input_data, num_warmup, num_benchmark; training=true)
                end
                
                # Expert usage analysis (CPU only for now)
                println("         📈 Analyzing expert usage patterns...")
                cpu_expert_usage = analyze_expert_usage(cpu_moe, input_data, "CPU (MixtureOfExperts)")
                
                # Store results
                test_results["cpu"] = Dict(
                    "inference" => cpu_inference,
                    "expert_usage" => cpu_expert_usage
                )
                test_results["gpu"] = Dict(
                    "inference" => gpu_inference
                )
                
                if test_training
                    test_results["cpu"]["training"] = cpu_training
                    test_results["gpu"]["training"] = gpu_training
                end
                
                # Compute comparison metrics
                inference_speedup = cpu_inference["mean_time_ms"] / gpu_inference["mean_time_ms"]
                throughput_improvement = gpu_inference["tokens_per_second"] / cpu_inference["tokens_per_second"]
                
                comparison_metrics = Dict(
                    "inference_speedup" => inference_speedup,
                    "throughput_improvement" => throughput_improvement,
                    "gpu_memory_efficiency" => gpu_inference["mean_memory_mb"] / cpu_inference["mean_memory_mb"],
                    "implementations_compared" => "MixtureOfExperts (CPU) vs CUDAMoE (GPU)"
                )
                
                if test_training
                    training_speedup = cpu_training["mean_time_ms"] / gpu_training["mean_time_ms"]
                    comparison_metrics["training_speedup"] = training_speedup
                end
                
                test_results["comparison"] = comparison_metrics
                
                # Performance summary
                println("         📊 Results Summary:")
                println("            Implementation: MixtureOfExperts (CPU) vs CUDAMoE (GPU)")
                println("            Inference speedup: $(inference_speedup:.2f)x")
                println("            Throughput improvement: $(throughput_improvement:.2f)x")
                println("            CPU: $(cpu_inference["tokens_per_second"]:.0f) tokens/sec")
                println("            GPU: $(gpu_inference["tokens_per_second"]:.0f) tokens/sec")
                if test_training
                    println("            Training speedup: $(training_speedup:.2f)x")
                end
                println()
                
                all_results["detailed_results"][test_key] = test_results
                
                # Clean up models for memory
                gpu_moe = nothing
                GC.gc()
                CUDA.functional() && CUDA.reclaim()
            end
            
            cpu_moe = nothing
            GC.gc()
        end
    end
    
    # Generate summary statistics
    all_results["summary"] = generate_summary_statistics(all_results["detailed_results"])
    
    # Save results to file
    println("💾 Saving results to: $output_file")
    open(output_file, "w") do f
        JSON.print(f, all_results, 2)
    end
    
    # Print final summary
    print_benchmark_summary(all_results["summary"])
    
    return all_results
end

"""
Generate summary statistics across all benchmark results
"""
function generate_summary_statistics(detailed_results::Dict)
    inference_speedups = Float64[]
    training_speedups = Float64[]
    throughput_improvements = Float64[]
    
    for (test_key, result) in detailed_results
        if haskey(result, "comparison")
            push!(inference_speedups, result["comparison"]["inference_speedup"])
            push!(throughput_improvements, result["comparison"]["throughput_improvement"])
            
            if haskey(result["comparison"], "training_speedup")
                push!(training_speedups, result["comparison"]["training_speedup"])
            end
        end
    end
    
    summary = Dict{String, Any}()
    summary["implementations"] = "MixtureOfExperts (CPU) vs CUDAMoE (GPU)"
    
    if !isempty(inference_speedups)
        summary["inference"] = Dict(
            "mean_speedup" => mean(inference_speedups),
            "median_speedup" => median(inference_speedups),
            "min_speedup" => minimum(inference_speedups),
            "max_speedup" => maximum(inference_speedups),
            "std_speedup" => std(inference_speedups)
        )
    end
    
    if !isempty(training_speedups)
        summary["training"] = Dict(
            "mean_speedup" => mean(training_speedups),
            "median_speedup" => median(training_speedups),
            "min_speedup" => minimum(training_speedups),
            "max_speedup" => maximum(training_speedups),
            "std_speedup" => std(training_speedups)
        )
    end
    
    if !isempty(throughput_improvements)
        summary["throughput"] = Dict(
            "mean_improvement" => mean(throughput_improvements),
            "median_improvement" => median(throughput_improvements),
            "min_improvement" => minimum(throughput_improvements),
            "max_improvement" => maximum(throughput_improvements),
            "std_improvement" => std(throughput_improvements)
        )
    end
    
    summary["test_count"] = length(detailed_results)
    
    return summary
end

"""
Print a formatted summary of benchmark results
"""
function print_benchmark_summary(summary::Dict)
    println("🎯 BENCHMARK SUMMARY")
    println("   Comparing: $(summary["implementations"])")
    println("=" ^ 60)
    
    if haskey(summary, "inference")
        inf = summary["inference"]
        println("📈 Inference Performance:")
        println("   Mean GPU speedup: $(inf["mean_speedup"]:.2f)x (±$(inf["std_speedup"]:.2f))")
        println("   Range: $(inf["min_speedup"]:.2f)x to $(inf["max_speedup"]:.2f)x")
        println("   Median: $(inf["median_speedup"]:.2f)x")
    end
    
    if haskey(summary, "training")
        train = summary["training"]
        println("🎓 Training Performance:")
        println("   Mean GPU speedup: $(train["mean_speedup"]:.2f)x (±$(train["std_speedup"]:.2f))")
        println("   Range: $(train["min_speedup"]:.2f)x to $(train["max_speedup"]:.2f)x")
        println("   Median: $(train["median_speedup"]:.2f)x")
    end
    
    if haskey(summary, "throughput")
        thr = summary["throughput"]
        println("⚡ Throughput Improvement:")
        println("   Mean improvement: $(thr["mean_improvement"]:.2f)x (±$(thr["std_improvement"]:.2f))")
        println("   Range: $(thr["min_improvement"]:.2f)x to $(thr["max_improvement"]:.2f)x")
        println("   Median: $(thr["median_improvement"]:.2f)x")
    end
    
    println("📊 Total tests completed: $(summary["test_count"])")
    println("=" ^ 60)
end

# Example usage and execution script
function main()
    println("🚀 MoE CPU vs GPU Comprehensive Benchmark Suite")
    println("   🖥️  CPU Implementation: MixtureOfExperts module")
    println("   🚀 GPU Implementation: CUDAMoE module")
    println("This benchmark will test realistic large language model configurations")
    println()
    
    # Run the comprehensive benchmark with explicit module separation
    results = run_comprehensive_benchmark(
        model_configs = ["Llama2-7B-style", "Llama2-13B-style"],  # Start with 7B and 13B
        expert_configs = [
            (8, 2, "8 experts, top-2"), 
            (16, 2, "16 experts, top-2"), 
            (32, 2, "32 experts, top-2")
        ],              # Multiple expert counts
        test_configs = [                                             # Realistic workloads
            (1, 512, "Single sequence"),   # Added description
            (8, 512, "Small batch"),       # Added description
            (16, 512, "Medium batch"),     # Added description
            (32, 256, "Large batch, shorter sequences"),  # Added description
            (64, 256, "Very large batch") # Added description
        ],
        num_warmup = 5,      # More warmup for stable results
        num_benchmark = 15,  # More samples for statistical significance
        test_training = true,
        verify_accuracy = true
    )
    
    println("✅ Benchmark completed successfully!")
    println("   CPU Implementation: MixtureOfExperts")
    println("   GPU Implementation: CUDAMoE")
    println("Results saved to JSON file for further analysis.")
    
    return results
end

# Uncomment to run the benchmark
results = main()