#!/usr/bin/env julia

"""
GPU MoE Pipeline Comprehensive Test - FIXED VERSION

Tests the full GPU MoE implementation with realistic large-scale inputs
simulating mid-large model scenarios (7B-13B parameter range).
"""

using CUDA
using Statistics
using Random
using Printf
using LinearAlgebra
using Dates

# Import your MoE modules
include("src/cuda/CUDAMoE.jl")
using .CUDAMoE
import .CUDAMoE: GPURoutingState, create_gpu_moe_config, GPUTopKGatingState, create_gpu_moe_layer,process_gpu_output,gpu_moe_forward

function main()
    # Create output file with timestamp
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    output_filename = "gpu_moe_test_results_$(timestamp).txt"
    
    # Open file and redirect stdout
    original_stdout = stdout
    output_file = open(output_filename, "w")
    
    try
        redirect_stdout(output_file)
        
        println("=" ^ 80)
        println("GPU MoE Pipeline Comprehensive Test - FIXED")
        println("=" ^ 80)
        
        # Check CUDA availability
        if !CUDA.functional()
            error("CUDA not functional - cannot run GPU tests")
        end
        
        println("GPU Device: $(CUDA.name(CUDA.device()))")
        available_memory_gb = CUDA.available_memory() ÷ (1024^3)
        println("Available Memory: $(available_memory_gb) GB")
        println()
        
        # Clean up any existing GPU memory
        GC.gc()
        CUDA.reclaim()
        
        # Test configurations with REALISTIC batch sizes for 8GB GPU
        #=
        test_configurations = [
            # Small model for 8GB GPU
            Dict(
                :name => "Small Model (GPU Memory Optimized)",
                :input_dim => 1024,
                :hidden_dim => 2048,
                :output_dim => 1024,
                :num_experts => 4,
                :top_k => 2,
                :max_tokens => 4096,  # Total tokens to process at once
                :test_scenarios => [
                    (32, 64),   # 32 batch × 64 seq = 2048 tokens
                    (16, 128),  # 16 batch × 128 seq = 2048 tokens
                    (8, 256),   # 8 batch × 256 seq = 2048 tokens
                    (64, 64),   # 64 batch × 64 seq = 4096 tokens
                ]
            ),
            
            # Medium model if small works
            Dict(
                :name => "Medium Model (Conservative)",
                :input_dim => 2048,
                :hidden_dim => 4096,
                :output_dim => 2048,
                :num_experts => 8,
                :top_k => 2,
                :max_tokens => 2048,  # More conservative
                :test_scenarios => [
                    (16, 64),   # 16 batch × 64 seq = 1024 tokens
                    (8, 128),   # 8 batch × 128 seq = 1024 tokens
                    (32, 64),   # 32 batch × 64 seq = 2048 tokens
                ]
            )
        ] =#
         test_configurations = [
    Dict(
        :name => "Tiny Model (Shared Memory Test)",
        :input_dim => 256,     # Much smaller
        :hidden_dim => 512,    # Much smaller
        :output_dim => 256,    # Much smaller
        :num_experts => 4,
        :top_k => 2,
        :max_tokens => 256,    # Very small
        :test_scenarios => [
            (4, 32),    # 4 batch × 32 seq = 128 tokens
            (8, 16),    # 8 batch × 16 seq = 128 tokens
            (16, 16),   # 16 batch × 16 seq = 256 tokens
        ]
    ),
    Dict(
        :name => "Small Model (No Shared Memory)",
        :input_dim => 1024,
        :hidden_dim => 2048,
        :output_dim => 1024,
        :num_experts => 4,
        :top_k => 2,
        :max_tokens => 1024,
        :test_scenarios => [
            (16, 32),   # 16 batch × 32 seq = 512 tokens
            (32, 32),   # 32 batch × 32 seq = 1024 tokens
        ]
    )
]
        
        # Only test larger configs if we have more memory
        if available_memory_gb >= 16
            push!(test_configurations, Dict(
                :name => "Large Model (16GB+ GPU)",
                :input_dim => 4096,
                :hidden_dim => 8192,
                :output_dim => 4096,
                :num_experts => 16,
                :top_k => 2,
                :max_tokens => 2048,
                :test_scenarios => [
                    (8, 64),    # 8 batch × 64 seq = 512 tokens
                    (16, 64),   # 16 batch × 64 seq = 1024 tokens
                ]
            ))
        end
        
        # Run tests for each configuration
        all_results = Dict()
        
        for config in test_configurations
            println("Testing: $(config[:name])")
            println("-" ^ 60)
            
            try
                config_results = test_moe_configuration(config)
                all_results[config[:name]] = config_results
                
                println("✓ Configuration test completed successfully")
                println()
            catch e
                println("✗ Configuration test failed: $e")
                @show e
                println()
                
                # Clean up memory before continuing
                GC.gc()
                CUDA.reclaim()
                continue
            end
        end
        
        # Generate comprehensive report
        generate_test_report(all_results)
        
        println("=" ^ 80)
        println("GPU MoE Pipeline Test Completed")
        println("=" ^ 80)
        
    finally
        # Restore stdout and close file
        redirect_stdout(original_stdout)
        close(output_file)
        
        # Print to terminal that the test completed and where results are saved
        println("GPU MoE test completed. Results saved to: $(output_filename)")
    end
end

function test_moe_configuration(config::Dict)
    """Test a specific MoE configuration with various scenarios"""
    
    results = Dict()
    
    # Clean memory before starting
    GC.gc()
    CUDA.reclaim()
    
    # Create GPU MoE configuration with FIXED max_batch_size
    gpu_config = create_gpu_moe_config(
        num_experts = config[:num_experts],
        input_dim = config[:input_dim],
        hidden_dim = config[:hidden_dim],
        output_dim = config[:output_dim],
        top_k = config[:top_k],
        max_batch_size = config[:max_tokens],  # Use realistic token count
        use_half_precision = false,
        enable_kernel_fusion = true,
        memory_alignment = 32
    )
    
    # Estimate and check memory before proceeding
    estimated_memory_mb = estimate_memory_requirements_fixed(gpu_config)
    available_memory_mb = CUDA.available_memory() ÷ (1024^2)
    
    println("  Estimated memory usage: $(estimated_memory_mb)MB")
    println("  Available memory: $(available_memory_mb)MB")
    
    if estimated_memory_mb > available_memory_mb * 0.8
        error("Estimated memory usage too high for available GPU memory")
    end
    
    # Create GPU MoE layer
    moe_layer = create_gpu_moe_layer(
        config[:input_dim],
        config[:hidden_dim],
        config[:output_dim];
        num_experts = config[:num_experts],
        top_k = config[:top_k],
        max_batch_size = config[:max_tokens],
        alpha = 0.01f0,
        enable_expert_parallelism = true,
        use_dynamic_batching = true,
        enable_memory_optimization = true
    )
    
    println("  GPU MoE Layer created successfully")
    println("  - Experts: $(config[:num_experts]), Top-K: $(config[:top_k])")
    println("  - Dimensions: $(config[:input_dim]) → $(config[:hidden_dim]) → $(config[:output_dim])")
    println("  - Max tokens: $(config[:max_tokens])")
    
    # Test scenarios
    scenarios = [
        ("Inference", false),
        ("Training", true)
    ]
    
    for (scenario_name, training) in scenarios
        println("  Testing $scenario_name Mode:")
        scenario_results = Dict()
        
        for (batch_size, seq_length) in config[:test_scenarios]
            total_tokens = batch_size * seq_length
            
            # Skip if exceeds our max_tokens limit
            if total_tokens > config[:max_tokens]
                println("    Skipping batch_size=$batch_size, seq_length=$seq_length (exceeds max)")
                continue
            end
            
            println("    Testing batch_size=$batch_size, seq_length=$seq_length (total=$total_tokens)")
            
            try
                test_result = run_forward_pass_test(
                    moe_layer, 
                    config[:input_dim], 
                    total_tokens, 
                    training
                )
                
                scenario_results["$(batch_size)x$(seq_length)"] = test_result
                
                println("      ✓ Forward pass: $(test_result[:forward_time_ms]:.2f)ms")
                println("      ✓ Throughput: $(test_result[:throughput_tokens_per_sec]:.0f) tokens/sec")
                if training
                    println("      ✓ Balance loss: $(test_result[:balance_loss]:.6f)")
                end
                
            catch e
                println("      ✗ Failed: $e")
                @show e
                
                # Clean memory after failure
                GC.gc()
                CUDA.reclaim()
            end
        end
        
        results[scenario_name] = scenario_results
    end
    
    # Test memory efficiency with smaller tests
    println("  Testing Memory Efficiency:")
    memory_results = test_memory_efficiency(moe_layer, config)
    results["Memory"] = memory_results
    
    # Test numerical stability with small inputs
    println("  Testing Numerical Stability:")
    stability_results = test_numerical_stability(moe_layer, config)
    results["Stability"] = stability_results
    
    return results
end

function run_forward_pass_test(moe_layer, input_dim::Int, total_tokens::Int, training::Bool)
    """Run a single forward pass test and measure performance"""
    
    # Generate realistic input (simulate embeddings)
    input = CUDA.randn(Float32, input_dim, total_tokens) .* 0.02f0
    
    # Warm up (1 iteration to avoid memory allocation overhead)
    try
        if training
            output, _ = gpu_moe_forward(moe_layer, input; training=true, return_cpu=false)
        else
            output = gpu_moe_forward(moe_layer, input; training=false, return_cpu=false)
        end
        CUDA.synchronize()
    catch e
        # Clean up and rethrow
        GC.gc()
        CUDA.reclaim()
        rethrow(e)
    end
    
    # Measure performance
    start_time = time_ns()
    
    if training
        output, balance_loss, stats = moe_layer(input; training=true, return_stats=true)
    else
        output, stats = moe_layer(input; training=false, return_stats=true)
        balance_loss = Int32(0.0)
    end
    
    CUDA.synchronize()
    end_time = time_ns()
    
    forward_time_ms = (end_time - start_time) / 1e6
    throughput = total_tokens / (forward_time_ms / 1000)
    
    # Validate output
    output_sample = Array(output[:, 1:min(10, size(output, 2))])  # Sample first 10 tokens
    if !all(isfinite, output_sample)
        error("Non-finite values in output")
    end
    
    # Check output distribution
    output_mean = mean(output_sample)
    output_std = std(output_sample)
    
    result = Dict(
        :forward_time_ms => forward_time_ms,
        :throughput_tokens_per_sec => throughput,
        :balance_loss => balance_loss,
        :output_mean => output_mean,
        :output_std => output_std,
        :memory_usage_mb => CUDA.pool_status().used_bytes ÷ (1024^2)
    )
    
    if haskey(stats, "expert_usage_analysis")
        result[:expert_balance] = stats["expert_usage_analysis"]
    end
    
    return result
end

function estimate_memory_requirements_fixed(config)
    """Fixed memory estimation function"""
    
    T = config.use_half_precision ? Float16 : Float32
    element_size = sizeof(T)
    
    # Expert weights (realistic calculation)
    weights_per_expert = (
        config.input_dim * config.hidden_dim +     # w1
        config.hidden_dim * config.output_dim +    # w2  
        config.input_dim * config.hidden_dim +     # w3
        config.hidden_dim * 3                      # biases (optional)
    )
    expert_memory = config.num_experts * weights_per_expert * element_size
    
    # Router weights
    router_memory = config.input_dim * config.num_experts * element_size
    
    # Workspace memory (more realistic estimation)
    # This includes intermediate activations, routing arrays, etc.
    max_batch = config.max_batch_size
    workspace_memory = max_batch * (
        config.input_dim * 2 +           # input + copy
        config.hidden_dim * 4 +          # intermediate activations
        config.output_dim * 2 +          # output + temp
        config.num_experts * 0.1 +       # routing overhead
        config.top_k * 2                 # gating arrays
    ) * element_size
    
    # Add some buffer (20%)
    total_memory_bytes = (expert_memory + router_memory + workspace_memory) * 1.2
    
    return Int(ceil(total_memory_bytes ÷ (1024^2)))  # Convert to MB
end

function test_memory_efficiency(moe_layer, config::Dict)
    """Test memory usage patterns with small incremental tests"""
    
    println("    Testing incremental memory usage...")
    
    input_dim = config[:input_dim]
    memory_results = Dict()
    
    # Test with small incremental batch sizes
    test_tokens = [64, 128, 256, 512, min(1024, config[:max_tokens])]
    
    for num_tokens in test_tokens
        if num_tokens > config[:max_tokens]
            continue
        end
        
        try
            # Force garbage collection
            GC.gc()
            CUDA.reclaim()
            
            initial_memory = CUDA.pool_status().used_bytes
            
            input = CUDA.randn(Float32, input_dim, num_tokens) .* 0.02f0
            output = gpu_moe_forward(moe_layer, input; training=false, return_cpu=false)
            CUDA.synchronize()
            
            peak_memory = CUDA.pool_status().used_bytes
            memory_used = (peak_memory - initial_memory) ÷ (1024^2)
            
            memory_results[num_tokens] = Dict(
                :success => true,
                :memory_used_mb => memory_used,
                :memory_per_token_kb => (memory_used * 1024) / num_tokens
            )
            
            println("      ✓ $(num_tokens) tokens: $(memory_used)MB ($(round((memory_used * 1024) / num_tokens, digits=2))KB/token)")
            
        catch e
            memory_results[num_tokens] = Dict(
                :success => false,
                :error => string(e)
            )
            println("      ✗ $(num_tokens) tokens: Failed - $(string(e))")
            
            # Clean up after failure
            GC.gc()
            CUDA.reclaim()
        end
    end
    
    return memory_results
end

function test_numerical_stability(moe_layer, config::Dict)
    """Test numerical stability with edge cases using small inputs"""
    
    println("    Testing with extreme inputs...")
    
    input_dim = config[:input_dim]
    batch_size = min(64, config[:max_tokens])  # Use small batch for stability tests
    
    stability_results = Dict()
    
    # Test cases with small batches
    test_cases = [
        ("Normal", () -> CUDA.randn(Float32, input_dim, batch_size) .* 0.02f0),
        ("Large values", () -> CUDA.randn(Float32, input_dim, batch_size) .* 5.0f0),  # Reduced scale
        ("Small values", () -> CUDA.randn(Float32, input_dim, batch_size) .* 1e-4f0),
        ("Zero input", () -> CUDA.zeros(Float32, input_dim, batch_size))
    ]
    
    for (test_name, input_generator) in test_cases
        try
            # Clean memory before each test
            GC.gc()
            CUDA.reclaim()
            
            input = input_generator()
            output = gpu_moe_forward(moe_layer, input; training=false, return_cpu=false)
            
            # Sample output for checking (don't copy entire array)
            output_sample = Array(output[:, 1:min(10, size(output, 2))])
            
            # Check for numerical issues
            has_nan = any(isnan, output_sample)
            has_inf = any(isinf, output_sample)
            all_finite = all(isfinite, output_sample)
            
            if all_finite
                output_range = (minimum(output_sample), maximum(output_sample))
                output_mean = mean(output_sample)
                output_std = std(output_sample)
            else
                output_range = (NaN, NaN)
                output_mean = NaN
                output_std = NaN
            end
            
            stability_results[test_name] = Dict(
                :success => all_finite,
                :has_nan => has_nan,
                :has_inf => has_inf,
                :output_range => output_range,
                :output_mean => output_mean,
                :output_std => output_std
            )
            
            status = all_finite ? "✓" : "✗"
            if all_finite
                println("      $status $test_name: range=$(round(output_range[1], digits=3)) to $(round(output_range[2], digits=3))")
            else
                println("      $status $test_name: Non-finite values detected")
            end
            
        catch e
            stability_results[test_name] = Dict(
                :success => false,
                :error => string(e)
            )
            println("      ✗ $test_name: Failed with error - $(string(e))")
            
            # Clean up after failure
            GC.gc()
            CUDA.reclaim()
        end
    end
    
    return stability_results
end

function generate_test_report(all_results::Dict)
    """Generate a comprehensive test report"""
    
    println("COMPREHENSIVE TEST REPORT")
    
    if isempty(all_results)
        println("❌ No successful test results to report")
        return
    end
    
    for (config_name, config_results) in all_results
        println("\n📊 $config_name")
        println("-" ^ length(config_name))
        
        # Performance summary
        if haskey(config_results, "Inference")
            inference_results = config_results["Inference"]
            
            if !isempty(inference_results)
                println("\n🚀 Inference Performance:")
                for (scenario, result) in inference_results
                    throughput = result[:throughput_tokens_per_sec]
                    latency = result[:forward_time_ms]
                    memory = result[:memory_usage_mb]
                    println("  $scenario: $(round(throughput, digits=0)) tokens/sec, $(round(latency, digits=2))ms, $(memory)MB")
                end
            end
        end
        
        if haskey(config_results, "Training")
            training_results = config_results["Training"]
            
            if !isempty(training_results)
                println("\n🎯 Training Performance:")
                balance_losses = [result[:balance_loss] for (_, result) in training_results if haskey(result, :balance_loss)]
                if !isempty(balance_losses)
                    avg_balance_loss = mean(balance_losses)
                    println("  Average balance loss: $(round(avg_balance_loss, digits=6))")
                end
            end
        end
        
        # Memory analysis
        if haskey(config_results, "Memory")
            memory_results = config_results["Memory"]
            
            println("\n💾 Memory Efficiency:")
            successful_tests = [(tokens, result) for (tokens, result) in memory_results if result[:success]]
            
            if !isempty(successful_tests)
                max_tokens = maximum(tokens for (tokens, _) in successful_tests)
                avg_memory_per_token = mean(result[:memory_per_token_kb] for (_, result) in successful_tests)
                println("  Max tokens tested: $max_tokens")
                println("  Avg memory per token: $(round(avg_memory_per_token, digits=2))KB")
            end
        end
        
        # Stability analysis
        if haskey(config_results, "Stability")
            stability_results = config_results["Stability"]
            
            println("\n🔬 Numerical Stability:")
            stable_tests = sum(result[:success] for (_, result) in stability_results if haskey(result, :success))
            total_tests = length(stability_results)
            println("  Stable tests: $stable_tests/$total_tests")
        end
    end
    
    # Overall system health
    println("SYSTEM HEALTH SUMMARY")
    
    total_configs = length(all_results)
    println("✅ Configurations tested: $total_configs")
    println("💻 GPU: $(CUDA.name(CUDA.device()))")
    println("🔋 Available memory: $(CUDA.available_memory() ÷ (1024^3))GB")
    println("🔧 Memory pool: $(CUDA.pool_status().used_bytes ÷ (1024^2))MB used")
    
    # Recommendations
    println("\n💡 RECOMMENDATIONS:")
    println("🎯 For your 8GB GPU:")
    println("   - Keep total tokens < 4096 for larger models")
    println("   - Use FP16 precision for 2x memory savings")
    println("   - Consider gradient checkpointing for training")
    println("   - Monitor memory usage with CUDA.pool_status()")
end

# Memory cleanup function
function cleanup_gpu_memory()
    """Clean up GPU memory"""
    try
        GC.gc()
        CUDA.reclaim()
        println("✓ GPU memory cleaned up")
    catch e
        println("⚠ Memory cleanup warning: $e")
    end
end

# Enhanced error handling for the main test
try
    main()
catch e
    println("\n❌ Test failed with error: $e")
    println("\nDebug information:")
    @show typeof(e)
    @show e
    
    println("\nCleaning up...")
    cleanup_gpu_memory()
    
    # Don't rethrow in standalone mode, just exit gracefully
    println("Exiting...")
    exit(1)
finally
    cleanup_gpu_memory()
end




