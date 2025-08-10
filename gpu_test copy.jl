using CUDA
using LinearAlgebra
using Printf
using Statistics

include("src/cuda/CUDAMoE.jl")
# Include CPU MoE library for comparison
include("src/MixtureOfExperts.jl")
using .CUDAMoE

using .MixtureOfExperts
function test_gpu_vs_cpu_components()
    println("🏁 GPU vs CPU Component Performance Comparison")
    println("=" ^ 70)
    
    if !CUDA.functional()
        println("❌ CUDA not functional")
        return false
    end
    
    # Test configuration
    input_dim, hidden_dim, output_dim = 512, 2048, 512
    batch_size, num_experts, k = 64, 8, 2
    
    gpu_config = CMoE.GPUMoEConfig{Float32}(input_dim, hidden_dim, output_dim, num_experts, k)
    
    println("Configuration: $input_dim → $hidden_dim → $output_dim, $num_experts experts, top-$k")
    println("GPU: $(CUDA.name(CUDA.device()))")
    println("Batch size: $batch_size")
    println()
    
    results = Dict{String, Dict{String, Float64}}()
    
    # Test 1: SiLU Activation Comparison
    println("🧮 Testing SiLU Activation...")
    
    # GPU SiLU test
    gpu_input = CUDA.randn(Float32, hidden_dim, batch_size)
    # Warmup
    for _ in 1:10
        CMoE.gpu_silu(gpu_input)
        CUDA.synchronize()
    end
    
    gpu_time = @elapsed begin
        for _ in 1:1000  # Increased iterations
            result_gpu = CMoE.gpu_silu(gpu_input)
            CUDA.synchronize()
        end
    end
    gpu_silu_ms = gpu_time
    
    # CPU SiLU test  
    cpu_input = randn(Float32, hidden_dim, batch_size)
    # Warmup
    for _ in 1:10
        cpu_input .* (1.0f0 ./ (1.0f0 .+ exp.(-cpu_input)))
    end
    
    cpu_time = @elapsed begin
        for _ in 1:1000  # Increased iterations
            # CPU SiLU: x * sigmoid(x) - store result to prevent optimization
            result_cpu = cpu_input .* (1.0f0 ./ (1.0f0 .+ exp.(-cpu_input)))
            # Touch result to prevent optimization
            sum(result_cpu)
        end
    end
    cpu_silu_ms = cpu_time
    
    speedup = cpu_silu_ms / gpu_silu_ms
    results["silu"] = Dict("gpu_ms" => gpu_silu_ms, "cpu_ms" => cpu_silu_ms, "speedup" => speedup)
    println("  GPU: $(round(gpu_silu_ms, digits=3)) ms")
    println("  CPU: $(round(cpu_silu_ms, digits=3)) ms") 
    println("  Speedup: $(round(speedup, digits=2))x")
    println()
    
    # Test 2: Softmax Comparison
    println("🎯 Testing Softmax...")
    
    # GPU Softmax test
    gpu_logits = CUDA.randn(Float32, num_experts, batch_size)
    # Warmup
    for _ in 1:10
        CMoE.gpu_softmax(gpu_logits; dims=1)
        CUDA.synchronize()
    end
    
    gpu_time = @elapsed begin
        for _ in 1:1000  # Increased iterations
            result_gpu = CMoE.gpu_softmax(gpu_logits; dims=1)
            CUDA.synchronize()
        end
    end
    gpu_softmax_ms = gpu_time
    
    # CPU Softmax test
    cpu_logits = randn(Float32, num_experts, batch_size)
    # Warmup
    for _ in 1:10
        max_vals = maximum(cpu_logits, dims=1)
        exp_vals = exp.(cpu_logits .- max_vals)
        exp_vals ./ sum(exp_vals, dims=1)
    end
    
    cpu_time = @elapsed begin
        for _ in 1:1000  # Increased iterations
            # Manual softmax implementation
            max_vals = maximum(cpu_logits, dims=1)
            exp_vals = exp.(cpu_logits .- max_vals)
            result_cpu = exp_vals ./ sum(exp_vals, dims=1)
            # Touch result to prevent optimization
            sum(result_cpu)
        end
    end
    cpu_softmax_ms = cpu_time
    
    speedup = cpu_softmax_ms / gpu_softmax_ms
    results["softmax"] = Dict("gpu_ms" => gpu_softmax_ms, "cpu_ms" => cpu_softmax_ms, "speedup" => speedup)
    println("  GPU: $(round(gpu_softmax_ms, digits=3)) ms")
    println("  CPU: $(round(cpu_softmax_ms, digits=3)) ms")
    println("  Speedup: $(round(speedup, digits=2))x")
    println()
    
    # Test 3: Expert Forward Pass Comparison
    println("🤖 Testing Expert Forward Pass...")
    
    # GPU Expert test
    gpu_w1 = CUDA.randn(Float32, input_dim, hidden_dim) .* 0.02f0
    gpu_w2 = CUDA.randn(Float32, hidden_dim, output_dim) .* 0.02f0
    gpu_w3 = CUDA.randn(Float32, input_dim, hidden_dim) .* 0.02f0
    gpu_expert_input = CUDA.randn(Float32, input_dim, batch_size)
    
    # Warmup
    for _ in 1:5
        gate = gpu_w1' * gpu_expert_input
        up = gpu_w3' * gpu_expert_input
        activated_gate = CMoE.gpu_silu(gate)
        gated = activated_gate .* up
        output = gpu_w2' * gated
        CUDA.synchronize()
    end
    
    gpu_time = @elapsed begin
        for _ in 1:100  # Reasonable iterations for larger operations
            gate = gpu_w1' * gpu_expert_input
            up = gpu_w3' * gpu_expert_input
            activated_gate = CMoE.gpu_silu(gate)
            gated = activated_gate .* up
            output_gpu = gpu_w2' * gated
            CUDA.synchronize()
        end
    end
    gpu_expert_ms = gpu_time
    
    # CPU Expert test
    cpu_w1 = Array(gpu_w1)
    cpu_w2 = Array(gpu_w2) 
    cpu_w3 = Array(gpu_w3)
    cpu_expert_input = Array(gpu_expert_input)
    
    # Warmup
    for _ in 1:5
        gate = cpu_w1' * cpu_expert_input
        up = cpu_w3' * cpu_expert_input
        activated_gate = gate .* (1.0f0 ./ (1.0f0 .+ exp.(-gate)))
        gated = activated_gate .* up
        cpu_w2' * gated
    end
    
    cpu_time = @elapsed begin
        for _ in 1:100
            # Gated expert computation: w2(silu(w1(x)) * w3(x))
            gate = cpu_w1' * cpu_expert_input
            up = cpu_w3' * cpu_expert_input
            # SiLU activation
            activated_gate = gate .* (1.0f0 ./ (1.0f0 .+ exp.(-gate)))
            gated = activated_gate .* up
            output_cpu = cpu_w2' * gated
            # Touch result to prevent optimization
            sum(output_cpu)
        end
    end
    cpu_expert_ms = cpu_time
    
    speedup = cpu_expert_ms / gpu_expert_ms
    results["expert"] = Dict("gpu_ms" => gpu_expert_ms, "cpu_ms" => cpu_expert_ms, "speedup" => speedup)
    println("  GPU: $(round(gpu_expert_ms, digits=3)) ms")
    println("  CPU: $(round(cpu_expert_ms, digits=3)) ms")
    println("  Speedup: $(round(speedup, digits=2))x")
    println()
    
    # Test 4: TopK Gating Comparison  
    println("🎯 Testing TopK Gating...")
    
    # GPU Gating test
    gpu_router_logits = CUDA.randn(Float32, num_experts, batch_size)
    gpu_expert_indices = CUDA.zeros(Int32, k, batch_size)
    gpu_expert_gates = CMoE.gpu_zeros(Float32, k, batch_size)
    
    # Warmup
    for _ in 1:5
        router_probs = CMoE.gpu_softmax(gpu_router_logits; dims=1)
        CMoE.launch_topk_selection_kernel!(gpu_expert_indices, gpu_expert_gates, router_probs)
        CUDA.synchronize()
    end
    
    gpu_time = @elapsed begin
        for _ in 1:100
            router_probs = CMoE.gpu_softmax(gpu_router_logits; dims=1)
            CMoE.launch_topk_selection_kernel!(gpu_expert_indices, gpu_expert_gates, router_probs)
            CUDA.synchronize()
        end
    end
    gpu_gating_ms = gpu_time
    
    # CPU Gating test
    cpu_router_logits = Array(gpu_router_logits)
    
    # Warmup
    for _ in 1:5
        max_vals = maximum(cpu_router_logits, dims=1)
        exp_vals = exp.(cpu_router_logits .- max_vals)
        router_probs = exp_vals ./ sum(exp_vals, dims=1)
        
        expert_indices = zeros(Int32, k, batch_size)
        expert_gates = zeros(Float32, k, batch_size)
        
        for b in 1:batch_size
            probs_col = router_probs[:, b]
            topk_indices = partialsortperm(probs_col, 1:k, rev=true)
            expert_indices[:, b] = topk_indices
            selected_probs = probs_col[topk_indices]
            expert_gates[:, b] = selected_probs ./ sum(selected_probs)
        end
    end
    
    cpu_time = @elapsed begin
        for _ in 1:100
            # CPU TopK gating
            # Softmax
            max_vals = maximum(cpu_router_logits, dims=1)
            exp_vals = exp.(cpu_router_logits .- max_vals)
            router_probs = exp_vals ./ sum(exp_vals, dims=1)
            
            # TopK selection for each batch
            expert_indices = zeros(Int32, k, batch_size)
            expert_gates = zeros(Float32, k, batch_size)
            
            for b in 1:batch_size
                probs_col = router_probs[:, b]
                # Get top-k indices
                topk_indices = partialsortperm(probs_col, 1:k, rev=true)
                expert_indices[:, b] = topk_indices
                
                # Renormalize gates
                selected_probs = probs_col[topk_indices]
                expert_gates[:, b] = selected_probs ./ sum(selected_probs)
            end
            
            # Touch results to prevent optimization
            sum(expert_indices) + sum(expert_gates)
        end
    end
    cpu_gating_ms = cpu_time
    
    speedup = cpu_gating_ms / gpu_gating_ms
    results["gating"] = Dict("gpu_ms" => gpu_gating_ms, "cpu_ms" => cpu_gating_ms, "speedup" => speedup)
    println("  GPU: $(round(gpu_gating_ms, digits=3)) ms")
    println("  CPU: $(round(cpu_gating_ms, digits=3)) ms")
    println("  Speedup: $(round(speedup, digits=2))x")
    println()
    
    # Test 5: Loss Computation Comparison
    println("📊 Testing Loss Computation...")
    
    # GPU Loss test
    gpu_expert_assignments = CUDA.zeros(Int32, k, batch_size)
    CMoE.gpu_random_categorical_sample!(gpu_expert_assignments, num_experts)
    gpu_router_probs = CMoE.gpu_softmax(CUDA.randn(Float32, num_experts, batch_size); dims=1)
    
    # Warmup
    for _ in 1:10
        expert_fractions = CMoE.gpu_reduce_mean(gpu_router_probs; dims=2)
        CMoE.gpu_reduce_sum(expert_fractions)
        CUDA.synchronize()
    end
    
    gpu_time = @elapsed begin
        for _ in 1:1000  # Increased iterations for small operation
            expert_fractions = CMoE.gpu_reduce_mean(gpu_router_probs; dims=2)
            loss_value = CMoE.gpu_reduce_sum(expert_fractions)
            CUDA.synchronize()
        end
    end
    gpu_loss_ms = gpu_time
    
    # CPU Loss test
    cpu_expert_assignments = Array(gpu_expert_assignments)
    cpu_router_probs = Array(gpu_router_probs)
    
    # Warmup
    for _ in 1:10
        expert_fractions = mean(cpu_router_probs, dims=2)
        sum(expert_fractions)
    end
    
    cpu_time = @elapsed begin
        for _ in 1:1000  # Increased iterations
            # Simple loss computation
            expert_fractions = mean(cpu_router_probs, dims=2)
            loss_value = sum(expert_fractions)
            # Touch result to prevent optimization
            loss_value + 0.0
        end
    end
    cpu_loss_ms = cpu_time
    
    speedup = cpu_loss_ms / gpu_loss_ms
    results["loss"] = Dict("gpu_ms" => gpu_loss_ms, "cpu_ms" => cpu_loss_ms, "speedup" => speedup)
    println("  GPU: $(round(gpu_loss_ms, digits=3)) ms")
    println("  CPU: $(round(cpu_loss_ms, digits=3)) ms")
    println("  Speedup: $(round(speedup, digits=2))x")
    println()
    
    # Rest of the analysis code remains the same...
    
    # Comprehensive Performance Summary
    println("📊 COMPREHENSIVE PERFORMANCE COMPARISON")
    println("=" ^ 70)
    
    @printf "%-15s %12s %12s %12s %12s\n" "Operation" "GPU (ms)" "CPU (ms)" "Speedup" "Status"
    println("-" ^ 70)
    
    total_gpu_time = 0.0
    total_cpu_time = 0.0
    all_working = true
    
    for (op_name, timings) in results
        gpu_ms = timings["gpu_ms"]
        cpu_ms = timings["cpu_ms"] 
        speedup = timings["speedup"]
        
        total_gpu_time += gpu_ms
        total_cpu_time += cpu_ms
        
        status = gpu_ms > 0 && speedup > 0.1 ? "✅" : "❌"
        if gpu_ms <= 0 || speedup <= 0.1
            all_working = false
        end
        
        @printf "%-15s %12.3f %12.3f %12.2fx %12s\n" titlecase(op_name) gpu_ms cpu_ms speedup status
    end
    
    println("-" ^ 70)
    overall_speedup = total_cpu_time / total_gpu_time
    @printf "%-15s %12.3f %12.3f %12.2fx %12s\n" "TOTAL" total_gpu_time total_cpu_time overall_speedup (all_working ? "✅" : "❌")
    
    println()
    
    # Part 2 Readiness Assessment
    if all_working && overall_speedup > 2.0
        println("🚀 PART 1 COMPLETE - READY FOR PART 2!")
        println("  ✅ All GPU components functional")
        println("  ✅ Significant performance improvements achieved")
        println("  ✅ $(round(overall_speedup, digits=2))x overall speedup vs CPU")
        println("\n  → Proceed to Part 2: MoE Layer Integration & Batch Processing")
        return true
    else
        println("❌ NOT READY FOR PART 2")
        println("  Fix remaining issues before proceeding")
        return false
    end
end
#test_gpu_vs_cpu_components()
function test_gpu_scaling_benefits()
    println("🔥 GPU Scaling Benefits Demonstration")
    println("=" ^ 60)
    
    # Test different scales
    test_configs = [
        (batch=64, experts=8, name="Small"),
        (batch=128, experts=16, name="Medium"), 
        (batch=256, experts=32, name="Large"),
        (batch=512, experts=64, name="XLarge")
    ]
    
    input_dim, hidden_dim, output_dim = 512, 2048, 512
    k = 2
    
    println("Testing GPU vs CPU at different scales...")
    println()
    
    for config in test_configs
        batch_size = config.batch
        num_experts = config.experts
        scale_name = config.name
        
        println("📏 $scale_name Scale: $num_experts experts, batch=$batch_size")
        
        # Test Softmax at this scale
        gpu_logits = CUDA.randn(Float32, num_experts, batch_size)
        cpu_logits = Array(gpu_logits)
        
        # GPU Softmax
        gpu_time = @elapsed begin
            for _ in 1:100
                result = CMoE.gpu_softmax(gpu_logits; dims=1)
                CUDA.synchronize()
            end
        end
        
        # CPU Softmax  
        cpu_time = @elapsed begin
            for _ in 1:100
                max_vals = maximum(cpu_logits, dims=1)
                exp_vals = exp.(cpu_logits .- max_vals)
                result = exp_vals ./ sum(exp_vals, dims=1)
                sum(result) # Prevent optimization
            end
        end
        
        softmax_speedup = cpu_time / gpu_time
        
        # Test Expert at this scale
        gpu_expert_input = CUDA.randn(Float32, input_dim, batch_size)
        cpu_expert_input = Array(gpu_expert_input)
        
        gpu_w1 = CUDA.randn(Float32, input_dim, hidden_dim) .* 0.02f0
        gpu_w2 = CUDA.randn(Float32, hidden_dim, output_dim) .* 0.02f0
        gpu_w3 = CUDA.randn(Float32, input_dim, hidden_dim) .* 0.02f0
        
        cpu_w1 = Array(gpu_w1)
        cpu_w2 = Array(gpu_w2)
        cpu_w3 = Array(gpu_w3)
        
        # GPU Expert
        gpu_time = @elapsed begin
            for _ in 1:50
                gate = gpu_w1' * gpu_expert_input
                up = gpu_w3' * gpu_expert_input
                activated_gate = CMoE.gpu_silu(gate)
                gated = activated_gate .* up
                output = gpu_w2' * gated
                CUDA.synchronize()
            end
        end
        
        # CPU Expert
        cpu_time = @elapsed begin
            for _ in 1:50
                gate = cpu_w1' * cpu_expert_input
                up = cpu_w3' * cpu_expert_input
                activated_gate = gate .* (1.0f0 ./ (1.0f0 .+ exp.(-gate)))
                gated = activated_gate .* up
                output = cpu_w2' * gated
                sum(output) # Prevent optimization
            end
        end
        
        expert_speedup = cpu_time / gpu_time
        
        @printf "  Softmax speedup: %6.2fx\n" softmax_speedup
        @printf "  Expert speedup:  %6.2fx\n" expert_speedup
        println()
    end
    
    println("💡 Key Insights:")
    println("  • Small operations: GPU overhead dominates")
    println("  • Large operations: GPU parallelism wins")  
    println("  • Real MoE workloads: GPU significantly faster")
    println("  • Part 2 will use larger scales where GPU excels")
end

# test2
#test_gpu_scaling_benefits()
function test_gpu_cpu_accuracy()
    println("🔍 GPU vs CPU Numerical Accuracy Verification")
    println("=" ^ 60)
    
    # Test configuration
    input_dim, hidden_dim, output_dim = 512, 2048, 512
    batch_size, num_experts, k = 32, 8, 2
    tolerance = 1e-5  # Acceptable floating point difference
    
    results = Dict{String, Bool}()
    
    # Test 1: SiLU Accuracy
    println("🧮 Testing SiLU Accuracy...")
    cpu_input = randn(Float32, hidden_dim, batch_size)
    gpu_input = CuArray(cpu_input)
    
    # Compute on both
    cpu_result = cpu_input .* (1.0f0 ./ (1.0f0 .+ exp.(-cpu_input)))
    gpu_result = Array(CMoE.gpu_silu(gpu_input))
    
    max_diff = maximum(abs.(cpu_result - gpu_result))
    mean_diff = mean(abs.(cpu_result - gpu_result))
    
    silu_accurate = max_diff < tolerance
    results["silu"] = silu_accurate
    
    println("  Max difference: $(max_diff)")
    println("  Mean difference: $(mean_diff)")
    println("  Status: $(silu_accurate ? "✅ ACCURATE" : "❌ INACCURATE")")
    println()
    
    # Test 2: Softmax Accuracy
    println("🎯 Testing Softmax Accuracy...")
    cpu_logits = randn(Float32, num_experts, batch_size)
    gpu_logits = CuArray(cpu_logits)
    
    # CPU softmax
    cpu_max_vals = maximum(cpu_logits, dims=1)
    cpu_exp_vals = exp.(cpu_logits .- cpu_max_vals)
    cpu_softmax = cpu_exp_vals ./ sum(cpu_exp_vals, dims=1)
    
    # GPU softmax
    gpu_softmax = Array(CMoE.gpu_softmax(gpu_logits; dims=1))
    
    max_diff = maximum(abs.(cpu_softmax - gpu_softmax))
    mean_diff = mean(abs.(cpu_softmax - gpu_softmax))
    
    # Check probabilities sum to 1
    cpu_sums = sum(cpu_softmax, dims=1)
    gpu_sums = sum(gpu_softmax, dims=1)
    sum_diff = maximum(abs.(cpu_sums .- 1.0f0)) + maximum(abs.(gpu_sums .- 1.0f0))
    
    softmax_accurate = max_diff < tolerance && sum_diff < tolerance
    results["softmax"] = softmax_accurate
    
    println("  Max difference: $(max_diff)")
    println("  Mean difference: $(mean_diff)")
    println("  Sum error: $(sum_diff)")
    println("  Status: $(softmax_accurate ? "✅ ACCURATE" : "❌ INACCURATE")")
    println()
    
    # Test 3: Expert Forward Pass Accuracy
    println("🤖 Testing Expert Forward Pass Accuracy...")
    
    # Same weights for both
    cpu_w1 = randn(Float32, input_dim, hidden_dim) .* 0.02f0
    cpu_w2 = randn(Float32, hidden_dim, output_dim) .* 0.02f0
    cpu_w3 = randn(Float32, input_dim, hidden_dim) .* 0.02f0
    cpu_input = randn(Float32, input_dim, batch_size)
    
    gpu_w1 = CuArray(cpu_w1)
    gpu_w2 = CuArray(cpu_w2)
    gpu_w3 = CuArray(cpu_w3)
    gpu_input = CuArray(cpu_input)
    
    # CPU expert computation
    cpu_gate = cpu_w1' * cpu_input
    cpu_up = cpu_w3' * cpu_input
    cpu_activated_gate = cpu_gate .* (1.0f0 ./ (1.0f0 .+ exp.(-cpu_gate)))
    cpu_gated = cpu_activated_gate .* cpu_up
    cpu_output = cpu_w2' * cpu_gated
    
    # GPU expert computation
    gpu_gate = gpu_w1' * gpu_input
    gpu_up = gpu_w3' * gpu_input
    gpu_activated_gate = CMoE.gpu_silu(gpu_gate)
    gpu_gated = gpu_activated_gate .* gpu_up
    gpu_output = Array(gpu_w2' * gpu_gated)
    
    max_diff = maximum(abs.(cpu_output - gpu_output))
    mean_diff = mean(abs.(cpu_output - gpu_output))
    relative_error = max_diff / (maximum(abs.(cpu_output)) + 1e-8)
    
    expert_accurate = max_diff < tolerance && relative_error < 0.01
    results["expert"] = expert_accurate
    
    println("  Max absolute difference: $(max_diff)")
    println("  Mean absolute difference: $(mean_diff)")
    println("  Relative error: $(relative_error)")
    println("  Status: $(expert_accurate ? "✅ ACCURATE" : "❌ INACCURATE")")
    println()
    
    # Test 4: TopK Gating Accuracy
    println("🎯 Testing TopK Gating Accuracy...")
    
    cpu_router_logits = randn(Float32, num_experts, batch_size)
    gpu_router_logits = CuArray(cpu_router_logits)
    
    # CPU TopK gating
    cpu_max_vals = maximum(cpu_router_logits, dims=1)
    cpu_exp_vals = exp.(cpu_router_logits .- cpu_max_vals)
    cpu_router_probs = cpu_exp_vals ./ sum(cpu_exp_vals, dims=1)
    
    cpu_expert_indices = zeros(Int32, k, batch_size)
    cpu_expert_gates = zeros(Float32, k, batch_size)
    
    for b in 1:batch_size
        probs_col = cpu_router_probs[:, b]
        topk_indices = partialsortperm(probs_col, 1:k, rev=true)
        cpu_expert_indices[:, b] = topk_indices
        selected_probs = probs_col[topk_indices]
        cpu_expert_gates[:, b] = selected_probs ./ sum(selected_probs)
    end
    
    # GPU TopK gating
    gpu_router_probs = CMoE.gpu_softmax(gpu_router_logits; dims=1)
    gpu_expert_indices = CUDA.zeros(Int32, k, batch_size)
    gpu_expert_gates = CMoE.gpu_zeros(Float32, k, batch_size)
    
    CMoE.launch_topk_selection_kernel!(gpu_expert_indices, gpu_expert_gates, gpu_router_probs)
    
    gpu_expert_indices_cpu = Array(gpu_expert_indices)
    gpu_expert_gates_cpu = Array(gpu_expert_gates)
    
    # Check if same experts selected (order might differ)
    indices_match = true
    gates_close = true
    
    for b in 1:batch_size
        cpu_set = Set(cpu_expert_indices[:, b])
        gpu_set = Set(gpu_expert_indices_cpu[:, b])
        
        if cpu_set != gpu_set
            indices_match = false
        end
        
        # Check gate sums are close to 1
        cpu_sum = sum(cpu_expert_gates[:, b])
        gpu_sum = sum(gpu_expert_gates_cpu[:, b])
        
        if abs(cpu_sum - 1.0f0) > tolerance || abs(gpu_sum - 1.0f0) > tolerance
            gates_close = false
        end
    end
    
    gating_accurate = indices_match && gates_close
    results["gating"] = gating_accurate
    
    println("  Expert selection match: $(indices_match ? "✅" : "❌")")
    println("  Gate normalization: $(gates_close ? "✅" : "❌")")
    println("  Status: $(gating_accurate ? "✅ ACCURATE" : "❌ INACCURATE")")
    println()
    
    # Test 5: Loss Computation Accuracy
    println("📊 Testing Loss Computation Accuracy...")
    
    cpu_router_probs = randn(Float32, num_experts, batch_size)
    cpu_router_probs = abs.(cpu_router_probs)  # Make positive
    cpu_router_probs = cpu_router_probs ./ sum(cpu_router_probs, dims=1)  # Normalize
    
    gpu_router_probs = CuArray(cpu_router_probs)
    
    # CPU loss computation
    cpu_expert_fractions = mean(cpu_router_probs, dims=2)
    cpu_loss = sum(cpu_expert_fractions)
    
    # GPU loss computation
    gpu_expert_fractions = Array(CMoE.gpu_reduce_mean(gpu_router_probs; dims=2))
    gpu_loss = CMoE.gpu_reduce_sum(CuArray(gpu_expert_fractions))
    gpu_loss_cpu = Array([gpu_loss])[1]
    
    loss_diff = abs(cpu_loss - gpu_loss_cpu)
    fraction_diff = maximum(abs.(cpu_expert_fractions - gpu_expert_fractions))
    
    loss_accurate = loss_diff < tolerance && fraction_diff < tolerance
    results["loss"] = loss_accurate
    
    println("  Loss difference: $(loss_diff)")
    println("  Fraction difference: $(fraction_diff)")
    println("  Status: $(loss_accurate ? "✅ ACCURATE" : "❌ INACCURATE")")
    println()
    
    # Summary
    println("🎯 ACCURACY VERIFICATION SUMMARY")
    println("=" ^ 50)
    
    passing_tests = sum(values(results))
    total_tests = length(results)
    
    for (test_name, passed) in results
        status = passed ? "✅" : "❌"
        println("  $status $(titlecase(test_name)): $(passed ? "ACCURATE" : "INACCURATE")")
    end
    
    println()
    println("Tests passed: $passing_tests/$total_tests")
    
    if passing_tests == total_tests
        println("🎉 ALL ACCURACY TESTS PASSED!")
        println("  ✅ GPU implementations are numerically equivalent to CPU")
        println("  ✅ Ready for production use")
        println("  ✅ Safe to proceed to Part 2")
        return true
    else
        println("❌ ACCURACY ISSUES DETECTED!")
        println("  Fix numerical differences before proceeding")
        return false
    end
end

# Add this to your test suite
println("\n" * "="^60)
test_gpu_cpu_accuracy()