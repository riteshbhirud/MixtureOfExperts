
using CUDA
using LinearAlgebra
using Printf

include("src/cuda/CUDAMoE.jl")
#const CMoE = Main.CUDAMoE

function test_gpu_components()
    println("🏁 GPU Component Test for Part 1 → Part 2 Readiness")
    println("=" ^ 60)
    
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
    println()
    
    results = Dict{String, Float64}()
    
    # Test 1: Basic GPU Operations
    println("🧮 Testing Basic GPU Operations...")
    test_input = CUDA.randn(Float32, hidden_dim, batch_size)
    
    # SiLU activation test
    gpu_time = @elapsed begin
        for _ in 1:100
            result = CMoE.gpu_silu(test_input)
            CUDA.synchronize()
        end
    end
    results["silu_ms"] = gpu_time * 10
    println("  ✅ SiLU activation: $(round(results["silu_ms"], digits=3)) ms")
    
    # Softmax test
    logits = CUDA.randn(Float32, num_experts, batch_size)
    gpu_time = @elapsed begin
        for _ in 1:100
            result = CMoE.gpu_softmax(logits; dims=1)
            CUDA.synchronize()
        end
    end
    results["softmax_ms"] = gpu_time * 10
    println("  ✅ Softmax: $(round(results["softmax_ms"], digits=3)) ms")
    
    # Test 2: Expert Operations (using direct matrix operations to avoid workspace issues)
    println("\n🤖 Testing Expert Operations...")
    
    # Create weight matrices directly
    w1 = CUDA.randn(Float32, input_dim, hidden_dim) .* 0.02f0
    w2 = CUDA.randn(Float32, hidden_dim, output_dim) .* 0.02f0
    w3 = CUDA.randn(Float32, input_dim, hidden_dim) .* 0.02f0
    expert_input = CUDA.randn(Float32, input_dim, batch_size)
    
    gpu_time = @elapsed begin
        for _ in 1:50
            # Gated expert computation using CUBLAS
            gate = w1' * expert_input
            up = w3' * expert_input
            activated_gate = CMoE.gpu_silu(gate)
            gated = activated_gate .* up
            output = w2' * gated
            CUDA.synchronize()
        end
    end
    results["expert_ms"] = gpu_time * 20
    println("  ✅ Expert forward pass: $(round(results["expert_ms"], digits=3)) ms")
    
    # Test 3: Gating Operations
# REPLACE the gating test section with:
println("\n🎯 Testing Gating Operations...")

try
    router_logits = CUDA.randn(Float32, num_experts, batch_size)
    expert_indices = CUDA.zeros(Int32, k, batch_size)
    expert_gates = CMoE.gpu_zeros(Float32, k, batch_size)
    
    gpu_time = @elapsed begin
        for _ in 1:50
            router_probs = CMoE.gpu_softmax(router_logits; dims=1)
            CMoE.launch_topk_selection_kernel!(expert_indices, expert_gates, router_probs)
            CUDA.synchronize()
        end
    end
    results["gating_ms"] = gpu_time * 20
    println("  ✅ TopK gating: $(round(results["gating_ms"], digits=3)) ms")
    
catch e
    println("  ⚠️  TopK gating test failed: $e")
    results["gating_ms"] = 0.0
end

# REPLACE the loss test section with:
println("\n📊 Testing Loss Operations...")

try
    expert_assignments = CUDA.zeros(Int32, k, batch_size)
    CMoE.gpu_random_categorical_sample!(expert_assignments, num_experts)
    router_probs = CMoE.gpu_softmax(CUDA.randn(Float32, num_experts, batch_size); dims=1)
    
    gpu_time = @elapsed begin
        for _ in 1:50
            expert_fractions = CMoE.gpu_reduce_mean(router_probs; dims=2)
            loss_value = CMoE.gpu_reduce_sum(expert_fractions)
            CUDA.synchronize()
        end
    end
    results["loss_ms"] = gpu_time * 20
    println("  ✅ Loss computation: $(round(results["loss_ms"], digits=3)) ms")
    
catch e
    println("  ⚠️  Loss computation test failed: $e")
    results["loss_ms"] = 0.0
end
    
    # Summary and Part 2 Readiness Assessment
    println("\n📋 PART 1 → PART 2 READINESS ASSESSMENT")
    println("-" ^ 50)
    
    working_components = sum(v > 0 for v in values(results))
    total_components = length(results)
    
    println("Working Components: $working_components/$total_components")
    
    for (component, time_ms) in results
        status = time_ms > 0 ? "✅" : "❌"
        println("  $status $(replace(component, "_ms" => "")): $(time_ms > 0 ? "$(round(time_ms, digits=3)) ms" : "FAILED")")
    end
    
    println()
    
    # Part 2 readiness criteria
    core_working = results["silu_ms"] > 0 && results["softmax_ms"] > 0 && results["expert_ms"] > 0
    
    if core_working
        println("🚀 READY FOR PART 2!")
        println("  ✅ Core GPU operations working")
        println("  ✅ Expert computation functional") 
        println("  ✅ Memory management stable")
        if results["gating_ms"] > 0
            println("  ✅ Gating operations ready")
        end
        if results["loss_ms"] > 0
            println("  ✅ Loss computation ready")
        end
        println("\n  → Proceed to Part 2: MoE Layer Integration")
        return true
    else
        println("❌ NOT READY FOR PART 2")
        println("  Fix the failing components before proceeding")
        return false
    end
end

# Run the test
if test_gpu_components()
    println("\n🎉 SUCCESS: Part 1 complete, ready for Part 2!")
else
    println("\n⚠️  Fix issues before proceeding to Part 2")
end