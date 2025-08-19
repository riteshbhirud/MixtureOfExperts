include("src/MixtureOfExperts.jl")
using .MixtureOfExperts
import .MixtureOfExperts: FluxStandardExpert, FluxGatedExpert, FluxRouter, FluxMoELayer, FluxTopKGating, FluxSwitchTransformerLoss,FluxMoEConfig,compute_gates,flux_silu
using LinearAlgebra
using Flux
println(" Starting Flux Integration Tests - Phases 1 & 2")
println("=" ^ 60)

function test_assert(condition, test_name, details="")
    if condition
        println(" PASS: $test_name")
        return true
    else
        println(" FAIL: $test_name")
        if !isempty(details)
            println("   Reason: $details")
        end
        return false
    end
end

function test_size_equals(actual, expected, test_name)
    condition = size(actual) == expected
    details = condition ? "" : "Expected size $expected, got $(size(actual))"
    return test_assert(condition, test_name, details)
end

function test_type_is(actual, expected_type, test_name)
    condition = actual isa expected_type
    details = condition ? "" : "Expected type $expected_type, got $(typeof(actual))"
    return test_assert(condition, test_name, details)
end

function test_approximately_equal(actual, expected, test_name; atol=1e-6)
    condition = isapprox(actual, expected, atol=atol)
    details = condition ? "" : "Expected ≈$expected, got $actual (tolerance=$atol)"
    return test_assert(condition, test_name, details)
end

println("\n Phase 1: Expert Layer Tests")
println("-" ^ 40)

println("\n Testing FluxStandardExpert...")

try
    expert = FluxStandardExpert(256, 1024, 256, Flux.relu; dropout=0.1f0)
    test_type_is(expert, FluxStandardExpert, "FluxStandardExpert construction")
    
    x_single = randn(Float32, 256)
    y_single = expert(x_single)
    test_size_equals(y_single, (256,), "FluxStandardExpert single input forward pass")
    
    x_batch = randn(Float32, 256, 32)
    y_batch = expert(x_batch; training=true)
    test_size_equals(y_batch, (256, 32), "FluxStandardExpert batch input forward pass")
    
    y_inference = expert(x_batch; training=false)
    test_size_equals(y_inference, (256, 32), "FluxStandardExpert inference mode")
    
    using Flux
    params = Flux.trainables(expert)
    test_assert(length(params) >= 2, "FluxStandardExpert parameter access", 
                "Should have at least 2 parameter arrays (weights from Dense layers)")
    
    println("✓ FluxStandardExpert tests completed")
    
catch e
    println(" FAIL: FluxStandardExpert tests - Error: $e")
end

println("\n Testing FluxGatedExpert...")

try
    expert = FluxGatedExpert(256, 1024, 256, flux_silu)
    test_type_is(expert, FluxGatedExpert, "FluxGatedExpert construction")
    
    x_single = randn(Float32, 256)
    y_single = expert(x_single)
    test_size_equals(y_single, (256,), "FluxGatedExpert single input forward pass")
    
    x_batch = randn(Float32, 256, 32)
    y_batch = expert(x_batch; training=true)
    test_size_equals(y_batch, (256, 32), "FluxGatedExpert batch input forward pass")
    
    using Flux
    params = Flux.trainables(expert)
    test_assert(length(params) >= 3, "FluxGatedExpert parameter access",
                "Should have at least 3 parameter arrays (weights from w1, w2, w3)")
    
    println("✓ FluxGatedExpert tests completed")
    
catch e
    println(" FAIL: FluxGatedExpert tests - Error: $e")
end

println("\n Phase 2: Router and MoE Layer Tests")
println("-" ^ 40)

println("\n Testing FluxTopKGating...")

try
    gating = FluxTopKGating(2)
    router_logits = randn(Float32, 8, 32)
    
    expert_indices, expert_gates, router_probs = compute_gates(gating, router_logits)
    
    test_size_equals(expert_indices, (2, 32), "FluxTopKGating expert_indices size")
    test_size_equals(expert_gates, (2, 32), "FluxTopKGating expert_gates size")
    test_size_equals(router_probs, (8, 32), "FluxTopKGating router_probs size")
    
    prob_sums = sum(router_probs, dims=1)
    all_close_to_one = all(isapprox.(prob_sums, 1.0, atol=1e-6))
    test_assert(all_close_to_one, "FluxTopKGating probability normalization",
                "Router probabilities should sum to 1 across experts")
    
    gates_normalized = true
    for t in 1:32
        if !isapprox(sum(expert_gates[:, t]), 1.0, atol=1e-6)
            gates_normalized = false
            break
        end
    end
    test_assert(gates_normalized, "FluxTopKGating gate normalization",
                "Expert gates should sum to 1 for each token")
    
    println("✓ FluxTopKGating tests completed")
    
catch e
    println(" FAIL: FluxTopKGating tests - Error: $e")
end

println("\n Testing FluxRouter...")

try
    gate = FluxTopKGating(2)
    router = FluxRouter(256, 8, gate; noise_scale=0.1f0)
    test_type_is(router, FluxRouter, "FluxRouter construction")
    
    x = randn(Float32, 256, 32)
    expert_indices, expert_gates, router_probs, router_logits = router(x; training=true)
    
    test_size_equals(expert_indices, (2, 32), "FluxRouter batch expert_indices")
    test_size_equals(expert_gates, (2, 32), "FluxRouter batch expert_gates")
    test_size_equals(router_probs, (8, 32), "FluxRouter batch router_probs")
    test_size_equals(router_logits, (8, 32), "FluxRouter batch router_logits")
    
    x_single = randn(Float32, 256)
    expert_indices_single, expert_gates_single, router_probs_single, router_logits_single = router(x_single; training=false)
    
    test_size_equals(expert_indices_single, (2, 1), "FluxRouter single expert_indices")
    test_size_equals(expert_gates_single, (2, 1), "FluxRouter single expert_gates")
    test_size_equals(router_probs_single, (8, 1), "FluxRouter single router_probs")
    test_size_equals(router_logits_single, (8, 1), "FluxRouter single router_logits")
    
    println("✓ FluxRouter tests completed")
    
catch e
    println(" FAIL: FluxRouter tests - Error: $e")
end

println("\n Testing FluxSwitchTransformerLoss...")

try
    loss_fn = FluxSwitchTransformerLoss(0.01f0)
    
    expert_indices = [1 2 3; 2 1 1]
    router_probs = rand(Float32, 3, 3)
    router_probs = router_probs ./ sum(router_probs, dims=1)
    
    loss = compute_loss(loss_fn, expert_indices, router_probs)
    test_type_is(loss, Float32, "FluxSwitchTransformerLoss output type")
    test_assert(loss >= 0, "FluxSwitchTransformerLoss non-negative", 
                "Loss should be non-negative, got $loss")
    
    println("✓ FluxSwitchTransformerLoss tests completed")
    
catch e
    println(" FAIL: FluxSwitchTransformerLoss tests - Error: $e")
end

println("\n Testing FluxMoELayer with Standard Experts...")

try
    config = FluxMoEConfig(
        input_dim=256,
        hidden_dim=1024, 
        output_dim=256,
        num_experts=4,
        top_k=2,
        expert_type=:standard,
        expert_dropout=0.1f0
    )
    
    moe = FluxMoELayer(config)
    test_type_is(moe, FluxMoELayer, "FluxMoELayer (standard) construction")
    
    x_batch = randn(Float32, 256, 16)
    y_batch, aux_loss = moe(x_batch; training=true)
    
    test_size_equals(y_batch, (256, 16), "FluxMoELayer (standard) batch output")
    test_type_is(aux_loss, Float32, "FluxMoELayer (standard) aux_loss type")
    test_assert(aux_loss >= 0, "FluxMoELayer (standard) aux_loss non-negative",
                "Auxiliary loss should be non-negative, got $aux_loss")
    
    y_inference, aux_loss_inference = moe(x_batch; training=false)
    test_size_equals(y_inference, (256, 16), "FluxMoELayer (standard) inference output")
    test_approximately_equal(aux_loss_inference, 0.0f0, "FluxMoELayer (standard) inference aux_loss")
    
    x_single = randn(Float32, 256)
    y_single, aux_loss_single = moe(x_single; training=true)
    test_size_equals(y_single, (256,), "FluxMoELayer (standard) single input")
    
    println("✓ FluxMoELayer (standard experts) tests completed")
    
catch e
    println(" FAIL: FluxMoELayer (standard experts) tests - Error: $e")
end

println("\n Testing FluxMoELayer with Gated Experts...")

try
    moe = FluxMoELayer(256, 1024, 256; 
                      num_experts=4, top_k=2, expert_type=:gated)
    test_type_is(moe, FluxMoELayer, "FluxMoELayer (gated) construction")
    
    x = randn(Float32, 256, 16)
    y, aux_loss = moe(x; training=true)
    
    test_size_equals(y, (256, 16), "FluxMoELayer (gated) output")
    test_type_is(aux_loss, Float32, "FluxMoELayer (gated) aux_loss type")
    test_assert(aux_loss >= 0, "FluxMoELayer (gated) aux_loss non-negative",
                "Auxiliary loss should be non-negative, got $aux_loss")
    
    println("✓ FluxMoELayer (gated experts) tests completed")
    
catch e
    println(" FAIL: FluxMoELayer (gated experts) tests - Error: $e")
end

println("\n Testing Flux Integration - Parameter Training...")

try
    using Flux
    
    moe = FluxMoELayer(64, 256, 64; num_experts=4, top_k=2)
    
    params = Flux.trainables(moe)
    test_assert(length(params) > 0, "Flux trainable parameters",
                "Should have trainable parameters, got $(length(params))")
    
    x = randn(Float32, 64, 8)
    
    function loss_fn(model, x)
        y, aux_loss = model(x; training=true)
        return sum(abs2, y) + aux_loss
    end
    
    loss_value, grads = Flux.withgradient(m -> loss_fn(m, x), moe)
    test_type_is(loss_value, Float32, "Flux gradient computation loss type")
    test_assert(grads[1] !== nothing, "Flux gradient computation",
                "Gradients should not be nothing")
    
    println("✓ Flux integration parameter training tests completed")
    
catch e
    println(" FAIL: Flux integration parameter training tests - Error: $e")
end

println("\n Performance and Memory Tests")
println("-" ^ 40)

println("\n Testing Expert Efficiency...")

try
    expert = FluxStandardExpert(512, 2048, 512)
    x = randn(Float32, 512, 64)
    
    expert(x)
    
    t_start = time()
    for i in 1:100
        y = expert(x)
    end
    t_expert = time() - t_start
    
    test_assert(t_expert < 10.0, "Expert efficiency timing",
                "Should complete 100 forward passes in <10s, took $(t_expert)s")
    
    println("✓ Expert efficiency test completed")
    
catch e
    println(" FAIL: Expert efficiency test - Error: $e")
end

println("\n Testing MoE Scaling...")

try
    for num_experts in [2, 4, 8]
        println("   Testing with $num_experts experts...")
        moe = FluxMoELayer(128, 512, 128; num_experts=num_experts, top_k=2)
        x = randn(Float32, 128, 32)
        
        y, aux_loss = moe(x; training=true)
        test_size_equals(y, (128, 32), "MoE scaling ($num_experts experts) output")
        test_assert(aux_loss >= 0, "MoE scaling ($num_experts experts) aux_loss",
                    "Auxiliary loss should be non-negative")
    end
    
    println("✓ MoE scaling test completed")
    
catch e
    println(" FAIL: MoE scaling test - Error: $e")
end

println("\n Edge Cases and Error Handling")
println("-" ^ 40)

println("\n Testing Empty Expert Assignment...")

try
    config = FluxMoEConfig(
        input_dim=32,
        hidden_dim=64,
        output_dim=32,
        num_experts=8,
        top_k=1
    )
    
    moe = FluxMoELayer(config)
    x = randn(Float32, 32, 4)
    
    y, aux_loss = moe(x; training=true)
    test_size_equals(y, (32, 4), "Empty expert assignment output")
    test_assert(!any(isnan, y), "Empty expert assignment NaN check",
                "Output should not contain NaN values")
    
    println("✓ Empty expert assignment test completed")
    
catch e
    println(" FAIL: Empty expert assignment test - Error: $e")
end

println("\n Testing Single Token Batch...")

try
    moe = FluxMoELayer(64, 128, 64; num_experts=4, top_k=2)
    x = randn(Float32, 64, 1)
    
    y, aux_loss = moe(x; training=true)
    test_size_equals(y, (64, 1), "Single token batch output")
    
    println("✓ Single token batch test completed")
    
catch e
    println(" FAIL: Single token batch test - Error: $e")
end

println("\n" ^ 2)
println(" All Flux Integration Tests (Phases 1 & 2) Completed!")
println("=" ^ 60)
println(" Phase 1: Expert layers (FluxStandardExpert, FluxGatedExpert)")
println(" Phase 2: Router, gating, load balancing, and complete MoE layer")
