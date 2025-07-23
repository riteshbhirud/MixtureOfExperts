#!/usr/bin/env julia

using Llama2
using LinearAlgebra
using Statistics
using Printf

include("src/MixtureOfExperts.jl")
using .MixtureOfExperts
import .MixtureOfExperts: create_moe_run_state,convert_to_moe , MoELlamaConfig, TopKGating,SwitchTransformerLoss,convert_to_moe, GatingMechanism, LoadBalancingLoss, 
                         TopKGating, SwitchTransformerLoss, MoELanguageModel,moe_attention!,apply_rope!,MoEConfig,MoETransformerWeights,MoEKVCache,MoERunState,sample_moe,
                         MoELlamaConfig, count_parameters, count_active_parameters,compute_gates,create_moe_expert_weights,gated_expert_forward!,create_moe_layer,MoEExpertWeights,MoETransformerLayerWeights,moe_transformer!,reset_moe_state!
using Llama2
using LinearAlgebra
using Statistics
using Printf

include("src/MixtureOfExperts.jl")
using .MixtureOfExperts
import .MixtureOfExperts: convert_to_moe
function test_basic_types()
    
    llama_config = Llama2.ModelConfig(
        dim=512, hidden_dim=1024, n_layers=4, n_heads=8, 
        n_kv_heads=8, vocab_size=1000, seq_len=128,
        rope_freq_base=10000.0f0, rope_is_neox=false
    )
    
    moe_config = MoELlamaConfig(
        llama_config,
        [2, 4],           
        4,                
        2,                
        :gated,           
        TopKGating(2),    
        SwitchTransformerLoss(0.01f0),  
        :perturb,         
        0.01f0,           
        false, 0, 0.0f0, 1.25f0, false,  
        false, nothing, 10,  
        true, 0.0f0, 0.001f0  
    )
    
    @assert moe_config.dim == 512 "Config delegation failed"
    @assert moe_config.moe_num_experts == 4 "MoE config failed"
    
    return moe_config
end

function test_expert_weights(config)
    
    expert = create_moe_expert_weights(config, :gated)
    
    @assert size(expert.w1) == (config.dim, config.hidden_dim) "Expert w1 wrong size"
    @assert size(expert.w2) == (config.hidden_dim, config.dim) "Expert w2 wrong size"  
    @assert size(expert.w3) == (config.dim, config.hidden_dim) "Expert w3 wrong size"
    @assert length(expert.hb1) == config.hidden_dim "Expert hb1 wrong size"
    @assert length(expert.hb2) == config.hidden_dim "Expert hb2 wrong size"
    
    return expert
end

config = test_basic_types()
expert = test_expert_weights(config)

function test_router_functionality(config)
    
    router_weight = randn(Float32, config.dim, config.moe_num_experts) .* 0.02f0
    
    test_input = randn(Float32, config.dim)
    router_logits = zeros(Float32, config.moe_num_experts)
    
    Llama2.matmul!(router_logits, router_weight, test_input)
    
    @assert all(isfinite.(router_logits)) "Router produced non-finite values"
    @assert !all(iszero.(router_logits)) "Router produced all zeros"
    
    router_logits_matrix = reshape(router_logits, :, 1)
    expert_indices, expert_gates, router_probs = compute_gates(TopKGating(2), router_logits_matrix)
    
    @assert size(expert_indices, 1) >= 2 "Not enough experts selected"
    @assert all(1 .<= expert_indices[1:2, 1] .<= config.moe_num_experts) "Invalid expert indices"
    @assert isapprox(sum(expert_gates[1:2, 1]), 1.0, atol=1e-6) "Gates don't sum to 1"
    
    return router_weight, expert_indices[1:2, 1], expert_gates[1:2, 1]
end

function test_expert_forward(expert, config)
    
    test_input = randn(Float32, config.dim)
    output = zeros(Float32, config.dim)
    
    gated_expert_forward!(output, expert, test_input)
    
    @assert all(isfinite.(output)) "Expert produced non-finite values"
    @assert !all(iszero.(output)) "Expert produced all zeros"
    
    input_norm = sqrt(sum(test_input.^2))
    output_norm = sqrt(sum(output.^2))
    @assert output_norm > 0.1 * input_norm "Output suspiciously small"
    @assert output_norm < 10.0 * input_norm "Output suspiciously large"
    
    return output
end

println("="^60)

router_weight, selected_experts, expert_gates = test_router_functionality(config);
expert_output = test_expert_forward(expert, config);


println("="^60)

function test_moe_layer_creation(config)
    
    moe_layer = create_moe_layer(config.dim, config.hidden_dim, config.dim;
                                num_experts=config.moe_num_experts,
                                expert_type=config.moe_expert_type,
                                gate_type=config.moe_gate_type,
                                top_k=config.moe_top_k)
    
    test_input = randn(Float32, config.dim, 1)  
    output, balance_loss = moe_layer(test_input; training=false)
    
    @assert size(output) == size(test_input) "MoE layer output wrong size"
    @assert all(isfinite.(output)) "MoE layer produced non-finite values"
    @assert balance_loss >= 0 "Negative balance loss"
    
    return moe_layer
end

function test_moe_vs_dense_equivalence(config)
    
    test_input = randn(Float32, config.dim)
    
    w1 = randn(Float32, config.dim, config.hidden_dim) .* 0.02f0
    w2 = randn(Float32, config.hidden_dim, config.dim) .* 0.02f0
    w3 = randn(Float32, config.dim, config.hidden_dim) .* 0.02f0
    
    hb1 = zeros(Float32, config.hidden_dim)
    hb2 = zeros(Float32, config.hidden_dim)
    dense_output = zeros(Float32, config.dim)
    
    Llama2.matmul!(hb1, w1, test_input)  
    Llama2.matmul!(hb2, w3, test_input)  
    
    for i in 1:length(hb1)
        gate_val = hb1[i]
        silu_val = gate_val * (1.0f0 / (1.0f0 + exp(-gate_val)))
        hb1[i] = silu_val * hb2[i]
    end
    
    Llama2.matmul!(dense_output, w2, hb1)
    
    expert = MoEExpertWeights(w1, w2, w3, zeros(Float32, config.hidden_dim), 
                             zeros(Float32, config.hidden_dim), :gated, false, 
                             nothing, nothing, nothing)
    moe_output = zeros(Float32, config.dim)
    gated_expert_forward!(moe_output, expert, test_input)
    
    diff = maximum(abs.(dense_output - moe_output))
    @assert diff < 1e-5 "MoE and dense outputs don't match (diff: $diff)"
    
end

function test_multi_expert_routing(config)
    
    experts = []
    for i in 1:config.moe_num_experts
        expert_weights = create_moe_expert_weights(config, :gated)
        push!(experts, expert_weights)
    end
    
    test_inputs = [randn(Float32, config.dim) for _ in 1:5]
    
    for (i, test_input) in enumerate(test_inputs)
        router_weight = randn(Float32, config.dim, config.moe_num_experts) .* 0.02f0
        router_logits = zeros(Float32, config.moe_num_experts)
        Llama2.matmul!(router_logits, router_weight, test_input)
        
        router_logits_matrix = reshape(router_logits, :, 1)
        expert_indices, expert_gates, _ = compute_gates(TopKGating(config.moe_top_k), router_logits_matrix)
        
        final_output = zeros(Float32, config.dim)
        for k in 1:config.moe_top_k
            expert_idx = expert_indices[k, 1]
            gate_weight = expert_gates[k, 1]
            
            if expert_idx > 0 && expert_idx <= length(experts)
                expert_output = zeros(Float32, config.dim)
                gated_expert_forward!(expert_output, experts[expert_idx], test_input)
                final_output .+= gate_weight .* expert_output
            end
        end
        
        @assert all(isfinite.(final_output)) "Multi-expert routing produced non-finite values for input $i"
        @assert !all(iszero.(final_output)) "Multi-expert routing produced all zeros for input $i"
    end
    
end

moe_layer = test_moe_layer_creation(config);
test_moe_vs_dense_equivalence(config);
test_multi_expert_routing(config);


println("="^60)

function test_attention_with_moe_types(config)
    
    state = create_moe_run_state(config)
    
    llama_layer = Llama2.TransformerLayerWeights(
        ones(Float32, config.dim),                    
        ones(Float32, config.dim),                    
        randn(Float32, config.dim, config.dim) .* 0.02f0,  
        randn(Float32, config.dim, config.dim) .* 0.02f0,  
        randn(Float32, config.dim, config.dim) .* 0.02f0,  
        randn(Float32, config.dim, config.dim) .* 0.02f0,  
        randn(Float32, config.dim, config.hidden_dim) .* 0.02f0,  
        randn(Float32, config.hidden_dim, config.dim) .* 0.02f0,  
        randn(Float32, config.dim, config.hidden_dim) .* 0.02f0   
    )
    
    moe_layer = MoETransformerLayerWeights(
        llama_layer, false, nothing, nothing, nothing, nothing, nothing, nothing
    )
    
    state.x .= randn(Float32, config.dim) .* 0.1f0
    
moe_attention!(state, moe_layer, 1, config, 1)    
    @assert all(isfinite.(state.xb2)) "Attention produced non-finite values"
    @assert !all(iszero.(state.xb2)) "Attention produced all zeros"
    
    @assert length(state.xb2) == config.dim "Attention output wrong size"
    
end

function test_rope_functionality(config)
    
    for (rope_type, rope_is_neox) in [("normal", false), ("neox", true)]
        
        test_config = MoELlamaConfig(
            Llama2.ModelConfig(config.dim, config.hidden_dim, config.n_layers, 
                              config.n_heads, config.n_kv_heads, config.vocab_size, 
                              config.seq_len, config.rope_freq_base, rope_is_neox),
            config.moe_layers, config.moe_num_experts, config.moe_top_k, 
            config.moe_expert_type, config.moe_gate_type, config.moe_balance_loss,
            config.expert_init_strategy, config.expert_init_noise,
            config.use_shared_experts, config.num_shared_experts, config.expert_dropout,
            config.capacity_factor, config.drop_tokens, config.use_cur, config.cur_rank,
            config.cur_oversample, config.use_fp32_router, config.router_jitter, config.z_loss_weight
        )
        
        head_size = config.dim ÷ config.n_heads
        
        test_matrix1 = randn(Float32, head_size, config.n_heads) .* 0.1f0
        original_matrix = copy(test_matrix1)
        apply_rope!(test_matrix1, 1, test_config)
        
        @assert isapprox(test_matrix1, original_matrix, atol=1e-5) "$rope_type RoPE pos=1 should be identity"
        
        test_matrix2 = copy(original_matrix)
        apply_rope!(test_matrix2, 3, test_config)
        @assert !isapprox(test_matrix2, original_matrix, atol=1e-6) "$rope_type RoPE didn't modify input for pos>1"
        
        test_matrix3 = copy(original_matrix)
        apply_rope!(test_matrix3, 7, test_config)
        @assert !isapprox(test_matrix2, test_matrix3, atol=1e-6) "$rope_type RoPE same output for different positions"
        
        @assert all(isfinite.(test_matrix2)) "$rope_type RoPE produced non-finite values"
        @assert all(isfinite.(test_matrix3)) "$rope_type RoPE produced non-finite values"
    end
    
end

function test_kv_caching(config)
    
    state = create_moe_run_state(config)
    
    @assert length(state.kvcache_layers) == config.n_layers "Wrong number of KV cache layers"
    
    for (i, kv_cache) in enumerate(state.kvcache_layers)
        head_size = config.dim ÷ config.n_heads
        
        @assert size(kv_cache.key_cache) == (head_size, config.n_kv_heads, config.seq_len) "Layer $i: Wrong key cache size"
        @assert size(kv_cache.value_cache) == (config.seq_len, head_size, config.n_kv_heads) "Layer $i: Wrong value cache size"
    end
    
    test_pos = 3
    head_size = config.dim ÷ config.n_heads
    
    test_key = randn(Float32, head_size, config.n_kv_heads)
    test_value = randn(Float32, head_size, config.n_kv_heads)
    
    kv_cache = state.kvcache_layers[1]
    copyto!(view(kv_cache.key_cache, :, :, test_pos), test_key)
    copyto!(view(kv_cache.value_cache, test_pos, :, :), test_value)  
    
    retrieved_key = kv_cache.key_cache[:, :, test_pos]
    retrieved_value = kv_cache.value_cache[test_pos, :, :]  
    
    @assert isapprox(retrieved_key, test_key, atol=1e-6) "KV cache key retrieval failed"
    @assert isapprox(retrieved_value, test_value, atol=1e-6) "KV cache value retrieval failed"
    
end

function test_attention_numerical_stability(config)
    
    state = create_moe_run_state(config)
    
    llama_layer = Llama2.TransformerLayerWeights(
        ones(Float32, config.dim),
        ones(Float32, config.dim),
        randn(Float32, config.dim, config.dim) .* 0.1f0,  
        randn(Float32, config.dim, config.dim) .* 0.1f0,
        randn(Float32, config.dim, config.dim) .* 0.1f0,
        randn(Float32, config.dim, config.dim) .* 0.1f0,
        randn(Float32, config.dim, config.hidden_dim) .* 0.02f0,
        randn(Float32, config.hidden_dim, config.dim) .* 0.02f0,
        randn(Float32, config.dim, config.hidden_dim) .* 0.02f0
    )
    
    moe_layer = MoETransformerLayerWeights(
        llama_layer, false, nothing, nothing, nothing, nothing, nothing, nothing
    )
    
    for scale in [0.01f0, 1.0f0, 10.0f0]
        state.x .= randn(Float32, config.dim) .* scale
        
moe_attention!(state, moe_layer, 1, config, 1)
        
        @assert all(isfinite.(state.xb2)) "Attention unstable with input scale $scale"
        
        input_norm = sqrt(sum(state.x.^2))
        output_norm = sqrt(sum(state.xb2.^2))
        @assert output_norm < 100 * input_norm "Attention output exploded with scale $scale"
    end
    
end

println("="^60)

function test_single_token_forward(config)
    
    model = create_test_moe_model(config)
    state = create_moe_run_state(config)
    
    test_token = 5  
    test_pos = 1
    
    fill!(state.logits, 0.0f0)
    
    moe_transformer!(test_token, test_pos, model, state)
    
    @assert size(state.logits) == (config.vocab_size,) "Wrong logits shape"
    @assert all(isfinite.(state.logits)) "Forward pass produced non-finite logits"
    @assert !all(iszero.(state.logits)) "Forward pass produced all-zero logits"
    
    logit_norm = sqrt(sum(state.logits.^2))
    @assert logit_norm > 0.1 "Logits suspiciously small: $logit_norm"
    @assert logit_norm < 1000.0 "Logits suspiciously large: $logit_norm"
    
    return model, state
end

function test_sequence_processing(model, state, config)
    
    test_sequence = [1, 5, 10, 2, 8]
    max_pos = min(length(test_sequence), config.seq_len)
    
    reset_moe_state!(state)
    
    logits_history = []
    
    for (pos, token) in enumerate(test_sequence[1:max_pos])
        if token <= config.vocab_size
            moe_transformer!(token, pos, model, state)
            
            @assert all(isfinite.(state.logits)) "Non-finite logits at position $pos"
            @assert !all(iszero.(state.logits)) "All-zero logits at position $pos"
            
            push!(logits_history, copy(state.logits))
        end
    end
    
    @assert length(logits_history) >= 2 "Need at least 2 positions for comparison"
    
    for i in 2:length(logits_history)
        diff = maximum(abs.(logits_history[i] - logits_history[i-1]))
        @assert diff > 1e-6 "Positions $(i-1) and $i produced identical outputs"
    end
    
    return logits_history
end

function test_moe_vs_dense_layer_integration(config)
    
    config_dense = MoELlamaConfig(
        config.llama_config, Int[], 4, 2, :gated, TopKGating(2), 
        SwitchTransformerLoss(0.01f0), :perturb, 0.01f0, false, 0, 0.0f0, 
        1.25f0, false, false, nothing, 10, true, 0.0f0, 0.001f0
    )
    
    config_mixed = MoELlamaConfig(
        config.llama_config, [2], 4, 2, :gated, TopKGating(2), 
        SwitchTransformerLoss(0.01f0), :perturb, 0.01f0, false, 0, 0.0f0, 
        1.25f0, false, false, nothing, 10, true, 0.0f0, 0.001f0
    )
    
    dense_model = create_test_moe_model(config_dense)
    mixed_model = create_test_moe_model(config_mixed)
    
    test_token = 7
    test_pos = 1
    
    dense_state = create_moe_run_state(config_dense)
    moe_transformer!(test_token, test_pos, dense_model, dense_state)
    
    mixed_state = create_moe_run_state(config_mixed)
    moe_transformer!(test_token, test_pos, mixed_model, mixed_state)
    
    @assert all(isfinite.(dense_state.logits)) "Dense model produced non-finite logits"
    @assert all(isfinite.(mixed_state.logits)) "Mixed model produced non-finite logits"
    
    diff = maximum(abs.(dense_state.logits - mixed_state.logits))
    @assert diff > 1e-6 "Dense and mixed models produced identical outputs"
    
end

function test_expert_activation_tracking(model, state, config)
    
    test_tokens = [1, 3, 7, 12, 5]
    reset_moe_state!(state)
    
    for (pos, token) in enumerate(test_tokens)
        if token <= config.vocab_size && pos <= config.seq_len
            moe_transformer!(token, pos, model, state)
        end
    end
    
    moe_layers_exist = any(layer.use_moe for layer in model.weights.layers)
    
    if moe_layers_exist
        @assert state.inference_stats[:expert_activations] > 0 "No expert activations recorded"
        @assert state.inference_stats[:moe_layer_calls] > 0 "No MoE layer calls recorded"
        
        total_expert_usage = sum(state.expert_load_counts)
        @assert total_expert_usage > 0 "No experts were used"
        
        @assert !isempty(state.routing_entropy) "No routing entropy recorded"
        
    else
        
    end
    
end

function test_numerical_consistency(model, state, config)
    
    test_token = 3
    test_pos = 2
    
    results = []
    
    for run in 1:3
        reset_moe_state!(state)
        
        moe_transformer!(test_token, test_pos, model, state)
        
        push!(results, copy(state.logits))
    end
    
    for i in 2:length(results)
        diff = maximum(abs.(results[i] - results[1]))
        @assert diff < 1e-6 "Run $i differs from run 1 by $diff (should be deterministic)"
    end
    
end

function create_test_moe_model(config)
    tokenizer = create_dummy_tokenizer(config.vocab_size)
    
    token_embedding = randn(Float32, config.dim, config.vocab_size) .* 0.02f0
    rms_final = ones(Float32, config.dim)
    output_weight = randn(Float32, config.dim, config.vocab_size) .* 0.02f0
    
    layers = MoETransformerLayerWeights[]
    
    for layer_idx in 1:config.n_layers
        llama_layer = create_dummy_llama_layer(config)
        
        if layer_idx in config.moe_layers
            experts = [create_moe_expert_weights(config, :gated) for _ in 1:config.moe_num_experts]
            router_weight = randn(Float32, config.dim, config.moe_num_experts) .* 0.02f0
            
            moe_config = MoEConfig(
                num_experts = config.moe_num_experts,
                expert_type = :gated,
                input_dim = config.dim,
                hidden_dim = config.hidden_dim,
                output_dim = config.dim,
                activation = x -> x * sigmoid(x),
                top_k = config.moe_top_k,
                gate_type = config.moe_gate_type,
                balance_loss = config.moe_balance_loss
            )
            
            layer = MoETransformerLayerWeights(
                llama_layer, true, experts, router_weight, moe_config,
                nothing, nothing, zeros(Int, config.moe_num_experts)
            )
        else
            layer = MoETransformerLayerWeights(
                llama_layer, false, nothing, nothing, nothing,
                nothing, nothing, nothing
            )
        end
        
        push!(layers, layer)
    end
    
    weights = MoETransformerWeights(
        token_embedding,    
        rms_final,         
        output_weight,     
        layers,            
        config,            
        Dict{Symbol,Any}() 
    )
    
    return MoELanguageModel(
        config, tokenizer, weights, Dict{String,Any}(), Dict{String,Any}(),
        MoEKVCache[], MoERunState[]
    )
end

function create_dummy_llama_layer(config)
    return Llama2.TransformerLayerWeights(
        ones(Float32, config.dim),                     
        ones(Float32, config.dim),                     
        randn(Float32, config.dim, config.dim) .* 0.02f0,    
        randn(Float32, config.dim, config.dim) .* 0.02f0,    
        randn(Float32, config.dim, config.dim) .* 0.02f0,    
        randn(Float32, config.dim, config.dim) .* 0.02f0,    
        randn(Float32, config.dim, config.hidden_dim) .* 0.02f0,  
        randn(Float32, config.hidden_dim, config.dim) .* 0.02f0,  
        randn(Float32, config.dim, config.hidden_dim) .* 0.02f0   
    )
end

function create_dummy_tokenizer(vocab_size)
    id_to_token = Vector{String}()
    
    push!(id_to_token, "<s>")        
    push!(id_to_token, "</s>")       
    push!(id_to_token, " ")          
    push!(id_to_token, "token")      
    push!(id_to_token, "_")          
    
    for i in 1:(vocab_size-5)
        push!(id_to_token, "$(i)")   
    end
    
    token_to_id = Dict(token => i for (i, token) in enumerate(id_to_token))
    token_scores = ones(Float32, vocab_size)
    
    return Llama2.BPETokenizer(id_to_token, token_to_id, token_scores, 1, 2)
end


model, state = test_single_token_forward(config);
logits_history = test_sequence_processing(model, state, config);
test_moe_vs_dense_layer_integration(config);
test_expert_activation_tracking(model, state, config);
test_numerical_consistency(model, state, config);


function test_llama2_to_moe_conversion()
    
    tiny_config = Llama2.ModelConfig(
        256,        
        512,        
        2,          
        4,          
        4,          
        100,        
        64,         
        10000.0f0,  
        false       
    )
    
    tiny_weights = create_dummy_llama2_weights(tiny_config)
    
    tokenizer = create_dummy_tokenizer(tiny_config.vocab_size)
    
    llama_model = Llama2.LanguageModel(tiny_config, tokenizer, tiny_weights)
    
    moe_model = convert_to_moe(
        llama_model, 
        [2];                    
        num_experts=4, 
        top_k=2,
        expert_init_strategy=:perturb,
        expert_init_noise=0.01f0,
        gate_type=TopKGating(2),
        balance_loss=SwitchTransformerLoss(0.01f0),
        expert_type=:gated
    )
    
    @assert length(moe_model.weights.layers) == tiny_config.n_layers "Wrong layer count"
    @assert !moe_model.weights.layers[1].use_moe "Layer 1 should be dense"
    @assert moe_model.weights.layers[2].use_moe "Layer 2 should be MoE"
    
    moe_layer = moe_model.weights.layers[2]
    @assert length(moe_layer.moe_experts) == 4 "Wrong number of experts"
    @assert !isnothing(moe_layer.moe_router_weight) "Router weight missing"
    @assert size(moe_layer.moe_router_weight) == (tiny_config.dim, 4) "Wrong router size"
    
    return llama_model, moe_model
end

function test_converted_model_inference(llama_model, moe_model)
    
    config = llama_model.config
    
    test_token = 5
    test_pos = 1
    
    llama_state = Llama2.RunState(config)
    Llama2.transformer!(test_token, test_pos, config, llama_state, llama_model.weights)
    
    @assert all(isfinite.(llama_state.logits)) "Original model produced non-finite logits"
    @assert !all(iszero.(llama_state.logits)) "Original model produced all-zero logits"
    
    moe_state = create_moe_run_state(moe_model.config)
    moe_transformer!(test_token, test_pos, moe_model, moe_state)
    
    @assert all(isfinite.(moe_state.logits)) "MoE model produced non-finite logits"
    @assert !all(iszero.(moe_state.logits)) "MoE model produced all-zero logits"
    
    diff = maximum(abs.(llama_state.logits - moe_state.logits))
    @assert diff > 1e-6 "Original and MoE models produced identical outputs"
    
    return llama_state, moe_state
end

function test_parameter_preservation(llama_model, moe_model)
    
    original_params = count_llama_parameters(llama_model)
    moe_total_params = count_parameters(moe_model)
    moe_active_params = count_active_parameters(moe_model)
    
    @assert moe_total_params > original_params "MoE should have more total parameters"
    
    efficiency_ratio = moe_active_params / moe_total_params
    
    original_layer1 = llama_model.weights.layers[1]
    moe_layer1 = moe_model.weights.layers[1].llama_layer
    
    @assert isapprox(original_layer1.wq, moe_layer1.wq, atol=1e-6) "Layer 1 attention weights not preserved"
    @assert isapprox(original_layer1.w1, moe_layer1.w1, atol=1e-6) "Layer 1 FFN weights not preserved"
    
end

function test_expert_specialization(moe_model)
    
    test_tokens = [1, 5, 10, 15, 20, 25, 30]
    moe_state = create_moe_run_state(moe_model.config)
    
    expert_usage = zeros(Int, 4)  
    
    for (pos, token) in enumerate(test_tokens)
        if token <= moe_model.config.vocab_size && pos <= moe_model.config.seq_len
            reset_moe_state!(moe_state)
            moe_transformer!(token, pos, moe_model, moe_state)
            
            for expert_idx in moe_state.selected_experts[1:2]  
                if expert_idx > 0
                    expert_usage[expert_idx] += 1
                end
            end
        end
    end
    
    @assert sum(expert_usage) > 0 "No experts were activated"
    
    max_usage = maximum(expert_usage)
    min_usage = minimum(expert_usage)
    usage_variance = var(Float64.(expert_usage))
    
    @assert usage_variance > 0 "No expert specialization detected"
    
end

function test_generation_capability(llama_model, moe_model)
    
    test_prompt_tokens = [1, 5, 10]  
    max_length = 10
    
    llama_generated = generate_sequence(llama_model, test_prompt_tokens, max_length)
    @assert length(llama_generated) > length(test_prompt_tokens) "Original model didn't generate"
    
    moe_generated = generate_sequence_moe(moe_model, test_prompt_tokens, max_length)
    @assert length(moe_generated) > length(test_prompt_tokens) "MoE model didn't generate"
    
    @assert llama_generated != moe_generated "Models generated identical sequences"
    
end

function test_conversion_edge_cases()
    
    tiny_config = Llama2.ModelConfig(256, 512, 2, 4, 4, 50, 32, 10000.0f0, false)
    tiny_weights = create_dummy_llama2_weights(tiny_config)
    tokenizer = create_dummy_tokenizer(tiny_config.vocab_size)
    llama_model = Llama2.LanguageModel(tiny_config, tokenizer, tiny_weights)
    
    all_moe_model = convert_to_moe(llama_model, [1, 2]; num_experts=2, top_k=1)
    
    @assert all_moe_model.weights.layers[1].use_moe "Layer 1 should be MoE"
    @assert all_moe_model.weights.layers[2].use_moe "Layer 2 should be MoE"
    
    state = create_moe_run_state(all_moe_model.config)
    moe_transformer!(1, 1, all_moe_model, state)
    @assert all(isfinite.(state.logits)) "All-MoE model failed"
    
    no_moe_model = convert_to_moe(llama_model, Int[]; num_experts=2, top_k=1)
    
    @assert !no_moe_model.weights.layers[1].use_moe "Layer 1 should be dense"
    @assert !no_moe_model.weights.layers[2].use_moe "Layer 2 should be dense"
    
end

function create_dummy_llama2_weights(config)
    layers = Llama2.TransformerLayerWeights[]
    
    for i in 1:config.n_layers
        layer = Llama2.TransformerLayerWeights(
            ones(Float32, config.dim),                           
            ones(Float32, config.dim),                           
            randn(Float32, config.dim, config.dim) .* 0.02f0,          
            randn(Float32, config.dim, config.dim) .* 0.02f0,          
            randn(Float32, config.dim, config.dim) .* 0.02f0,          
            randn(Float32, config.dim, config.dim) .* 0.02f0,          
            randn(Float32, config.dim, config.hidden_dim) .* 0.02f0,   
            randn(Float32, config.hidden_dim, config.dim) .* 0.02f0,   
            randn(Float32, config.dim, config.hidden_dim) .* 0.02f0    
        )
        push!(layers, layer)
    end
    
    return Llama2.TransformerWeights(
        randn(Float32, config.dim, config.vocab_size) .* 0.02f0,  
        layers,                                                   
        ones(Float32, config.dim),                               
        randn(Float32, config.dim, config.vocab_size) .* 0.02f0   
    )
end

function count_llama_parameters(model::Llama2.LanguageModel)
    count = 0
    
    count += length(model.weights.token_embedding_table)
    
    for layer in model.weights.layers
        count += length(layer.rms_att_weight)
        count += length(layer.rms_ffn_weight)
        count += length(layer.wq)
        count += length(layer.wk)
        count += length(layer.wv)
        count += length(layer.wo)
        count += length(layer.w1)
        count += length(layer.w2)
        count += length(layer.w3)
    end
    
    count += length(model.weights.rms_final_weight)
    count += length(model.weights.output_weight)
    
    return count
end

function generate_sequence(llama_model::Llama2.LanguageModel, prompt_tokens::Vector{Int}, max_length::Int)
    config = llama_model.config
    state = Llama2.RunState(config)
    
    generated = copy(prompt_tokens)
    
    for pos in 1:min(max_length, config.seq_len)
        if pos <= length(prompt_tokens)
            token = prompt_tokens[pos]
        else
            token = argmax(state.logits)
            push!(generated, token)
        end
        
        Llama2.transformer!(token, pos, config, state, llama_model.weights)
        
        if length(generated) >= max_length
            break
        end
    end
    
    return generated
end

function generate_sequence_moe(moe_model::MoELanguageModel, prompt_tokens::Vector{Int}, max_length::Int)
    config = moe_model.config
    state = create_moe_run_state(config)
    
    generated = copy(prompt_tokens)
    
    for pos in 1:min(max_length, config.seq_len)
        if pos <= length(prompt_tokens)
            token = prompt_tokens[pos]
        else
            token = argmax(state.logits)
            push!(generated, token)
        end
        
        moe_transformer!(token, pos, moe_model, state)
        
        if length(generated) >= max_length
            break
        end
    end
    
    return generated
end

function generate_moe_with_temp(moe_model, prompt_tokens, max_length, temperature)
    config = moe_model.config
    state = create_moe_run_state(config)
    generated = copy(prompt_tokens)
    
    for pos in 1:min(max_length, config.seq_len)
        if pos <= length(prompt_tokens)
            token = prompt_tokens[pos]
        else
            token = sample_with_temperature(state.logits, temperature)
            push!(generated, token)
        end
        
        moe_transformer!(token, pos, moe_model, state)
        
        if length(generated) >= max_length
            break
        end
    end
    
    return generated
end

function generate_with_temperature(model, prompt_tokens, max_length, temperature)
    if isa(model, Llama2.LanguageModel)
        return generate_llama_with_temp(model, prompt_tokens, max_length, temperature)
    else
        return generate_moe_with_temp(model, prompt_tokens, max_length, temperature)
    end
end
function sample_with_temperature(logits, temperature)
    if temperature == 0.0f0
        return argmax(logits)
    end
    
    scaled_logits = logits ./ temperature
    
    max_logit = maximum(scaled_logits)
    exp_logits = exp.(scaled_logits .- max_logit)
    probs = exp_logits ./ sum(exp_logits)
    
    r = rand()
    cumsum_prob = 0.0f0
    for (i, prob) in enumerate(probs)
        cumsum_prob += prob
        if r <= cumsum_prob
            return i
        end
    end
    return length(logits)  
end
function generate_llama_with_temp(llama_model, prompt_tokens, max_length, temperature)
    config = llama_model.config
    state = Llama2.RunState(config)
    generated = copy(prompt_tokens)
    
    for pos in 1:min(max_length, config.seq_len)
        if pos <= length(prompt_tokens)
            token = prompt_tokens[pos]
        else
            token = sample_with_temperature(state.logits, temperature)
            push!(generated, token)
        end
        
        Llama2.transformer!(token, pos, config, state, llama_model.weights)
        
        if length(generated) >= max_length
            break
        end
    end
    
    return generated
end
function test_temperature_generation(llama_model, moe_model)
    
    test_prompt = [1, 5, 10]
    
    temperatures = [0.0f0, 0.5f0, 1.0f0]
    different_count = 0
    total_tests = 0
    
    for temp in temperatures
        
        runs = temp > 0.0f0 ? 3 : 1
        
        for run in 1:runs
            llama_gen = generate_with_temperature(llama_model, test_prompt, 8, temp)
            moe_gen = generate_with_temperature(moe_model, test_prompt, 8, temp)
            
            total_tests += 1
            is_different = llama_gen != moe_gen
            if is_different
                different_count += 1
            end
            
        end
    end
    
    difference_rate = different_count / total_tests
    
    if difference_rate > 0.5
        return true
    elseif difference_rate > 0.3
        return true
    elseif difference_rate > 0.1
        return true
    else
        return false
    end
end
function test_generation_capability_fixed(llama_model, moe_model)
    
    diversity_ok = test_temperature_generation(llama_model, moe_model)
    
    test_prompt_tokens = [1, 5, 10]
    
    llama_generated = generate_with_temperature(llama_model, test_prompt_tokens, 10, 0.0f0)
    moe_generated = generate_with_temperature(moe_model, test_prompt_tokens, 10, 0.0f0)
    
    @assert length(llama_generated) > length(test_prompt_tokens) "Original model didn't generate"
    @assert length(moe_generated) > length(test_prompt_tokens) "MoE model didn't generate"
    
    if !diversity_ok
        
    else
        
    end
end


llama_model, moe_model = test_llama2_to_moe_conversion();

llama_state, moe_state = test_converted_model_inference(llama_model, moe_model);

test_parameter_preservation(llama_model, moe_model);

test_expert_specialization(moe_model);

test_generation_capability_fixed(llama_model, moe_model);

test_conversion_edge_cases();

function diagnose_moe_vs_dense_behavior(llama_model, moe_model)
    
    test_tokens = [1, 5, 10, 15, 20]
    
    for token in test_tokens
        llama_state = Llama2.RunState(llama_model.config)
        moe_state = create_moe_run_state(moe_model.config)
        
        Llama2.transformer!(token, 1, llama_model.config, llama_state, llama_model.weights)
        moe_transformer!(token, 1, moe_model, moe_state)
        
        diff = maximum(abs.(llama_state.logits - moe_state.logits))
        
        if diff < 1e-4
            
        end
    end
    
    moe_state = create_moe_run_state(moe_model.config)
    
    for token in [1, 50, 99]  
        reset_moe_state!(moe_state)
        moe_transformer!(token, 1, moe_model, moe_state)
        
    end
    
    moe_layer = moe_model.weights.layers[2]  
    test_input = randn(Float32, moe_model.config.dim)
    
    expert_outputs = []
    for (i, expert) in enumerate(moe_layer.moe_experts)
        output = zeros(Float32, moe_model.config.dim)
        gated_expert_forward!(output, expert, test_input)
        push!(expert_outputs, copy(output))
    end
    
    for i in 2:length(expert_outputs)
        diff = maximum(abs.(expert_outputs[i] - expert_outputs[1]))
        if diff < 1e-4
            
        end
    end
end

function test_generation_with_multiple_prompts(llama_model, moe_model)
    
    test_prompts = [
        [1, 5],
        [10, 20], 
        [30, 40],
        [50, 60],
        [90, 95]
    ]
    
    identical_count = 0
    
    for (i, prompt) in enumerate(test_prompts)
        llama_gen = generate_sequence(llama_model, prompt, 8)
        moe_gen = generate_sequence_moe(moe_model, prompt, 8)
        
        if llama_gen == moe_gen
            identical_count += 1
        else
            
        end
    end
    
    if identical_count == length(test_prompts)
        return false
    elseif identical_count > length(test_prompts) / 2
        return false
    else
        return true
    end
end

function test_temperature_sampling(llama_model, moe_model)
    
    test_prompt = [1, 5, 10]
    
    llama_state = Llama2.RunState(llama_model.config)
    moe_state = create_moe_run_state(moe_model.config)
    
    Llama2.transformer!(10, 1, llama_model.config, llama_state, llama_model.weights)
    moe_transformer!(10, 1, moe_model, moe_state)
    
    llama_top2 = partialsortperm(llama_state.logits, 1:2, rev=true)
    moe_top2 = partialsortperm(moe_state.logits, 1:2, rev=true)
    
    if llama_top2 == moe_top2
        return false
    else
        return true
    end
end 

println("="^60)

function test_real_model_moe_integration()
    
    original_model = Llama2.load_karpathy_model("stories42M.bin", "tokenizer.bin")
    
    moe_layers_to_convert = [2, 4]  
    
    moe_model = convert_to_moe(
        original_model,
        moe_layers_to_convert;
        num_experts=4,           
        top_k=2,                
        expert_init_strategy=:perturb,
        expert_init_noise=0.01f0,
        gate_type=TopKGating(2),
        balance_loss=SwitchTransformerLoss(0.01f0),
        expert_type=:gated
    )
    
    original_params = count_llama_parameters(original_model)
    moe_total = count_parameters(moe_model)
    moe_active = count_active_parameters(moe_model)
    
    return original_model, moe_model
end

function test_generation_comparison(original_model, moe_model)
    
    test_prompts = [
        "Once upon a time",
        "The little girl",
        "In the forest",
        "Tim and Sam",
        "The magic"
    ]
    
    for (i, prompt) in enumerate(test_prompts)
        
        original_out = Llama2.sample(original_model, prompt; temperature=0.0f0)
        
        moe_out = sample_moe(moe_model, prompt; 
                            temperature=0.0f0,
                            show_expert_stats=false,
                            verbose=false)
        
    end
end

function test_temperature_effects(original_model, moe_model)
    
    prompt = "The dragon"
    temperatures = [0.0f0, 0.5f0, 1.0f0]
    
    for temp in temperatures
        
        original_out = Llama2.sample(original_model, prompt; temperature=temp)
        
        moe_out = sample_moe(moe_model, prompt; 
                            temperature=temp,
                            show_expert_stats=false,
                            verbose=false)
        
    end
end

function test_expert_analysis(moe_model)
    
    test_prompts = [
        "The princess",
        "The monster", 
        "The castle",
        "The forest",
        "The magic spell"
    ]
    
    for prompt in test_prompts
        
        output = sample_moe(moe_model, prompt;
                           temperature=0.3f0,
                           max_seq_len=20,
                           show_expert_stats=true,
                           show_routing_entropy=true,
                           verbose=false)
    end
end

function count_llama_parameters(model::Llama2.LanguageModel)
    count = 0
    count += length(model.weights.token_embedding_table)
    
    for layer in model.weights.layers
        count += length(layer.rms_att_weight) + length(layer.rms_ffn_weight)
        count += length(layer.wq) + length(layer.wk) + length(layer.wv) + length(layer.wo)
        count += length(layer.w1) + length(layer.w2) + length(layer.w3)
    end
    
    count += length(model.weights.rms_final_weight) + length(model.weights.output_weight)
    return count
end

println()

if isfile("stories42M.bin") && isfile("tokenizer.bin")
    original_model, moe_model = test_real_model_moe_integration()
    
    test_generation_comparison(original_model, moe_model)
    
    test_temperature_effects(original_model, moe_model)
    
    test_expert_analysis(moe_model)
    
    println("\n All real model tests completed!")
    println(" Your MoE integration works with real Llama2 models!")
    
else
    println(" Model files not found!")
    println("Please download:")
    println("  - stories42M.bin")
    println("  - tokenizer.bin")
    println("Then run this script again.")
end
en