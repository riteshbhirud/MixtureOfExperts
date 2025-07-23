"""
Utility Functions for MoE-Llama2 Integration

This module provides save/load, analysis, debugging, and comparison utilities
for MoE-enabled Llama2 models.
"""

using Serialization
using JSON3
using Dates

"""
    save_moe_model(model::MoELanguageModel, filename::String; format::Symbol = :julia)

Save MoE model to disk with comprehensive metadata.

# Arguments
- `model`: MoE model to save
- `filename`: Output file path
- `format`: Save format (:julia for .jls, :json for metadata, :both for both)
"""
function save_moe_model(model::MoELanguageModel, filename::String; format::Symbol = :julia)
    println("Saving MoE model to $filename...")
    
    if format ∈ [:julia, :both]
        save_julia_format(model, filename)
    end
    
    if format ∈ [:json, :both]
        metadata_file = replace(filename, r"\.[^.]*$" => "_metadata.json")
        save_metadata_json(model, metadata_file)
    end
    
    println(" Model saved successfully!!")
    return nothing
end

"""
    save_julia_format(model::MoELanguageModel, filename::String)

Save model in Julia serialization format.
"""
function save_julia_format(model::MoELanguageModel, filename::String)
    save_data = Dict{String, Any}()
    
    save_data["config"] = serialize_config(model.config)
    
    save_data["tokenizer"] = model.tokenizer
    
    save_data["weights"] = serialize_weights(model.weights)
    
    save_data["metadata"] = Dict(
        "model_type" => "MoELanguageModel",
        "save_timestamp" => string(now()),
        "library_version" => "1.0.0",
        "original_model_info" => model.original_model_info,
        "conversion_info" => model.moe_conversion_info
    )
    
    save_data["statistics"] = compute_model_statistics(model)
    
    open(filename, "w") do f
        serialize(f, save_data)
    end
    
    return nothing
end

"""
    load_moe_model(filename::String) -> MoELanguageModel

Load MoE model from disk.
"""
function load_moe_model(filename::String)
    println("Loading MoE model from $filename...")
    
    if !isfile(filename)
        throw(ArgumentError("File not found: $filename"))
    end
    
    save_data = open(filename, "r") do f
        deserialize(f)
    end
    
    validate_save_data(save_data)
    
    config = deserialize_config(save_data["config"])
    
    weights = deserialize_weights(save_data["weights"], config)
    
    tokenizer = save_data["tokenizer"]
    
    metadata = save_data["metadata"]
    original_info = get(metadata, "original_model_info", Dict{String, Any}())
    conversion_info = get(metadata, "conversion_info", Dict{String, Any}())
    
    model = MoELanguageModel(
        config,
        tokenizer,
        weights,
        original_info,
        conversion_info,
        MoEKVCache[],    
        MoERunState[]    
    )
    
    println("✓ Model loaded successfully")
    println("  Save timestamp: $(get(metadata, "save_timestamp", "unknown"))")
    println("  Model layers: $(config.n_layers)")
    println("  MoE layers: $(length(config.moe_layers))")
    
    return model
end

"""
    serialize_config(config::MoELlamaConfig) -> Dict

Serialize MoE configuration to dictionary.
"""
function serialize_config(config::MoELlamaConfig)
    return Dict{String, Any}(
        "dim" => config.dim,
        "hidden_dim" => config.hidden_dim,
        "n_layers" => config.n_layers,
        "n_heads" => config.n_heads,
        "n_kv_heads" => config.n_kv_heads,
        "vocab_size" => config.vocab_size,
        "seq_len" => config.seq_len,
        "rope_freq_base" => config.rope_freq_base,
        "rope_is_neox" => config.rope_is_neox,
        
        "moe_layers" => collect(config.moe_layers),
        "moe_num_experts" => config.moe_num_experts,
        "moe_top_k" => config.moe_top_k,
        "moe_expert_type" => config.moe_expert_type,
        "expert_init_strategy" => config.expert_init_strategy,
        "expert_init_noise" => config.expert_init_noise,
        "use_shared_experts" => config.use_shared_experts,
        "num_shared_experts" => config.num_shared_experts,
        "expert_dropout" => config.expert_dropout,
        "capacity_factor" => config.capacity_factor,
        "drop_tokens" => config.drop_tokens,
        "use_cur" => config.use_cur,
        "cur_rank" => config.cur_rank,
        "cur_oversample" => config.cur_oversample,
        "use_fp32_router" => config.use_fp32_router,
        "router_jitter" => config.router_jitter,
        "z_loss_weight" => config.z_loss_weight,
        
        "gate_type_name" => string(typeof(config.moe_gate_type)),
        "balance_loss_name" => string(typeof(config.moe_balance_loss))
    )
end

"""
    deserialize_config(config_dict::Dict) -> MoELlamaConfig

Reconstruct MoE configuration from dictionary.
"""
function deserialize_config(config_dict::Dict)
    llama_config = Llama2.ModelConfig(
        config_dict["dim"],
        config_dict["hidden_dim"],
        config_dict["n_layers"],
        config_dict["n_heads"],
        config_dict["n_kv_heads"],
        config_dict["vocab_size"],
        config_dict["seq_len"],
        config_dict["rope_freq_base"],
        config_dict["rope_is_neox"]
    )
    
    gate_type = reconstruct_gate_type(
        config_dict["gate_type_name"], 
        config_dict["moe_top_k"]
    )
    
    balance_loss = reconstruct_balance_loss(
        config_dict["balance_loss_name"]
    )
    
    return MoELlamaConfig(
        llama_config,
        config_dict["moe_layers"],
        config_dict["moe_num_experts"],
        config_dict["moe_top_k"],
        Symbol(config_dict["moe_expert_type"]),
        gate_type,
        balance_loss,
        Symbol(config_dict["expert_init_strategy"]),
        config_dict["expert_init_noise"],
        config_dict["use_shared_experts"],
        config_dict["num_shared_experts"],
        config_dict["expert_dropout"],
        config_dict["capacity_factor"],
        config_dict["drop_tokens"],
        config_dict["use_cur"],
        config_dict["cur_rank"],
        config_dict["cur_oversample"],
        config_dict["use_fp32_router"],
        config_dict["router_jitter"],
        config_dict["z_loss_weight"]
    )
end

"""
    serialize_weights(weights::MoETransformerWeights) -> Dict

Serialize model weights to dictionary format.
"""
function serialize_weights(weights::MoETransformerWeights)
    weights_dict = Dict{String, Any}()
    
    weights_dict["token_embedding_table"] = copy(weights.token_embedding_table)
    weights_dict["rms_final_weight"] = copy(weights.rms_final_weight)
    weights_dict["output_weight"] = copy(weights.output_weight)
    
    weights_dict["layers"] = []
    
    for (layer_idx, layer) in enumerate(weights.layers)
        layer_dict = serialize_layer_weights(layer, layer_idx)
        push!(weights_dict["layers"], layer_dict)
    end
    
    return weights_dict
end

"""
    serialize_layer_weights(layer::MoETransformerLayerWeights, layer_idx::Int) -> Dict

Serialize a single layer's weights.
"""
function serialize_layer_weights(layer::MoETransformerLayerWeights, layer_idx::Int)
    layer_dict = Dict{String, Any}(
        "layer_index" => layer_idx,
        "use_moe" => layer.use_moe
    )
    
    llama_layer = layer.llama_layer
    layer_dict["attention"] = Dict(
        "rms_att_weight" => copy(llama_layer.rms_att_weight),
        "wq" => copy(llama_layer.wq),
        "wk" => copy(llama_layer.wk),
        "wv" => copy(llama_layer.wv),
        "wo" => copy(llama_layer.wo)
    )
    
    if layer.use_moe
        layer_dict["moe"] = serialize_moe_layer(layer)
    else
        layer_dict["ffn"] = Dict(
            "rms_ffn_weight" => copy(llama_layer.rms_ffn_weight),
            "w1" => copy(llama_layer.w1),
            "w2" => copy(llama_layer.w2),
            "w3" => copy(llama_layer.w3)
        )
    end
    
    return layer_dict
end

"""
    serialize_moe_layer(layer::MoETransformerLayerWeights) -> Dict

Serialize MoE-specific components of a layer.
"""
function serialize_moe_layer(layer::MoETransformerLayerWeights)
    moe_dict = Dict{String, Any}(
        "router_weight" => copy(layer.moe_router_weight),
        "num_experts" => length(layer.moe_experts),
        "experts" => []
    )
    
    for (expert_idx, expert) in enumerate(layer.moe_experts)
        expert_dict = Dict{String, Any}(
            "expert_index" => expert_idx,
            "expert_type" => expert.expert_type,
            "is_cur_compressed" => expert.is_cur_compressed,
            "w1" => copy(expert.w1),
            "w2" => copy(expert.w2),
            "w3" => copy(expert.w3)
        )
        
        if expert.is_cur_compressed && !isnothing(expert.cur_c)
            expert_dict["cur_components"] = Dict(
                "C" => copy(expert.cur_c),
                "U" => copy(expert.cur_u),
                "R" => copy(expert.cur_r)
            )
        end
        
        push!(moe_dict["experts"], expert_dict)
    end
    
    if !isnothing(layer.shared_experts)
        moe_dict["shared_experts"] = []
        for shared_expert in layer.shared_experts
            shared_dict = Dict{String, Any}(
                "expert_type" => shared_expert.expert_type,
                "w1" => copy(shared_expert.w1),
                "w2" => copy(shared_expert.w2),
                "w3" => copy(shared_expert.w3)
            )
            push!(moe_dict["shared_experts"], shared_dict)
        end
    end
    
    return moe_dict
end

"""
    deserialize_weights(weights_dict::Dict, config::MoELlamaConfig) -> MoETransformerWeights

Reconstruct model weights from dictionary.
"""
function deserialize_weights(weights_dict::Dict, config::MoELlamaConfig)
    token_embedding_table = weights_dict["token_embedding_table"]
    rms_final_weight = weights_dict["rms_final_weight"]
    output_weight = weights_dict["output_weight"]
    
    layers = MoETransformerLayerWeights[]
    
    for layer_dict in weights_dict["layers"]
        layer = deserialize_layer_weights(layer_dict, config)
        push!(layers, layer)
    end
    
    return MoETransformerWeights(
        token_embedding_table,
        layers,
        rms_final_weight,
        output_weight,
        config,
        Dict{Symbol, Any}()  
    )
end

"""
    deserialize_layer_weights(layer_dict::Dict, config::MoELlamaConfig) -> MoETransformerLayerWeights

Reconstruct a single layer from dictionary.
"""
function deserialize_layer_weights(layer_dict::Dict, config::MoELlamaConfig)
    use_moe = layer_dict["use_moe"]
    
    att_dict = layer_dict["attention"]
    
    if use_moe

        ffn_dict = Dict(
            "rms_ffn_weight" => zeros(Float32, config.dim),
            "w1" => zeros(Float32, config.dim, config.hidden_dim),
            "w2" => zeros(Float32, config.hidden_dim, config.dim),
            "w3" => zeros(Float32, config.dim, config.hidden_dim)
        )
    else
        ffn_dict = layer_dict["ffn"]
    end
    
    llama_layer = Llama2.TransformerLayerWeights(
        att_dict["rms_att_weight"],
        ffn_dict["rms_ffn_weight"],
        att_dict["wq"],
        att_dict["wk"],
        att_dict["wv"],
        att_dict["wo"],
        ffn_dict["w1"],
        ffn_dict["w2"],
        ffn_dict["w3"]
    )
    
    if use_moe
        moe_dict = layer_dict["moe"]
        router_weight = moe_dict["router_weight"]
        
        experts = MoEExpertWeights[]
        for expert_dict in moe_dict["experts"]
            expert = deserialize_expert(expert_dict, config)
            push!(experts, expert)
        end
        
        shared_experts = if haskey(moe_dict, "shared_experts")
            shared_experts_list = MoEExpertWeights[]
            for shared_dict in moe_dict["shared_experts"]
                shared_expert = deserialize_expert(shared_dict, config)
                push!(shared_experts_list, shared_expert)
            end
            shared_experts_list
        else
            nothing
        end
        
        layer_moe_config = MoEConfig(
            num_experts = length(experts),
            expert_type = config.moe_expert_type,
            input_dim = config.dim,
            hidden_dim = config.hidden_dim,
            output_dim = config.dim,
            activation = x -> x * sigmoid(x),
            top_k = config.moe_top_k,
            gate_type = config.moe_gate_type,
            balance_loss = config.moe_balance_loss
        )
        
        return MoETransformerLayerWeights(
            llama_layer,
            true,
            experts,
            router_weight,
            layer_moe_config,
            shared_experts,
            nothing,
            zeros(Int, length(experts))
        )
    else
        return MoETransformerLayerWeights(
            llama_layer,
            false,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing
        )
    end
end

"""
    deserialize_expert(expert_dict::Dict, config::MoELlamaConfig) -> MoEExpertWeights

Reconstruct expert weights from dictionary.
"""
function deserialize_expert(expert_dict::Dict, config::MoELlamaConfig)
    expert_type = Symbol(expert_dict["expert_type"])
    is_cur_compressed = expert_dict["is_cur_compressed"]
    
    w1 = expert_dict["w1"]
    w2 = expert_dict["w2"]
    w3 = expert_dict["w3"]
    
    hb1 = zeros(Float32, config.hidden_dim)
    hb2 = zeros(Float32, config.hidden_dim)
    
    cur_c = cur_u = cur_r = nothing
    if haskey(expert_dict, "cur_components")
        cur_comp = expert_dict["cur_components"]
        cur_c = cur_comp["C"]
        cur_u = cur_comp["U"]
        cur_r = cur_comp["R"]
    end
    
    return MoEExpertWeights(
        w1, w2, w3,
        hb1, hb2,
        expert_type,
        is_cur_compressed,
        cur_c, cur_u, cur_r
    )
end

"""
Helper functions for type reconstruction
"""

"""
    reconstruct_gate_type(type_name::String, top_k::Int) -> GatingMechanism

Reconstruct gating mechanism from type name.
"""
function reconstruct_gate_type(type_name::String, top_k::Int)
    if occursin("TopKGating", type_name)
        return TopKGating(top_k)
    elseif occursin("SwitchGating", type_name)
        return SwitchGating()
    elseif occursin("RandomGating", type_name)
        return RandomGating(top_k)
    else
        @warn "Unknown gate type: $type_name, defaulting to TopKGating"
        return TopKGating(top_k)
    end
end

"""
    reconstruct_balance_loss(type_name::String) -> LoadBalancingLoss

Reconstruct load balancing loss from type name.
"""
function reconstruct_balance_loss(type_name::String)
    if occursin("SwitchTransformerLoss", type_name)
        return SwitchTransformerLoss(0.01f0)
    elseif occursin("NoBalancingLoss", type_name)
        return NoBalancingLoss()
    else
        @warn "Unknown balance loss type: $type_name, defaulting to SwitchTransformerLoss"
        return SwitchTransformerLoss(0.01f0)
    end
end

"""
Analysis and comparison utilities
"""

"""
    get_expert_stats(model::MoELanguageModel, tokens::Vector{Int}) -> Dict

Analyze expert usage for given token sequence.
"""
function get_expert_stats(model::MoELanguageModel, tokens::Vector{Int})
    config = model.config
    state = create_moe_run_state(config)
    
    moe_layers = get_moe_layer_indices(model)
    expert_usage = Dict{Int, Vector{Int}}()
    routing_entropy = Dict{Int, Vector{Float32}}()
    
    for layer_idx in moe_layers
        expert_usage[layer_idx] = zeros(Int, config.moe_num_experts)
        routing_entropy[layer_idx] = Float32[]
    end
    
    for (pos, token) in enumerate(tokens)
        if pos > config.seq_len
            break
        end
        
        moe_transformer!(token, pos, model, state)
        
        for layer_idx in moe_layers
            for k in 1:config.moe_top_k
                expert_idx = state.selected_experts[k]
                if expert_idx > 0
                    expert_usage[layer_idx][expert_idx] += 1
                end
            end
        end
    end
    
    return Dict{String, Any}(
        "expert_usage" => expert_usage,
        "routing_entropy" => routing_entropy,
        "total_tokens" => length(tokens),
        "moe_layers" => moe_layers
    )
end

"""
    compare_models(original_model::Llama2.LanguageModel, moe_model::MoELanguageModel,
                  test_prompt::String; num_comparisons::Int = 5) -> Dict

Compare outputs between original and MoE models.
"""
function compare_models(original_model::Llama2.LanguageModel, 
                       moe_model::MoELanguageModel,
                       test_prompt::String; 
                       num_comparisons::Int = 5)
    
    println("Comparing original vs MoE model outputs...")
    println("Prompt: \"$test_prompt\"")
    println("="^60)
    
    results = Dict{String, Any}(
        "prompt" => test_prompt,
        "comparisons" => [],
        "parameter_counts" => Dict(
            "original" => count_llama_parameters(original_model),
            "moe_total" => count_parameters(moe_model),
            "moe_active" => count_active_parameters(moe_model)
        )
    )
    
    for i in 1:num_comparisons
        println("\n--- Comparison $i ---")
        
        print("Original: ")
        original_output = Llama2.sample(original_model, test_prompt; 
                                       temperature=0.9f0, max_seq_len=50)
        
        print("MoE:      ")
        moe_output = sample_moe(moe_model, test_prompt;
                               temperature=0.9f0, max_seq_len=50,
                               show_expert_stats=false)
        
        comparison = Dict(
            "original_output" => original_output,
            "moe_output" => moe_output,
            "length_diff" => length(moe_output) - length(original_output)
        )
        
        push!(results["comparisons"], comparison)
    end
    
    return results
end

"""
    model_info(model::MoELanguageModel) -> Dict

Get comprehensive information about MoE model.
"""
function model_info(model::MoELanguageModel)
    config = model.config
    weights = model.weights
    
    param_counts = Dict{String, Int}()
    param_counts["total"] = count_parameters(model)
    param_counts["active"] = count_active_parameters(model)
    param_counts["embedding"] = length(weights.token_embedding_table)
    param_counts["output"] = length(weights.output_weight)
    
    layer_info = []
    for (i, layer) in enumerate(weights.layers)
        if layer.use_moe
            expert_params = length(layer.moe_experts[1].w1) + 
                           length(layer.moe_experts[1].w2) + 
                           length(layer.moe_experts[1].w3)
            
            layer_dict = Dict(
                "index" => i,
                "type" => "moe",
                "num_experts" => length(layer.moe_experts),
                "expert_params" => expert_params,
                "router_params" => length(layer.moe_router_weight),
                "total_params" => length(layer.moe_experts) * expert_params + length(layer.moe_router_weight)
            )
        else
            llama_layer = layer.llama_layer
            layer_dict = Dict(
                "index" => i,
                "type" => "dense",
                "ffn_params" => length(llama_layer.w1) + length(llama_layer.w2) + length(llama_layer.w3)
            )
        end
        
        push!(layer_info, layer_dict)
    end
    
    return Dict{String, Any}(
        "model_type" => "MoELanguageModel",
        "config" => Dict(
            "dim" => config.dim,
            "hidden_dim" => config.hidden_dim,
            "n_layers" => config.n_layers,
            "vocab_size" => config.vocab_size,
            "moe_layers" => config.moe_layers,
            "moe_num_experts" => config.moe_num_experts,
            "moe_top_k" => config.moe_top_k
        ),
        "parameters" => param_counts,
        "layers" => layer_info,
        "efficiency" => Dict(
            "parameter_efficiency" => param_counts["active"] / param_counts["total"],
            "moe_layer_fraction" => length(config.moe_layers) / config.n_layers
        )
    )
end

"""
    compute_model_statistics(model::MoELanguageModel) -> Dict

Compute detailed model statistics.
"""
function compute_model_statistics(model::MoELanguageModel)
    info = model_info(model)
    
    return Dict{String, Any}(
        "total_parameters" => info["parameters"]["total"],
        "active_parameters" => info["parameters"]["active"],
        "parameter_efficiency" => info["efficiency"]["parameter_efficiency"],
        "moe_layers" => length(model.config.moe_layers),
        "dense_layers" => model.config.n_layers - length(model.config.moe_layers),
        "experts_per_layer" => model.config.moe_num_experts,
        "routing_factor" => model.config.moe_top_k / model.config.moe_num_experts
    )
end

"""
    save_metadata_json(model::MoELanguageModel, filename::String)

Save model metadata in JSON format for external tools.
"""
function save_metadata_json(model::MoELanguageModel, filename::String)
    metadata = Dict{String, Any}(
        "model_info" => model_info(model),
        "original_model_info" => model.original_model_info,
        "conversion_info" => model.moe_conversion_info,
        "save_timestamp" => string(now())
    )
    
    open(filename, "w") do f
        JSON3.pretty(f, metadata)
    end
    
    return nothing
end

"""
    validate_save_data(save_data::Dict)

Validate loaded save data for consistency.
"""
function validate_save_data(save_data::Dict)
    required_keys = ["config", "tokenizer", "weights", "metadata"]
    
    for key in required_keys
        if !haskey(save_data, key)
            throw(ArgumentError("Invalid save file: missing key '$key'"))
        end
    end
    
    metadata = save_data["metadata"]
    if get(metadata, "model_type", "") != "MoELanguageModel"
        @warn "Save file may not be a MoELanguageModel"
    end
    
    return true
end

"""
    get_routing_stats(model::MoELanguageModel, state::MoERunState) -> Dict

Get detailed routing statistics from current state.
"""
function get_routing_stats(model::MoELanguageModel, state::MoERunState)
    return Dict{String, Any}(
        "selected_experts" => copy(state.selected_experts),
        "expert_gates" => copy(state.expert_gates),
        "router_logits" => copy(state.router_logits),
        "expert_load_counts" => copy(state.expert_load_counts),
        "routing_entropy" => copy(state.routing_entropy),
        "total_activations" => state.inference_stats[:expert_activations],
        "moe_layer_calls" => state.inference_stats[:moe_layer_calls]
    )
end