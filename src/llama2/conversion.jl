"""
Llama2 to MoE Model Conversion

This module handles converting existing Llama2 models to MoE models by replacing
specified FFN layers with MoE layers while preserving all other functionality.
"""

"""
    convert_to_moe(model::Llama2.LanguageModel, moe_layers::Vector{Int};
                   num_experts::Int = 8,
                   top_k::Int = 2,
                   expert_init_strategy::Symbol = :perturb,
                   expert_init_noise::Float32 = 0.01f0,
                   gate_type::GatingMechanism = TopKGating(top_k),
                   balance_loss::LoadBalancingLoss = SwitchTransformerLoss(0.01f0),
                   expert_type::Symbol = :gated,
                   use_shared_experts::Bool = false,
                   num_shared_experts::Int = 0,
                   use_cur::Bool = false,
                   cur_rank::Union{Int, Nothing} = nothing,
                   kwargs...) -> MoELanguageModel

Convert an existing Llama2 model to MoE by replacing specified layers.

# Arguments
- `model`: Original Llama2.LanguageModel
- `moe_layers`: Vector of layer indices to convert to MoE (1-based)
- `num_experts`: Number of experts per MoE layer
- `top_k`: Number of experts to activate per token
- `expert_init_strategy`: How to initialize expert weights (:copy, :perturb, :split, :random)
- `expert_init_noise`: Noise level for :perturb strategy
- `gate_type`: Gating mechanism from MoE library
- `balance_loss`: Load balancing loss function
- `expert_type`: Type of experts (:standard, :gated, :cur)
- `use_shared_experts`: Whether to use shared experts (DeepSeek-style)
- `num_shared_experts`: Number of shared experts
- `use_cur`: Whether to use CUR compression for experts
- `cur_rank`: Rank for CUR decomposition (nothing = auto)

# Returns
- `MoELanguageModel`: Converted model with MoE layers
"""
function convert_to_moe(model::Llama2.LanguageModel, moe_layers::Vector{Int};
                       num_experts::Int = 8,
                       top_k::Int = 2,
                       expert_init_strategy::Symbol = :perturb,
                       expert_init_noise::Float32 = 0.01f0,
                       gate_type::GatingMechanism = TopKGating(top_k),
                       balance_loss::LoadBalancingLoss = SwitchTransformerLoss(0.01f0),
                       expert_type::Symbol = :gated,
                       use_shared_experts::Bool = false,
                       num_shared_experts::Int = 0,
                       use_cur::Bool = false,
                       cur_rank::Union{Int, Nothing} = nothing,
                       expert_dropout::Float32 = 0.0f0,
                       capacity_factor::Float32 = 1.25f0,
                       drop_tokens::Bool = false,
                       use_fp32_router::Bool = true,
                       router_jitter::Float32 = 0.0f0,
                       z_loss_weight::Float32 = 0.001f0,
                       cur_oversample::Int = 10,
                       kwargs...)
    
    println("Converting Llama2 model to MoE...")
    println("  Original layers: $(model.config.n_layers)")
    println("  MoE layers: $moe_layers")
    println("  Experts per MoE layer: $num_experts")
    println("  Expert type: $expert_type")
    println("  Initialization strategy: $expert_init_strategy")
    
    validate_conversion_inputs(model, moe_layers, num_experts, top_k, expert_init_strategy)
    
    moe_config = create_moe_llama_config(
        model.config, moe_layers, num_experts, top_k, expert_init_strategy,
        expert_init_noise, gate_type, balance_loss, expert_type,
        use_shared_experts, num_shared_experts, use_cur, cur_rank,
        expert_dropout, capacity_factor, drop_tokens, use_fp32_router,
        router_jitter, z_loss_weight, cur_oversample
    )
    
    converted_layers = convert_transformer_layers(model, moe_config, moe_layers, expert_init_strategy)
    
moe_weights = MoETransformerWeights(
    copy(model.weights.token_embedding_table),  
    copy(model.weights.rms_final_weight),      
    copy(model.weights.output_weight),         
    converted_layers,                          
    moe_config,                                
    create_conversion_info(model, moe_layers, expert_init_strategy)  
)
    
    moe_model = MoELanguageModel(
        moe_config,
        model.tokenizer,  
        moe_weights,
        extract_original_model_info(model),
        create_moe_conversion_info(moe_layers, num_experts, expert_init_strategy),
        MoEKVCache[],     
        MoERunState[]     
    )
    
    println(" Model conversion completed successfully")
    println("  Total parameters: $(count_parameters(moe_model))")
    println("  Active parameters: $(count_active_parameters(moe_model))")
    
    return moe_model
end

"""
    create_moe_llama_config(llama_config, moe_layers, num_experts, top_k, ...)

Create MoELlamaConfig from original Llama2 config and MoE parameters.
"""
function create_moe_llama_config(llama_config::Llama2.ModelConfig,
                                moe_layers::Vector{Int},
                                num_experts::Int,
                                top_k::Int,
                                expert_init_strategy::Symbol,
                                expert_init_noise::Float32,
                                gate_type::GatingMechanism,
                                balance_loss::LoadBalancingLoss,
                                expert_type::Symbol,
                                use_shared_experts::Bool,
                                num_shared_experts::Int,
                                use_cur::Bool,
                                cur_rank::Union{Int, Nothing},
                                expert_dropout::Float32,
                                capacity_factor::Float32,
                                drop_tokens::Bool,
                                use_fp32_router::Bool,
                                router_jitter::Float32,
                                z_loss_weight::Float32,
                                cur_oversample::Int)
    
    return MoELlamaConfig(
        llama_config,
        moe_layers,
        num_experts,
        top_k,
        expert_type,
        gate_type,
        balance_loss,
        expert_init_strategy,
        expert_init_noise,
        use_shared_experts,
        num_shared_experts,
        expert_dropout,
        capacity_factor,
        drop_tokens,
        use_cur,
        cur_rank,
        cur_oversample,
        use_fp32_router,
        router_jitter,
        z_loss_weight
    )
end

"""
    convert_transformer_layers(model, moe_config, moe_layers, expert_init_strategy)

Convert transformer layers to MoE format.
"""
function convert_transformer_layers(model::Llama2.LanguageModel,
                                   moe_config::MoELlamaConfig,
                                   moe_layers::Vector{Int},
                                   expert_init_strategy::Symbol)
    original_layers = model.weights.layers
    converted_layers = MoETransformerLayerWeights[]
    
    for (layer_idx, original_layer) in enumerate(original_layers)
        if layer_idx in moe_layers
            println("  Converting layer $layer_idx to MoE...")
            moe_layer = convert_layer_to_moe(original_layer, moe_config, expert_init_strategy)
            push!(converted_layers, moe_layer)
        else
            println("  Preserving dense layer $layer_idx...")
            dense_layer = preserve_dense_layer(original_layer)
            push!(converted_layers, dense_layer)
        end
    end
    
    return converted_layers
end

"""
    convert_layer_to_moe(original_layer, moe_config, expert_init_strategy)

Convert a single transformer layer to MoE.
"""
function convert_layer_to_moe(original_layer::Llama2.TransformerLayerWeights,
                             moe_config::MoELlamaConfig,
                             expert_init_strategy::Symbol)
    
    experts = create_experts_from_layer(original_layer, moe_config, expert_init_strategy)
    
   
    router_weight = create_router_weights(moe_config)
    
    shared_experts = if moe_config.use_shared_experts && moe_config.num_shared_experts > 0
        create_shared_experts(original_layer, moe_config, expert_init_strategy)
    else
        nothing
    end
    
    layer_moe_config = MoEConfig(
        num_experts = moe_config.moe_num_experts,
        expert_type = moe_config.moe_expert_type,
        input_dim = moe_config.dim,
        hidden_dim = moe_config.hidden_dim,
        output_dim = moe_config.dim,
        activation = x -> x * sigmoid(x),  
        expert_dropout = moe_config.expert_dropout,
        gate_type = moe_config.moe_gate_type,
        top_k = moe_config.moe_top_k,
        balance_loss = moe_config.moe_balance_loss,
        capacity_factor = moe_config.capacity_factor,
        drop_tokens = moe_config.drop_tokens,
        use_cur = moe_config.use_cur,
        cur_rank = moe_config.cur_rank,
        num_shared_experts = length(shared_experts === nothing ? [] : shared_experts)
    )
    
    return MoETransformerLayerWeights(
        original_layer,  
        true,           
        experts,
        router_weight,
        layer_moe_config,
        shared_experts,
        nothing,        
        zeros(Int, moe_config.moe_num_experts)  
    )
end

"""
    preserve_dense_layer(original_layer)

Preserve a layer as dense (non-MoE).
"""
function preserve_dense_layer(original_layer::Llama2.TransformerLayerWeights)
    return MoETransformerLayerWeights(
        original_layer,
        false,         
        nothing,        
        nothing,       
        nothing,       
        nothing,        
        nothing,        
        nothing        
    )
end

"""
    create_experts_from_layer(original_layer, moe_config, expert_init_strategy)

Create MoE experts from original layer weights using specified strategy.
"""
function create_experts_from_layer(original_layer::Llama2.TransformerLayerWeights,
                                  moe_config::MoELlamaConfig,
                                  expert_init_strategy::Symbol)
    experts = MoEExpertWeights[]
    
    for expert_idx in 1:moe_config.moe_num_experts
        expert = create_single_expert(original_layer, moe_config, expert_init_strategy, expert_idx)
        push!(experts, expert)
    end
    
    return experts
end

"""
    create_single_expert(original_layer, moe_config, strategy, expert_idx)

Create a single expert using the specified initialization strategy.
"""
function create_single_expert(original_layer::Llama2.TransformerLayerWeights,
                             moe_config::MoELlamaConfig,
                             strategy::Symbol,
                             expert_idx::Int)
    
    dim = moe_config.dim
    hidden_dim = moe_config.hidden_dim
    
    if strategy == :copy
        w1 = copy(original_layer.w1)
        w2 = copy(original_layer.w2)
        w3 = copy(original_layer.w3)
        
    elseif strategy == :perturb
        noise_scale = moe_config.expert_init_noise
        w1 = copy(original_layer.w1) .+ randn(Float32, size(original_layer.w1)) .* noise_scale
        w2 = copy(original_layer.w2) .+ randn(Float32, size(original_layer.w2)) .* noise_scale
        w3 = copy(original_layer.w3) .+ randn(Float32, size(original_layer.w3)) .* noise_scale
        
    elseif strategy == :split
        experts_per_dim = ceil(Int, moe_config.moe_num_experts)
        start_idx = ((expert_idx - 1) * hidden_dim ÷ experts_per_dim) + 1
        end_idx = min(expert_idx * hidden_dim ÷ experts_per_dim, hidden_dim)
        
        w1 = zeros(Float32, dim, hidden_dim)
        w2 = zeros(Float32, hidden_dim, dim)
        w3 = zeros(Float32, dim, hidden_dim)
        
        w1[:, start_idx:end_idx] = original_layer.w1[:, start_idx:end_idx]
        w2[start_idx:end_idx, :] = original_layer.w2[start_idx:end_idx, :]
        w3[:, start_idx:end_idx] = original_layer.w3[:, start_idx:end_idx]
        
    elseif strategy == :random
        σ = sqrt(2.0f0 / dim)
        w1 = randn(Float32, dim, hidden_dim) .* σ
        w2 = randn(Float32, hidden_dim, dim) .* σ
        w3 = randn(Float32, dim, hidden_dim) .* σ
        
    else
        throw(ArgumentError("Unknown expert initialization strategy: $strategy"))
    end
    
    if moe_config.use_cur && moe_config.moe_expert_type == :cur
        return create_cur_expert(w1, w2, w3, moe_config)
    else
        return MoEExpertWeights(
            w1, w2, w3,
            zeros(Float32, hidden_dim),  
            zeros(Float32, hidden_dim),  
            moe_config.moe_expert_type,
            false,                      
            nothing, nothing, nothing    
        )
    end
end

function create_cur_expert(w1::AbstractMatrix, w2::AbstractMatrix, w3::AbstractMatrix,
                          moe_config::MoELlamaConfig)
    
    rank = something(moe_config.cur_rank, moe_config.hidden_dim ÷ 4)
    oversample = moe_config.cur_oversample
    
    w1_cur = cur_decompose(w1; rank=rank, oversample=oversample)
    w2_cur = cur_decompose(w2; rank=rank, oversample=oversample)
    w3_cur = cur_decompose(w3; rank=rank, oversample=oversample)
    
    return MoEExpertWeights(
        w1_cur.C, w2_cur.C, w3_cur.C,
        zeros(Float32, moe_config.hidden_dim),  
        zeros(Float32, moe_config.hidden_dim),  
        :cur,
        true,                                  
        w1_cur.C, w1_cur.U, w1_cur.R        
    )
end

"""
    create_router_weights(moe_config)

Create router weight matrix with proper dimensions for Llama2 compatibility.
"""
function create_router_weights(moe_config::MoELlamaConfig)
    dim = moe_config.dim
    num_experts = moe_config.moe_num_experts
    

    σ = sqrt(2.0f0 / dim)
    router_weight = randn(Float32, dim, num_experts) .* σ
    
    return router_weight
end

"""
    create_shared_experts(original_layer, moe_config, expert_init_strategy)

Create shared experts for DeepSeek-style architecture.
"""
function create_shared_experts(original_layer::Llama2.TransformerLayerWeights,
                              moe_config::MoELlamaConfig,
                              expert_init_strategy::Symbol)
    
    shared_experts = MoEExpertWeights[]
    
    for i in 1:moe_config.num_shared_experts
        expert = create_single_expert(original_layer, moe_config, expert_init_strategy, i)
        push!(shared_experts, expert)
    end
    
    return shared_experts
end

"""
    validate_conversion_inputs(model, moe_layers, num_experts, top_k, strategy)

Validate inputs for model conversion.
"""
function validate_conversion_inputs(model::Llama2.LanguageModel,
                                   moe_layers::Vector{Int},
                                   num_experts::Int,
                                   top_k::Int,
                                   strategy::Symbol)
    
    n_layers = model.config.n_layers
    for layer_idx in moe_layers
        if layer_idx < 1 || layer_idx > n_layers
            throw(ArgumentError("Layer index $layer_idx out of range [1, $n_layers]"))
        end
    end
    
    if num_experts < 1
        throw(ArgumentError("num_experts must be positive"))
    end
    
    if top_k < 1 || top_k > num_experts
        throw(ArgumentError("top_k must be in range [1, $num_experts]"))
    end
    
    valid_strategies = [:copy, :perturb, :split, :random]
    if strategy ∉ valid_strategies
        throw(ArgumentError("Invalid strategy $strategy. Must be one of $valid_strategies"))
    end
    
    return true
end

"""
    extract_original_model_info(model)

Extract metadata from original model for tracking.
"""
function extract_original_model_info(model::Llama2.LanguageModel)
    return Dict{String, Any}(
        "dim" => model.config.dim,
        "n_layers" => model.config.n_layers,
        "n_heads" => model.config.n_heads,
        "vocab_size" => model.config.vocab_size,
        "seq_len" => model.config.seq_len,
        "rope_freq_base" => model.config.rope_freq_base,
        "rope_is_neox" => model.config.rope_is_neox,
        "parameter_count" => count_llama_parameters(model)
    )
end

"""
    create_moe_conversion_info(moe_layers, num_experts, strategy)

Create metadata about MoE conversion.
"""
function create_moe_conversion_info(moe_layers::Vector{Int},
                                   num_experts::Int,
                                   strategy::Symbol)
    return Dict{String, Any}(
        "moe_layers" => copy(moe_layers),
        "num_experts" => num_experts,
        "initialization_strategy" => strategy,
        "conversion_timestamp" => string(now()),
        "moe_library_version" => "1.0.0" 
    )
end

"""
    create_conversion_info(model, moe_layers, strategy)

Create detailed conversion tracking information.
"""
function create_conversion_info(model::Llama2.LanguageModel,
                               moe_layers::Vector{Int},
                               strategy::Symbol)
    return Dict{Symbol, Any}(
        :original_model => extract_original_model_info(model),
        :moe_conversion => create_moe_conversion_info(moe_layers, 8, strategy), 
        :layer_mapping => Dict(
            i => (i in moe_layers ? "moe" : "dense") 
            for i in 1:model.config.n_layers
        )
    )
end

"""
Utility functions for parameter counting and analysis
"""

"""
    count_llama_parameters(model::Llama2.LanguageModel)

Count parameters in original Llama2 model.
"""
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

"""
    count_parameters(model::MoELanguageModel)

Count total parameters in MoE model.
"""
function count_parameters(model::MoELanguageModel)
    count = 0
    
    count += length(model.weights.token_embedding_table)
    count += length(model.weights.rms_final_weight)
    count += length(model.weights.output_weight)
    
    for layer in model.weights.layers
        count += length(layer.llama_layer.rms_att_weight)
        count += length(layer.llama_layer.wq)
        count += length(layer.llama_layer.wk)
        count += length(layer.llama_layer.wv)
        count += length(layer.llama_layer.wo)
        
        if layer.use_moe
            count += length(layer.moe_router_weight)
            for expert in layer.moe_experts
                count += length(expert.w1)
                count += length(expert.w2)
                count += length(expert.w3)
            end
            
            if !isnothing(layer.shared_experts)
                for expert in layer.shared_experts
                    count += length(expert.w1)
                    count += length(expert.w2)
                    count += length(expert.w3)
                end
            end
        else
            count += length(layer.llama_layer.rms_ffn_weight)
            count += length(layer.llama_layer.w1)
            count += length(layer.llama_layer.w2)
            count += length(layer.llama_layer.w3)
        end
    end
    
    return count
end

"""
    count_active_parameters(model::MoELanguageModel)

Count parameters that are active during inference (considering top-k routing).
"""
function count_active_parameters(model::MoELanguageModel)
    count = 0
    
    count += length(model.weights.token_embedding_table)
    count += length(model.weights.rms_final_weight)
    count += length(model.weights.output_weight)
    
    for layer in model.weights.layers
        count += length(layer.llama_layer.rms_att_weight)
        count += length(layer.llama_layer.wq)
        count += length(layer.llama_layer.wk)
        count += length(layer.llama_layer.wv)
        count += length(layer.llama_layer.wo)
        
        if layer.use_moe
            count += length(layer.moe_router_weight)
            
            top_k = layer.moe_config.top_k
            expert_params = length(layer.moe_experts[1].w1) + 
                           length(layer.moe_experts[1].w2) + 
                           length(layer.moe_experts[1].w3)
            count += top_k * expert_params
            
            if !isnothing(layer.shared_experts)
                for expert in layer.shared_experts
                    count += length(expert.w1)
                    count += length(expert.w2)
                    count += length(expert.w3)
                end
            end
        else
            count += length(layer.llama_layer.rms_ffn_weight)
            count += length(layer.llama_layer.w1)
            count += length(layer.llama_layer.w2)
            count += length(layer.llama_layer.w3)
        end
    end
    
    return count
end