"""
Llama2 Integration Types

This file defines wrapper types that extend Llama2 functionality with MoE capabilities
without modifying the original Llama2 library.
"""

"""
    MoELlamaConfig

Extended configuration that wraps Llama2.ModelConfig with MoE settings.
Preserves all original Llama2 config fields while adding MoE-specific parameters.
"""
struct MoELlamaConfig
    llama_config::Llama2.ModelConfig
    
    moe_layers::Vector{Int}                   
    moe_num_experts::Int                       
    moe_top_k::Int                           
    moe_expert_type::Symbol                   
    moe_gate_type::GatingMechanism            
    moe_balance_loss::LoadBalancingLoss       
    
    expert_init_strategy::Symbol            
    expert_init_noise::Float32               
    
    use_shared_experts::Bool                  
    num_shared_experts::Int                  
    expert_dropout::Float32                  
    capacity_factor::Float32                  
    drop_tokens::Bool                        
    
    use_cur::Bool                            
    cur_rank::Union{Int, Nothing}           
    cur_oversample::Int                     
    
    use_fp32_router::Bool                    
    router_jitter::Float32                  
    z_loss_weight::Float32                   
end

function Base.getproperty(config::MoELlamaConfig, name::Symbol)
    if hasfield(MoELlamaConfig, name)
        return getfield(config, name)
    else
        return getproperty(config.llama_config, name)
    end
end

"""
    MoEExpertWeights

Represents weights for a single MoE expert, matching Llama2's gated FFN structure.
Supports multiple expert types: standard, gated, and CUR-compressed.
"""
struct MoEExpertWeights{W1, W2, W3, B1, B2}
    w1::W1             
    w2::W2              
    w3::W3              
    
    hb1::B1            
    hb2::B2            
    
    expert_type::Symbol
    is_cur_compressed::Bool
    
    cur_c::Union{Nothing, AbstractMatrix}   
    cur_u::Union{Nothing, AbstractMatrix}    
    cur_r::Union{Nothing, AbstractMatrix}   
end

"""
    MoETransformerLayerWeights

Extended transformer layer that can be either dense (original Llama2) or MoE.
Uses composition to wrap original Llama2 layer while adding MoE capabilities.
"""
struct MoETransformerLayerWeights
    llama_layer::Llama2.TransformerLayerWeights
    
    use_moe::Bool                          
    
    moe_experts::Union{Nothing, Vector{MoEExpertWeights}}     
    moe_router_weight::Union{Nothing, AbstractMatrix}        
    moe_config::Union{Nothing, MoEConfig}                     
    
    shared_experts::Union{Nothing, Vector{MoEExpertWeights}}  
    
    auxiliary_loss_state::Union{Nothing, Dict{Symbol, Any}}   
    expert_usage_stats::Union{Nothing, Vector{Int}}         
end

"""
    MoETransformerWeights  

Complete transformer weights with mixed dense/MoE layers.
Wraps Llama2.TransformerWeights while replacing specified layers with MoE.
"""
struct MoETransformerWeights
    token_embedding_table::AbstractMatrix   
    rms_final_weight::Vector{Float32}     
    output_weight::AbstractMatrix         
    
    layers::Vector{MoETransformerLayerWeights}
    
    config::MoELlamaConfig
    conversion_info::Dict{Symbol, Any}      
end

"""
    MoEKVCache

Extended KV cache that preserves Llama2's caching strategy.
Identical to Llama2.KVCache but wrapped for type consistency.
"""
struct MoEKVCache
    llama_kv_cache::Llama2.KVCache
end

Base.getproperty(cache::MoEKVCache, name::Symbol) = getproperty(cache.llama_kv_cache, name)
Base.setproperty!(cache::MoEKVCache, name::Symbol, value) = setproperty!(cache.llama_kv_cache, name, value)

"""
    MoERunState

Extended run state that includes all Llama2 buffers plus MoE-specific buffers.
Preserves all Llama2 computation patterns while adding MoE routing state.
"""
struct MoERunState
    llama_state::Llama2.RunState
    
    router_logits::Vector{Float32}          
    expert_gates::Vector{Float32}          
    selected_experts::Vector{Int}            
    expert_outputs::Vector{Vector{Float32}}  
    moe_temp_buffer::Vector{Float32}        
    
    auxiliary_loss_values::Vector{Float32}  
    routing_entropy::Vector{Float32}        
    expert_load_counts::Vector{Int}         
    
    inference_stats::Dict{Symbol, Any}     
end

function Base.getproperty(state::MoERunState, name::Symbol)
    if hasfield(MoERunState, name)
        return getfield(state, name)
    else
        return getproperty(state.llama_state, name)
    end
end

"""
    MoELanguageModel

Complete language model with MoE integration.
Wraps Llama2.LanguageModel while replacing FFN layers with MoE where specified.
"""
struct MoELanguageModel{TOK<:Llama2.Tokenizer}
    config::MoELlamaConfig
    tokenizer::TOK                          
    weights::MoETransformerWeights
    
    original_model_info::Dict{String, Any}  
    moe_conversion_info::Dict{String, Any} 
    
    kv_cache_pool::Vector{MoEKVCache}       
    state_pool::Vector{MoERunState}         
end

Base.getproperty(model::MoELanguageModel, name::Symbol) = 
    hasfield(MoELanguageModel, name) ? getfield(model, name) : getproperty(model.tokenizer, name)

"""
    Expert creation utilities
"""

"""
    create_moe_expert_weights(config::MoELlamaConfig, expert_type::Symbol = :gated)

Create expert weights matching the specified type and configuration.
"""
function create_moe_expert_weights(config::MoELlamaConfig, expert_type::Symbol = :gated)
    dim = config.dim
    hidden_dim = config.hidden_dim
    
    σ = sqrt(2.0f0 / dim)
    
    if expert_type == :cur && config.use_cur
        rank = something(config.cur_rank, hidden_dim ÷ 4)
        
        w1_full = randn(Float32, dim, hidden_dim) .* σ
        w2_full = randn(Float32, hidden_dim, dim) .* σ  
        w3_full = randn(Float32, dim, hidden_dim) .* σ
        
        w1_cur = cur_decompose(w1_full; rank=rank, oversample=config.cur_oversample)
        w2_cur = cur_decompose(w2_full; rank=rank, oversample=config.cur_oversample)
        w3_cur = cur_decompose(w3_full; rank=rank, oversample=config.cur_oversample)
        
        return MoEExpertWeights(
            w1_cur.C, w2_cur.C, w3_cur.C,
            zeros(Float32, hidden_dim), zeros(Float32, hidden_dim),
            :cur, true,
            w1_cur.C, w1_cur.U, w1_cur.R  
        )
    else
        return MoEExpertWeights(
            randn(Float32, dim, hidden_dim) .* σ,      
            randn(Float32, hidden_dim, dim) .* σ,     
            randn(Float32, dim, hidden_dim) .* σ,      
            zeros(Float32, hidden_dim),               
            zeros(Float32, hidden_dim),               
            expert_type, false,
            nothing, nothing, nothing                 
        )
    end
end

"""
    create_moe_run_state(config::MoELlamaConfig)

Create run state with all necessary buffers for MoE computation.
"""
function create_moe_run_state(config::MoELlamaConfig)
    llama_state = Llama2.RunState(config.llama_config)
    
    num_experts = config.moe_num_experts
    top_k = config.moe_top_k
    dim = config.dim
    
    return MoERunState(
        llama_state,
        zeros(Float32, num_experts),                  
        zeros(Float32, top_k),                        
        zeros(Int, top_k),                            
        [zeros(Float32, dim) for _ in 1:num_experts],  
        zeros(Float32, dim),                          
        Float32[],                                     
        Float32[],                                    
        zeros(Int, num_experts),                      
        Dict{Symbol, Any}(
            :total_tokens => 0,
            :moe_layer_calls => 0,
            :expert_activations => 0,
            :routing_time => 0.0,
            :expert_compute_time => 0.0
        )
    )
end

"""
Utility functions for type checking and validation
"""

"""
    is_moe_layer(layer::MoETransformerLayerWeights)

Check if a layer uses MoE (vs dense FFN).
"""
is_moe_layer(layer::MoETransformerLayerWeights) = layer.use_moe

"""
    get_expert_count(layer::MoETransformerLayerWeights)

Get number of experts in a MoE layer (0 for dense layers).
"""
function get_expert_count(layer::MoETransformerLayerWeights)
    return layer.use_moe ? length(layer.moe_experts) : 0
end

"""
    get_moe_layer_indices(model::MoELanguageModel)

Get indices of all MoE layers in the model.
"""
function get_moe_layer_indices(model::MoELanguageModel)
    return [i for (i, layer) in enumerate(model.weights.layers) if is_moe_layer(layer)]
end

"""
    validate_matrix_dimensions(config::MoELlamaConfig)

Validate that all matrix dimensions are correct for Llama2 compatibility.
Critical for ensuring matmul! operations work correctly.
"""
function validate_matrix_dimensions(config::MoELlamaConfig)
    dim = config.dim
    hidden_dim = config.hidden_dim
    vocab_size = config.vocab_size
    num_experts = config.moe_num_experts
    
    checks = [
        (true, "dim must be positive"),
        (hidden_dim > 0, "hidden_dim must be positive"), 
        (vocab_size > 0, "vocab_size must be positive"),
        (num_experts > 0, "num_experts must be positive"),
        (config.moe_top_k <= num_experts, "top_k must not exceed num_experts"),
        (dim % config.n_heads == 0, "dim must be divisible by n_heads"),

    ]
    
    for (condition, message) in checks
        if !condition
            throw(ArgumentError("Invalid configuration: $message"))
        end
    end
    
    return true
end