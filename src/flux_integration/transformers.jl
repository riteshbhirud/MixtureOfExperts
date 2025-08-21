"""
MoE-Transformer Integration with Transformers.jl

Production-ready integration that replaces FFN layers with FluxMoELayer in transformer architectures.
Follows exact Transformers.jl patterns and conventions for seamless compatibility.
"""

using Flux
using Flux: Dense, LayerNorm, Dropout, Chain, @layer
using Transformers
using Transformers.Layers
using Transformers.Layers: PreNormResidual, DropoutLayer, SelfAttention, 
                           TransformerBlock, Transformer, CompositeEmbedding,
                           Embed, FixedLenPositionEmbed, EmbedDecoder, Branch,
                           collect_outputs, apply_on_namedtuple, return_hidden_state,
                           Architecture, LayerStruct
using Functors
using Statistics: mean

"""
    MoETransformerConfig

Configuration for MoE-enabled Transformer models following GPT-2 architecture.
Combines standard transformer parameters with MoE-specific settings.
"""
Base.@kwdef struct MoETransformerConfig
    vocab_size::Int = 50257
    max_position_embeddings::Int = 1024
    hidden_size::Int = 768
    num_hidden_layers::Int = 12
    num_attention_heads::Int = 12
    intermediate_size::Int = 3072
    
    hidden_act::Function = gelu
    hidden_dropout::Float32 = Float32(0.1)
    attention_dropout::Float32 = Float32(0.1)
    layer_norm_epsilon::Float32 = Float32(1e-5)
    initializer_range::Float32 = Float32(0.02)
    
    num_experts::Int = 8
    expert_top_k::Int = 2
    expert_type::Symbol = :gated
    moe_dropout::Float32 = Float32(0.0)
    balance_loss_weight::Float32 = Float32(0.01)
    z_loss_weight::Float32 = Float32(0.001)
    router_noise_scale::Float32 = Float32(0.0)
    router_use_noise_network::Bool = false
    router_use_fp32::Bool = true
    
    use_cache::Bool = true
    gradient_checkpointing::Bool = false
end

"""
    MoELayerWrapper

Wrapper that makes FluxMoELayer compatible with Transformers.jl architecture patterns.
Inherits from Architecture to ensure proper NamedTuple handling by apply_on_namedtuple.
"""
struct MoELayerWrapper{L} <: Architecture
    moe_layer::L
    aux_loss_storage::Ref{Float32}
end

@layer MoELayerWrapper trainable=(moe_layer,)

function MoELayerWrapper(moe_layer::FluxMoELayer)
    return MoELayerWrapper(moe_layer, Ref(Float32(0.0)))
end

Transformers.Layers.argument_names(::MoELayerWrapper) = (:hidden_state,)

function (wrapper::MoELayerWrapper)(nt::NamedTuple)
    hidden_state = nt.hidden_state
    
    training = get(nt, :training, false)
    

    original_size = size(hidden_state)
    
    if ndims(hidden_state) == 3
        features, seq_len, batch_size = original_size
        reshaped_input = reshape(hidden_state, features, seq_len * batch_size)
        
        output, aux_loss = wrapper.moe_layer(reshaped_input; training=training)
        
        output = reshape(output, features, seq_len, batch_size)
        
    elseif ndims(hidden_state) == 2
        output, aux_loss = wrapper.moe_layer(hidden_state; training=training)
        
    else
        error("MoELayerWrapper: Unsupported input dimensions $(ndims(hidden_state)). Expected 2D or 3D.")
    end
    
    wrapper.aux_loss_storage[] = aux_loss
    
    result_nt = (hidden_state = output,)
    
    other_fields = Base.structdiff(nt, NamedTuple{(:hidden_state,)})
    
    return merge(result_nt, other_fields)
end

"""
    AuxiliaryLossCollector

Thread-safe auxiliary loss collection for MoE layers.
Stores references to all MoE layer wrappers for centralized loss aggregation.
"""
struct AuxiliaryLossCollector
    moe_wrappers::Vector{MoELayerWrapper}
end

function AuxiliaryLossCollector()
    return AuxiliaryLossCollector(MoELayerWrapper[])
end

function register_moe_wrapper!(collector::AuxiliaryLossCollector, wrapper::MoELayerWrapper)
    push!(collector.moe_wrappers, wrapper)
    return wrapper
end

function collect_aux_losses(collector::AuxiliaryLossCollector)
    total_loss = Float32(0.0)
    for wrapper in collector.moe_wrappers
        total_loss += wrapper.aux_loss_storage[]
    end
    return total_loss
end

function reset_aux_losses!(collector::AuxiliaryLossCollector)
    for wrapper in collector.moe_wrappers
        wrapper.aux_loss_storage[] = Float32(0.0)
    end
end

"""
    create_moe_transformer_block(config::MoETransformerConfig, aux_collector::AuxiliaryLossCollector; dropout_prob=nothing)

Create a TransformerBlock with MoE feedforward layer following exact Transformers.jl patterns.
This is the only function that differs from standard GPT-2 - replaces FFN with MoE.
"""
function create_moe_transformer_block(config::MoETransformerConfig, aux_collector::AuxiliaryLossCollector; dropout_prob=nothing)
    p = something(dropout_prob, config.hidden_dropout)
    
    head_dim = config.hidden_size ÷ config.num_attention_heads
    sa = SelfAttention(
        config.num_attention_heads, 
        config.hidden_size, 
        head_dim;
        dropout = config.attention_dropout,
        causal = true,  
        return_score = false
    )
    
    sa_ln = LayerNorm(config.hidden_size; eps=config.layer_norm_epsilon)
    sa_wrapped = PreNormResidual(DropoutLayer(sa, p), sa_ln)
    
    moe_config = FluxMoEConfig(
        num_experts = config.num_experts,
        expert_type = config.expert_type,
        input_dim = config.hidden_size,
        hidden_dim = config.intermediate_size,
        output_dim = config.hidden_size,
        activation = config.hidden_act,
        expert_dropout = config.moe_dropout,
        expert_bias = true,
        top_k = config.expert_top_k,
        noise_scale = config.router_noise_scale,
        use_noise_network = config.router_use_noise_network,
        use_fp32_router = config.router_use_fp32,
        balance_loss_weight = config.balance_loss_weight,
        z_loss_weight = config.z_loss_weight,
        init = (args...) -> Flux.truncated_normal(args...; std=config.initializer_range)
    )
    
    moe_layer = FluxMoELayer(moe_config)
    moe_wrapper = MoELayerWrapper(moe_layer)
    
    register_moe_wrapper!(aux_collector, moe_wrapper)
    
    ff_ln = LayerNorm(config.hidden_size; eps=config.layer_norm_epsilon)
    ff_wrapped = PreNormResidual(DropoutLayer(moe_wrapper, p), ff_ln)
    
    return TransformerBlock(sa_wrapped, ff_wrapped)
end

"""
    MoETransformerModel

Core MoE-enabled transformer model following exact Transformers.jl patterns.
Uses existing CompositeEmbedding and Transformer structures.
"""
struct MoETransformerModel{E, D, L} <: LayerStruct
    embed::E
    decoder::D
    ln_f::L  
    config::MoETransformerConfig
    aux_collector::AuxiliaryLossCollector
end

@layer MoETransformerModel trainable=(embed, decoder, ln_f)

function MoETransformerModel(config::MoETransformerConfig; collect_output=false)
    vocab_size, dims, max_pos = config.vocab_size, config.hidden_size, config.max_position_embeddings
    factor = config.initializer_range
    
    token_weight = randn(Float32, dims, vocab_size) * factor
    pos_weight = randn(Float32, dims, max_pos) * factor
    
    embed = CompositeEmbedding(
        token = Embed(token_weight),
        position = FixedLenPositionEmbed(pos_weight)
    )
    
    p = config.hidden_dropout > 0 ? config.hidden_dropout : nothing
    embed = DropoutLayer(embed, p)
    
    aux_collector = AuxiliaryLossCollector()
    
    blocks = [create_moe_transformer_block(config, aux_collector) for _ in 1:config.num_hidden_layers]
    
    collect_f = collect_output ? Layers.collect_outputs : nothing
    decoder = Transformer(Tuple(blocks), collect_f)
    
    ln_f = LayerNorm(config.hidden_size; eps=config.layer_norm_epsilon)
    
    return MoETransformerModel(embed, decoder, ln_f, config, aux_collector)
end

"""
    propagate_training_flag(nt::NamedTuple, training::Bool)

Helper function to ensure training flag propagates through all transformer layers.
"""
function propagate_training_flag(nt::NamedTuple, training::Bool)
    if training
        return merge(nt, (training = true,))
    else
        return Base.structdiff(nt, NamedTuple{(:training,)})
    end
end

"""
Forward pass for MoETransformerModel following exact Transformers.jl NamedTuple patterns.
"""
function (model::MoETransformerModel)(nt::NamedTuple)
    training = get(nt, :training, false)
    
    reset_aux_losses!(model.aux_collector)
    
    embedded_nt = model.embed(nt)
    
    embedded_nt = propagate_training_flag(embedded_nt, training)
    
    decoder_nt = model.decoder(embedded_nt)
    
    output_nt = apply_on_namedtuple(model.ln_f, decoder_nt)
    
    aux_loss = collect_aux_losses(model.aux_collector)
    
    return merge(output_nt, (aux_loss = aux_loss,))
end

"""
    MoETransformerLMHeadModel

MoE transformer with language modeling head following HuggingFace patterns.
"""
struct MoETransformerLMHeadModel{M, H} <: LayerStruct
    model::M
    lm_head::H
    config::MoETransformerConfig
end

@layer MoETransformerLMHeadModel trainable=(model, lm_head)

function MoETransformerLMHeadModel(config::MoETransformerConfig)
    model = MoETransformerModel(config)
    

    token_embed_layer = model.embed.layer.token  
    lmhead = EmbedDecoder(token_embed_layer)
    
    lm_head = Branch{(:logit,), (:hidden_state,)}(lmhead)
    
    return MoETransformerLMHeadModel(model, lm_head, config)
end

function (model::MoETransformerLMHeadModel)(nt::NamedTuple)
    transformer_output = model.model(nt)
    
    lm_output = model.lm_head(transformer_output)
    
    return merge(lm_output, (aux_loss = transformer_output.aux_loss,))
end


"""
    create_moe_transformer_model(config::MoETransformerConfig; lm_head=true)

Create MoE transformer model with proper Transformers.jl compatibility.
"""
function create_moe_transformer_model(config::MoETransformerConfig; lm_head=true)
    if lm_head
        return MoETransformerLMHeadModel(config)
    else
        return MoETransformerModel(config)
    end
end


"""
    moe_transformer_base_config(; kwargs...)

Base MoE Transformer configuration (768d model).
"""
function moe_transformer_base_config(; kwargs...)
    return MoETransformerConfig(
        vocab_size = 50257,
        max_position_embeddings = 1024,
        hidden_size = 768,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        intermediate_size = 3072,
        num_experts = 8,
        expert_top_k = 2,
        expert_type = :gated,
        balance_loss_weight = Float32(0.01),
        z_loss_weight = Float32(0.001);
        kwargs...
    )
end

"""
    moe_transformer_medium_config(; kwargs...)

Medium MoE Transformer configuration (1024d model).
"""
function moe_transformer_medium_config(; kwargs...)
    return MoETransformerConfig(
        vocab_size = 50257,
        max_position_embeddings = 1024,
        hidden_size = 1024,
        num_hidden_layers = 24,
        num_attention_heads = 16,
        intermediate_size = 4096,
        num_experts = 16,
        expert_top_k = 2,
        expert_type = :gated,
        balance_loss_weight = Float32(0.01),
        z_loss_weight = Float32(0.001);
        kwargs...
    )
end

"""
    moe_transformer_large_config(; kwargs...)

Large MoE Transformer configuration (1280d model).
"""
function moe_transformer_large_config(; kwargs...)
    return MoETransformerConfig(
        vocab_size = 50257,
        max_position_embeddings = 1024,
        hidden_size = 1280,
        num_hidden_layers = 36,
        num_attention_heads = 20,
        intermediate_size = 5120,
        num_experts = 32,
        expert_top_k = 2,
        expert_type = :gated,
        balance_loss_weight = Float32(0.01),
        z_loss_weight = Float32(0.001);
        kwargs...
    )
end


"""
    count_moe_parameters(model::Union{MoETransformerModel, MoETransformerLMHeadModel})

Count parameters separating shared vs expert parameters for analysis.
"""
function count_moe_parameters(model::Union{MoETransformerModel, MoETransformerLMHeadModel})
    base_model = model isa MoETransformerLMHeadModel ? model.model : model
    
    shared_params = 0
    expert_params = 0
    
    shared_params += sum(length, Flux.trainables(base_model.embed))
    shared_params += sum(length, Flux.trainables(base_model.ln_f))
    
    for wrapper in base_model.aux_collector.moe_wrappers
        expert_params += sum(length, Flux.trainables(wrapper.moe_layer))
    end
    
    all_decoder_params = sum(length, Flux.trainables(base_model.decoder))
    shared_params += all_decoder_params - expert_params
    
    # Count LM head if present (tied weights don't add parameters)

    if model isa MoETransformerLMHeadModel
        # LM head uses tied weights, so no additional parameters
        # shared_params += sum(length, Flux.trainables(model.lm_head))
    end
    
    total_params = shared_params + expert_params
    
    return (
        total = total_params,
        shared = shared_params,
        expert = expert_params,
        expert_ratio = expert_params / total_params
    )
end

"""
    create_moe_training_loss(model_output, targets; aux_loss_weight=Float32(1.0))

Create training loss combining language modeling and auxiliary losses.
Uses numerically stable vectorized log-softmax cross-entropy computation.
"""
function create_moe_training_loss(model_output, targets; aux_loss_weight::Float32=Float32(1.0))
    logits = model_output.logit  
    aux_loss = get(model_output, :aux_loss, Float32(0.0))
    

    
    vocab_size, batch_size, seq_length = size(logits)
    

    logits_flat = reshape(logits, vocab_size, batch_size * seq_length)
    
    targets_flat = reshape(Int.(targets), batch_size * seq_length)
    
    log_probs = Flux.logsoftmax(logits_flat, dims=1)  
    

    N = length(targets_flat)
    linear_indices = targets_flat .+ (0:N-1) .* vocab_size
    
    target_log_probs = log_probs[linear_indices]
    
    lm_loss = -mean(target_log_probs)
    
    total_loss = lm_loss + aux_loss_weight * aux_loss
    
    return (
        total_loss = total_loss,
        lm_loss = lm_loss,
        aux_loss = aux_loss,
        aux_loss_weight = aux_loss_weight
    )
end

"""
    prepare_transformer_inputs(input_ids::AbstractMatrix; position_ids=nothing)

Prepare inputs in the correct NamedTuple format for Transformers.jl models.
"""
function prepare_transformer_inputs(input_ids::AbstractMatrix; position_ids=nothing)
    batch_size, seq_length = size(input_ids)
    
    if position_ids === nothing
        position_ids = repeat(0:(seq_length-1), 1, batch_size)'
    end
    
    return (
        token = input_ids,
        position = position_ids .+ 1  
    )
end

"""
    initialize_moe_transformer_weights!(model::Union{MoETransformerModel, MoETransformerLMHeadModel}, config::MoETransformerConfig)

Initialize weights following Transformer conventions.
"""
function initialize_moe_transformer_weights!(model::Union{MoETransformerModel, MoETransformerLMHeadModel}, config::MoETransformerConfig)
    base_model = model isa MoETransformerLMHeadModel ? model.model : model
    
    for param in Flux.trainables(base_model.embed)
        if param isa AbstractMatrix
            Flux.truncated_normal!(param; std=config.initializer_range)
        end
    end
    
    for param in Flux.trainables(base_model.ln_f)
        if param isa AbstractVector
            fill!(param, Float32(1.0))  
        end
    end
    

    
    return model
end


function Base.show(io::IO, config::MoETransformerConfig)
    print(io, "MoETransformerConfig(")
    print(io, "$(config.hidden_size)d, ")
    print(io, "$(config.num_hidden_layers)L, ")
    print(io, "$(config.num_experts)E, ")
    print(io, "top$(config.expert_top_k))")
end

function Base.show(io::IO, model::MoETransformerModel)
    params = count_moe_parameters(model)
    print(io, "MoETransformerModel(")
    print(io, "$(model.config.num_hidden_layers)L, ")
    print(io, "$(model.config.num_experts)E, ")
    print(io, "$(params.total) params)")
end

function Base.show(io::IO, model::MoETransformerLMHeadModel)
    params = count_moe_parameters(model)
    print(io, "MoETransformerLMHeadModel(")
    print(io, "$(model.config.num_hidden_layers)L, ")
    print(io, "$(model.config.num_experts)E, ")
    print(io, "$(params.total) params)")
end