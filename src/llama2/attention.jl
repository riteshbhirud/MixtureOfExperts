"""
Complete Attention Implementation for MoE-Llama2 Integration

This module provides full attention computation that exactly matches Llama2's 
implementation, including proper RoPE and KV caching.
"""

"""
    moe_attention!(state::MoERunState, layer::MoETransformerLayerWeights, 
                   pos::Int, config::MoELlamaConfig, layer_idx::Int)

Complete multi-head attention computation with RoPE and KV caching.
Exactly matches Llama2's attention implementation.
"""
function moe_attention!(state::MoERunState, layer::MoETransformerLayerWeights, 
                       pos::Int, config::MoELlamaConfig, layer_idx::Int)  
    llama_layer = layer.llama_layer
    
    Llama2.rmsnorm!(state.xb, state.x, llama_layer.rms_att_weight)
    
    Llama2.matmul!(state.q, llama_layer.wq, state.xb)
    Llama2.matmul!(state.k, llama_layer.wk, state.xb) 
    Llama2.matmul!(state.v, llama_layer.wv, state.xb)
    

    kv_cache = state.kvcache_layers[layer_idx]
    
    head_size = config.dim ÷ config.n_heads
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    
    q_reshaped = reshape(state.q, head_size, n_heads)
    
    k_reshaped = reshape(state.k, head_size, n_kv_heads)
    v_reshaped = reshape(state.v, head_size, n_kv_heads)
    
    apply_rope!(q_reshaped, pos, config)
    apply_rope!(k_reshaped, pos, config)
    
    copyto!(view(kv_cache.key_cache, :, :, pos), k_reshaped)
    copyto!(view(kv_cache.value_cache, pos, :, :), v_reshaped)
    
    compute_attention_output!(state, kv_cache, q_reshaped, pos, config)
    
    Llama2.matmul!(state.xb2, llama_layer.wo, state.xb)
    
    return nothing
end

"""
    apply_rope!(x::AbstractMatrix{Float32}, pos::Int, config::MoELlamaConfig)

Apply Rotary Position Embedding (RoPE) to query or key vectors.
Supports both normal and NeoX variants based on config.rope_is_neox.
"""
function apply_rope!(x::AbstractMatrix{Float32}, pos::Int, config::MoELlamaConfig)
    if config.rope_is_neox
        apply_rope_neox!(x, pos, config)
    else
        apply_rope_normal!(x, pos, config)
    end
    return nothing
end

"""
    apply_rope_normal!(x::AbstractMatrix{Float32}, pos::Int, config::MoELlamaConfig)

Apply standard RoPE as used in original Llama.
Uses complex number representation for rotation.
"""
function apply_rope_normal!(x::AbstractMatrix{Float32}, pos::Int, config::MoELlamaConfig)
    x_complex = reinterpret(ComplexF32, x)
    head_size_div2, n_heads = size(x_complex)
    
    freq_base = config.rope_freq_base
    freq_scale = 1.0f0
    
    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))
    
    @inbounds for head in 1:n_heads
        theta = freq_scale * (pos - 1)  
        
        for i in 1:head_size_div2
            x_complex[i, head] *= cis(theta)
            theta *= theta_scale
        end
    end
    
    return nothing
end

"""
    apply_rope_neox!(x::AbstractMatrix{Float32}, pos::Int, config::MoELlamaConfig)

Apply NeoX-style RoPE as used in some model variants.
Rotates pairs of adjacent elements directly.
"""
function apply_rope_neox!(x::AbstractMatrix{Float32}, pos::Int, config::MoELlamaConfig)
    head_size, n_heads = size(x)
    head_size_div2 = head_size ÷ 2
    
    freq_base = config.rope_freq_base
    freq_scale = 1.0f0
    
    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))
    
    @inbounds for head in 1:n_heads
        theta = freq_scale * (pos - 1)  
        
        for i in 1:head_size_div2
            idx1 = i
            idx2 = head_size_div2 + i
            
            x1 = x[idx1, head]
            x2 = x[idx2, head]
            
            cos_theta = cos(theta)
            sin_theta = sin(theta)
            
            x[idx1, head] = x1 * cos_theta - x2 * sin_theta
            x[idx2, head] = x1 * sin_theta + x2 * cos_theta
            
            theta *= theta_scale
        end
    end
    
    return nothing
end

"""
    compute_attention_output!(state::MoERunState, kv_cache::Llama2.KVCache,
                             q::AbstractMatrix, pos::Int, config::MoELlamaConfig)

Compute attention weights and combine with values.
Implements efficient attention computation with proper scaling and softmax.
"""
function compute_attention_output!(state::MoERunState, kv_cache::Llama2.KVCache,
                                  q::AbstractMatrix, pos::Int, config::MoELlamaConfig)
    head_size = config.dim ÷ config.n_heads
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    
    n_gqa = n_heads ÷ n_kv_heads  
    
    att = reshape(view(state.att, 1:(n_heads * pos)), pos, n_heads)
    
    compute_attention_weights!(att, kv_cache.key_cache, q, pos, n_gqa)
    
    attention_scale = inv(sqrt(Float32(head_size)))
    att .*= attention_scale
    
    for h in 1:n_heads
        for t in (pos + 1):size(att, 1)
            if t <= size(att, 1)
                att[t, h] = -Inf32
            end
        end
        
        softmax_inplace!(view(att, 1:pos, h))
    end
    
    xb_reshaped = reshape(state.xb, head_size, n_heads)
    fill!(xb_reshaped, 0.0f0)
    
    combine_attention_values!(xb_reshaped, kv_cache.value_cache, att, pos, n_gqa)
    
    return nothing
end

"""
    compute_attention_weights!(att::AbstractMatrix, key_cache::AbstractArray,
                              q::AbstractMatrix, pos::Int, n_gqa::Int)

Compute attention weights = Q * K^T for all positions up to pos.
Handles Group Query Attention (GQA) where multiple query heads share KV heads.
"""
function compute_attention_weights!(att::AbstractMatrix, key_cache::AbstractArray,
                                   q::AbstractMatrix, pos::Int, n_gqa::Int)
    n_heads = size(q, 2)
    head_size = size(q, 1)
    
    kv_head = 1 
    
    @inbounds for h in 1:n_heads
        @fastmath for t in 1:pos
            score = 0.0f0
            for i in 1:head_size
                score += q[i, h] * key_cache[i, kv_head, t]
            end
            att[t, h] = score
        end
        
        if h % n_gqa == 0
            kv_head += 1
        end
    end
    
    return nothing
end

"""
    combine_attention_values!(output::AbstractMatrix, value_cache::AbstractArray,
                             att::AbstractMatrix, pos::Int, n_gqa::Int)

Combine attention weights with values: output = Σ(att[t] * value[t]).
Handles Group Query Attention (GQA) mapping.
"""
function combine_attention_values!(output::AbstractMatrix, value_cache::AbstractArray,
                                  att::AbstractMatrix, pos::Int, n_gqa::Int)
    n_heads = size(output, 2)
    head_size = size(output, 1)
    
    kv_head = 1  
    
    @inbounds for h in 1:n_heads
        @fastmath for i in 1:head_size
            sum_val = 0.0f0
            for t in 1:pos
                sum_val += att[t, h] * value_cache[t, i, kv_head]
            end
            output[i, h] = sum_val
        end
        
        if h % n_gqa == 0
            kv_head += 1
        end
    end
    
    return nothing
end

"""
    softmax_inplace!(x::AbstractVector)

Compute softmax in-place with numerical stability.
Uses the standard exp(x - max(x)) / sum(exp(x - max(x))) formula.
"""
function softmax_inplace!(x::AbstractVector)
    if isempty(x)
        return nothing
    end
    
    x_max = maximum(x)
    
    for i in eachindex(x)
        x[i] = exp(x[i] - x_max)
    end
    
    x_sum = sum(x)
    if x_sum > 0
        for i in eachindex(x)
            x[i] /= x_sum
        end
    end
    
    return nothing
end

"""
    precompute_rope_cache(config::MoELlamaConfig)

Precompute RoPE rotation angles for efficient inference.
Returns cache that can be used to avoid repeated trigonometric computations.
"""
function precompute_rope_cache(config::MoELlamaConfig)
    seq_len = config.seq_len
    head_size = config.dim ÷ config.n_heads
    head_size_div2 = head_size ÷ 2
    
    freq_base = config.rope_freq_base
    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))
    
    cache = Dict{Symbol, Any}()
    
    if config.rope_is_neox
        cos_cache = zeros(Float32, head_size_div2, seq_len)
        sin_cache = zeros(Float32, head_size_div2, seq_len)
        
        for pos in 1:seq_len
            theta = 1.0f0 * (pos - 1)
            for i in 1:head_size_div2
                cos_cache[i, pos] = cos(theta)
                sin_cache[i, pos] = sin(theta)
                theta *= theta_scale
            end
        end
        
        cache[:cos_cache] = cos_cache
        cache[:sin_cache] = sin_cache
    else
        exp_cache = zeros(ComplexF32, head_size_div2, seq_len)
        
        for pos in 1:seq_len
            theta = 1.0f0 * (pos - 1)
            for i in 1:head_size_div2
                exp_cache[i, pos] = cis(theta)
                theta *= theta_scale
            end
        end
        
        cache[:exp_cache] = exp_cache
    end
    
    return cache
end

"""
    apply_rope_cached!(x::AbstractMatrix, pos::Int, config::MoELlamaConfig, cache::Dict)

Apply RoPE using precomputed cache for improved performance.
"""
function apply_rope_cached!(x::AbstractMatrix, pos::Int, config::MoELlamaConfig, cache::Dict)
    if config.rope_is_neox
        apply_rope_neox_cached!(x, pos, cache)
    else
        apply_rope_normal_cached!(x, pos, cache)
    end
    return nothing
end

function apply_rope_normal_cached!(x::AbstractMatrix, pos::Int, cache::Dict)
    x_complex = reinterpret(ComplexF32, x)
    exp_cache = cache[:exp_cache]
    head_size_div2, n_heads = size(x_complex)
    
    @inbounds for head in 1:n_heads
        for i in 1:head_size_div2
            x_complex[i, head] *= exp_cache[i, pos]
        end
    end
    
    return nothing
end

function apply_rope_neox_cached!(x::AbstractMatrix, pos::Int, cache::Dict)
    cos_cache = cache[:cos_cache]
    sin_cache = cache[:sin_cache]
    head_size, n_heads = size(x)
    head_size_div2 = head_size ÷ 2
    
    @inbounds for head in 1:n_heads
        for i in 1:head_size_div2
            idx1 = i
            idx2 = head_size_div2 + i
            
            x1 = x[idx1, head]
            x2 = x[idx2, head]
            
            cos_val = cos_cache[i, pos]
            sin_val = sin_cache[i, pos]
            
            x[idx1, head] = x1 * cos_val - x2 * sin_val
            x[idx2, head] = x1 * sin_val + x2 * cos_val
        end
    end
    
    return nothing
end