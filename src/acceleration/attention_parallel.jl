using CUDA
using LinearAlgebra

struct ParallelAttentionConfig
    parallel_heads::Bool
    fused_qkv_projection::Bool
    flash_attention::Bool
    memory_efficient::Bool
    use_tensor_cores::Bool
    sequence_parallel::Bool
    head_grouping::Symbol
end

function ParallelAttentionConfig(device::AbstractDevice, num_heads::Int, seq_len::Int)
    if device isa GPUDevice
        compute_capability = CUDA.capability(device.device_id)
        has_tensor_cores = compute_capability >= v"7.0"
        
        return ParallelAttentionConfig(
            true,
            num_heads >= 8,
            seq_len > 512,
            true,
            has_tensor_cores,
            seq_len > 1024,
            :balanced
        )
    else
        return ParallelAttentionConfig(
            Threads.nthreads() >= num_heads,
            false,
            false,
            false,
            false,
            false,
            :sequential
        )
    end
end

struct AttentionWorkspace
    device::AbstractDevice
    query_buffer::AbstractArray
    key_buffer::AbstractArray
    value_buffer::AbstractArray
    attention_weights::AbstractArray
    output_buffer::AbstractArray
    temp_buffers::Dict{Symbol, AbstractArray}
    rope_cache::Union{Dict, Nothing}
end

function AttentionWorkspace(device::AbstractDevice, batch_size::Int, seq_len::Int,
                          dim::Int, num_heads::Int, head_size::Int)
    
    if device isa GPUDevice
        return with_device(device) do
            AttentionWorkspace(
                device,
                CUDA.zeros(Float32, head_size, num_heads, seq_len, batch_size),
                CUDA.zeros(Float32, head_size, num_heads, seq_len, batch_size),
                CUDA.zeros(Float32, head_size, num_heads, seq_len, batch_size),
                CUDA.zeros(Float32, seq_len, seq_len, num_heads, batch_size),
                CUDA.zeros(Float32, dim, seq_len, batch_size),
                Dict{Symbol, AbstractArray}(
                    :qkv_temp => CUDA.zeros(Float32, dim * 3, seq_len, batch_size),
                    :rope_temp => CUDA.zeros(ComplexF32, head_size ÷ 2, num_heads, seq_len, batch_size),
                    :softmax_temp => CUDA.zeros(Float32, seq_len, num_heads, batch_size),
                    :head_output => CUDA.zeros(Float32, head_size, num_heads, seq_len, batch_size)
                ),
                nothing
            )
        end
    else
        AttentionWorkspace(
            device,
            zeros(Float32, head_size, num_heads, seq_len, batch_size),
            zeros(Float32, head_size, num_heads, seq_len, batch_size),
            zeros(Float32, head_size, num_heads, seq_len, batch_size),
            zeros(Float32, seq_len, seq_len, num_heads, batch_size),
            zeros(Float32, dim, seq_len, batch_size),
            Dict{Symbol, AbstractArray}(
                :qkv_temp => zeros(Float32, dim * 3, seq_len, batch_size),
                :softmax_temp => zeros(Float32, seq_len, num_heads, batch_size),
                :head_output => zeros(Float32, head_size, num_heads, seq_len, batch_size)
            ),
            nothing
        )
    end
end

function parallel_attention_forward!(output::AbstractArray, input::AbstractArray,
                                   wq::AbstractMatrix, wk::AbstractMatrix, wv::AbstractMatrix, wo::AbstractMatrix,
                                   kv_cache, pos::Int, config, attention_config::ParallelAttentionConfig,
                                   workspace::AttentionWorkspace)
    
    device = workspace.device
    
    if device isa GPUDevice
        parallel_attention_forward_gpu!(output, input, wq, wk, wv, wo, kv_cache, 
                                      pos, config, attention_config, workspace)
    else
        parallel_attention_forward_cpu!(output, input, wq, wk, wv, wo, kv_cache, 
                                      pos, config, attention_config, workspace)
    end
end

function parallel_attention_forward_gpu!(output::CuArray, input::CuArray,
                                       wq::AbstractMatrix, wk::AbstractMatrix, wv::AbstractMatrix, wo::AbstractMatrix,
                                       kv_cache, pos::Int, config, attention_config::ParallelAttentionConfig,
                                       workspace::AttentionWorkspace)
    
    batch_size = size(input, 2)
    seq_len = size(input, 1)
    dim = config.dim
    num_heads = config.n_heads
    head_size = dim ÷ num_heads
    
    if attention_config.fused_qkv_projection
        compute_fused_qkv_gpu!(workspace, input, wq, wk, wv, dim)
    else
        compute_separate_qkv_gpu!(workspace, input, wq, wk, wv)
    end
    
    if pos > 1
        apply_rope_parallel_gpu!(workspace.query_buffer, workspace.key_buffer, 
                               pos, config, workspace)
    end
    
    if attention_config.parallel_heads
        compute_attention_parallel_heads_gpu!(workspace, pos, config, attention_config)
    else
        compute_attention_sequential_gpu!(workspace, pos, config)
    end
    
    project_output_gpu!(output, workspace.output_buffer, wo)
    
    CUDA.synchronize()
end

function compute_fused_qkv_gpu!(workspace::AttentionWorkspace, input::CuArray,
                              wq::AbstractMatrix, wk::AbstractMatrix, wv::AbstractMatrix, dim::Int)
    
    qkv_weight = get_buffer(Float32, (dim * 3, dim), workspace.device)
    
    try
        qkv_weight[1:dim, :] = wq
        qkv_weight[dim+1:2*dim, :] = wk
        qkv_weight[2*dim+1:3*dim, :] = wv
        
        CUDA.CUBLAS.gemm!('N', 'N', 1.0f0, qkv_weight, input, 0.0f0, workspace.temp_buffers[:qkv_temp])
        
        split_qkv_gpu!(workspace, workspace.temp_buffers[:qkv_temp], dim)
        
    finally
        return_buffer(qkv_weight, workspace.device)
    end
end

function split_qkv_gpu!(workspace::AttentionWorkspace, qkv_output::CuArray, dim::Int)
    batch_size = size(qkv_output, 3)
    seq_len = size(qkv_output, 2)
    head_size = size(workspace.query_buffer, 1)
    num_heads = size(workspace.query_buffer, 2)
    
    threads = (16, 8, 4)
    blocks = (cld(head_size, 16), cld(num_heads, 8), cld(seq_len, 4))
    
    CUDA.@cuda threads=threads blocks=blocks split_qkv_kernel!(
        workspace.query_buffer, workspace.key_buffer, workspace.value_buffer,
        qkv_output, dim, head_size, num_heads
    )
end

function split_qkv_kernel!(query::CuDeviceArray{Float32, 4}, key::CuDeviceArray{Float32, 4}, 
                          value::CuDeviceArray{Float32, 4}, qkv::CuDeviceArray{Float32, 3},
                          dim::Int, head_size::Int, num_heads::Int)
    
    h = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    head = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    seq = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    if h <= head_size && head <= num_heads && seq <= size(qkv, 2)
        for batch in 1:size(qkv, 3)
            idx = (head - 1) * head_size + h
            
            query[h, head, seq, batch] = qkv[idx, seq, batch]
            key[h, head, seq, batch] = qkv[idx + dim, seq, batch]
            value[h, head, seq, batch] = qkv[idx + 2 * dim, seq, batch]
        end
    end
    
    return nothing
end

function compute_separate_qkv_gpu!(workspace::AttentionWorkspace, input::CuArray,
                                 wq::AbstractMatrix, wk::AbstractMatrix, wv::AbstractMatrix)
    
    q_flat = reshape(workspace.query_buffer, :, size(workspace.query_buffer, 4))
    k_flat = reshape(workspace.key_buffer, :, size(workspace.key_buffer, 4))
    v_flat = reshape(workspace.value_buffer, :, size(workspace.value_buffer, 4))
    
    CUDA.CUBLAS.gemm!('N', 'N', 1.0f0, wq, input, 0.0f0, q_flat)
    CUDA.CUBLAS.gemm!('N', 'N', 1.0f0, wk, input, 0.0f0, k_flat)
    CUDA.CUBLAS.gemm!('N', 'N', 1.0f0, wv, input, 0.0f0, v_flat)
end

function apply_rope_parallel_gpu!(query::CuArray, key::CuArray, pos::Int, 
                                config, workspace::AttentionWorkspace)
    
    if isnothing(workspace.rope_cache)
        workspace.rope_cache = precompute_rope_cache_gpu(config, workspace.device)
    end
    
    head_size = size(query, 1)
    num_heads = size(query, 2)
    batch_size = size(query, 4)
    
    if config.rope_is_neox
        apply_rope_neox_parallel_gpu!(query, key, pos, config.rope_freq_base, head_size)
    else
        query_complex = reinterpret(ComplexF32, workspace.temp_buffers[:rope_temp])
        key_complex = reinterpret(ComplexF32, workspace.temp_buffers[:rope_temp])
        
        copyto!(query_complex, reinterpret(ComplexF32, query))
        copyto!(key_complex, reinterpret(ComplexF32, key))
        
        launch_rope!(query_complex, pos, config.rope_freq_base, head_size ÷ 2)
        launch_rope!(key_complex, pos, config.rope_freq_base, head_size ÷ 2)
        
        copyto!(query, reinterpret(Float32, query_complex))
        copyto!(key, reinterpret(Float32, key_complex))
    end
end

function apply_rope_neox_parallel_gpu!(query::CuArray, key::CuArray, pos::Int, 
                                     freq_base::Float32, head_size::Int)
    
    threads = (32, 8, 1)
    blocks = (cld(head_size ÷ 2, 32), cld(size(query, 2), 8), size(query, 4))
    
    CUDA.@cuda threads=threads blocks=blocks rope_neox_kernel!(
        query, key, Int32(pos), freq_base, head_size
    )
end

function rope_neox_kernel!(query::CuDeviceArray{Float32, 4}, key::CuDeviceArray{Float32, 4},
                         pos::Int32, freq_base::Float32, head_size::Int)
    
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    head = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    batch = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    head_size_div2 = head_size ÷ 2
    
    if i <= head_size_div2 && head <= size(query, 2) && batch <= size(query, 4)
        theta = Float32(pos - 1) * (freq_base ^ (-2.0f0 * (i - 1) / head_size))
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        
        for seq in 1:size(query, 3)
            q1 = query[i, head, seq, batch]
            q2 = query[i + head_size_div2, head, seq, batch]
            query[i, head, seq, batch] = q1 * cos_theta - q2 * sin_theta
            query[i + head_size_div2, head, seq, batch] = q1 * sin_theta + q2 * cos_theta
            
            k1 = key[i, head, seq, batch]
            k2 = key[i + head_size_div2, head, seq, batch]
            key[i, head, seq, batch] = k1 * cos_theta - k2 * sin_theta
            key[i + head_size_div2, head, seq, batch] = k1 * sin_theta + k2 * cos_theta
        end
    end
    
    return nothing
end

function compute_attention_parallel_heads_gpu!(workspace::AttentionWorkspace, pos::Int, 
                                             config, attention_config::ParallelAttentionConfig)
    
    num_heads = size(workspace.query_buffer, 2)
    batch_size = size(workspace.query_buffer, 4)
    seq_len = min(pos, size(workspace.query_buffer, 3))
    head_size = size(workspace.query_buffer, 1)
    
    scale = inv(sqrt(Float32(head_size)))
    
    if attention_config.flash_attention && seq_len > 512
        compute_flash_attention_gpu!(workspace, pos, scale)
    else
        compute_standard_attention_gpu!(workspace, pos, scale)
    end
end

function compute_standard_attention_gpu!(workspace::AttentionWorkspace, pos::Int, scale::Float32)
    query = workspace.query_buffer
    key = workspace.key_buffer
    value = workspace.value_buffer
    attention_weights = workspace.attention_weights
    
    launch_attention_weights!(attention_weights, query, key, scale)
    
    apply_causal_mask_gpu!(attention_weights, pos)
    
    apply_softmax_attention_gpu!(attention_weights, pos)
    
    launch_attention_combine!(workspace.temp_buffers[:head_output], attention_weights, value)
    
    reshape_head_output_gpu!(workspace.output_buffer, workspace.temp_buffers[:head_output])
end

function apply_causal_mask_gpu!(attention_weights::CuArray, pos::Int)
    threads = (16, 16, 4)
    blocks = (cld(pos, 16), cld(pos, 16), cld(size(attention_weights, 3), 4))
    
    CUDA.@cuda threads=threads blocks=blocks causal_mask_kernel!(attention_weights, pos)
end

function causal_mask_kernel!(weights::CuDeviceArray{Float32, 4}, pos::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    head = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    if i <= pos && j <= pos && head <= size(weights, 3)
        for batch in 1:size(weights, 4)
            if i < j
                weights[i, j, head, batch] = -Inf32
            end
        end
    end
    
    return nothing
end

function apply_softmax_attention_gpu!(attention_weights::CuArray, pos::Int)
    threads = (256, 1, 1)
    blocks = (cld(size(attention_weights, 3), 1), size(attention_weights, 4), 1)
    
    CUDA.@cuda threads=threads blocks=blocks softmax_attention_kernel!(attention_weights, pos)
end

function softmax_attention_kernel!(weights::CuDeviceArray{Float32, 4}, pos::Int)
    head = blockIdx().x
    batch = blockIdx().y
    
    if head <= size(weights, 3) && batch <= size(weights, 4)
        for i in 1:pos
            max_val = -Inf32
            for j in 1:i
                max_val = max(max_val, weights[i, j, head, batch])
            end
            
            sum_exp = 0.0f0
            for j in 1:i
                val = exp(weights[i, j, head, batch] - max_val)
                weights[i, j, head, batch] = val
                sum_exp += val
            end
            
            for j in 1:i
                weights[i, j, head, batch] /= sum_exp
            end
        end
    end
    
    return nothing
end

function reshape_head_output_gpu!(output::CuArray, head_output::CuArray)
    head_size = size(head_output, 1)
    num_heads = size(head_output, 2)
    seq_len = size(head_output, 3)
    batch_size = size(head_output, 4)
    
    threads = (32, 8, 4)
    blocks = (cld(head_size * num_heads, 32), cld(seq_len, 8), cld(batch_size, 4))
    
    CUDA.@cuda threads=threads blocks=blocks reshape_head_kernel!(output, head_output, head_size, num_heads)
end

function reshape_head_kernel!(output::CuDeviceArray{Float32, 3}, 
                            head_output::CuDeviceArray{Float32, 4},
                            head_size::Int, num_heads::Int)
    
    dim_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    seq_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    batch_idx = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    if dim_idx <= head_size * num_heads && seq_idx <= size(output, 2) && batch_idx <= size(output, 3)
        head = (dim_idx - 1) ÷ head_size + 1
        h = (dim_idx - 1) % head_size + 1
        
        output[dim_idx, seq_idx, batch_idx] = head_output[h, head, seq_idx, batch_idx]
    end
    
    return nothing
end

function project_output_gpu!(final_output::CuArray, attention_output::CuArray, wo::AbstractMatrix)
    output_flat = reshape(final_output, size(final_output, 1), :)
    attention_flat = reshape(attention_output, size(attention_output, 1), :)
    
    CUDA.CUBLAS.gemm!('N', 'N', 1.0f0, wo, attention_flat, 0.0f0, output_flat)
end

function parallel_attention_forward_cpu!(output::AbstractArray, input::AbstractArray,
                                       wq::AbstractMatrix, wk::AbstractMatrix, wv::AbstractMatrix, wo::AbstractMatrix,
                                       kv_cache, pos::Int, config, attention_config::ParallelAttentionConfig,
                                       workspace::AttentionWorkspace)
    
    compute_qkv_cpu!(workspace, input, wq, wk, wv)
    
    if pos > 1
        apply_rope_cpu!(workspace.query_buffer, workspace.key_buffer, pos, config)
    end
    
    if attention_config.parallel_heads && Threads.nthreads() >= config.n_heads
        compute_attention_parallel_heads_cpu!(workspace, pos, config)
    else
        compute_attention_sequential_cpu!(workspace, pos, config)
    end
    
    project_output_cpu!(output, workspace.output_buffer, wo)
end

function compute_qkv_cpu!(workspace::AttentionWorkspace, input::AbstractArray,
                        wq::AbstractMatrix, wk::AbstractMatrix, wv::AbstractMatrix)
    
    q_flat = reshape(workspace.query_buffer, :, size(workspace.query_buffer, 4))
    k_flat = reshape(workspace.key_buffer, :, size(workspace.key_buffer, 4))
    v_flat = reshape(workspace.value_buffer, :, size(workspace.value_buffer, 4))
    
    mul!(q_flat, wq, input)
    mul!(k_flat, wk, input)
    mul!(v_flat, wv, input)
end

function apply_rope_cpu!(query::AbstractArray, key::AbstractArray, pos::Int, config)
    head_size = size(query, 1)
    num_heads = size(query, 2)
    batch_size = size(query, 4)
    
    if config.rope_is_neox
        apply_rope_neox_cpu!(query, key, pos, config.rope_freq_base, head_size)
    else
        apply_rope_normal_cpu!(query, key, pos, config.rope_freq_base, head_size)
    end
end

function apply_rope_normal_cpu!(query::AbstractArray, key::AbstractArray, pos::Int, 
                              freq_base::Float32, head_size::Int)
    
    head_size_div2 = head_size ÷ 2
    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))
    
    for batch in 1:size(query, 4)
        for head in 1:size(query, 2)
            for seq in 1:size(query, 3)
                theta = Float32(pos - 1)
                
                for i in 1:head_size_div2
                    cos_theta = cos(theta)
                    sin_theta = sin(theta)
                    
                    q_real = query[2*i-1, head, seq, batch]
                    q_imag = query[2*i, head, seq, batch]
                    query[2*i-1, head, seq, batch] = q_real * cos_theta - q_imag * sin_theta
                    query[2*i, head, seq, batch] = q_real * sin_theta + q_imag * cos_theta
                    
                    k_real = key[2*i-1, head, seq, batch]
                    k_imag = key[2*i, head, seq, batch]
                    key[2*i-1, head, seq, batch] = k_real * cos_theta - k_imag * sin_theta
                    key[2*i, head, seq, batch] = k_real * sin_theta + k_imag * cos_theta
                    
                    theta *= theta_scale
                end
            end
        end
    end
end

function apply_rope_neox_cpu!(query::AbstractArray, key::AbstractArray, pos::Int, 
                            freq_base::Float32, head_size::Int)
    
    head_size_div2 = head_size ÷ 2
    
    for batch in 1:size(query, 4)
        for head in 1:size(query, 2)
            for seq in 1:size(query, 3)
                for i in 1:head_size_div2
                    theta = Float32(pos - 1) * (freq_base ^ (-2.0f0 * (i - 1) / head_size))
                    cos_theta = cos(theta)
                    sin_theta = sin(theta)
                    
                    q1 = query[i, head, seq, batch]
                    q2 = query[i + head_size_div2, head, seq, batch]
                    query[i, head, seq, batch] = q1 * cos_theta - q2 * sin_theta
                    query[i + head_size_div2, head, seq, batch] = q1 * sin_theta + q2 * cos_theta
                    
                    k1 = key[i, head, seq, batch]
                    k2 = key[i + head_size_div2, head, seq, batch]
                    key[i, head, seq, batch] = k1 * cos_theta - k2 * sin_theta
                    key[i + head_size_div2, head, seq, batch] = k1 * sin_theta + k2 * cos_theta
                end
            end
        end
    end
end

function compute_attention_parallel_heads_cpu!(workspace::AttentionWorkspace, pos::Int, config)
    num_heads = size(workspace.query_buffer, 2)
    batch_size = size(workspace.query_buffer, 4)
    head_size = size(workspace.query_buffer, 1)
    scale = inv(sqrt(Float32(head_size)))
    
    Threads.@threads for head in 1:num_heads
        for batch in 1:batch_size
            compute_single_head_attention_cpu!(workspace, head, batch, pos, scale)
        end
    end
end

function compute_single_head_attention_cpu!(workspace::AttentionWorkspace, head::Int, 
                                          batch::Int, pos::Int, scale::Float32)
    
    query = view(workspace.query_buffer, :, head, :, batch)
    key = view(workspace.key_buffer, :, head, :, batch)
    value = view(workspace.value_buffer, :, head, :, batch)
    attention_weights = view(workspace.attention_weights, :, :, head, batch)
    output = view(workspace.temp_buffers[:head_output], :, head, :, batch)
    
    for i in 1:pos
        for j in 1:i
            score = 0.0f0
            for d in 1:size(query, 1)
                score += query[d, i] * key[d, j]
            end
            attention_weights[i, j] = score * scale
        end
        
        for j in (i+1):size(attention_weights, 2)
            attention_weights[i, j] = -Inf32
        end
        
        max_val = maximum(view(attention_weights, i, 1:i))
        sum_exp = 0.0f0
        for j in 1:i
            val = exp(attention_weights[i, j] - max_val)
            attention_weights[i, j] = val
            sum_exp += val
        end
        
        for j in 1:i
            attention_weights[i, j] /= sum_exp
        end
        
        for d in 1:size(output, 1)
            sum_val = 0.0f0
            for j in 1:i
                sum_val += attention_weights[i, j] * value[d, j]
            end
            output[d, i] = sum_val
        end
    end
end

function compute_attention_sequential_cpu!(workspace::AttentionWorkspace, pos::Int, config)
    for head in 1:size(workspace.query_buffer, 2)
        for batch in 1:size(workspace.query_buffer, 4)
            compute_single_head_attention_cpu!(workspace, head, batch, pos, 
                                             inv(sqrt(Float32(size(workspace.query_buffer, 1)))))
        end
    end
end

function project_output_cpu!(final_output::AbstractArray, attention_output::AbstractArray, wo::AbstractMatrix)
    output_flat = reshape(final_output, size(final_output, 1), :)
    attention_flat = reshape(attention_output, size(attention_output, 1), :)
    
    mul!(output_flat, wo, attention_flat)
end

function precompute_rope_cache_gpu(config, device::GPUDevice)
    return with_device(device) do
        seq_len = config.seq_len
        head_size = config.dim ÷ config.n_heads
        head_size_div2 = head_size ÷ 2
        
        freq_base = config.rope_freq_base
        theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))
        
        if config.rope_is_neox
            cos_cache = CUDA.zeros(Float32, head_size_div2, seq_len)
            sin_cache = CUDA.zeros(Float32, head_size_div2, seq_len)
            
            for pos in 1:seq_len
                for i in 1:head_size_div2
                    theta = Float32(pos - 1) * (freq_base ^ (-2.0f0 * (i - 1) / head_size))
                    cos_cache[i, pos] = cos(theta)
                    sin_cache[i, pos] = sin(theta)
                end
            end
            
            return Dict(:cos_cache => cos_cache, :sin_cache => sin_cache)
        else
            exp_cache = CUDA.zeros(ComplexF32, head_size_div2, seq_len)
            
            for pos in 1:seq_len
                theta = Float32(pos - 1)
                for i in 1:head_size_div2
                    exp_cache[i, pos] = ComplexF32(cos(theta), sin(theta))
                    theta *= theta_scale
                end
            end
            
            return Dict(:exp_cache => exp_cache)
        end
    end
end