using CUDA

function expert_forward_kernel!(output::CuDeviceArray{Float32, 2},
                               w1::CuDeviceArray{Float32, 2},
                               w2::CuDeviceArray{Float32, 2},
                               w3::CuDeviceArray{Float32, 2},
                               input::CuDeviceArray{Float32, 2},
                               temp1::CuDeviceArray{Float32, 2},
                               temp2::CuDeviceArray{Float32, 2})
    
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i <= size(output, 1) && j <= size(output, 2)
        gate_sum = 0.0f0
        up_sum = 0.0f0
        
        for k in 1:size(input, 1)
            gate_sum += w1[i, k] * input[k, j]
            up_sum += w3[i, k] * input[k, j]
        end
        
        gate_val = gate_sum * (1.0f0 / (1.0f0 + exp(-gate_sum)))
        temp1[i, j] = gate_val * up_sum
        
        sync_threads()
        
        output_sum = 0.0f0
        for k in 1:size(temp1, 1)
            output_sum += w2[i, k] * temp1[k, j]
        end
        
        output[i, j] = output_sum
    end
    
    return nothing
end

function launch_expert_forward!(output::CuArray{Float32, 2},
                               w1::CuArray{Float32, 2},
                               w2::CuArray{Float32, 2},
                               w3::CuArray{Float32, 2},
                               input::CuArray{Float32, 2},
                               temp1::CuArray{Float32, 2},
                               temp2::CuArray{Float32, 2})
    
    threads_x = min(32, size(output, 1))
    threads_y = min(32, size(output, 2))
    threads = (threads_x, threads_y)
    
    blocks_x = cld(size(output, 1), threads_x)
    blocks_y = cld(size(output, 2), threads_y)
    blocks = (blocks_x, blocks_y)
    
    CUDA.@cuda threads=threads blocks=blocks expert_forward_kernel!(
        output, w1, w2, w3, input, temp1, temp2)
end

function router_forward_kernel!(output::CuDeviceArray{Float32, 2},
                              weight::CuDeviceArray{Float32, 2},
                              input::CuDeviceArray{Float32, 2})
    
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i <= size(output, 1) && j <= size(output, 2)
        sum_val = 0.0f0
        for k in 1:size(input, 1)
            sum_val += weight[i, k] * input[k, j]
        end
        output[i, j] = sum_val
    end
    
    return nothing
end

function launch_router_forward!(output::CuArray{Float32, 2},
                               weight::CuArray{Float32, 2},
                               input::CuArray{Float32, 2})
    
    threads_x = min(32, size(output, 1))
    threads_y = min(32, size(output, 2))
    threads = (threads_x, threads_y)
    
    blocks_x = cld(size(output, 1), threads_x)
    blocks_y = cld(size(output, 2), threads_y)
    blocks = (blocks_x, blocks_y)
    
    CUDA.@cuda threads=threads blocks=blocks router_forward_kernel!(
        output, weight, input)
end

function topk_kernel!(indices::CuDeviceArray{Int32, 2},
                     values::CuDeviceArray{Float32, 2},
                     input::CuDeviceArray{Float32, 2},
                     k::Int32)
    
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if j <= size(input, 2)
        col = view(input, :, j)
        n = length(col)
        
        for pos in 1:k
            max_idx = 1
            max_val = col[1]
            
            for i in 2:n
                if col[i] > max_val
                    max_val = col[i]
                    max_idx = i
                end
            end
            
            indices[pos, j] = max_idx
            values[pos, j] = max_val
            col[max_idx] = -Inf32
        end
    end
    
    return nothing
end

function launch_topk!(indices::CuArray{Int32, 2},
                     values::CuArray{Float32, 2},
                     input::CuArray{Float32, 2},
                     k::Int)
    
    threads = min(256, size(input, 2))
    blocks = cld(size(input, 2), threads)
    
    CUDA.@cuda threads=threads blocks=blocks topk_kernel!(
        indices, values, input, Int32(k))
end

function softmax_kernel!(output::CuDeviceArray{Float32, 2},
                        input::CuDeviceArray{Float32, 2})
    
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if j <= size(input, 2)
        max_val = -Inf32
        for i in 1:size(input, 1)
            max_val = max(max_val, input[i, j])
        end
        
        sum_exp = 0.0f0
        for i in 1:size(input, 1)
            val = exp(input[i, j] - max_val)
            output[i, j] = val
            sum_exp += val
        end
        
        for i in 1:size(input, 1)
            output[i, j] /= sum_exp
        end
    end
    
    return nothing
end

function launch_softmax!(output::CuArray{Float32, 2},
                        input::CuArray{Float32, 2})
    
    threads = min(256, size(input, 2))
    blocks = cld(size(input, 2), threads)
    
    CUDA.@cuda threads=threads blocks=blocks softmax_kernel!(output, input)
end

function weighted_sum_kernel!(output::CuDeviceArray{Float32, 2},
                             expert_outputs::CuDeviceArray{Float32, 3},
                             weights::CuDeviceArray{Float32, 2},
                             indices::CuDeviceArray{Int32, 2})
    
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i <= size(output, 1) && j <= size(output, 2)
        sum_val = 0.0f0
        
        for k in 1:size(indices, 1)
            expert_idx = indices[k, j]
            if expert_idx > 0
                weight = weights[k, j]
                sum_val += weight * expert_outputs[i, j, expert_idx]
            end
        end
        
        output[i, j] = sum_val
    end
    
    return nothing
end

function launch_weighted_sum!(output::CuArray{Float32, 2},
                             expert_outputs::CuArray{Float32, 3},
                             weights::CuArray{Float32, 2},
                             indices::CuArray{Int32, 2})
    
    threads_x = min(32, size(output, 1))
    threads_y = min(32, size(output, 2))
    threads = (threads_x, threads_y)
    
    blocks_x = cld(size(output, 1), threads_x)
    blocks_y = cld(size(output, 2), threads_y)
    blocks = (blocks_x, blocks_y)
    
    CUDA.@cuda threads=threads blocks=blocks weighted_sum_kernel!(
        output, expert_outputs, weights, indices)
end

function attention_weights_kernel!(output::CuDeviceArray{Float32, 3},
                                 query::CuDeviceArray{Float32, 3},
                                 key::CuDeviceArray{Float32, 3},
                                 scale::Float32)
    
    seq_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    seq_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    head = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    if seq_i <= size(output, 1) && seq_j <= size(output, 2) && head <= size(output, 3)
        if seq_i <= seq_j
            sum_val = 0.0f0
            for d in 1:size(query, 1)
                sum_val += query[d, seq_i, head] * key[d, seq_j, head]
            end
            output[seq_i, seq_j, head] = sum_val * scale
        else
            output[seq_i, seq_j, head] = -Inf32
        end
    end
    
    return nothing
end

function launch_attention_weights!(output::CuArray{Float32, 3},
                                  query::CuArray{Float32, 3},
                                  key::CuArray{Float32, 3},
                                  scale::Float32)
    
    threads_x = min(16, size(output, 1))
    threads_y = min(16, size(output, 2))
    threads_z = min(4, size(output, 3))
    threads = (threads_x, threads_y, threads_z)
    
    blocks_x = cld(size(output, 1), threads_x)
    blocks_y = cld(size(output, 2), threads_y)
    blocks_z = cld(size(output, 3), threads_z)
    blocks = (blocks_x, blocks_y, blocks_z)
    
    CUDA.@cuda threads=threads blocks=blocks attention_weights_kernel!(
        output, query, key, scale)
end

function attention_combine_kernel!(output::CuDeviceArray{Float32, 3},
                                 weights::CuDeviceArray{Float32, 3},
                                 values::CuDeviceArray{Float32, 3})
    
    d = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    seq = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    head = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    if d <= size(output, 1) && seq <= size(output, 2) && head <= size(output, 3)
        sum_val = 0.0f0
        for s in 1:size(weights, 2)
            sum_val += weights[seq, s, head] * values[d, s, head]
        end
        output[d, seq, head] = sum_val
    end
    
    return nothing
end

function launch_attention_combine!(output::CuArray{Float32, 3},
                                  weights::CuArray{Float32, 3},
                                  values::CuArray{Float32, 3})
    
    threads_x = min(16, size(output, 1))
    threads_y = min(16, size(output, 2))
    threads_z = min(4, size(output, 3))
    threads = (threads_x, threads_y, threads_z)
    
    blocks_x = cld(size(output, 1), threads_x)
    blocks_y = cld(size(output, 2), threads_y)
    blocks_z = cld(size(output, 3), threads_z)
    blocks = (blocks_x, blocks_y, blocks_z)
    
    CUDA.@cuda threads=threads blocks=blocks attention_combine_kernel!(
        output, weights, values)
end

function rope_kernel!(x::CuDeviceArray{ComplexF32, 3},
                     pos::Int32,
                     freq_base::Float32,
                     theta_scale::Float32)
    
    d = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    head = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    batch = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    if d <= size(x, 1) && head <= size(x, 2) && batch <= size(x, 3)
        theta = Float32(pos - 1) * (theta_scale ^ (d - 1))
        rotation = ComplexF32(cos(theta), sin(theta))
        x[d, head, batch] *= rotation
    end
    
    return nothing
end

function launch_rope!(x::CuArray{ComplexF32, 3},
                     pos::Int,
                     freq_base::Float32,
                     head_size_div2::Int)
    
    theta_scale = freq_base ^ (-1.0f0 / Float32(head_size_div2))
    
    threads_x = min(32, size(x, 1))
    threads_y = min(8, size(x, 2))
    threads_z = min(4, size(x, 3))
    threads = (threads_x, threads_y, threads_z)
    
    blocks_x = cld(size(x, 1), threads_x)
    blocks_y = cld(size(x, 2), threads_y)
    blocks_z = cld(size(x, 3), threads_z)
    blocks = (blocks_x, blocks_y, blocks_z)
    
    CUDA.@cuda threads=threads blocks=blocks rope_kernel!(
        x, Int32(pos), freq_base, theta_scale)
end