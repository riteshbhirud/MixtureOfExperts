using AMDGPU

function topk_kernel!(expert_indices, expert_gates, router_probs, k, num_experts, batch_size)
    tid = AMDGPU.workitemIdx().x + (AMDGPU.workgroupIdx().x - 1) * AMDGPU.workgroupDim().x
    
    if tid <= batch_size
        gate_sum = 0.0f0
        
        for slot in 1:k
            max_val = -Inf32
            max_idx = 1
            
            for expert in 1:num_experts
                val = router_probs[expert, tid]
                
                already_selected = false
                for prev_slot in 1:(slot-1)
                    if expert_indices[prev_slot, tid] == expert
                        already_selected = true
                        break
                    end
                end
                
                if !already_selected && val > max_val
                    max_val = val
                    max_idx = expert
                end
            end
            
            expert_indices[slot, tid] = max_idx
            expert_gates[slot, tid] = max_val
            gate_sum += max_val
        end
        
        if gate_sum > 0.0f0
            for slot in 1:k
                expert_gates[slot, tid] /= gate_sum
            end
        end
    end
    
    return nothing
end

function count_assignments_kernel!(expert_counts, expert_indices, k, num_experts, batch_size)
    tid = AMDGPU.workitemIdx().x + (AMDGPU.workgroupIdx().x - 1) * AMDGPU.workgroupDim().x
    
    if tid <= batch_size
        for i in 1:k
            expert_idx = expert_indices[i, tid]
            if expert_idx > 0 && expert_idx <= num_experts
                AMDGPU.atomic_add!(expert_counts, expert_idx, 1.0f0)
            end
        end
    end
    
    return nothing
end

function apply_gates_kernel!(output, expert_output, expert_indices, expert_gates, 
                           expert_id, k, output_dim, batch_size)
    tid = AMDGPU.workitemIdx().x + (AMDGPU.workgroupIdx().x - 1) * AMDGPU.workgroupDim().x
    
    if tid <= batch_size
        gate_weight = 0.0f0
        for slot in 1:k
            if expert_indices[slot, tid] == expert_id
                gate_weight = expert_gates[slot, tid]
                break
            end
        end
        
        if gate_weight > 0.0f0
            for dim in 1:output_dim
                AMDGPU.atomic_add!(output, (dim-1)*batch_size + tid, gate_weight * expert_output[dim, tid])
            end
        end
    end
    
    return nothing
end