using CUDA
using LinearAlgebra

struct ParallelExpertConfig
    parallel_execution::Bool
    max_concurrent_experts::Int
    use_streams::Bool
    memory_efficient::Bool
    load_balancing::Bool
    expert_grouping::Symbol
end

function ParallelExpertConfig(device::AbstractDevice, num_experts::Int)
    if device isa GPUDevice
        mp_count = CUDA.attribute(device.device_id, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        max_concurrent = min(num_experts, mp_count * 2)
        
        return ParallelExpertConfig(
            true,
            max_concurrent,
            true,
            true,
            true,
            :balanced
        )
    else
        return ParallelExpertConfig(
            true,
            min(num_experts, Threads.nthreads()),
            false,
            false,
            false,
            :sequential
        )
    end
end

struct ExpertExecutionPlan
    expert_groups::Vector{Vector{Int}}
    execution_order::Vector{Int}
    memory_allocation::Dict{Int, Symbol}
    stream_assignment::Dict{Int, Int}
    load_estimates::Vector{Float64}
end

function create_execution_plan(experts, selected_experts::AbstractMatrix, 
                             gate_weights::AbstractMatrix, config::ParallelExpertConfig)
    
    num_experts = length(experts)
    batch_size = size(selected_experts, 2)
    
    expert_loads = compute_expert_loads(selected_experts, gate_weights, num_experts)
    
    if config.expert_grouping == :balanced
        expert_groups = create_balanced_groups(expert_loads, config.max_concurrent_experts)
    elseif config.expert_grouping == :sequential
        expert_groups = [[i] for i in 1:num_experts]
    else
        expert_groups = create_memory_groups(experts, config.max_concurrent_experts)
    end
    
    execution_order = optimize_execution_order(expert_groups, expert_loads)
    memory_allocation = assign_memory_slots(expert_groups, config.memory_efficient)
    stream_assignment = assign_streams(expert_groups, config.use_streams)
    
    return ExpertExecutionPlan(
        expert_groups, execution_order, memory_allocation, 
        stream_assignment, expert_loads
    )
end

function compute_expert_loads(selected_experts::AbstractMatrix, 
                            gate_weights::AbstractMatrix, num_experts::Int)
    loads = zeros(Float64, num_experts)
    
    for col in 1:size(selected_experts, 2)
        for row in 1:size(selected_experts, 1)
            expert_idx = selected_experts[row, col]
            if expert_idx > 0 && expert_idx <= num_experts
                loads[expert_idx] += gate_weights[row, col]
            end
        end
    end
    
    return loads
end

function create_balanced_groups(expert_loads::Vector{Float64}, max_groups::Int)
    num_experts = length(expert_loads)
    expert_indices = sortperm(expert_loads, rev=true)
    
    groups = [Int[] for _ in 1:max_groups]
    group_loads = zeros(Float64, max_groups)
    
    for expert_idx in expert_indices
        min_group = argmin(group_loads)
        push!(groups[min_group], expert_idx)
        group_loads[min_group] += expert_loads[expert_idx]
    end
    
    return filter(!isempty, groups)
end

function create_memory_groups(experts, max_groups::Int)
    num_experts = length(experts)
    group_size = cld(num_experts, max_groups)
    
    groups = Vector{Int}[]
    for i in 1:max_groups
        start_idx = (i - 1) * group_size + 1
        end_idx = min(i * group_size, num_experts)
        if start_idx <= num_experts
            push!(groups, collect(start_idx:end_idx))
        end
    end
    
    return groups
end

function optimize_execution_order(expert_groups::Vector{Vector{Int}}, 
                                expert_loads::Vector{Float64})
    group_loads = [sum(expert_loads[group]) for group in expert_groups]
    return sortperm(group_loads, rev=true)
end

function assign_memory_slots(expert_groups::Vector{Vector{Int}}, memory_efficient::Bool)
    assignment = Dict{Int, Symbol}()
    
    if memory_efficient
        for (group_idx, group) in enumerate(expert_groups)
            slot = Symbol("slot_$(group_idx % 4 + 1)")
            for expert_idx in group
                assignment[expert_idx] = slot
            end
        end
    else
        for (group_idx, group) in enumerate(expert_groups)
            for expert_idx in group
                assignment[expert_idx] = Symbol("slot_$expert_idx")
            end
        end
    end
    
    return assignment
end

function assign_streams(expert_groups::Vector{Vector{Int}}, use_streams::Bool)
    assignment = Dict{Int, Int}()
    
    if use_streams
        num_streams = min(8, length(expert_groups))
        for (group_idx, group) in enumerate(expert_groups)
            stream_id = (group_idx - 1) % num_streams + 1
            for expert_idx in group
                assignment[expert_idx] = stream_id
            end
        end
    else
        for (group_idx, group) in enumerate(expert_groups)
            for expert_idx in group
                assignment[expert_idx] = 1
            end
        end
    end
    
    return assignment
end

function parallel_expert_forward!(output::AbstractArray, experts, input::AbstractArray,
                                selected_experts::AbstractMatrix, gate_weights::AbstractMatrix,
                                config::ParallelExpertConfig, device::AbstractDevice)
    
    plan = create_execution_plan(experts, selected_experts, gate_weights, config)
    
    if device isa GPUDevice
        parallel_expert_forward_gpu!(output, experts, input, selected_experts, 
                                    gate_weights, plan, device)
    else
        parallel_expert_forward_cpu!(output, experts, input, selected_experts, 
                                    gate_weights, plan)
    end
end

function parallel_expert_forward_gpu!(output::CuArray, experts, input::CuArray,
                                     selected_experts::AbstractMatrix, gate_weights::AbstractMatrix,
                                     plan::ExpertExecutionPlan, device::GPUDevice)
    
    batch_size = size(input, 2)
    dim = size(input, 1)
    
    expert_outputs = Dict{Int, CuArray}()
    streams = [CUDA.CuStream() for _ in 1:maximum(values(plan.stream_assignment))]
    
    fill!(output, 0.0f0)
    
    try
        for group_order in plan.execution_order
            expert_group = plan.expert_groups[group_order]
            
            group_tasks = []
            for expert_idx in expert_group
                stream_id = plan.stream_assignment[expert_idx]
                task = @async begin
                    CUDA.stream!(streams[stream_id]) do
                        compute_expert_gpu!(expert_idx, experts[expert_idx], input, 
                                          expert_outputs, dim, batch_size)
                    end
                end
                push!(group_tasks, task)
            end
            
            for task in group_tasks
                wait(task)
            end
        end
        
        CUDA.synchronize()
        
        combine_expert_outputs_gpu!(output, expert_outputs, selected_experts, 
                                   gate_weights, batch_size)
        
    finally
        for stream in streams
            CUDA.unsafe_destroy!(stream)
        end
    end
end

function compute_expert_gpu!(expert_idx::Int, expert, input::CuArray,
                           expert_outputs::Dict{Int, CuArray}, dim::Int, batch_size::Int)
    
    expert_output = get_buffer(Float32, (dim, batch_size), get_device())
    temp1 = get_buffer(Float32, (size(expert.w1, 1), batch_size), get_device())
    temp2 = get_buffer(Float32, (size(expert.w1, 1), batch_size), get_device())
    
    try
        if expert.expert_type == :gated
            launch_expert_forward!(expert_output, expert.w1, expert.w2, expert.w3,
                                 input, temp1, temp2)
        else
            CUDA.@sync begin
                mul!(temp1, expert.w1, input)
                mul!(temp2, expert.w3, input)
                
                temp1 .= temp1 .* sigmoid.(temp1) .* temp2
                mul!(expert_output, expert.w2, temp1)
            end
        end
        
        expert_outputs[expert_idx] = expert_output
        
    catch e
        return_buffer(expert_output, get_device())
        return_buffer(temp1, get_device())
        return_buffer(temp2, get_device())
        rethrow(e)
    finally
        return_buffer(temp1, get_device())
        return_buffer(temp2, get_device())
    end
end

function combine_expert_outputs_gpu!(output::CuArray, expert_outputs::Dict{Int, CuArray},
                                   selected_experts::AbstractMatrix, gate_weights::AbstractMatrix,
                                   batch_size::Int)
    
    for col in 1:batch_size
        for row in 1:size(selected_experts, 1)
            expert_idx = selected_experts[row, col]
            if expert_idx > 0 && haskey(expert_outputs, expert_idx)
                weight = gate_weights[row, col]
                expert_out = expert_outputs[expert_idx]
                
                CUDA.@cuda threads=256 blocks=cld(size(output, 1), 256) accumulate_kernel!(
                    output, expert_out, weight, col
                )
            end
        end
    end
    
    for expert_out in values(expert_outputs)
        return_buffer(expert_out, get_device())
    end
end

function accumulate_kernel!(output::CuDeviceArray{Float32, 2}, 
                          expert_output::CuDeviceArray{Float32, 2},
                          weight::Float32, col::Int)
    
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if i <= size(output, 1)
        output[i, col] += weight * expert_output[i, col]
    end
    
    return nothing
end

function parallel_expert_forward_cpu!(output::AbstractArray, experts, input::AbstractArray,
                                     selected_experts::AbstractMatrix, gate_weights::AbstractMatrix,
                                     plan::ExpertExecutionPlan)
    
    batch_size = size(input, 2)
    dim = size(input, 1)
    
    expert_outputs = Dict{Int, Array{Float32, 2}}()
    
    fill!(output, 0.0f0)
    
    for group_order in plan.execution_order
        expert_group = plan.expert_groups[group_order]
        
        if length(expert_group) == 1
            expert_idx = expert_group[1]
            compute_expert_cpu!(expert_idx, experts[expert_idx], input, expert_outputs)
        else
            Threads.@threads for expert_idx in expert_group
                compute_expert_cpu!(expert_idx, experts[expert_idx], input, expert_outputs)
            end
        end
    end
    
    combine_expert_outputs_cpu!(output, expert_outputs, selected_experts, gate_weights)
end

function compute_expert_cpu!(expert_idx::Int, expert, input::AbstractArray,
                           expert_outputs::Dict{Int, Array{Float32, 2}})
    
    batch_size = size(input, 2)
    dim = size(input, 1)
    hidden_dim = size(expert.w1, 1)
    
    expert_output = zeros(Float32, dim, batch_size)
    temp1 = zeros(Float32, hidden_dim, batch_size)
    temp2 = zeros(Float32, hidden_dim, batch_size)
    
    if expert.expert_type == :gated
        mul!(temp1, expert.w1, input)
        mul!(temp2, expert.w3, input)
        
        @inbounds for i in eachindex(temp1)
            gate_val = temp1[i]
            silu_val = gate_val * (1.0f0 / (1.0f0 + exp(-gate_val)))
            temp1[i] = silu_val * temp2[i]
        end
        
        mul!(expert_output, expert.w2, temp1)
    else
        mul!(temp1, expert.w1, input)
        @inbounds for i in eachindex(temp1)
            temp1[i] = max(0.0f0, temp1[i])
        end
        mul!(expert_output, expert.w2, temp1)
    end
    
    expert_outputs[expert_idx] = expert_output
end

function combine_expert_outputs_cpu!(output::AbstractArray, expert_outputs::Dict{Int, Array{Float32, 2}},
                                   selected_experts::AbstractMatrix, gate_weights::AbstractMatrix)
    
    batch_size = size(output, 2)
    
    for col in 1:batch_size
        for row in 1:size(selected_experts, 1)
            expert_idx = selected_experts[row, col]
            if expert_idx > 0 && haskey(expert_outputs, expert_idx)
                weight = gate_weights[row, col]
                expert_out = expert_outputs[expert_idx]
                
                @inbounds for dim in 1:size(output, 1)
                    output[dim, col] += weight * expert_out[dim, col]
                end
            end
        end
    end
end

struct ExpertLoadBalancer
    expert_usage_history::Vector{Vector{Float64}}
    load_threshold::Float64
    rebalance_frequency::Int
    iteration_count::Int
end

function ExpertLoadBalancer(num_experts::Int, load_threshold::Float64 = 0.8)
    return ExpertLoadBalancer(
        [Float64[] for _ in 1:num_experts],
        load_threshold,
        100,
        0
    )
end

function update_load_balancer!(balancer::ExpertLoadBalancer, expert_loads::Vector{Float64})
    balancer.iteration_count += 1
    
    for (i, load) in enumerate(expert_loads)
        push!(balancer.expert_usage_history[i], load)
        
        if length(balancer.expert_usage_history[i]) > 1000
            balancer.expert_usage_history[i] = balancer.expert_usage_history[i][end-500:end]
        end
    end
end

function should_rebalance(balancer::ExpertLoadBalancer)
    if balancer.iteration_count % balancer.rebalance_frequency != 0
        return false
    end
    
    if any(isempty, balancer.expert_usage_history)
        return false
    end
    
    recent_loads = [mean(history[max(1, end-50):end]) for history in balancer.expert_usage_history]
    load_variance = var(recent_loads)
    
    return load_variance > balancer.load_threshold
end

function suggest_expert_redistribution(balancer::ExpertLoadBalancer)
    recent_loads = [mean(history[max(1, end-50):end]) for history in balancer.expert_usage_history]
    
    overloaded_experts = findall(x -> x > balancer.load_threshold, recent_loads)
    underloaded_experts = findall(x -> x < 0.2, recent_loads)
    
    suggestions = Dict{Symbol, Vector{Int}}(
        :split => overloaded_experts,
        :merge => underloaded_experts,
        :reweight => Int[]
    )
    
    return suggestions
end