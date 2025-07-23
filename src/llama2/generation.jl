"""
Text Generation with MoE Integration

This module provides text generation capabilities for MoE-Llama2 models,
including expert usage tracking and analysis.
"""

"""
    sample_moe(model::MoELanguageModel, prompt::String = "";
               temperature::Float32 = 0.9f0,
               stop_on_special_token::Bool = true,
               max_seq_len::Int = typemax(Int),
               bos_token::Bool = true,
               show_expert_stats::Bool = false,
               show_routing_entropy::Bool = false,
               expert_usage_threshold::Float32 = 0.01f0,
               verbose::Bool = false) -> String

Generate text using MoE model with expert usage tracking.

Compatible with Llama2.sample() interface while adding MoE-specific features.

# Arguments
- `model`: MoE language model
- `prompt`: Input text prompt
- `temperature`: Sampling temperature (0.0 = greedy, higher = more random)
- `stop_on_special_token`: Stop on BOS/EOS tokens
- `max_seq_len`: Maximum sequence length
- `bos_token`: Add BOS token at start
- `show_expert_stats`: Display expert usage statistics
- `show_routing_entropy`: Show routing entropy over time
- `expert_usage_threshold`: Threshold for reporting expert usage
- `verbose`: Show detailed generation info

# Returns
- Generated text string
"""
function sample_moe(model::MoELanguageModel, prompt::String = "";
                   temperature::Float32 = 0.9f0,
                   stop_on_special_token::Bool = true,
                   max_seq_len::Int = typemax(Int),
                   bos_token::Bool = true,
                   show_expert_stats::Bool = false,
                   show_routing_entropy::Bool = false,
                   expert_usage_threshold::Float32 = 0.01f0,
                   verbose::Bool = false)
    
    if !bos_token && isempty(prompt)
        throw(ArgumentError("Prompt cannot be empty if bos_token = false"))
    end
    
    config = model.config
    tokenizer = model.tokenizer
    
    prompt_tokens = Llama2.encode(prompt, tokenizer)
    
    state = create_moe_run_state(config)
    
    if bos_token
        pushfirst!(prompt_tokens, tokenizer.bos_token_id)
    end
    
    generation_stats = init_generation_stats(model)
    generated_text = IOBuffer()
    
    if !bos_token && !isempty(prompt_tokens)
        print(tokenizer.id_to_token[prompt_tokens[1]])
    end
    
    time_start = time_ns()
    token = prompt_tokens[1]
    generated_seq_len = 0
    
    for pos in 1:min(config.seq_len, max_seq_len)
        if verbose && pos <= 3
            println("\n--- Position $pos ---")
            println("Token: $token ($(tokenizer.id_to_token[token]))")
        end
        
        moe_transformer!(token, pos, model, state)
        generated_seq_len += 1
        
        track_expert_usage!(generation_stats, state, model, pos)
        
        if pos + 1 <= length(prompt_tokens)
            next_token = prompt_tokens[pos + 1]
            if verbose && pos <= 3
                println("Next token from prompt: $next_token")
            end
        else
            next_token = sample_next_token(state.logits, temperature, config.vocab_size)
            if verbose && pos <= 3
                println("Generated token: $next_token (prob: $(softmax(state.logits)[next_token]))")
            end
        end
        
        if stop_on_special_token && 
           (next_token == tokenizer.bos_token_id || next_token == tokenizer.eos_token_id)
            if verbose
                println("Stopping on special token: $next_token")
            end
            break
        end
        
        next_str = tokenizer.id_to_token[next_token]
        print(next_str)
        write(generated_text, next_str)
        
        token = next_token
    end
    
    println()  
    
    time_end = time_ns()
    generation_time = (time_end - time_start) / 1e9
    
    print_generation_stats(generation_stats, state, generation_time, generated_seq_len)
    
    if show_expert_stats
        print_expert_stats(generation_stats, model, expert_usage_threshold)
    end
    
    if show_routing_entropy
        print_routing_entropy(state)
    end
    
    return String(take!(generated_text))
end

"""
    sample_moe_batch(model::MoELanguageModel, prompts::Vector{String};
                     kwargs...) -> Vector{String}

Generate text for multiple prompts in batch.
"""
function sample_moe_batch(model::MoELanguageModel, prompts::Vector{String};
                         temperature::Float32 = 0.9f0,
                         max_seq_len::Int = 512,
                         show_progress::Bool = true,
                         kwargs...)
    
    if isempty(prompts)
        return String[]
    end
    
    results = String[]
    
    if show_progress
        println("Generating $(length(prompts)) sequences...")
    end
    
    for (i, prompt) in enumerate(prompts)
        if show_progress
            print("[$i/$(length(prompts))] ")
        end
        
        result = sample_moe(model, prompt; 
                           temperature=temperature,
                           max_seq_len=max_seq_len,
                           show_expert_stats=false,
                           verbose=false,
                           kwargs...)
        
        push!(results, result)
        
        if show_progress
            println("✓")
        end
    end
    
    return results
end

"""
    sample_next_token(logits::Vector{Float32}, temperature::Float32, vocab_size::Int) -> Int

Sample next token from logits using temperature scaling.
"""
function sample_next_token(logits::Vector{Float32}, temperature::Float32, vocab_size::Int)
    if temperature == 0.0f0
        return argmax(logits)
    else
        scaled_logits = logits ./ temperature
        
        max_logit = maximum(scaled_logits)
        scaled_logits .-= max_logit
        
        exp_logits = exp.(scaled_logits)
        probs = exp_logits ./ sum(exp_logits)
        
        return sample_from_probs(probs)
    end
end

"""
    sample_from_probs(probs::Vector{Float32}) -> Int

Sample index from probability distribution.
"""
function sample_from_probs(probs::Vector{Float32})
    r = rand(Float32)
    cumsum_prob = 0.0f0
    
    for (i, prob) in enumerate(probs)
        cumsum_prob += prob
        if r <= cumsum_prob
            return i
        end
    end
    
    # Fallback 

    return length(probs)
end

"""
    init_generation_stats(model::MoELanguageModel) -> Dict

Initialize tracking structures for generation statistics.
"""
function init_generation_stats(model::MoELanguageModel)
    config = model.config
    
    moe_layer_indices = get_moe_layer_indices(model)
    
    stats = Dict{Symbol, Any}(
        :expert_usage => Dict{Int, Vector{Int}}(),  
        :routing_decisions => Dict{Int, Vector{Vector{Int}}}(), 
        :gate_weights => Dict{Int, Vector{Vector{Float32}}}(), 
        :moe_layers => moe_layer_indices,
        :total_positions => 0,
        :start_time => time()
    )
    
    for layer_idx in moe_layer_indices
        stats[:expert_usage][layer_idx] = zeros(Int, config.moe_num_experts)
        stats[:routing_decisions][layer_idx] = Vector{Int}[]
        stats[:gate_weights][layer_idx] = Vector{Float32}[]
    end
    
    return stats
end

"""
    track_expert_usage!(stats::Dict, state::MoERunState, model::MoELanguageModel, pos::Int)

Track expert usage for current position.
"""
function track_expert_usage!(stats::Dict, state::MoERunState, model::MoELanguageModel, pos::Int)
    stats[:total_positions] += 1
    
    #SIMPLIFIED FOR NOW!!
    for layer_idx in stats[:moe_layers]
        selected = copy(state.selected_experts[1:model.config.moe_top_k])
        weights = copy(state.expert_gates[1:model.config.moe_top_k])
        
        push!(stats[:routing_decisions][layer_idx], selected)
        push!(stats[:gate_weights][layer_idx], weights)
        
        for expert_idx in selected
            if expert_idx > 0
                stats[:expert_usage][layer_idx][expert_idx] += 1
            end
        end
    end
end

"""
    print_generation_stats(stats::Dict, state::MoERunState, generation_time::Float64, seq_len::Int)

Print basic generation performance statistics.
"""
function print_generation_stats(stats::Dict, state::MoERunState, generation_time::Float64, seq_len::Int)
    println("-------")
    @printf "Generation time: %.3f seconds\n" generation_time
    @printf "Tokens generated: %d\n" seq_len
    @printf "Tokens per second: %.2f\n" (seq_len / generation_time)
    
    total_expert_activations = state.inference_stats[:expert_activations]
    moe_layer_calls = state.inference_stats[:moe_layer_calls]
    
    if moe_layer_calls > 0
        @printf "MoE layer calls: %d\n" moe_layer_calls
        @printf "Expert activations: %d\n" total_expert_activations
        @printf "Avg experts per MoE call: %.2f\n" (total_expert_activations / moe_layer_calls)
        
        routing_time = state.inference_stats[:routing_time]
        expert_time = state.inference_stats[:expert_compute_time]
        @printf "Routing time: %.3f%% of total\n" (routing_time / generation_time * 100)
        @printf "Expert compute time: %.3f%% of total\n" (expert_time / generation_time * 100)
    end
end

"""
    print_expert_stats(stats::Dict, model::MoELanguageModel, threshold::Float32)

Print detailed expert usage statistics.
"""
function print_expert_stats(stats::Dict, model::MoELanguageModel, threshold::Float32)
    println("\n--- Expert Usage Statistics ---")
    
    total_positions = stats[:total_positions]
    
    for layer_idx in stats[:moe_layers]
        usage_counts = stats[:expert_usage][layer_idx]
        total_activations = sum(usage_counts)
        
        println("Layer $layer_idx:")
        println("  Total expert activations: $total_activations")
        
        for (expert_idx, count) in enumerate(usage_counts)
            if count > 0
                percentage = count / total_activations * 100
                if percentage >= threshold * 100
                    @printf "    Expert %2d: %4d activations (%.1f%%)\n" expert_idx count percentage
                end
            end
        end
        
        if total_activations > 0
            expected_per_expert = total_activations / model.config.moe_num_experts
            max_deviation = maximum(abs.(usage_counts .- expected_per_expert))
            balance_score = 1.0 - (max_deviation / expected_per_expert)
            @printf "  Load balance score: %.3f (1.0 = perfect)\n" balance_score
        end
        
        println()
    end
end

"""
    print_routing_entropy(state::MoERunState)

Print routing entropy analysis.
"""
function print_routing_entropy(state::MoERunState)
    entropies = state.routing_entropy
    
    if isempty(entropies)
        println("No routing entropy data available")
        return
    end
    
    println("\n--- Routing Entropy Analysis ---")
    @printf "Mean entropy: %.3f\n" mean(entropies)
    @printf "Std entropy: %.3f\n" std(entropies)
    @printf "Min entropy: %.3f\n" minimum(entropies)
    @printf "Max entropy: %.3f\n" maximum(entropies)
    
    if length(entropies) > 1
        println("Recent entropy values:")
        start_idx = max(1, length(entropies) - 19)
        for (i, entropy) in enumerate(entropies[start_idx:end])
            pos = start_idx + i - 1
            @printf "  Pos %3d: %.3f\n" pos entropy
        end
    end
end

"""
Advanced generation features
"""

"""
    sample_moe_with_constraints(model::MoELanguageModel, prompt::String;
                                forbidden_tokens::Vector{Int} = Int[],
                                required_tokens::Vector{Int} = Int[],
                                max_repeated_tokens::Int = 3,
                                kwargs...) -> String

Generate text with additional constraints.
"""
function sample_moe_with_constraints(model::MoELanguageModel, prompt::String;
                                    forbidden_tokens::Vector{Int} = Int[],
                                    required_tokens::Vector{Int} = Int[],
                                    max_repeated_tokens::Int = 3,
                                    temperature::Float32 = 0.9f0,
                                    kwargs...)
    
    config = model.config
    tokenizer = model.tokenizer
    
    prompt_tokens = Llama2.encode(prompt, tokenizer)
    state = create_moe_run_state(config)
    
    generated_tokens = Int[]
    generated_text = IOBuffer()
    
    token = prompt_tokens[1]
    
    for pos in 1:min(config.seq_len, get(kwargs, :max_seq_len, 512))
        moe_transformer!(token, pos, model, state)
        
        if pos + 1 <= length(prompt_tokens)
            next_token = prompt_tokens[pos + 1]
        else
            constrained_logits = apply_constraints(
                copy(state.logits),
                forbidden_tokens,
                required_tokens,
                generated_tokens,
                max_repeated_tokens
            )
            
            next_token = sample_next_token(constrained_logits, temperature, config.vocab_size)
            push!(generated_tokens, next_token)
        end
        
        if next_token == tokenizer.eos_token_id
            break
        end
        
        next_str = tokenizer.id_to_token[next_token]
        print(next_str)
        write(generated_text, next_str)
        
        token = next_token
    end
    
    println()
    return String(take!(generated_text))
end

"""
    apply_constraints(logits::Vector{Float32}, forbidden::Vector{Int}, 
                     required::Vector{Int}, history::Vector{Int}, 
                     max_repeat::Int) -> Vector{Float32}

Apply generation constraints to logits.
"""
function apply_constraints(logits::Vector{Float32},
                          forbidden::Vector{Int},
                          required::Vector{Int},
                          history::Vector{Int},
                          max_repeat::Int)
    
    for token in forbidden
        if 1 <= token <= length(logits)
            logits[token] = -Inf32
        end
    end
    
    if length(history) >= max_repeat
        recent_tokens = history[end-max_repeat+1:end]
        if all(t -> t == recent_tokens[1], recent_tokens)
            if 1 <= recent_tokens[1] <= length(logits)
                logits[recent_tokens[1]] = -Inf32
            end
        end
    end
    
    if !isempty(required)
        boost_factor = 2.0f0
        for token in required
            if 1 <= token <= length(logits)
                logits[token] += boost_factor
            end
        end
    end
    
    return logits
end

"""
    analyze_generation_patterns(model::MoELanguageModel, prompts::Vector{String};
                                num_samples::Int = 3) -> Dict

Analyze generation patterns across multiple runs.
"""
function analyze_generation_patterns(model::MoELanguageModel, prompts::Vector{String};
                                    num_samples::Int = 3,
                                    temperature::Float32 = 0.9f0)
    
    analysis = Dict{String, Any}()
    
    for (prompt_idx, prompt) in enumerate(prompts)
        println("Analyzing prompt $prompt_idx: \"$(prompt[1:min(50, end)])...\"")
        
        prompt_analysis = Dict{String, Any}(
            "generations" => String[],
            "expert_usage" => Dict{Int, Vector{Int}}(),
            "routing_entropy" => Float32[],
            "generation_times" => Float64[]
        )
        
        for sample_idx in 1:num_samples
            print("  Sample $sample_idx...")
            
            start_time = time()
            result = sample_moe(model, prompt;
                               temperature=temperature,
                               show_expert_stats=false,
                               verbose=false)
            generation_time = time() - start_time
            
            push!(prompt_analysis["generations"], result)
            push!(prompt_analysis["generation_times"], generation_time)
            
            println(" ✓")
        end
        
        analysis["prompt_$prompt_idx"] = prompt_analysis
    end
    
    return analysis
end

"""
    compare_temperature_effects(model::MoELanguageModel, prompt::String;
                               temperatures::Vector{Float32} = [0.0f0, 0.5f0, 0.9f0, 1.2f0])

Compare generation at different temperatures.
"""
function compare_temperature_effects(model::MoELanguageModel, prompt::String;
                                    temperatures::Vector{Float32} = [0.0f0, 0.5f0, 0.9f0, 1.2f0])
    
    println("Comparing temperature effects for prompt: \"$prompt\"")
    println("="^60)
    
    for temp in temperatures
        println("\nTemperature: $temp")
        println("-"^30)
        
        result = sample_moe(model, prompt;
                           temperature=temp,
                           show_expert_stats=false,
                           max_seq_len=100)
        
        println("Generated: \"$result\"")
    end
end