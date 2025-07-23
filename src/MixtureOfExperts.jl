module MixtureOfExperts

using Flux
using NNlib
using LinearAlgebra
using CUDA
using Random
using Statistics
using StatsBase
using ChainRulesCore
using Printf
using Statistics

# Core MoE exports 
export GatingMechanism, LoadBalancingLoss, Expert
export MoEConfig, MoELayer, Router

# Gating mechanisms 
export RandomGating                     
export TopKGating, SwitchGating        
export StochasticTopKGating
export ExpertChoiceGating              
export SoftMoEGating                    
export HashGating                       

# Load balancing 
export NoBalancingLoss                  
export SwitchTransformerLoss           
export DeepSeekLoss                     
export AuxiliaryFreeLoss               
export ZLoss                            

# Experts 
export StandardExpert, CURExpert, GatedExpert

# Core functions 
export create_moe_config, create_moe_layer
export compute_gates, compute_loss
export load_balance_score
export reset_stats!

const LLAMA2_AVAILABLE = try
    import Llama2
    true
catch
    false
end

if LLAMA2_AVAILABLE
    import Llama2
    
    export MoELlamaModel, MoELlamaConfig, MoETransformerWeights, MoETransformerLayerWeights
    export MoERunState, MoEKVCache
    export convert_to_moe, sample_moe, sample_moe_batch
    export save_moe_model, load_moe_model
    export get_expert_stats, get_routing_stats, compare_models
    export create_moe_llama_config, validate_moe_model
    export expert_usage_analysis, routing_entropy_analysis
    export create_moe_expert_weights, create_moe_run_state
end

include("gating/base.jl")
include("gating/simple.jl")
include("gating/topk.jl")
include("gating/switch.jl")
include("gating/expert_choice.jl")
include("gating/advanced.jl")

include("balancing/losses.jl")
include("balancing/auxiliary_free.jl")

include("experts/standard.jl")
include("experts/cur.jl")
include("experts/gated.jl")

include("core/utils.jl")
include("core/router.jl")
include("core/moe_layer.jl")

if LLAMA2_AVAILABLE
    include("llama2/types.jl")
    include("llama2/attention.jl")
    include("llama2/inference.jl")
    include("llama2/conversion.jl")
    include("llama2/generation.jl")
    include("llama2/utils.jl")
else
    @warn "Llama2 package not found. Llama2 integration features will not be available. Install Llama2.jl to enable full functionality."
end

end 