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

export GatingMechanism, LoadBalancingLoss, Expert
export MoEConfig, MoELayer, Router

export RandomGating                     
export TopKGating, SwitchGating        
export StochasticTopKGating
export ExpertChoiceGating              
export SoftMoEGating                    
export HashGating                       

export NoBalancingLoss                  
export SwitchTransformerLoss           
export DeepSeekLoss                     
export AuxiliaryFreeLoss               
export ZLoss                            

# Experts
export StandardExpert, CURExpert, GatedExpert

# Main functions
export create_moe_config, create_moe_layer
export compute_gates, compute_loss
export load_balance_score

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



end
