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

export cur_decompose, compute_leverage_scores, compute_row_leverage_scores, sample_by_scores, initialize_cur_directly, suggest_rank, convert_dense_to_cur, CURExpert, cur_matmul, GatedCURExpert, convert_expert_to_cur, compression_ratio, test_cur_expert_approximation, test_cur_in_moe_context, generate_realistic_test_inputs, select_cur_rank, validate_cur_decomposition, validate_cur_expert

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

export CudaMoEConfig, CudaMoELayer, CudaRouter, CudaStandardExpert, CudaGatedExpert
export create_cuda_moe, cuda_moe_forward!
export to_cuda, to_cpu, get_cuda_expert_stats
export generate_realistic_input, quick_test, run_comprehensive_benchmark
export REALISTIC_CONFIGS, BATCH_SCENARIOS

const CUDA_AVAILABLE = try
    import CUDA
    CUDA.functional()
catch
    false
end

if CUDA_AVAILABLE
    import CUDA
    import NNlib
    
    export cuda_router_forward!, cuda_standard_expert_forward!, cuda_gated_expert_forward!
    export cuda_process_experts!, cuda_compute_balance_loss
    
    include("cuda/kernels.jl")
    include("cuda/types.jl")
    include("cuda/router.jl")
    include("cuda/experts.jl")
    include("cuda/layer.jl")
    include("cuda/utils.jl")
   # include("cuda/configs.jl")
    #include("cuda/benchmarks.jl")
else
    @warn "CUDA not available. GPU acceleration features will not be available. Install CUDA.jl and ensure GPU drivers are properly installed to enable GPU acceleration."
end

export AMDGPUMoEConfig, AMDGPUMoELayer, AMDGPURouter, AMDGPUStandardExpert, AMDGPUGatedExpert
export create_amdgpu_moe, amdgpu_moe_forward!
export to_amdgpu, get_amdgpu_expert_stats
export generate_realistic_input_amdgpu, quick_test_amdgpu, run_comprehensive_benchmark_amdgpu
export REALISTIC_AMDGPU_CONFIGS, BATCH_SCENARIOS_AMDGPU

const AMDGPU_AVAILABLE = try
    import AMDGPU
    AMDGPU.functional()
catch
    false
end

if AMDGPU_AVAILABLE
    import AMDGPU
    import NNlib
    
    export amdgpu_router_forward!, amdgpu_standard_expert_forward!, amdgpu_gated_expert_forward!
    export amdgpu_process_experts!, amdgpu_compute_balance_loss
    
    include("amdgpu/kernels.jl")
    include("amdgpu/types.jl")
    include("amdgpu/router.jl")
    include("amdgpu/experts.jl")
    include("amdgpu/layer.jl")
    include("amdgpu/utils.jl")
  #  include("amdgpu/configs.jl")
   # include("amdgpu/benchmarks.jl")
else
    @warn "AMDGPU not available. AMD GPU acceleration features will not be available. Install AMDGPU.jl and ensure AMD GPU drivers are properly installed to enable AMD GPU acceleration."
end
end 