
module CUDAMoE

using CUDA
using CUDA.CUSPARSE
using CUDA.CUBLAS
using CUDA.CURAND
using LinearAlgebra
using Statistics
using Random
using Printf

if !CUDA.functional()
    @error "CUDA is not functional. GPU acceleration will not be available."
end

export GPUMoEConfig, GPUGatedExpertWeights, GPUTopKGatingState, GPUSwitchLossState
export GPUDeviceInfo, GPUMemoryInfo, GPUKernelConfig

export gpu_softmax!, gpu_softmax, gpu_silu!, gpu_silu
export gpu_elementwise_multiply!, gpu_matrix_vector_multiply!
export gpu_reduce_sum, gpu_reduce_mean, gpu_argmax, gpu_topk!
export gpu_memory_info, gpu_device_info, gpu_synchronize
export allocate_gpu_workspace, free_gpu_workspace
export gpu_zeros, gpu_ones, gpu_randn, gpu_copy!

export GPUGatedExpert, gpu_gated_expert_forward!, gpu_gated_expert_backward!

export GPUTopKGating, gpu_topk_gating_forward!, gpu_compute_router_logits!
export gpu_topk_selection!, gpu_renormalize_gates!

export GPUSwitchTransformerLoss, gpu_switch_loss_forward!
export gpu_compute_expert_fractions!, gpu_compute_load_balance_loss!
export gpu_random_categorical_sample!
export create_gpu_switch_loss, copy_loss_to_gpu
export get_loss_performance_stats, reset_loss_performance_stats!
export benchmark_loss_computation, AdaptiveLossScaling
export monitor_expert_balance, analyze_loss_history

export GPUMoELayer, GPUMoEConfig
export create_gpu_moe_config, create_gpu_moe_layer
export gpu_moe_forward, gpu_moe_forward_batch

# Export routing utilities
export GPURoutingState, GPURoutingInfo
export organize_token_routing!, analyze_routing_efficiency
export validate_routing_info, get_routing_performance_stats
export reset_routing_performance_stats!

# Export integration and conversion utilities
export convert_cpu_moe_to_gpu, convert_gpu_moe_to_cpu
export convert_cpu_expert_to_gpu, convert_gpu_expert_to_cpu
export convert_cpu_router_to_gpu
export prepare_gpu_input, process_gpu_output

# Export configuration optimization
export optimize_gpu_config_for_hardware
export estimate_memory_requirements, validate_gpu_moe_config

# Export device management
export set_gpu_device_for_moe, get_optimal_gpu_device

# Export performance and diagnostics
export get_moe_performance_report, reset_moe_performance_stats!
export diagnose_gpu_moe_setup, test_gpu_moe_performance
export get_current_moe_statistics, update_moe_statistics!

# Export workspace management
export allocate_moe_workspace!, free_moe_workspace!

# Export component creation utilities
export create_gpu_experts, create_gpu_gating, create_gpu_switch_loss
export create_gpu_moe_layer_from_components


include("types.jl")
include("utils.jl")

include("kernels/expert_kernels.jl")
include("kernels/gating_kernels.jl") 
include("kernels/loss_kernels.jl")

include("experts/gated_expert.jl")
include("gating/topk_gating.jl")
include("losses/switch_loss.jl")
include("moe_layer.jl")
include("routing.jl") 
include("integration.jl")



end 