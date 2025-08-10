
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

export GPUTokenAssignment, create_token_assignment, route_tokens_to_experts!
export combine_expert_outputs!, validate_token_assignment, get_assignment_statistics
export optimize_assignment_memory!
export GPUBatchConfig, GPUBatchWorkspace, get_batch_workspace, release_batch_workspace!
export allocate_tensor_from_pool, return_tensor_to_pool!, get_memory_pool_statistics
export clear_memory_pools!, optimize_batch_size, estimate_batch_memory_usage
export create_batch_streams, select_optimal_kernel_config

export GPUMoELayerConfig, GPUMoELayer, gpu_moe_forward!, create_gpu_moe_layer
export reset_moe_layer_stats!, optimize_moe_layer!, validate_moe_layer, benchmark_moe_layer
export get_moe_layer_stats, process_experts_parallel!

export GPUMoE, create_gpu_moe, convert_cpu_moe_to_gpu
export optimize_gpu_moe!, benchmark_gpu_moe, validate_gpu_moe
export get_gpu_moe_info, save_gpu_moe, load_gpu_moe
export transfer_weights_cpu_to_gpu!


include("types.jl")
include("utils.jl")

include("kernels/expert_kernels.jl")
include("kernels/gating_kernels.jl") 
include("kernels/loss_kernels.jl")

include("experts/gated_expert.jl")
include("gating/topk_gating.jl")
include("losses/switch_loss.jl")

include("routing.jl")
include("batching.jl")

include("moe_layer.jl")

include("integration.jl")

end 