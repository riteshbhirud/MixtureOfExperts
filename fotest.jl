# Add this to src/cuda/utils.jl or run it directly
using CUDA
include("src/cuda/CUDAMoE.jl")
using .CUDAMoE
import .CUDAMoE: GPURoutingState, create_gpu_moe_config, GPUTopKGatingState, create_gpu_moe_layer,process_gpu_output 
# Add this simple test at the start of your main() function
function quick_gpu_test()
    println("=== Quick GPU Test ===")
    try
        config = create_gpu_moe_config(
            input_dim=256, hidden_dim=512, output_dim=256,
            num_experts=4, top_k=2, max_batch_size=64
        )
        
        moe_layer = create_gpu_moe_layer(
            256, 512, 256;
            num_experts=4, top_k=2, max_batch_size=64
        )
        
        input = CUDA.randn(Float32, 256, 32)
        output = moe_layer(input; training=false)
        
        output_cpu = Array(output)
        println("✅ GPU MoE working! Output shape: $(size(output_cpu))")
        println("   Sample values: $(output_cpu[1:3, 1])")
        return true
    catch e
        println("❌ GPU test failed: $e")
        return false
    end
end

quick_gpu_test()
# Call this at the start of main()
if !quick_gpu_test()
    return
end