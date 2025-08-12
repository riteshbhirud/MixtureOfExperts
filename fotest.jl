include("src/cuda/CUDAMoE.jl")
using .CUDAMoE
create_gpu_moe_config(num_experts=8, input_dim=768, hidden_dim=3072, output_dim=768)