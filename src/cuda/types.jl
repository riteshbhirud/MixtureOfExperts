using CUDA

struct CudaMoEConfig
    num_experts::Int
    expert_type::Symbol
    input_dim::Int
    hidden_dim::Int
    output_dim::Int
    top_k::Int
    noise_scale::Float32
    use_noise_network::Bool
    balance_weight::Float32
end

function CudaMoEConfig(;
    num_experts::Int = 8,
    expert_type::Symbol = :gated,
    input_dim::Int = 768,
    hidden_dim::Int = 3072,
    output_dim::Int = 768,
    top_k::Int = 2,
    noise_scale::Float32 = 0.0f0,
    use_noise_network::Bool = false,
    balance_weight::Float32 = 0.01f0
)
    return CudaMoEConfig(num_experts, expert_type, input_dim, hidden_dim, 
                        output_dim, top_k, noise_scale, use_noise_network, 
                        balance_weight)
end

mutable struct CudaRouter{W, N}
    weight::W
    noise_weight::N
    noise_scale::Float32
    use_fp32::Bool
    k::Int
end

function CudaRouter(input_dim::Int, num_experts::Int, k::Int;
                   noise_scale::Float32 = 0.0f0,
                   use_noise_network::Bool = false,
                   use_fp32::Bool = true)
    
    weight = CUDA.randn(Float32, num_experts, input_dim) .* sqrt(2.0f0 / input_dim)
    noise_weight = use_noise_network ? 
        CUDA.randn(Float32, num_experts, input_dim) .* sqrt(2.0f0 / input_dim) : 
        nothing
    
    return CudaRouter(weight, noise_weight, noise_scale, use_fp32, k)
end

struct CudaStandardExpert{W1, W2, B1, B2}
    w1::W1
    w2::W2
    b1::B1
    b2::B2
    input_dim::Int
    hidden_dim::Int
    output_dim::Int
end

function CudaStandardExpert(input_dim::Int, hidden_dim::Int, output_dim::Int;
                           bias::Bool = true)
    σ1 = sqrt(2.0f0 / input_dim)
    σ2 = sqrt(2.0f0 / hidden_dim)
    
    w1 = CUDA.randn(Float32, hidden_dim, input_dim) .* σ1
    w2 = CUDA.randn(Float32, output_dim, hidden_dim) .* σ2
    
    b1 = bias ? CUDA.zeros(Float32, hidden_dim, 1) : nothing
    b2 = bias ? CUDA.zeros(Float32, output_dim, 1) : nothing
    
    return CudaStandardExpert(w1, w2, b1, b2, input_dim, hidden_dim, output_dim)
end

struct CudaGatedExpert{W1, W2, W3}
    w1::W1
    w2::W2
    w3::W3
    input_dim::Int
    hidden_dim::Int
    output_dim::Int
end

function CudaGatedExpert(input_dim::Int, hidden_dim::Int, output_dim::Int)
    σ = sqrt(2.0f0 / input_dim)
    
    w1 = CUDA.randn(Float32, hidden_dim, input_dim) .* σ
    w2 = CUDA.randn(Float32, output_dim, hidden_dim) .* σ
    w3 = CUDA.randn(Float32, hidden_dim, input_dim) .* σ
    
    return CudaGatedExpert(w1, w2, w3, input_dim, hidden_dim, output_dim)
end

struct CudaMoELayer{E, R}
    experts::E
    router::R
    config::CudaMoEConfig
    router_logits_buf::CuArray{Float32, 2}
    router_probs_buf::CuArray{Float32, 2}
    expert_indices_buf::CuArray{Int32, 2}
    expert_gates_buf::CuArray{Float32, 2}
    expert_counts_buf::CuArray{Float32, 1}
end

function CudaMoELayer(config::CudaMoEConfig)
    experts = create_cuda_experts(config)
    router = CudaRouter(config.input_dim, config.num_experts, config.top_k;
                       noise_scale = config.noise_scale,
                       use_noise_network = config.use_noise_network)
    
    max_batch = 1024
    router_logits_buf = CUDA.zeros(Float32, config.num_experts, max_batch)
    router_probs_buf = CUDA.zeros(Float32, config.num_experts, max_batch)
    expert_indices_buf = CUDA.zeros(Int32, config.top_k, max_batch)
    expert_gates_buf = CUDA.zeros(Float32, config.top_k, max_batch)
    expert_counts_buf = CUDA.zeros(Float32, config.num_experts)
    
    return CudaMoELayer(experts, router, config, router_logits_buf, 
                       router_probs_buf, expert_indices_buf, expert_gates_buf,
                       expert_counts_buf)
end

function create_cuda_experts(config::CudaMoEConfig)
    experts = []
    for i in 1:config.num_experts
        if config.expert_type == :gated
            expert = CudaGatedExpert(config.input_dim, config.hidden_dim, config.output_dim)
        else
            expert = CudaStandardExpert(config.input_dim, config.hidden_dim, config.output_dim)
        end
        push!(experts, expert)
    end
    return experts
end