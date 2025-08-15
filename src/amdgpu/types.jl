using AMDGPU

struct AMDGPUMoEConfig
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

function AMDGPUMoEConfig(;
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
    return AMDGPUMoEConfig(num_experts, expert_type, input_dim, hidden_dim, 
                          output_dim, top_k, noise_scale, use_noise_network, 
                          balance_weight)
end

mutable struct AMDGPURouter{W, N}
    weight::W
    noise_weight::N
    noise_scale::Float32
    use_fp32::Bool
    k::Int
end

function AMDGPURouter(input_dim::Int, num_experts::Int, k::Int;
                     noise_scale::Float32 = 0.0f0,
                     use_noise_network::Bool = false,
                     use_fp32::Bool = true)
    
    weight = AMDGPU.randn(Float32, num_experts, input_dim) .* sqrt(2.0f0 / input_dim)
    noise_weight = use_noise_network ? 
        AMDGPU.randn(Float32, num_experts, input_dim) .* sqrt(2.0f0 / input_dim) : 
        nothing
    
    return AMDGPURouter(weight, noise_weight, noise_scale, use_fp32, k)
end

struct AMDGPUStandardExpert{W1, W2, B1, B2}
    w1::W1
    w2::W2
    b1::B1
    b2::B2
    input_dim::Int
    hidden_dim::Int
    output_dim::Int
end

function AMDGPUStandardExpert(input_dim::Int, hidden_dim::Int, output_dim::Int;
                             bias::Bool = true)
    σ1 = sqrt(2.0f0 / input_dim)
    σ2 = sqrt(2.0f0 / hidden_dim)
    
    w1 = AMDGPU.randn(Float32, hidden_dim, input_dim) .* σ1
    w2 = AMDGPU.randn(Float32, output_dim, hidden_dim) .* σ2
    
    b1 = bias ? AMDGPU.zeros(Float32, hidden_dim, 1) : nothing
    b2 = bias ? AMDGPU.zeros(Float32, output_dim, 1) : nothing
    
    return AMDGPUStandardExpert(w1, w2, b1, b2, input_dim, hidden_dim, output_dim)
end

struct AMDGPUGatedExpert{W1, W2, W3}
    w1::W1
    w2::W2
    w3::W3
    input_dim::Int
    hidden_dim::Int
    output_dim::Int
end

function AMDGPUGatedExpert(input_dim::Int, hidden_dim::Int, output_dim::Int)
    σ = sqrt(2.0f0 / input_dim)
    
    w1 = AMDGPU.randn(Float32, hidden_dim, input_dim) .* σ
    w2 = AMDGPU.randn(Float32, output_dim, hidden_dim) .* σ
    w3 = AMDGPU.randn(Float32, hidden_dim, input_dim) .* σ
    
    return AMDGPUGatedExpert(w1, w2, w3, input_dim, hidden_dim, output_dim)
end

struct AMDGPUMoELayer{E, R}
    experts::E
    router::R
    config::AMDGPUMoEConfig
    router_logits_buf::ROCArray{Float32, 2}
    router_probs_buf::ROCArray{Float32, 2}
    expert_indices_buf::ROCArray{Int32, 2}
    expert_gates_buf::ROCArray{Float32, 2}
    expert_counts_buf::ROCArray{Float32, 1}
end

function AMDGPUMoELayer(config::AMDGPUMoEConfig)
    experts = create_amdgpu_experts(config)
    router = AMDGPURouter(config.input_dim, config.num_experts, config.top_k;
                         noise_scale = config.noise_scale,
                         use_noise_network = config.use_noise_network)
    
    max_batch = 1024
    router_logits_buf = AMDGPU.zeros(Float32, config.num_experts, max_batch)
    router_probs_buf = AMDGPU.zeros(Float32, config.num_experts, max_batch)
    expert_indices_buf = AMDGPU.zeros(Int32, config.top_k, max_batch)
    expert_gates_buf = AMDGPU.zeros(Float32, config.top_k, max_batch)
    expert_counts_buf = AMDGPU.zeros(Float32, config.num_experts)
    
    return AMDGPUMoELayer(experts, router, config, router_logits_buf, 
                         router_probs_buf, expert_indices_buf, expert_gates_buf,
                         expert_counts_buf)
end

function create_amdgpu_experts(config::AMDGPUMoEConfig)
    experts = []
    for i in 1:config.num_experts
        if config.expert_type == :gated
            expert = AMDGPUGatedExpert(config.input_dim, config.hidden_dim, config.output_dim)
        else
            expert = AMDGPUStandardExpert(config.input_dim, config.hidden_dim, config.output_dim)
        end
        push!(experts, expert)
    end
    return experts
end