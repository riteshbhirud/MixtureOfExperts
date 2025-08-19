"""
    FluxRouter(input_dim, num_experts, gate_type; 
               noise_scale=0.0f0, use_noise_network=false, 
               use_fp32=true, init=Flux.glorot_uniform)

Flux-compatible router for expert selection using Dense layers.

# Arguments
- `input_dim::Int`: Input dimension
- `num_experts::Int`: Number of experts to route between
- `gate_type::GatingMechanism`: Gating mechanism (e.g., FluxTopKGating(2))
- `noise_scale::Float32`: Scale of noise added during training (default: 0.0f0)
- `use_noise_network::Bool`: Whether to use learnable noise (default: false)
- `use_fp32::Bool`: Whether to ensure FP32 computation (default: true)
- `init`: Weight initialization function (default: Flux.glorot_uniform)

"""
struct FluxRouter{W, N, G}
    weight::W             
    noise_weight::N      
    gate_type::G        
    noise_scale::Float32
    use_fp32::Bool      
end

@layer FluxRouter trainable=(weight, noise_weight)

function FluxRouter(input_dim::Int, num_experts::Int, gate_type::GatingMechanism;
                   noise_scale::Float32=0.0f0,
                   use_noise_network::Bool=false,
                   use_fp32::Bool=true,
                   init=Flux.glorot_uniform)
    
    weight = Dense(input_dim => num_experts; bias=false, init=init)
    
    noise_weight = use_noise_network ? 
                   Dense(input_dim => num_experts; bias=false, init=init) : 
                   nothing
    
    return FluxRouter(weight, noise_weight, gate_type, noise_scale, use_fp32)
end

function (router::FluxRouter)(x::AbstractVecOrMat; training::Bool=false)
    if x isa AbstractVector
        x = reshape(x, :, 1)
    end
    
    if router.use_fp32 && eltype(x) != Float32
        x = Float32.(x)
    end
    
    router_logits = router.weight(x)
    
    if training && router.noise_scale > 0
        if !isnothing(router.noise_weight)
            noise_logits = router.noise_weight(x)
            noise = randn(Float32, size(router_logits)) * Flux.softplus.(noise_logits)
            router_logits = router_logits + router.noise_scale * noise
        else
            noise = randn(Float32, size(router_logits)) * router.noise_scale
            router_logits = router_logits + noise

        end
    end
    
    expert_indices, expert_gates, router_probs = compute_gates(router.gate_type, router_logits)
    
    return expert_indices, expert_gates, router_probs, router_logits
end

function Base.show(io::IO, router::FluxRouter)
    input_dim = size(router.weight.weight, 2)
    num_experts = size(router.weight.weight, 1)
    print(io, "FluxRouter($input_dim => $num_experts, $(typeof(router.gate_type).name))")
end