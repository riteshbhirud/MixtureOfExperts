# MixtureOfExperts.jl

A comprehensive Julia library for Mixture of Experts (MoE) architectures with seamless integration to existing models, particularly Llama2.jl. This library implements state-of-the-art MoE techniques from recent research including Switch Transformer, DeepSeek V3, and Stanford CS336 methodologies.

## Features

- **Multiple Gating Mechanisms**: TopK, Switch, Expert Choice, Soft MoE, Hash routing
- **Expert Architectures**: Standard FFN, Gated FFN (Llama-style), CUR compressed experts  
- **Load Balancing**: Switch Transformer loss, DeepSeek variants, auxiliary-free balancing
- **Llama2 Integration**: Convert existing Llama2 models to MoE with preserved functionality
- **Advanced Features**: Shared experts, expert virtualization, routing analysis

## Quick Start

### Basic MoE Layer

```julia
using MixtureOfExperts

config = create_moe_config(
    num_experts=8,
    expert_type=:gated,
    input_dim=512,
    hidden_dim=2048,
    output_dim=512,
    top_k=2,
    gate_type=TopKGating(2),
    balance_loss=SwitchTransformerLoss(0.01f0)
)

moe_layer = MoELayer(config)

input = randn(Float32, 512, 32)
output, balance_loss = moe_layer(input; training=true)
```

### Llama2 Model Conversion

```julia
using Llama2
using MixtureOfExperts

original_model = Llama2.load_karpathy_model("stories42M.bin", "tokenizer.bin")

moe_model = convert_to_moe(
    original_model,
    [2, 4, 6];
    num_experts=8,
    top_k=2,
    expert_init_strategy=:perturb,
    expert_init_noise=0.01f0,
    gate_type=TopKGating(2),
    balance_loss=SwitchTransformerLoss(0.01f0),
    expert_type=:gated
)

println("Original parameters: $(count_llama_parameters(original_model))")
println("MoE total parameters: $(count_parameters(moe_model))")
println("MoE active parameters: $(count_active_parameters(moe_model))")
```

### Text Generation with MoE

```julia
prompt = "Once upon a time"

original_output = Llama2.sample(original_model, prompt; temperature=0.9f0)

moe_output = sample_moe(moe_model, prompt; 
                       temperature=0.9f0,
                       show_expert_stats=true,
                       show_routing_entropy=true)
```

### Expert Usage Analysis

```julia
test_prompts = [
    "The dragon flew",
    "In the forest", 
    "The magic spell",
    "Once upon a time"
]

for prompt in test_prompts
    println("Prompt: \"$prompt\"")
    
    result = sample_moe(moe_model, prompt;
                       temperature=0.3f0,
                       max_seq_len=50,
                       show_expert_stats=true,
                       expert_usage_threshold=0.05f0)
    
    println("Generated: \"$result\"\n")
end
```

## Core API Reference

### Model Conversion

#### `convert_to_moe(model, moe_layers; kwargs...)`

Convert existing Llama2 model to MoE by replacing specified layers.

**Arguments:**
- `model::Llama2.LanguageModel`: Original model to convert
- `moe_layers::Vector{Int}`: Layer indices to convert to MoE (1-based)

**Key Options:**
- `num_experts::Int=8`: Number of experts per MoE layer
- `top_k::Int=2`: Number of experts to activate per token
- `expert_init_strategy::Symbol=:perturb`: Weight initialization (`:copy`, `:perturb`, `:split`, `:random`)
- `expert_type::Symbol=:gated`: Expert architecture (`:standard`, `:gated`, `:cur`)
- `gate_type::GatingMechanism=TopKGating(top_k)`: Routing mechanism
- `balance_loss::LoadBalancingLoss=SwitchTransformerLoss(0.01f0)`: Load balancing

**Returns:** `MoELanguageModel`

### Text Generation

#### `sample_moe(model, prompt; kwargs...)`

Generate text using MoE model with expert tracking.

**Arguments:**
- `model::MoELanguageModel`: MoE model for generation
- `prompt::String=""`: Input text prompt

**Key Options:**
- `temperature::Float32=0.9f0`: Sampling temperature
- `max_seq_len::Int=typemax(Int)`: Maximum sequence length
- `show_expert_stats::Bool=false`: Display expert usage statistics
- `show_routing_entropy::Bool=false`: Show routing entropy analysis
- `expert_usage_threshold::Float32=0.01f0`: Threshold for reporting expert usage

**Returns:** `String` (generated text)

### Model Analysis

#### `count_parameters(model::MoELanguageModel)`
Count total parameters in MoE model.

#### `count_active_parameters(model::MoELanguageModel)`  
Count parameters active during inference (considering top-k routing).

#### `get_expert_stats(model, tokens)`
Analyze expert usage patterns for given token sequence.

### Configuration

#### `create_moe_config(; kwargs...)`

Create MoE layer configuration with sensible defaults.

**Key Options:**
- `num_experts::Int=8`: Number of experts
- `expert_type::Symbol=:standard`: Expert architecture
- `top_k::Int=2`: Experts to activate
- `gate_type::GatingMechanism=TopKGating(2)`: Routing mechanism
- `balance_loss::LoadBalancingLoss=SwitchTransformerLoss(0.01f0)`: Load balancing

## Gating Mechanisms

### TopKGating
Stanford CS336 implementation with softmax renormalization:
```julia
gate = TopKGating(k=2)
```

### SwitchGating  
Switch Transformer (k=1 special case):
```julia
gate = SwitchGating()
```

### ExpertChoiceGating
Experts select tokens instead of tokens selecting experts:
```julia
gate = ExpertChoiceGating(capacity_factor=1.25f0)
```

### Advanced Routing
```julia
gate = SoftMoEGating(k=2, λ=1.0f0)
gate = HashGating(k=2, num_experts=8)
gate = SharedExpertGating(num_shared=2, base_gate=TopKGating(2))
```

## Expert Types

### StandardExpert
Basic 2-layer FFN with configurable activation:
```julia
expert = StandardExpert(input_dim, hidden_dim, output_dim, gelu; dropout=0.1f0)
```

### GatedExpert  
Llama-style gated FFN: `w2(silu(w1(x)) * w3(x))`:
```julia
expert = GatedExpert(input_dim, hidden_dim, output_dim, silu)
```

### CURExpert
Compressed expert using CUR decomposition:
```julia
expert = CURExpert(input_dim, hidden_dim, output_dim, gelu; rank=64)
```

## Load Balancing

### SwitchTransformerLoss
Original Switch Transformer auxiliary loss:
```julia
loss = SwitchTransformerLoss(α=0.01f0)
```

### DeepSeekLoss
DeepSeek V1/V2 variants with device-aware balancing:
```julia
loss = DeepSeekLoss(α=0.01f0, balance_type=:device)
```

### AuxiliaryFreeLoss
DeepSeek V3 innovation with online bias learning:
```julia
loss = AuxiliaryFreeLoss(num_experts=8, learning_rate=0.01f0)
```

## File Structure

```
src/
├── MixtureOfExperts.jl          # Main module
├── gating/                      # Routing mechanisms
│   ├── base.jl                  # Abstract types and interfaces
│   ├── simple.jl                # RandomGating (testing/baseline)
│   ├── topk.jl                  # TopKGating (Stanford CS336)
│   ├── switch.jl                # SwitchGating, JitterGating
│   ├── expert_choice.jl         # ExpertChoiceGating
│   └── advanced.jl              # SoftMoE, HashGating, SharedExpert
├── experts/                     # Expert architectures
│   ├── standard.jl              # Basic 2-layer FFN
│   ├── gated.jl                 # Llama-style gated FFN
│   └── cur.jl                   # CUR decomposition experts
├── balancing/                   # Load balancing losses
│   ├── losses.jl                # Switch, DeepSeek, Z-loss
│   └── auxiliary_free.jl        # DeepSeek V3 innovation
├── core/                        # Core MoE components
│   ├── router.jl                # Neural routing network
│   ├── moe_layer.jl             # Main MoE layer implementation
│   └── utils.jl                 # Utility functions
└── llama2/                      # Llama2.jl integration
    ├── types.jl                 # MoE wrapper types
    ├── conversion.jl            # convert_to_moe functionality
    ├── inference.jl             # MoE transformer forward pass
    ├── attention.jl             # Attention with RoPE support
    ├── generation.jl            # sample_moe and text generation
    └── utils.jl                 # Save/load, analysis utilities
```

## Advanced Usage

### Custom Gating Mechanism

```julia
struct MyGating <: GatingMechanism
    k::Int
    temperature::Float32
end

function compute_gates(gate::MyGating, router_logits::AbstractMatrix)
    scaled_logits = router_logits ./ gate.temperature
    router_probs = softmax(scaled_logits; dims=1)
    
    expert_indices = zeros(Int, gate.k, size(router_logits, 2))
    expert_gates = zeros(Float32, gate.k, size(router_logits, 2))
    
    for i in 1:size(router_logits, 2)
        topk_indices = partialsortperm(router_probs[:, i], 1:gate.k, rev=true)
        expert_indices[:, i] = topk_indices
        expert_gates[:, i] = router_probs[topk_indices, i] ./ sum(router_probs[topk_indices, i])
    end
    
    return expert_indices, expert_gates, router_probs
end
```

### Batch Generation

```julia
prompts = [
    "The dragon",
    "In the castle", 
    "Magic spell",
    "Forest adventure"
]

results = sample_moe_batch(moe_model, prompts;
                          temperature=0.7f0,
                          max_seq_len=100,
                          show_progress=true)

for (prompt, result) in zip(prompts, results)
    println("\"$prompt\" → \"$result\"")
end
```

### Model Saving and Loading

```julia
save_moe_model(moe_model, "my_moe_model.jls")

loaded_model = load_moe_model("my_moe_model.jls")

metadata = model_info(loaded_model)
println("Model info: $metadata")
```

### Comparative Analysis

```julia
comparison = compare_models(original_model, moe_model, "The brave knight")

for (i, comp) in enumerate(comparison["comparisons"])
    println("Run $i:")
    println("  Original: $(comp["original_output"])")
    println("  MoE:      $(comp["moe_output"])")
    println()
end
```

## Research Implementation Notes

This library implements techniques from:

- **Switch Transformer** (Fedus et al., 2022): Core MoE architecture and load balancing
- **DeepSeek V1-V3** (2024): Shared experts, auxiliary-free balancing, advanced routing
- **Stanford CS336** (2024): Mathematical formulations and routing algorithms
- **Expert Choice Routing** (Zhou et al., 2022): Alternative routing paradigm
- **CUR Decomposition**: Memory-efficient expert compression

### GPU Acceleration (CUDA)

The library provides full GPU acceleration using CUDA for significantly improved performance on NVIDIA GPUs.

#### Quick Start

```julia
using MixtureOfExperts

gpu_moe = create_cuda_moe(
    num_experts = 8,
    expert_type = :gated,
    input_dim = 768,
    hidden_dim = 3072,
    output_dim = 768,
    top_k = 2
)

gpu_input = CUDA.randn(Float32, 768, 32)  
gpu_output, balance_loss = cuda_moe_forward!(gpu_moe, gpu_input; training=true)

cpu_moe, gpu_moe, sync_success = create_synchronized_moe_pair(
    CudaMoEConfig(
        num_experts = 8,
        expert_type = :gated,
        input_dim = 768,
        hidden_dim = 3072,
        output_dim = 768,
        top_k = 2
    )
)

gpu_tensor = to_cuda(cpu_tensor)
cpu_tensor = to_cpu(gpu_tensor)

expert_counts, usage_percentages = get_cuda_expert_stats(expert_indices, num_experts)

test_input = generate_realistic_input(input_dim=768, batch_size=32)
config = CudaMoEConfig(
    num_experts = 16,
    expert_type = :gated,          
    input_dim = 1024,
    hidden_dim = 4096,
    output_dim = 1024,
    top_k = 2,
    noise_scale = 0.0f0,           
    use_noise_network = false,    
    balance_weight = 0.01f0        
)

gpu_moe = CudaMoELayer(config)
```