include("src/MixtureOfExperts.jl")
using .MixtureOfExperts
import .MixtureOfExperts: convert_expert_to_cur, test_cur_expert_approximation, generate_realistic_test_inputs
using LinearAlgebra
using Statistics

# Test configuration
input_dim = 768
hidden_dim = 3072
output_dim = 768
batch_size = 128

# Generate realistic test inputs
test_inputs = generate_realistic_test_inputs(input_dim, batch_size; scale=1.0f0)

# Create original expert
original_expert = GatedExpert(input_dim, hidden_dim, output_dim, silu)

# Test different compression ratios
compression_ratios = [0.5f0, 0.4f0, 0.3f0, 0.25f0, 0.2f0]

println("Testing CUR compression on single expert:")
println("="^60)

for ratio in compression_ratios
    cur_expert = convert_expert_to_cur(original_expert; rank_ratio=ratio)
    result = test_cur_expert_approximation(original_expert, cur_expert, test_inputs)
    
    # Calculate actual compression
    orig_params = sum(length, [original_expert.w1.weight, original_expert.w2.weight, original_expert.w3.weight])
    cur_params = sum(length, [cur_expert.cur_w1.C, cur_expert.cur_w1.U, cur_expert.cur_w1.R,
                             cur_expert.cur_w2.C, cur_expert.cur_w2.U, cur_expert.cur_w2.R,
                             cur_expert.cur_w3.C, cur_expert.cur_w3.U, cur_expert.cur_w3.R])
    actual_compression = orig_params / cur_params
    
    println("\nRank ratio: $ratio ($(actual_compression)x compression)")
    println("  Relative error: $(round(result.relative_error, digits=4))")
    println("  Cosine similarity: $(round(result.cosine_similarity, digits=4))")
    println("  Max absolute error: $(round(result.max_absolute_error, digits=4))")
    println("  PASSED: $(result.passed)")
end