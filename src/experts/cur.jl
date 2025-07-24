using Flux
"""
    CUR decomposition for expert weight compression
"""
struct CURDecomposition{C, U, R}
    C::C  
    U::U 
    R::R  
    rank::Int
end

Flux.@functor CURDecomposition (C, U, R)

"""
    cur_decompose(A::AbstractMatrix; rank::Int, oversample::Int=10, regularization::Float32=1f-6)

Perform CUR decomposition on matrix A with numerical stability fixes.
"""
function cur_decompose(A::AbstractMatrix; rank::Int, oversample::Int = 10, regularization::Float32 = 1f-6)
    m, n = size(A)
    
    # Fix 2: Clamp rank to valid range
    max_rank = min(m, n) - 1  # -1 for numerical stability
    if rank > max_rank
        @warn "Requested rank $rank exceeds maximum $max_rank, clamping"
        rank = max_rank
    end
    
    # Compute leverage scores
    col_scores = compute_leverage_scores(A, rank)         
    row_scores = compute_row_leverage_scores(A, rank)      
    
    # Ensure we don't exceed matrix dimensions
    num_cols = min(rank + oversample, n)
    num_rows = min(rank + oversample, m)
    
    # Sample columns and rows
    col_indices = sample_by_scores(col_scores, num_cols)
    row_indices = sample_by_scores(row_scores, num_rows)
    
    # Extract submatrices
    C = A[:, col_indices]
    R = A[row_indices, :]
    W = A[row_indices, col_indices]
    
    # Fix 3: Bias correction for unbiased approximation
    n_samples_col = length(col_indices)
    n_samples_row = length(row_indices)
    
    scale_C = sqrt(n / n_samples_col)
    scale_R = sqrt(m / n_samples_row)
    
    C = C * scale_C
    R = scale_R * R
    
    # Fix 1: Numerically stable computation of U using SVD
    U_w, S_w, V_w = svd(W)
    
    # Apply regularization to avoid division by very small values
    S_inv = 1.0f0 ./ (S_w .+ regularization)
    
    # Truncate to rank for additional stability
    k = min(rank, length(S_w))
    S_inv_truncated = zeros(Float32, size(S_w))
    S_inv_truncated[1:k] = S_inv[1:k]
    
    U = V_w * Diagonal(S_inv_truncated) * U_w'
    
    # Apply inverse scaling to U
    U = U / (scale_C * scale_R)
    
    return CURDecomposition(C, U, R, rank)
end

function compute_leverage_scores(A::AbstractMatrix, rank::Int)
    U, s, V = svd(A)
    k = min(rank, size(V, 2))
    
    scores = zeros(Float32, size(A, 2))
    for j in 1:size(A, 2)
        scores[j] = sum(abs2, V[j, 1:k])  
    end
    
    # Normalize
    scores = scores ./ sum(scores)
    
    # Add small epsilon to ensure all columns have non-zero probability
    scores = scores .+ 1f-8
    scores = scores ./ sum(scores)
    
    return scores
end

function compute_row_leverage_scores(A::AbstractMatrix, rank::Int)
    U, s, V = svd(A)
    k = min(rank, size(U, 2))
    
    scores = zeros(Float32, size(A, 1))
    for j in 1:size(A, 1)
        scores[j] = sum(abs2, U[j, 1:k])  
    end
    
    # Normalize
    scores = scores ./ sum(scores)
    
    # Add small epsilon
    scores = scores .+ 1f-8
    scores = scores ./ sum(scores)
    
    return scores
end

function sample_by_scores(scores::AbstractVector, num_samples::Int)
    n = length(scores)
    num_samples = min(num_samples, n)
    
    indices = StatsBase.sample(1:n, Weights(scores), num_samples, replace=false)
    return sort(indices)
end

"""
    initialize_cur_directly(out_dim::Int, in_dim::Int, rank::Int, σ::Float32)

Fix 4: Initialize CUR factors directly without wasteful decomposition of random weights.
"""
function initialize_cur_directly(out_dim::Int, in_dim::Int, rank::Int, σ::Float32)
    # Clamp rank
    max_rank = min(out_dim, in_dim) - 1
    if rank > max_rank
        @warn "Requested rank $rank exceeds maximum $max_rank, clamping"
        rank = max_rank
    end
    
    # Initialize factors with proper scaling
    C = randn(Float32, out_dim, rank) .* σ
    R = randn(Float32, rank, in_dim) .* σ
    
    # Apply QR for orthogonal initialization (better conditioning)
    if out_dim >= rank
        Q_c, _ = qr(C)
        C = Matrix(Q_c)[:, 1:rank] .* σ
    end
    
    if in_dim >= rank
        Q_r, _ = qr(R')
        R = Matrix(Q_r)'[1:rank, :] .* σ
    end
    
    # Initialize U as identity (will be learned during training)
    U = Matrix{Float32}(I, rank, rank)
    
    return CURDecomposition(C, U, R, rank)
end

"""
    suggest_rank(A::AbstractMatrix; energy_threshold::Float32 = 0.95f0)

Fix 6: Helper to choose rank based on preserved energy (singular values).
"""
function suggest_rank(A::AbstractMatrix; energy_threshold::Float32 = 0.95f0)
    _, S, _ = svd(A)
    total_energy = sum(S.^2)
    cumsum_energy = cumsum(S.^2)
    
    rank = findfirst(x -> x >= energy_threshold * total_energy, cumsum_energy)
    
    # Ensure valid rank
    max_rank = min(size(A)...) - 1
    rank = min(rank, max_rank)
    
    return rank
end

"""
    convert_dense_to_cur(dense_weight::AbstractMatrix; rank_ratio::Float32 = 0.25f0)

Fix 5: Convert pretrained dense weights to CUR format.
"""
function convert_dense_to_cur(dense_weight::AbstractMatrix; rank_ratio::Float32 = 0.25f0, 
                             energy_threshold::Union{Float32, Nothing} = nothing)
    if !isnothing(energy_threshold)
        rank = suggest_rank(dense_weight, energy_threshold=energy_threshold)
    else
        rank = Int(round(minimum(size(dense_weight)) * rank_ratio))
    end
    
    return cur_decompose(dense_weight, rank=rank)
end

"""
    CURExpert - Expert using CUR decomposition
"""
struct CURExpert{C1, C2, A} <: Expert
    cur_w1::C1
    cur_w2::C2
    activation::A
end

Flux.@functor CURExpert (cur_w1, cur_w2)

function CURExpert(input_dim::Int, hidden_dim::Int, output_dim::Int, 
                   activation; rank::Int)
    σ1 = sqrt(2.0f0 / input_dim)
    σ2 = sqrt(2.0f0 / hidden_dim)
    
    # Fix 4: Use direct initialization instead of decomposing random weights
    cur_w1 = initialize_cur_directly(hidden_dim, input_dim, rank, σ1)
    cur_w2 = initialize_cur_directly(output_dim, hidden_dim, rank, σ2)
    
    return CURExpert(cur_w1, cur_w2, activation)
end

"""
    CURExpert from pretrained dense expert
"""
function CURExpert(dense_expert::StandardExpert; rank_ratio::Float32 = 0.25f0)
    cur_w1 = convert_dense_to_cur(dense_expert.w1.weight, rank_ratio=rank_ratio)
    cur_w2 = convert_dense_to_cur(dense_expert.w2.weight, rank_ratio=rank_ratio)
    
    return CURExpert(cur_w1, cur_w2, dense_expert.activation)
end

function (expert::CURExpert)(x; training::Bool = false)
    h = cur_matmul(expert.cur_w1, x)
    h = expert.activation.(h)
    y = cur_matmul(expert.cur_w2, h)
    return y
end

function cur_matmul(cur::CURDecomposition, x::AbstractVecOrMat)
    # x is (in_features, batch) or (in_features,)
    # R is (rank, in_features), so R * x is (rank, batch)
    temp1 = cur.R * x
    # U is (rank, rank), so U * temp1 is (rank, batch)
    temp2 = cur.U * temp1
    # C is (out_features, rank), so C * temp2 is (out_features, batch)
    return cur.C * temp2
end

"""
    GatedCURExpert - Gated expert with CUR decomposition
"""
struct GatedCURExpert{C1, C2, C3, A} <: Expert
    cur_w1::C1
    cur_w2::C2
    cur_w3::C3
    activation::A
end

Flux.@functor GatedCURExpert (cur_w1, cur_w2, cur_w3)

function GatedCURExpert(input_dim::Int, hidden_dim::Int, output_dim::Int, 
                       activation; rank::Int)
    σ = sqrt(2.0f0 / input_dim)
    
    # Fix 4: Use direct initialization
    cur_w1 = initialize_cur_directly(hidden_dim, input_dim, rank, σ)
    cur_w2 = initialize_cur_directly(output_dim, hidden_dim, rank, σ)
    cur_w3 = initialize_cur_directly(hidden_dim, input_dim, rank, σ)
    
    return GatedCURExpert(cur_w1, cur_w2, cur_w3, activation)
end

"""
    convert_expert_to_cur(expert::GatedExpert; rank_ratio::Float32 = 0.25f0)

Fix 5: Convert pretrained gated expert to CUR format.
"""
function convert_expert_to_cur(expert::GatedExpert; rank_ratio::Float32 = 0.25f0,
                              energy_threshold::Union{Float32, Nothing} = nothing)
    cur_w1 = convert_dense_to_cur(expert.w1.weight, rank_ratio=rank_ratio, 
                                  energy_threshold=energy_threshold)
    cur_w2 = convert_dense_to_cur(expert.w2.weight, rank_ratio=rank_ratio, 
                                  energy_threshold=energy_threshold)
    cur_w3 = convert_dense_to_cur(expert.w3.weight, rank_ratio=rank_ratio, 
                                  energy_threshold=energy_threshold)
    
    return GatedCURExpert(cur_w1, cur_w2, cur_w3, expert.activation)
end

function (expert::GatedCURExpert)(x; training::Bool = false)
    gate = cur_matmul(expert.cur_w1, x)
    gate = expert.activation.(gate)
    up = cur_matmul(expert.cur_w3, x)
    h = gate .* up
    return cur_matmul(expert.cur_w2, h)
end

"""
Example usage for proper CUR testing in MoE context:

```julia
# 1. Create test inputs (realistic, not random)
test_inputs = generate_realistic_test_inputs(768, 128; scale=1.0f0)

# 2. Test single expert compression
original_expert = GatedExpert(768, 3072, 768, silu)
cur_expert = convert_expert_to_cur(original_expert; rank_ratio=0.4f0)

result = test_cur_expert_approximation(original_expert, cur_expert, test_inputs)
println("Relative error: (result.relative_error)")  # Should be < 0.01
println("Cosine similarity: (result.cosine_similarity)")  # Should be > 0.99

# 3. Automatically find best rank
auto_result = select_cur_rank(original_expert, test_inputs; 
                             target_error=0.01f0,
                             min_compression=0.25f0)
println("Suggested compression: (auto_result.compression_ratio)")

# 4. Test in full MoE context
moe_config = MoEConfig(num_experts=8, use_cur=false, ...)
moe_original = MoELayer(moe_config)

moe_config_cur = MoEConfig(num_experts=8, use_cur=true, cur_rank=256, ...)
moe_cur = MoELayer(moe_config_cur)

moe_result = test_cur_in_moe_context(moe_original, moe_cur, test_inputs)
println("MoE output difference: (moe_result.output_diff)")
println("Routing stability: (moe_result.routing_stability)")
```
"""

"""
    compression_ratio(cur::CURDecomposition)

Calculate the compression ratio achieved by CUR decomposition.
"""
function compression_ratio(cur::CURDecomposition)
    original_params = size(cur.C, 1) * size(cur.R, 2)
    cur_params = length(cur.C) + length(cur.U) + length(cur.R)
    return original_params / cur_params
end

"""
    test_cur_expert_approximation(original_expert, cur_expert, test_inputs)

Test functional approximation quality of CUR-compressed expert.
This is the CORRECT way to test neural network compression.
"""
function test_cur_expert_approximation(original_expert, cur_expert, test_inputs::AbstractMatrix)
    # Get outputs from both experts
    y_original = original_expert(test_inputs; training=false)
    y_cur = cur_expert(test_inputs; training=false)
    
    # Key metrics that actually matter:
    relative_output_error = norm(y_original - y_cur) / (norm(y_original) + 1e-8)
    
    # Cosine similarity (are outputs pointing in same direction?)
    y_orig_vec = vec(y_original)
    y_cur_vec = vec(y_cur)
    cosine_similarity = dot(y_orig_vec, y_cur_vec) / 
                       (norm(y_orig_vec) * norm(y_cur_vec) + 1e-8)
    
    max_absolute_error = maximum(abs.(y_original - y_cur))
    
    return (
        relative_error = relative_output_error,
        cosine_similarity = cosine_similarity,
        max_absolute_error = max_absolute_error,
        passed = relative_output_error < 0.01f0 && cosine_similarity > 0.99f0
    )
end

"""
    test_cur_in_moe_context(moe_layer_original, moe_layer_with_cur, test_inputs)

Test CUR compression in full MoE context, checking both output quality and routing stability.
"""
function test_cur_in_moe_context(moe_layer_original::MoELayer, moe_layer_with_cur::MoELayer, 
                                test_inputs::AbstractMatrix)
    # Get outputs from both MoE layers
    output_orig, _ = moe_layer_original(test_inputs; training=false)
    output_cur, _ = moe_layer_with_cur(test_inputs; training=false)
    
    # Output difference
    moe_output_difference = norm(output_orig - output_cur) / (norm(output_orig) + 1e-8)
    
    # Check if routing decisions are preserved
    _, _, routing_probs_orig, _ = moe_layer_original.router(test_inputs; training=false)
    _, _, routing_probs_cur, _ = moe_layer_with_cur.router(test_inputs; training=false)
    
    routing_difference = norm(routing_probs_orig - routing_probs_cur) / (norm(routing_probs_orig) + 1e-8)
    routing_stability = 1.0f0 - routing_difference
    
    # Expert selection similarity (are same experts being chosen?)
    expert_indices_orig, _, _, _ = moe_layer_original.router(test_inputs; training=false)
    expert_indices_cur, _, _, _ = moe_layer_with_cur.router(test_inputs; training=false)
    selection_similarity = mean(expert_indices_orig .== expert_indices_cur)
    
    return (
        output_diff = moe_output_difference,
        routing_stability = routing_stability,
        selection_similarity = selection_similarity,
        passed = moe_output_difference < 0.01f0 && routing_stability > 0.95f0
    )
end

"""
    generate_realistic_test_inputs(input_dim::Int, batch_size::Int = 32; 
                                  scale::Float32 = 1.0f0)

Generate test inputs with realistic activation statistics for neural networks.
"""
function generate_realistic_test_inputs(input_dim::Int, batch_size::Int = 32; 
                                       scale::Float32 = 1.0f0)
    # Neural network activations often have specific statistics
    # Use normal distribution with appropriate scaling
    test_inputs = randn(Float32, input_dim, batch_size) .* scale
    
    # Add some correlation structure (common in real activations)
    if input_dim > 10
        # Create some correlated features
        n_correlated = input_dim ÷ 10
        base_features = randn(Float32, n_correlated, batch_size)
        for i in 1:n_correlated
            for j in 1:9
                idx = i * 10 + j
                if idx <= input_dim
                    test_inputs[idx, :] = 0.7f0 * base_features[i, :] + 
                                         0.3f0 * test_inputs[idx, :]
                end
            end
        end
    end
    
    return test_inputs
end

"""
    select_cur_rank(expert::Expert, test_inputs::AbstractMatrix; 
                    target_error::Float32 = 0.01f0,
                    min_compression::Float32 = 0.3f0)

Automatically select CUR rank to achieve target approximation error.
"""
function select_cur_rank(expert::Union{StandardExpert, GatedExpert}, 
                        test_inputs::AbstractMatrix;
                        target_error::Float32 = 0.01f0,
                        min_compression::Float32 = 0.3f0)
    
    # Get original output for comparison
    y_original = expert(test_inputs; training=false)
    
    # Try different compression ratios
    if isa(expert, GatedExpert)
        test_ratios = [0.7f0, 0.5f0, 0.4f0, 0.3f0, 0.25f0, 0.2f0]
    else
        test_ratios = [0.7f0, 0.6f0, 0.5f0, 0.4f0, 0.3f0]
    end
    
    results = []
    
    for ratio in test_ratios
        if ratio < min_compression
            continue
        end
        
        # Convert to CUR with this ratio
        cur_expert = convert_expert_to_cur(expert; rank_ratio=ratio)
        
        # Test approximation quality
        test_result = test_cur_expert_approximation(expert, cur_expert, test_inputs)
        
        push!(results, (
            ratio = ratio,
            error = test_result.relative_error,
            similarity = test_result.cosine_similarity,
            passed = test_result.passed
        ))
        
        # If we achieved target, we can stop
        if test_result.relative_error < target_error
            suggested_rank = if isa(expert, GatedExpert)
                (
                    w1 = cur_expert.cur_w1.rank,
                    w2 = cur_expert.cur_w2.rank,
                    w3 = cur_expert.cur_w3.rank
                )
            else
                (
                    w1 = cur_expert.cur_w1.rank,
                    w2 = cur_expert.cur_w2.rank
                )
            end
            
            return (
                suggested_rank = suggested_rank,
                compression_ratio = ratio,
                expected_error = test_result.relative_error,
                test_results = results
            )
        end
    end
    
    # If no ratio achieved target, return best one
    best_idx = argmin([r.error for r in results])
    best_result = results[best_idx]
    
    @warn "Could not achieve target error $target_error. Best error: $(best_result.error) at ratio $(best_result.ratio)"
    
    return (
        suggested_rank = nothing,
        compression_ratio = best_result.ratio,
        expected_error = best_result.error,
        test_results = results
    )
end

"""
    compression_ratio(cur::CURDecomposition)

Calculate the compression ratio achieved by CUR decomposition.
"""
function compression_ratio(cur::CURDecomposition)
    original_params = size(cur.C, 1) * size(cur.R, 2)
    cur_params = length(cur.C) + length(cur.U) + length(cur.R)
    return original_params / cur_params
end

# Deprecated - DO NOT USE for neural network testing
# reconstruction_error(cur, original) = error("Use test_cur_expert_approximation instead!")

"""
    validate_cur_decomposition(cur::CURDecomposition; check_values::Bool = true)

Validate CUR decomposition structure and optionally check for numerical issues.
"""
function validate_cur_decomposition(cur::CURDecomposition; check_values::Bool = true)
    C_rows, C_cols = size(cur.C)
    U_rows, U_cols = size(cur.U)
    R_rows, R_cols = size(cur.R)
    
    # Structural checks
    @assert C_cols == U_rows "C columns ($C_cols) must match U rows ($U_rows)"
    @assert U_cols == R_rows "U columns ($U_cols) must match R rows ($R_rows)"
    @assert U_rows == U_cols "U must be square, got ($U_rows, $U_cols)"
    @assert cur.rank == C_cols "Stored rank ($(cur.rank)) must match C columns ($C_cols)"
    @assert cur.rank == R_rows "Stored rank ($(cur.rank)) must match R rows ($R_rows)"
    @assert cur.rank <= min(C_rows, R_cols) "Rank exceeds original matrix dimensions"
    
    # Check for empty matrices
    @assert C_rows > 0 && C_cols > 0 "C matrix is empty"
    @assert R_rows > 0 && R_cols > 0 "R matrix is empty"
    
    if check_values
        # Check for numerical issues
        @assert all(isfinite, cur.C) "C contains non-finite values (NaN or Inf)"
        @assert all(isfinite, cur.U) "U contains non-finite values (NaN or Inf)"
        @assert all(isfinite, cur.R) "R contains non-finite values (NaN or Inf)"
        
        # Check U conditioning (important for numerical stability)
        U_cond = cond(cur.U)
        if U_cond > 1e6
            @warn "U matrix is poorly conditioned (condition number: $U_cond)"
        end
        
        # Check for reasonable value ranges (for neural networks)
        max_val = max(maximum(abs, cur.C), maximum(abs, cur.U), maximum(abs, cur.R))
        if max_val > 1e3
            @warn "Large values detected (max absolute value: $max_val)"
        end
    end
    
    return true
end

"""
    validate_cur_expert(expert::Union{CURExpert, GatedCURExpert})

Validate all CUR decompositions in an expert.
"""
function validate_cur_expert(expert::CURExpert)
    validate_cur_decomposition(expert.cur_w1)
    validate_cur_decomposition(expert.cur_w2)
    return true
end

function validate_cur_expert(expert::GatedCURExpert)
    validate_cur_decomposition(expert.cur_w1)
    validate_cur_decomposition(expert.cur_w2)
    validate_cur_decomposition(expert.cur_w3)
    return true
end
export cur_decompose, compute_leverage_scores, compute_row_leverage_scores, sample_by_scores, initialize_cur_directly, suggest_rank, convert_dense_to_cur, CURExpert, cur_matmul, GatedCURExpert, convert_expert_to_cur, compression_ratio, test_cur_expert_approximation, test_cur_in_moe_context, generate_realistic_test_inputs, select_cur_rank, validate_cur_decomposition, validate_cur_expert
