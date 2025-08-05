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
    cur_decompose(A::AbstractMatrix; rank::Int, oversample::Int=10)

Perform CUR decomposition on matrix A.
"""
function cur_decompose(A::AbstractMatrix; rank::Int, oversample::Int = 10)
    m, n = size(A)
    
    col_scores = compute_leverage_scores(A, rank)         
    row_scores = compute_row_leverage_scores(A, rank)      
    
    num_cols = min(rank + oversample, n)
    num_rows = min(rank + oversample, m)
    
    col_indices = sample_by_scores(col_scores, num_cols)
    row_indices = sample_by_scores(row_scores, num_rows)
    
    C = A[:, col_indices]
    R = A[row_indices, :]
    W = A[row_indices, col_indices]
    
    U = pinv(W)
    
    return CURDecomposition(C, U, R, rank)
end

function compute_leverage_scores(A::AbstractMatrix, rank::Int)
    U, s, V = svd(A)
    k = min(rank, size(V, 2))
    
    scores = zeros(Float32, size(A, 2))
    for j in 1:size(A, 2)
        scores[j] = sum(abs2, V[j, 1:k])  
    end
    
    return scores ./ sum(scores)
end

function compute_row_leverage_scores(A::AbstractMatrix, rank::Int)
    U, s, V = svd(A)
    k = min(rank, size(U, 2))
    
    scores = zeros(Float32, size(A, 1))
    for j in 1:size(A, 1)
        scores[j] = sum(abs2, U[j, 1:k])  
    end
    
    return scores ./ sum(scores)
end

function sample_by_scores(scores::AbstractVector, num_samples::Int)
    n = length(scores)
    num_samples = min(num_samples, n)
    
    indices = StatsBase.sample(1:n, Weights(scores), num_samples, replace=false)
    return sort(indices)
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
    
    w1 = randn(Float32, hidden_dim, input_dim) .* σ1
    w2 = randn(Float32, output_dim, hidden_dim) .* σ2
    
    cur_w1 = cur_decompose(w1, rank=rank)
    cur_w2 = cur_decompose(w2, rank=rank)
    
    return CURExpert(cur_w1, cur_w2, activation)
end

function (expert::CURExpert)(x; training::Bool = false)
    h = cur_matmul(expert.cur_w1, x)
    h = expert.activation.(h)
    y = cur_matmul(expert.cur_w2, h)
    return y
end

function cur_matmul(cur::CURDecomposition, x::AbstractVecOrMat)
    temp1 = cur.R * x
    temp2 = cur.U * temp1
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
    
    w1 = randn(Float32, hidden_dim, input_dim) .* σ
    w2 = randn(Float32, output_dim, hidden_dim) .* σ
    w3 = randn(Float32, hidden_dim, input_dim) .* σ
    
    cur_w1 = cur_decompose(w1, rank=rank)
    cur_w2 = cur_decompose(w2, rank=rank)
    cur_w3 = cur_decompose(w3, rank=rank)
    
    return GatedCURExpert(cur_w1, cur_w2, cur_w3, activation)
end

function (expert::GatedCURExpert)(x; training::Bool = false)
    gate = cur_matmul(expert.cur_w1, x)
    gate = expert.activation.(gate)
    up = cur_matmul(expert.cur_w3, x)
    h = gate .* up
    return cur_matmul(expert.cur_w2, h)
end
