"""
GPU Utilities for CUDA MoE Implementation

Essential GPU utility functions including optimized softmax, matrix operations,
reduction kernels, and memory management utilities with custom CUDA kernels
for maximum performance.
"""

# GPU device information utilities
function gpu_device_info()
    if !CUDA.functional()
        throw(ErrorException("CUDA not functional"))
    end
    return GPUDeviceInfo()
end

function gpu_memory_info()
    if !CUDA.functional()
        throw(ErrorException("CUDA not functional"))
    end
    return GPUMemoryInfo()
end

function gpu_synchronize()
    if CUDA.functional()
        CUDA.synchronize()
    end
end

# Memory allocation utilities with alignment
function gpu_zeros(::Type{T}, dims...; aligned::Bool = false, alignment::Int = 32) where T
    if aligned && length(dims) >= 2
        aligned_dims = (align_dimension(dims[1], alignment), dims[2:end]...)
        result = CUDA.zeros(T, aligned_dims...)
        return view(result, 1:dims[1], axes(result)[2:end]...)
    else
        return CUDA.zeros(T, dims...)
    end
end

function gpu_ones(::Type{T}, dims...; aligned::Bool = false, alignment::Int = 32) where T
    if aligned && length(dims) >= 2
        aligned_dims = (align_dimension(dims[1], alignment), dims[2:end]...)
        result = CUDA.ones(T, aligned_dims...)
        return view(result, 1:dims[1], axes(result)[2:end]...)
    else
        return CUDA.ones(T, dims...)
    end
end

function gpu_randn(::Type{T}, dims...; aligned::Bool = false, alignment::Int = 32) where T
    if aligned && length(dims) >= 2
        aligned_dims = (align_dimension(dims[1], alignment), dims[2:end]...)
        result = CUDA.randn(T, aligned_dims...)
        return view(result, 1:dims[1], axes(result)[2:end]...)
    else
        return CUDA.randn(T, dims...)
    end
end

function gpu_copy!(dest::CuArray{T}, src::CuArray{T}) where T
    if size(dest) != size(src)
        throw(DimensionMismatch("Destination and source arrays must have the same size"))
    end
    copyto!(dest, src)
    return dest
end

# Optimized softmax implementation with numerically stable computation
function gpu_softmax!(output::CuMatrix{T}, input::CuMatrix{T}; 
                     dims::Int = 1, epsilon::T = T(1e-8)) where T<:AbstractFloat
    
    if size(output) != size(input)
        throw(DimensionMismatch("Output and input matrices must have the same size"))
    end
    
    if dims == 1
        # Softmax along columns (over rows)
        num_cols = size(input, 2)
        
        # Launch kernel for each column
        kernel_config = GPUKernelConfig(num_cols)
        
        @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid stream=kernel_config.stream gpu_softmax_kernel_colwise!(
            output, input, size(input, 1), size(input, 2), epsilon
        )
        
    elseif dims == 2
        # Softmax along rows (over columns)
        num_rows = size(input, 1)
        
        kernel_config = GPUKernelConfig(num_rows)
        
        @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid stream=kernel_config.stream gpu_softmax_kernel_rowwise!(
            output, input, size(input, 1), size(input, 2), epsilon
        )
        
    else
        throw(ArgumentError("dims must be 1 or 2"))
    end
    
    CUDA.synchronize()
    return output
end

function gpu_softmax(input::CuMatrix{T}; dims::Int = 1, epsilon::T = T(1e-8)) where T<:AbstractFloat
    output = similar(input)
    return gpu_softmax!(output, input; dims=dims, epsilon=epsilon)
end

# CUDA kernel for column-wise softmax (softmax over rows)
function gpu_softmax_kernel_colwise!(output::CuDeviceMatrix{T}, input::CuDeviceMatrix{T}, 
                                     num_rows::Int, num_cols::Int, epsilon::T) where T
    
    col_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if col_idx <= num_cols
        # Find maximum value in this column for numerical stability
        max_val = input[1, col_idx]
        for row_idx in 2:num_rows
            val = input[row_idx, col_idx]
            if val > max_val
                max_val = val
            end
        end
        
        # Compute sum of exponentials
        exp_sum = T(0)
        for row_idx in 1:num_rows
            exp_val = exp(input[row_idx, col_idx] - max_val)
            exp_sum += exp_val
        end
        
        # Ensure numerical stability
        exp_sum = max(exp_sum, epsilon)
        
        # Compute softmax values
        for row_idx in 1:num_rows
            exp_val = exp(input[row_idx, col_idx] - max_val)
            output[row_idx, col_idx] = exp_val / exp_sum
        end
    end
    
    return nothing
end

# CUDA kernel for row-wise softmax (softmax over columns)
function gpu_softmax_kernel_rowwise!(output::CuDeviceMatrix{T}, input::CuDeviceMatrix{T}, 
                                     num_rows::Int, num_cols::Int, epsilon::T) where T
    
    row_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if row_idx <= num_rows
        # Find maximum value in this row for numerical stability
        max_val = input[row_idx, 1]
        for col_idx in 2:num_cols
            val = input[row_idx, col_idx]
            if val > max_val
                max_val = val
            end
        end
        
        # Compute sum of exponentials
        exp_sum = T(0)
        for col_idx in 1:num_cols
            exp_val = exp(input[row_idx, col_idx] - max_val)
            exp_sum += exp_val
        end
        
        # Ensure numerical stability
        exp_sum = max(exp_sum, epsilon)
        
        # Compute softmax values
        for col_idx in 1:num_cols
            exp_val = exp(input[row_idx, col_idx] - max_val)
            output[row_idx, col_idx] = exp_val / exp_sum
        end
    end
    
    return nothing
end

# SiLU (Swish) activation function optimized for GPU
function gpu_silu!(output::CuArray{T}, input::CuArray{T}) where T<:AbstractFloat
    if size(output) != size(input)
        throw(DimensionMismatch("Output and input arrays must have the same size"))
    end
    
    total_elements = length(input)
    kernel_config = GPUKernelConfig(total_elements)
    
    @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid stream=kernel_config.stream gpu_silu_kernel!(
        output, input, total_elements
    )
    
    CUDA.synchronize()
    return output
end

function gpu_silu(input::CuArray{T}) where T<:AbstractFloat
    output = similar(input)
    return gpu_silu!(output, input)
end

# CUDA kernel for SiLU activation: x * sigmoid(x)
function gpu_silu_kernel!(output::CuDeviceArray{T}, input::CuDeviceArray{T}, 
                         total_elements::Int) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= total_elements
        x = input[idx]
        sigmoid_x = T(1) / (T(1) + exp(-x))
        output[idx] = x * sigmoid_x
    end
    
    return nothing
end

# Element-wise multiplication optimized for GPU
function gpu_elementwise_multiply!(output::CuArray{T}, a::CuArray{T}, b::CuArray{T}) where T<:AbstractFloat
    if size(output) != size(a) || size(output) != size(b)
        throw(DimensionMismatch("All arrays must have the same size"))
    end
    
    total_elements = length(output)
    kernel_config = GPUKernelConfig(total_elements)
    
    @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid stream=kernel_config.stream gpu_elementwise_multiply_kernel!(
        output, a, b, total_elements
    )
    
    CUDA.synchronize()
    return output
end

# CUDA kernel for element-wise multiplication
function gpu_elementwise_multiply_kernel!(output::CuDeviceArray{T}, a::CuDeviceArray{T}, 
                                        b::CuDeviceArray{T}, total_elements::Int) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= total_elements
        output[idx] = a[idx] * b[idx]
    end
    
    return nothing
end

# Matrix-vector multiplication with broadcasting
function gpu_matrix_vector_multiply!(output::CuMatrix{T}, matrix::CuMatrix{T}, 
                                    vector::CuVector{T}; transpose_matrix::Bool = false) where T<:AbstractFloat
    
    if transpose_matrix
        if size(matrix, 1) != length(vector) || size(output) != (size(matrix, 2), 1)
            throw(DimensionMismatch("Dimension mismatch for transposed matrix-vector multiplication"))
        end
        CUBLAS.gemv!('T', T(1), matrix, vector, T(0), vec(output))
    else
        if size(matrix, 2) != length(vector) || size(output) != (size(matrix, 1), 1)
            throw(DimensionMismatch("Dimension mismatch for matrix-vector multiplication"))
        end
        CUBLAS.gemv!('N', T(1), matrix, vector, T(0), vec(output))
    end
    
    return output
end

# Parallel reduction operations
function gpu_reduce_sum(input::CuArray{T}; dims = nothing) where T<:AbstractFloat
    if isnothing(dims)
        # Reduce entire array to scalar
        return CUDA.reduce(+, input)
    else
        # Reduce along specified dimensions
        return CUDA.reduce(+, input; dims=dims)
    end
end

function gpu_reduce_mean(input::CuArray{T}; dims = nothing) where T<:AbstractFloat
    if isnothing(dims)
        # Mean of entire array
        return CUDA.reduce(+, input) / length(input)
    else
        # Mean along specified dimensions - use more robust approach
        if dims == 1
            # Reduce along rows (over columns)
            return dropdims(mean(input, dims=1), dims=1)
        elseif dims == 2
            # Reduce along columns (over rows) 
            return dropdims(mean(input, dims=2), dims=2)
        else
            # General case
            return dropdims(mean(input, dims=dims), dims=dims)
        end
    end
end
function gpu_reduce_max(input::CuArray{T}; dims = nothing) where T<:AbstractFloat
    if isnothing(dims)
        return CUDA.reduce(max, input)
    else
        return CUDA.reduce(max, input; dims=dims)
    end
end

function gpu_reduce_min(input::CuArray{T}; dims = nothing) where T<:AbstractFloat
    if isnothing(dims)
        return CUDA.reduce(min, input)
    else
        return CUDA.reduce(min, input; dims=dims)
    end
end

# Argmax implementation for GPU
function gpu_argmax(input::CuMatrix{T}; dims::Int = 1) where T<:AbstractFloat
    if dims == 1
        # Argmax along columns (over rows)
        num_cols = size(input, 2)
        output = CUDA.zeros(Int32, 1, num_cols)
        
        kernel_config = GPUKernelConfig(num_cols)
        
        @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid stream=kernel_config.stream gpu_argmax_colwise_kernel!(
            output, input, size(input, 1), size(input, 2)
        )
        
    elseif dims == 2
        # Argmax along rows (over columns)
        num_rows = size(input, 1)
        output = CUDA.zeros(Int32, num_rows, 1)
        
        kernel_config = GPUKernelConfig(num_rows)
        
        @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid stream=kernel_config.stream gpu_argmax_rowwise_kernel!(
            output, input, size(input, 1), size(input, 2)
        )
        
    else
        throw(ArgumentError("dims must be 1 or 2"))
    end
    
    CUDA.synchronize()
    return output
end

# CUDA kernel for column-wise argmax
function gpu_argmax_colwise_kernel!(output::CuDeviceMatrix{Int32}, input::CuDeviceMatrix{T}, 
                                   num_rows::Int, num_cols::Int) where T
    
    col_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if col_idx <= num_cols
        max_val = input[1, col_idx]
        max_idx = Int32(1)
        
        for row_idx in 2:num_rows
            val = input[row_idx, col_idx]
            if val > max_val
                max_val = val
                max_idx = Int32(row_idx)
            end
        end
        
        output[1, col_idx] = max_idx
    end
    
    return nothing
end

# CUDA kernel for row-wise argmax
function gpu_argmax_rowwise_kernel!(output::CuDeviceMatrix{Int32}, input::CuDeviceMatrix{T}, 
                                   num_rows::Int, num_cols::Int) where T
    
    row_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if row_idx <= num_rows
        max_val = input[row_idx, 1]
        max_idx = Int32(1)
        
        for col_idx in 2:num_cols
            val = input[row_idx, col_idx]
            if val > max_val
                max_val = val
                max_idx = Int32(col_idx)
            end
        end
        
        output[row_idx, 1] = max_idx
    end
    
    return nothing
end

# Top-K selection with efficient GPU implementation
function gpu_topk!(indices::CuMatrix{Int32}, values::CuMatrix{T}, 
                  input::CuMatrix{T}, k::Int; dims::Int = 1, 
                  largest::Bool = true) where T<:AbstractFloat
    
    if k <= 0
        throw(ArgumentError("k must be positive"))
    end
    
    if dims == 1
        # Top-k along columns (over rows)
        num_cols = size(input, 2)
        input_rows = size(input, 1)
        
        if k > input_rows
            throw(ArgumentError("k cannot be larger than the dimension being sorted"))
        end
        
        if size(indices) != (k, num_cols) || size(values) != (k, num_cols)
            throw(DimensionMismatch("Output arrays have incorrect size"))
        end
        
        kernel_config = GPUKernelConfig(num_cols)
        
        @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid stream=kernel_config.stream gpu_topk_colwise_kernel!(
            indices, values, input, input_rows, num_cols, k, largest
        )
        
    elseif dims == 2
        # Top-k along rows (over columns)
        num_rows = size(input, 1)
        input_cols = size(input, 2)
        
        if k > input_cols
            throw(ArgumentError("k cannot be larger than the dimension being sorted"))
        end
        
        if size(indices) != (num_rows, k) || size(values) != (num_rows, k)
            throw(DimensionMismatch("Output arrays have incorrect size"))
        end
        
        kernel_config = GPUKernelConfig(num_rows)
        
        @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid stream=kernel_config.stream gpu_topk_rowwise_kernel!(
            indices, values, input, num_rows, input_cols, k, largest
        )
        
    else
        throw(ArgumentError("dims must be 1 or 2"))
    end
    
    CUDA.synchronize()
    return indices, values
end

# CUDA kernel for column-wise top-k selection (using insertion sort for small k)
function gpu_topk_colwise_kernel!(indices::CuDeviceMatrix{Int32}, values::CuDeviceMatrix{T},
                                 input::CuDeviceMatrix{T}, num_rows::Int, num_cols::Int,
                                 k::Int, largest::Bool) where T
    
    col_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if col_idx <= num_cols
        # Initialize top-k arrays for this column
        for i in 1:k
            if largest
                values[i, col_idx] = T(-Inf)
            else
                values[i, col_idx] = T(Inf)
            end
            indices[i, col_idx] = Int32(0)
        end
        
        # Process each element in the column
        for row_idx in 1:num_rows
            val = input[row_idx, col_idx]
            
            # Check if this value should be in top-k
            should_insert = false
            insert_pos = k + 1
            
            if largest
                for i in 1:k
                    if val > values[i, col_idx]
                        should_insert = true
                        insert_pos = i
                        break
                    end
                end
            else
                for i in 1:k
                    if val < values[i, col_idx]
                        should_insert = true
                        insert_pos = i
                        break
                    end
                end
            end
            
            # Insert the value if needed
            if should_insert
                # Shift elements down
                for i in k:-1:(insert_pos + 1)
                    values[i, col_idx] = values[i - 1, col_idx]
                    indices[i, col_idx] = indices[i - 1, col_idx]
                end
                
                # Insert new value
                values[insert_pos, col_idx] = val
                indices[insert_pos, col_idx] = Int32(row_idx)
            end
        end
    end
    
    return nothing
end

# CUDA kernel for row-wise top-k selection
function gpu_topk_rowwise_kernel!(indices::CuDeviceMatrix{Int32}, values::CuDeviceMatrix{T},
                                 input::CuDeviceMatrix{T}, num_rows::Int, num_cols::Int,
                                 k::Int, largest::Bool) where T
    
    row_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if row_idx <= num_rows
        # Initialize top-k arrays for this row
        for i in 1:k
            if largest
                values[row_idx, i] = T(-Inf)
            else
                values[row_idx, i] = T(Inf)
            end
            indices[row_idx, i] = Int32(0)
        end
        
        # Process each element in the row
        for col_idx in 1:num_cols
            val = input[row_idx, col_idx]
            
            # Check if this value should be in top-k
            should_insert = false
            insert_pos = k + 1
            
            if largest
                for i in 1:k
                    if val > values[row_idx, i]
                        should_insert = true
                        insert_pos = i
                        break
                    end
                end
            else
                for i in 1:k
                    if val < values[row_idx, i]
                        should_insert = true
                        insert_pos = i
                        break
                    end
                end
            end
            
            # Insert the value if needed
            if should_insert
                # Shift elements right
                for i in k:-1:(insert_pos + 1)
                    values[row_idx, i] = values[row_idx, i - 1]
                    indices[row_idx, i] = indices[row_idx, i - 1]
                end
                
                # Insert new value
                values[row_idx, insert_pos] = val
                indices[row_idx, insert_pos] = Int32(col_idx)
            end
        end
    end
    
    return nothing
end

# Workspace management for large operations
function allocate_gpu_workspace(required_mb::Int)
    if !CUDA.functional()
        throw(ErrorException("CUDA not functional"))
    end
    
    required_bytes = required_mb * 1024 * 1024
    available_bytes = CUDA.available_memory()
    
    if required_bytes > available_bytes
        # Try garbage collection first
        GC.gc()
        CUDA.reclaim()
        available_bytes = CUDA.available_memory()
        
        if required_bytes > available_bytes
            throw(OutOfMemoryError("Cannot allocate $(required_mb)MB, only $(available_bytes ÷ (1024^2))MB available"))
        end
    end
    
    return CUDA.zeros(UInt8, required_bytes)
end

function free_gpu_workspace(workspace::CuArray)
    # Julia's GC will handle this automatically, but we can help
    workspace = nothing
    GC.gc()
    CUDA.reclaim()
end

# Numerical stability utilities
function gpu_clamp!(output::CuArray{T}, input::CuArray{T}, min_val::T, max_val::T) where T<:AbstractFloat
    if size(output) != size(input)
        throw(DimensionMismatch("Output and input arrays must have the same size"))
    end
    
    total_elements = length(input)
    kernel_config = GPUKernelConfig(total_elements)
    
    @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid stream=kernel_config.stream gpu_clamp_kernel!(
        output, input, min_val, max_val, total_elements
    )
    
    CUDA.synchronize()
    return output
end

function gpu_clamp_kernel!(output::CuDeviceArray{T}, input::CuDeviceArray{T}, 
                          min_val::T, max_val::T, total_elements::Int) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= total_elements
        val = input[idx]
        if val < min_val
            output[idx] = min_val
        elseif val > max_val
            output[idx] = max_val
        else
            output[idx] = val
        end
    end
    
    return nothing
end

# Check for numerical issues
function gpu_check_finite(input::AbstractArray{T}) where T<:AbstractFloat
    # Use CUDA reduction to check if all elements are finite
    finite_count = CUDA.reduce(+, isfinite.(input))
    total_count = length(input)
    return finite_count == total_count
end

function gpu_check_range(input::CuArray{T}, min_val::T, max_val::T) where T<:AbstractFloat
    # Check if all elements are within the specified range
    in_range = (input .>= min_val) .& (input .<= max_val)
    return all(in_range)
end

# Performance monitoring utilities
mutable struct GPUPerfCounter
    operation_name::String
    total_time_ms::Float64
    call_count::Int
    last_time_ms::Float64
    min_time_ms::Float64
    max_time_ms::Float64
    
    function GPUPerfCounter(name::String)
        return new(name, 0.0, 0, 0.0, Inf, 0.0)
    end
end

const _gpu_perf_counters = Dict{String, GPUPerfCounter}()

macro gpu_time(name, expr)
    quote
        local start_time = time_ns()
        local result = $(esc(expr))
        CUDA.synchronize()
        local end_time = time_ns()
        local elapsed_ms = (end_time - start_time) / 1e6
        
        # Update performance counter
        counter_name = $(esc(name))
        if !haskey(_gpu_perf_counters, counter_name)
            _gpu_perf_counters[counter_name] = GPUPerfCounter(counter_name)
        end
        
        counter = _gpu_perf_counters[counter_name]
        counter.total_time_ms += elapsed_ms
        counter.call_count += 1
        counter.last_time_ms = elapsed_ms
        counter.min_time_ms = min(counter.min_time_ms, elapsed_ms)
        counter.max_time_ms = max(counter.max_time_ms, elapsed_ms)
        
        result
    end
end

function gpu_perf_report()
    println("GPU Performance Report:")
    println("=" ^ 80)
    @printf "%-30s %8s %12s %12s %12s %12s\n" "Operation" "Calls" "Total (ms)" "Avg (ms)" "Min (ms)" "Max (ms)"
    println("-" ^ 80)
    
    for (name, counter) in _gpu_perf_counters
        avg_time = counter.total_time_ms / counter.call_count
        @printf "%-30s %8d %12.3f %12.3f %12.3f %12.3f\n" name counter.call_count counter.total_time_ms avg_time counter.min_time_ms counter.max_time_ms
    end
    
    println("=" ^ 80)
end

function gpu_clear_perf_counters!()
    empty!(_gpu_perf_counters)
end

# ADD this function to utils.jl for proper GPU random sampling:

"""
    gpu_random_categorical_sample!(output::CuMatrix{Int32}, num_categories::Int)

Generate random categorical samples on GPU (1 to num_categories).
"""
function gpu_random_categorical_sample!(output::CuMatrix{Int32}, num_categories::Int)
    rows, cols = size(output)
    total_elements = rows * cols
    
    # Generate uniform random numbers and convert to categorical
    uniform_random = CUDA.rand(Float32, rows, cols)
    
    kernel_config = GPUKernelConfig(total_elements)
    
    @cuda threads=kernel_config.threads_per_block blocks=kernel_config.blocks_per_grid gpu_categorical_sample_kernel!(
        output, uniform_random, num_categories, total_elements
    )
    
    CUDA.synchronize()
    return output
end

# ADD the corresponding kernel:
function gpu_categorical_sample_kernel!(
    output::CuDeviceMatrix{Int32},
    uniform_random::CuDeviceMatrix{Float32},
    num_categories::Int,
    total_elements::Int
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= total_elements
        # Convert linear index to 2D
        row = ((idx - 1) % size(output, 1)) + 1
        col = ((idx - 1) ÷ size(output, 1)) + 1
        
        # Convert uniform [0,1] to categorical [1, num_categories]
        category = Int32(floor(uniform_random[row, col] * Float32(num_categories))) + Int32(1)
        
        # Ensure within bounds
        if category > num_categories
            category = Int32(num_categories)
        end
        
        output[row, col] = category
    end
    
    return nothing
end

