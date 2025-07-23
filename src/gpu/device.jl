using CUDA
using Adapt

abstract type AbstractDevice end

struct CPUDevice <: AbstractDevice end
struct GPUDevice <: AbstractDevice 
    device_id::Int
end

const CPU_DEVICE = CPUDevice()

mutable struct DeviceState
    current_device::AbstractDevice
    gpu_available::Bool
    gpu_devices::Vector{GPUDevice}
    memory_pools::Dict{AbstractDevice, Any}
end

const DEVICE_STATE = DeviceState(
    CPU_DEVICE,
    false,
    GPUDevice[],
    Dict{AbstractDevice, Any}()
)

function __init_device__()
    DEVICE_STATE.gpu_available = CUDA.functional()
    
    if DEVICE_STATE.gpu_available
        for i in 0:(CUDA.ndevices() - 1)
            push!(DEVICE_STATE.gpu_devices, GPUDevice(i))
        end
        
        if !isempty(DEVICE_STATE.gpu_devices)
            DEVICE_STATE.current_device = DEVICE_STATE.gpu_devices[1]
        end
    end
    
    initialize_memory_pools!()
end

function initialize_memory_pools!()
    DEVICE_STATE.memory_pools[CPU_DEVICE] = nothing
    
    for gpu_device in DEVICE_STATE.gpu_devices
        CUDA.device!(gpu_device.device_id)
        DEVICE_STATE.memory_pools[gpu_device] = CUDA.MemoryPool()
    end
end

function get_device()
    return DEVICE_STATE.current_device
end

function set_device!(device::AbstractDevice)
    if device isa GPUDevice
        if !DEVICE_STATE.gpu_available
            @warn "GPU not available, falling back to CPU"
            device = CPU_DEVICE
        elseif device.device_id >= length(DEVICE_STATE.gpu_devices)
            @warn "GPU device $(device.device_id) not available, using GPU 0"
            device = DEVICE_STATE.gpu_devices[1]
        else
            CUDA.device!(device.device_id)
        end
    end
    
    DEVICE_STATE.current_device = device
end

function with_device(f, device::AbstractDevice)
    old_device = get_device()
    try
        set_device!(device)
        return f()
    finally
        set_device!(old_device)
    end
end

function gpu_available()
    return DEVICE_STATE.gpu_available
end

function cpu_device()
    return CPU_DEVICE
end

function gpu_device(id::Int = 0)
    if !DEVICE_STATE.gpu_available
        error("GPU not available")
    end
    
    if id >= length(DEVICE_STATE.gpu_devices)
        error("GPU device $id not available")
    end
    
    return DEVICE_STATE.gpu_devices[id + 1]
end

function auto_device()
    return DEVICE_STATE.gpu_available ? DEVICE_STATE.gpu_devices[1] : CPU_DEVICE
end

function device_count()
    return length(DEVICE_STATE.gpu_devices)
end

function current_device_id()
    device = get_device()
    return device isa GPUDevice ? device.device_id : -1
end

function to_device(x, device::CPUDevice)
    return cpu(x)
end

function to_device(x, device::GPUDevice)
    return with_device(device) do
        gpu(x)
    end
end

function to_device(x)
    return to_device(x, get_device())
end

function is_on_device(x, device::CPUDevice)
    return x isa Array
end

function is_on_device(x, device::GPUDevice)
    return x isa CuArray
end

function is_on_current_device(x)
    return is_on_device(x, get_device())
end

function ensure_device(x, device::AbstractDevice)
    return is_on_device(x, device) ? x : to_device(x, device)
end

function ensure_current_device(x)
    return ensure_device(x, get_device())
end

function device_synchronize()
    device = get_device()
    if device isa GPUDevice
        CUDA.synchronize()
    end
end

function device_memory_info()
    device = get_device()
    if device isa GPUDevice
        return CUDA.memory_status()
    else
        return (free = 0, total = 0, used = 0)
    end
end

function optimal_batch_size(model_size::Int, sequence_length::Int = 512)
    device = get_device()
    
    if device isa CPUDevice
        return min(16, max(1, div(1024 * 1024 * 100, model_size * sequence_length)))
    else
        mem_info = device_memory_info()
        available_memory = mem_info.free
        estimated_memory_per_sample = model_size * sequence_length * 4
        safety_factor = 0.8
        
        max_batch = floor(Int, available_memory * safety_factor / estimated_memory_per_sample)
        return min(128, max(1, max_batch))
    end
end