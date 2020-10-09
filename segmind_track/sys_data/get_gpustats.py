import GPUtil


def gpu_metrics():
    gpu_data = {}
    gpus = GPUtil.getGPUs()

    for i, gpu in enumerate(gpus):
        gpu_data[f'GPU_load_{i}'] = gpu.load
        gpu_data[f'GPU_Memory_util_{i}'] = gpu.memoryUtil

    return gpu_data
