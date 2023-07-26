import pynvml as pynvml
import psutil

def check_gpu_usage(process_exceptions=['Xorg'], user_exceptions=['bla123'], min_memory=5, base_on_memory=True, base_on_process=True):
    # Process exceptions -> we don't care about such procs
    # User exceptions -> we care ONLY about procs of this user
    pynvml.nvmlInit()
    # print ("Driver Version:", pynvml.nvmlSystemGetDriverVersion())
    deviceCount = pynvml.nvmlDeviceGetCount()
    free_gpus = []
    mem_list = []
    for i in range(deviceCount):

        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem.free/(1024**3)
        
        if base_on_memory and free_memory < min_memory:
            continue

        free = True 
        if base_on_process:
            procs = [*pynvml.nvmlDeviceGetComputeRunningProcesses(handle), *pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)]
            for p in procs:
                try:
                    process = psutil.Process(p.pid)
                except psutil.NoSuchProcess:
                    continue

                if process.name not in process_exceptions and process.username() in user_exceptions:
                    free = False
                    break
        if free:
            free_gpus.append(str(i))
            mem_list.append(free_memory)


    #sort gpus by free memory
    sorted_data = sorted(list(zip(free_gpus, mem_list)), key=lambda x: x[1], reverse=True)
    free_gpus, mem_list = zip(*sorted_data)
    # print(f"GPUs:{free_gpus} Free")
    # print(f"free memory: {mem_list}")
    pynvml.nvmlShutdown()
    cuda_str = 'cuda:'+ free_gpus[0]
    return cuda_str


if __name__ == "__main__":
    print(check_gpu_usage())
    