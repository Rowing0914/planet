import os
import nvidia_smi
import tensorflow as tf
from eager_setup import eager_setup

nvidia_smi.nvmlInit()  # to get information regarding gpus


def get_gpu_info(gpu_id=None):
    """ Get gpu-info regarding gpu_id
    :param gpu_id: gpu bus id
    :return mem_used: used memory in MiB
    :return mem_total: total memory in MiB
    """
    if gpu_id is None:
        gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(int(gpu_id))
    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    mem_used = mem_res.used / (1024 ** 2)
    mem_total = mem_res.total / (1024 ** 2)
    return mem_used, mem_total, gpu_id


def print_gpu_info(gpu_id=None):
    """ Print gpu-info regarding gpu_id on console
    :param gpu_id: gpu bus id
    """
    if gpu_id is None:
        gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
    mem_used, mem_total, gpu_id = get_gpu_info(gpu_id=int(gpu_id))
    print("GPU({}): {:.2f}MiB / {:.2f}MiB".format(gpu_id, mem_used, mem_total))


if __name__ == '__main__':
    eager_setup()

    x = tf.random.normal(shape=(100, 1000))

    for id in range(nvidia_smi.nvmlDeviceGetCount()):
        mem_used, mem_total, gpu_id = get_gpu_info(gpu_id=id)
        print("GPU({}): {:.2f}MiB / {:.2f}MiB".format(id, mem_used, mem_total))
