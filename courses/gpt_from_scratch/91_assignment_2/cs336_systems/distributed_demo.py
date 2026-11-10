import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return device


def distributed_demo(rank, world_size, data_size):
    device = setup(rank, world_size)
    try:
        data = torch.zeros(data_size * 1_000_000 // 4, dtype=torch.float32, device=device)

        for _ in range(5):
            dist.all_reduce(data)
        torch.cuda.synchronize(device)

        max_times_ms = []
        for _ in range(10):
            dist.barrier()
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            dist.all_reduce(data)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000

            local_time = torch.tensor([elapsed_ms], dtype=torch.float64, device=device)
            all_times = [torch.zeros_like(local_time) for _ in range(world_size)]
            dist.all_gather(all_times, local_time)
            torch.cuda.synchronize(device)

            if rank == 0:
                times_ms = [t.item() for t in all_times]
                max_times_ms.append(max(times_ms))
                print(f"各 rank 耗时 (ms): {times_ms}")

        if rank == 0:
            print(f"每次最大耗时的平均值 (ms): {sum(max_times_ms) / len(max_times_ms):.3f}")
    finally:
        dist.destroy_process_group()


def benchmark():
    gpu_count = torch.cuda.device_count()
    data_size_list = [1, 10, 100, 1000]

    for world_size in range(2, 7):
        if world_size > gpu_count:
            print(f"需要 {world_size} 张 GPU，当前可见 {gpu_count} 张，停止测试。")
            break

        for data_size in data_size_list:
            print(f"world size {world_size}, data size {data_size} MB")
            mp.spawn(
                distributed_demo,
                args=(world_size, data_size),
                nprocs=world_size,
                join=True,
            )


if __name__ == "__main__":
    benchmark()
