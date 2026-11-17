"""单机多进程测量朴素 DDP 的训练步骤及梯度同步时间。"""

import argparse
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.improved_ddp import ImprovedDDPModule
from cs336_systems.naive_ddp import NaiveDDPModule


def run_worker(rank, args):
    device = torch.device(f"cuda:{rank}" if args.backend == "nccl" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)
    try:
        with Path(__file__).with_name("model_config.yaml").open(encoding="utf-8") as f:
            config = yaml.safe_load(f)[args.model]
        torch.manual_seed(0)
        ddp_class = ImprovedDDPModule if args.gradient_sync == "flat" else NaiveDDPModule
        model = ddp_class(BasicsTransformerLM(
            vocab_size=args.vocab_size, context_length=args.context_length, **config,
        ).to(device))
        optimizer = AdamW(model.parameters())
        torch.manual_seed(rank + 1)
        x = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
        targets = torch.randint(args.vocab_size, x.shape, device=device)

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        measurements = []
        for step in range(args.warmup + args.steps):
            # 屏障放在计时之外，使各进程从相近时间开始训练步骤。
            synchronize()
            dist.barrier()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(model(x), targets)
            loss.backward()
            synchronize()
            communication_start = time.perf_counter()
            model.finish_gradient_synchronization()
            synchronize()
            communication_time = time.perf_counter() - communication_start
            optimizer.step()
            synchronize()
            step_time = time.perf_counter() - start
            if step >= args.warmup:
                # 分别取各进程的最大耗时，反映最慢进程的耗时。
                times = torch.tensor([step_time, communication_time], dtype=torch.float64, device=device)
                dist.all_reduce(times, op=dist.ReduceOp.MAX)
                measurements.append(times.cpu().tolist())

        if rank == 0:
            step_mean = statistics.mean(row[0] for row in measurements)
            communication_mean = statistics.mean(row[1] for row in measurements)
            print(f"model={args.model}, gradient_sync={args.gradient_sync}, processes={args.world_size}, batch_size_per_process={args.batch_size}, "
                  f"context_length={args.context_length}, warmup={args.warmup}, steps={args.steps}")
            print(f"step: {step_mean * 1000:.3f} ms")
            print(f"gradient synchronization: {communication_mean * 1000:.3f} ms "
                  f"({communication_mean / step_mean * 100:.2f}% of step time)")
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["small", "medium", "large", "xl", "10B"], default="xl")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4, help="每个进程的 batch size")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--gradient-sync", choices=["individual", "flat"], default="individual")
    parser.add_argument("--port", default="29500")
    args = parser.parse_args()
    if min(args.world_size, args.batch_size, args.context_length, args.vocab_size, args.steps) <= 0 or args.warmup < 0:
        parser.error("进程数、维度和测量步数必须为正数，预热步数不能为负数")
    if args.backend == "nccl" and torch.cuda.device_count() < args.world_size:
        parser.error("可用 GPU 数量不足")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = args.port
    mp.spawn(run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
