"""统一测量普通、torch.compile 和 Triton Attention 的耗时与显存。

--preset compiled：原编译对比作业，batch=8、FP32、无因果掩码、100 次均值。
--preset flash：FlashAttention 作业，batch=1、因果掩码、do_bench 中位数。
--preset all：在 FlashAttention 作业的相同条件下比较三种实现。
"""

import argparse
import csv
import gc
import itertools
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch

from cs336_systems.flash_attention_core import FlashAttentionTriton


PRESETS = {
    "compiled": dict(seq_lens=[256, 1024, 4096, 8192, 16384], dtypes=["float32"],
                     batch_size=8, causal=False, timer="steps", implementations=["pytorch", "compiled"]),
    "flash": dict(seq_lens=[2**i for i in range(7, 17)], dtypes=["bfloat16", "float32"],
                  batch_size=1, causal=True, timer="triton", implementations=["pytorch", "triton"]),
    "all": dict(seq_lens=[2**i for i in range(7, 17)], dtypes=["bfloat16", "float32"],
                batch_size=1, causal=True, timer="triton", implementations=["pytorch", "compiled", "triton"]),
}


def pytorch_attention(q, k, v, is_causal=True):
    """非因果模式沿用原作业实现；显式计算注意力矩阵，不调用融合 SDPA。"""
    if not is_causal:
        from cs336_basics.model import scaled_dot_product_attention
        return scaled_dot_product_attention(q, k, v)
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    rows = torch.arange(q.shape[-2], device=q.device)
    cols = torch.arange(k.shape[-2], device=k.device)
    scores = scores.masked_fill(rows[:, None] < cols[None, :], float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def benchmark_case(attention, q, k, v, grad_output, *, timer="triton", steps=100,
                   warmup_steps=5, warmup_ms=25, rep_ms=100):
    def synchronize():
        if q.is_cuda:
            torch.cuda.synchronize(q.device)

    def forward_backward():
        output = attention(q, k, v)
        torch.autograd.grad(output, (q, k, v), grad_output)

    result = {}
    if timer == "triton":
        from triton.testing import do_bench

        def measure(fn):
            return do_bench(fn, warmup=warmup_ms, rep=rep_ms, quantiles=[0.5])[0]

        result["forward_ms"] = measure(lambda: attention(q, k, v))
        output = attention(q, k, v)
        result["backward_ms"] = measure(
            lambda output=output: torch.autograd.grad(output, (q, k, v), grad_output, retain_graph=True)
        )
        del output
        result["forward_backward_ms"] = measure(forward_backward)
    else:
        for _ in range(warmup_steps):
            forward_backward()
        timings = {"forward": [], "backward": [], "forward_backward": []}
        for _ in range(steps):
            synchronize()
            start = time.perf_counter()
            output = attention(q, k, v)
            synchronize()
            timings["forward"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            torch.autograd.grad(output, (q, k, v), grad_output)
            synchronize()
            timings["backward"].append((time.perf_counter() - start) * 1000)
            del output
        for _ in range(steps):
            synchronize()
            start = time.perf_counter()
            forward_backward()
            synchronize()
            timings["forward_backward"].append((time.perf_counter() - start) * 1000)
        for name, values in timings.items():
            result[f"{name}_ms"] = statistics.mean(values)
            result[f"{name}_std_ms"] = statistics.stdev(values) if steps > 1 else 0.0

    if q.is_cuda:
        # 单独测量一次完整前后向的显存，排除 do_bench 自身计时缓存的占用。
        synchronize()
        torch.cuda.reset_peak_memory_stats(q.device)
        output = attention(q, k, v)
        synchronize()
        result["memory_before_backward_bytes"] = torch.cuda.memory_allocated(q.device)
        torch.autograd.grad(output, (q, k, v), grad_output)
        synchronize()
        result["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(q.device)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="flash")
    parser.add_argument("--implementations", nargs="+", choices=["pytorch", "compiled", "triton"])
    parser.add_argument("--seq-lens", nargs="+", type=int)
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--dtypes", nargs="+", choices=["bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--timer", choices=["steps", "triton"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--warmup-ms", type=float, default=25)
    parser.add_argument("--rep-ms", type=float, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, help="同时保存 CSV，包含显存不足的配置")
    args = parser.parse_args()
    for name, value in PRESETS[args.preset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if (any(x <= 0 for x in args.seq_lens + args.dims + [args.batch_size, args.steps])
            or args.warmup_steps < 0 or args.warmup_ms <= 0 or args.rep_ms <= 0):
        parser.error("输入尺寸、测量次数及计时时长必须大于 0，预热次数不能小于 0")
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用；小规模检查可使用 --device cpu --timer steps --implementations pytorch")
    if args.device.type != "cuda" and (args.timer == "triton" or "triton" in args.implementations):
        parser.error("Triton 实现和 do_bench 计时需要 CUDA；CPU 请使用 steps 计时及 pytorch/compiled 实现")
    if args.warmup_steps == 0 and "compiled" in args.implementations and args.timer == "steps":
        print("预热次数为 0，首次编译耗时将计入结果。", file=sys.stderr)
    return args


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("highest")
    statistic = "median" if args.timer == "triton" else "mean"
    device_name = torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else str(args.device)
    print(f"Device: {device_name}; PyTorch: {torch.__version__}; preset={args.preset}; "
          f"batch={args.batch_size}; causal={args.causal}; {statistic} latency (ms)", file=sys.stderr)
    fields = ["seq_len", "dim", "dtype", "batch_size", "causal", "implementation", "device", "timer", "statistic",
              "forward_ms", "backward_ms", "forward_backward_ms", "forward_std_ms", "backward_std_ms",
              "forward_backward_std_ms", "memory_before_backward_bytes", "peak_memory_allocated_bytes", "status"]
    output_file = args.output.open("w", newline="") if args.output else None
    streams = [sys.stdout] + ([output_file] if output_file else [])
    writers = [csv.DictWriter(stream, fieldnames=fields) for stream in streams]
    try:
        for writer in writers:
            writer.writeheader()
        for seq_len, dim, dtype_name in itertools.product(args.seq_lens, args.dims, args.dtypes):
            for name in args.implementations:
                row = dict(seq_len=seq_len, dim=dim, dtype=dtype_name, batch_size=args.batch_size,
                           causal=args.causal, implementation=name, device=str(args.device), timer=args.timer, statistic=statistic)
                q = k = v = grad_output = attention = None
                try:
                    # 所有实现使用相同输入，随机生成不计入测量。
                    torch.manual_seed(0)
                    shape = (args.batch_size, seq_len, dim)
                    dtype = getattr(torch, dtype_name)
                    q, k, v = [torch.randn(shape, device=args.device, dtype=dtype, requires_grad=True) for _ in range(3)]
                    grad_output = torch.randn_like(q)
                    if name == "triton":
                        def attention(q, k, v):
                            return FlashAttentionTriton.apply(q, k, v, args.causal)
                    else:
                        attention = partial(pytorch_attention, is_causal=args.causal)
                        if name == "compiled":
                            attention = torch.compile(attention)
                    row.update(benchmark_case(attention, q, k, v, grad_output, timer=args.timer, steps=args.steps,
                                              warmup_steps=args.warmup_steps, warmup_ms=args.warmup_ms, rep_ms=args.rep_ms))
                    row["status"] = "ok"
                except torch.cuda.OutOfMemoryError:
                    row["status"] = "OOM"
                finally:
                    del q, k, v, grad_output, attention
                    gc.collect()
                    if args.device.type == "cuda":
                        with torch.cuda.device(args.device):
                            torch.cuda.empty_cache()
                for writer, stream in zip(writers, streams):
                    writer.writerow(row)
                    stream.flush()
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()
