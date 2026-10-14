import gc
import statistics
import timeit

import torch

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.benchmark import render_duration


batch_size = 8

d_model_list = [16, 32, 64, 128]
seq_len_list = [256, 1024, 4096, 8192, 16384]

def benchmark_inputs(q, k, v, step, warmup_steps, compile):
    attention = torch.compile(scaled_dot_product_attention) if compile else scaled_dot_product_attention

    device = q.device
    grad_output = torch.randn_like(v)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup_steps):
        q.grad = k.grad = v.grad = None
        output = attention(q, k, v)
        synchronize()
        output.backward(grad_output)
        synchronize()
        del output

    q.grad = k.grad = v.grad = None
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings = {"forward": [], "backward": []}
    memory_before_backward = None
    for index in range(step):
        q.grad = k.grad = v.grad = None
        synchronize()
        start = timeit.default_timer()
        output = attention(q, k, v)
        synchronize()
        timings["forward"].append(timeit.default_timer() - start)

        if index == 0 and device.type == "cuda":
            memory_before_backward = torch.cuda.memory_allocated(device)
            print(f"CUDA memory before backward: {memory_before_backward / 1024**2:.2f} MiB")

        start = timeit.default_timer()
        output.backward(grad_output)
        synchronize()
        timings["backward"].append(timeit.default_timer() - start)
        del output

    summary = {}
    for name, values in timings.items():
        summary[name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        print(f"{name}: mean={render_duration(summary[name]['mean'])}, "
              f"std={render_duration(summary[name]['std'])} ({step} passes)")
    summary["memory_before_backward_bytes"] = memory_before_backward
    if device.type == "cuda":
        summary["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        print(f"peak CUDA memory: allocated={summary['peak_memory_allocated_bytes'] / 1024**2:.2f} MiB")
    return summary


def benchmark_attention(
    device="cuda",
    dtype=torch.float32,
    step=100,
    warmup_steps=5,
    d_models=None,
    seq_lens=None,
    batch_size=batch_size,
    compile=False,
):
    """使用 (batch_size, seq_len, d_model) 的随机 QKV 测量注意力。"""
    if step <= 0 or warmup_steps < 0:
        raise ValueError("step 必须大于 0，warmup_steps 不能小于 0")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；可传 device='cpu' 检查小规模配置")
    results = {}
    for d_model in d_model_list if d_models is None else d_models:
        for seq_len in seq_len_list if seq_lens is None else seq_lens:
            print(f"\nd_model={d_model}, seq_len={seq_len}, batch_size={batch_size}, "
                  f"device={device}, dtype={dtype}, compile={compile}")
            q = k = v = None
            try:
                torch.manual_seed(0)
                shape = (batch_size, seq_len, d_model)
                q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                results[d_model, seq_len] = benchmark_inputs(
                    q, k, v, step=step, warmup_steps=warmup_steps, compile=compile
                )
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA OOM: skipping d_model={d_model}, seq_len={seq_len}")
            finally:
                del q, k, v
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    eager_results = benchmark_attention()
    compiled_results = benchmark_attention(compile=True)
    print("\nd_model  seq_len  eager forward  compiled forward  eager backward  compiled backward  forward speedup  backward speedup")
    for d_model in d_model_list:
        for seq_len in seq_len_list:
            key = (d_model, seq_len)
            if key not in eager_results or key not in compiled_results:
                print(f"{d_model:>7}  {seq_len:>7}  OOM or unavailable")
                continue
            eager = eager_results[key]
            compiled = compiled_results[key]
            print(
                f"{d_model:>7}  {seq_len:>7}  "
                f"{render_duration(eager['forward']['mean']):>13}  "
                f"{render_duration(compiled['forward']['mean']):>16}  "
                f"{render_duration(eager['backward']['mean']):>14}  "
                f"{render_duration(compiled['backward']['mean']):>17}  "
                f"{eager['forward']['mean'] / compiled['forward']['mean']:>14.2f}x  "
                f"{eager['backward']['mean'] / compiled['backward']['mean']:>15.2f}x"
            )

