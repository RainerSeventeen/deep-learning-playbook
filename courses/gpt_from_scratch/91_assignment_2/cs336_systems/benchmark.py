from contextlib import nullcontext
from pathlib import Path
import statistics
import timeit

import torch.cuda.nvtx as nvtx
import yaml
from einops import einsum
import torch

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW


vocab_size = 10_000
context_length = 512
batch_size = 4
model_types_to_benchmark = ("small", "medium",)
# model_types_to_benchmark = ("small", "medium", "large", "xl", "10B")

benchmark_mode = "full"


def nvtx_range(message, device):
    return nvtx.range(message) if torch.device(device).type == "cuda" else nullcontext()


def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """给朴素注意力的主要计算段添加 Nsight 可筛选的 NVTX 标记。"""
    with nvtx_range("scaled_dot_product_attention", Q.device):
        with nvtx_range("attention_scores", Q.device):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / K.shape[-1] ** 0.5

        if mask is not None:
            with nvtx_range("attention_mask", Q.device):
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx_range("attention_softmax", Q.device):
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx_range("attention_output", Q.device):
            return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")


def model_benchmark(model, step, device, warmup_steps=5, opti=None, mode="full"):
    """构造随机数据，测量前向、反向和参数更新的端到端耗时。"""
    if step <= 0:
        raise ValueError("step 必须大于 0")
    if opti is None:
        raise ValueError("空 opti")
    if mode not in {"forward", "forward_backward", "full"}:
        raise ValueError(f"未知 mode: {mode}")

    device = torch.device(device)
    # if next(model.parameters()).device != device:
    #     raise ValueError("模型和输入必须位于同一设备")
    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
    )

    def time_operation(operation):
        # CUDA 调用异步执行；两次同步使 default_timer 测得这一操作的端到端时间。
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = timeit.default_timer()
        result = operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return result, timeit.default_timer() - start_time

    # warm up
    for _ in range(warmup_steps):
        opti.zero_grad()
        if mode == "forward":
            with torch.no_grad():
                model(x)
        else:
            logits = model(x)
            loss = cross_entropy(logits, x)
            loss.backward()
            if mode == "full":
                opti.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings = {"forward": []}
    if mode != "forward":
        timings["backward"] = []
    if mode == "full":
        timings["optimizer_step"] = []

    # nsys 的 --nvtx-capture=profile 从此处开始追踪信息，因此预热不会进入报告
    with nvtx_range("profile", device):
        for _ in range(step):
            with nvtx_range("training_step", device):
                opti.zero_grad()
                with nvtx_range("forward", device):
                    if mode == "forward":
                        with torch.no_grad():
                            logits, duration = time_operation(lambda: model(x))
                    else:
                        logits, duration = time_operation(lambda: model(x))
                timings["forward"].append(duration)

                if mode != "forward":
                    with nvtx_range("loss", device):
                        loss, _ = time_operation(lambda: cross_entropy(logits, x))
                    with nvtx_range("backward", device):
                        _, duration = time_operation(loss.backward)
                    timings["backward"].append(duration)

                    if mode == "full":
                        with nvtx_range("optimizer_step", device):
                            _, duration = time_operation(opti.step)
                        timings["optimizer_step"].append(duration)

    summary = {
        name: {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}
        for name, values in timings.items()
    }
    for name, values in summary.items():
        print(
            f"{name}: mean={render_duration(values['mean'])}, "
            f"std={render_duration(values['std'])} ({step} steps)"
        )
    return summary


def render_duration(duration: float) -> str:
    if duration < 1e-3:
        return f"{duration * 1e6:.2f}us"
    if duration < 1:
        return f"{duration * 1e3:.2f}ms"
    return f"{duration:.2f}s"
    

if __name__ == "__main__":
    with Path(__file__).with_name("model_config.yaml").open("r", encoding="utf-8") as f:
        model_configs = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    for model_type in model_types_to_benchmark:
        config = model_configs[model_type]
        print(f"\n{model_type} ({device}, context_length={context_length}, mode={benchmark_mode})")
        gpt_lm = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
        ).to(device)
        opti = AdamW(gpt_lm.parameters())
        model_benchmark(
            model=gpt_lm,
            step=10,
            device=device,
            warmup_steps=5,
            opti=opti,
            mode=benchmark_mode,
        )
        del opti, gpt_lm
        if device == "cuda":
            torch.cuda.empty_cache()
