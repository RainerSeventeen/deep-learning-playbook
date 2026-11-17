from contextlib import nullcontext
from pathlib import Path
import gc
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
model_types_to_benchmark = ("medium",)
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


def model_benchmark(
    model,
    step,
    device,
    warmup_steps=5,
    opti=None,
    mode="full",
    autocast_dtype=None,
    scaler=None,
    memory_profile=False,
    input=None,
    targets=None,
    loss_fn=cross_entropy,
    grad_output=None,
):
    """测量前向、反向和参数更新；自定义输入可配损失函数或上游梯度。"""
    if step <= 0:
        raise ValueError("step 必须大于 0")
    if mode == "full" and opti is None:
        raise ValueError("空 opti")
    if mode not in {"forward", "forward_backward", "full"}:
        raise ValueError(f"未知 mode: {mode}")
    if warmup_steps < 0:
        raise ValueError("warmup_steps 不能小于 0")
    if mode != "forward" and input is not None and grad_output is None and targets is None:
        raise ValueError("自定义 input 的反向测试需要 targets 或 grad_output")
    if mode != "forward" and grad_output is None and loss_fn is None:
        raise ValueError("使用 targets 时需要 loss_fn")

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    x = input if input is not None else torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
    )
    if targets is None and input is None:
        targets = x
    if x.device != device:
        raise ValueError("input 必须位于指定 device")

    def zero_grad():
        if opti is not None:
            opti.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)
        if x.is_leaf and x.requires_grad:
            x.grad = None

    def autocast_context():
        if autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)

    def backward(value):
        if scaler is None:
            value.backward(grad_output)
        else:
            scaler.scale(value).backward(grad_output)

    def backward_value(output):
        return output if grad_output is not None else loss_fn(output, targets)

    def optimizer_step():
        if scaler is None:
            opti.step()
        else:
            scaler.step(opti)
            scaler.update()

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
        zero_grad()
        if mode == "forward":
            with torch.no_grad():
                with autocast_context():
                    model(x)
        else:
            with autocast_context():
                logits = model(x)
                loss = backward_value(logits)
            backward(loss)
            if mode == "full":
                optimizer_step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # memory profile
    if memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    timings = {"forward": []}
    if mode != "forward":
        timings["backward"] = []
    if mode == "full":
        timings["optimizer_step"] = []

    # nsys 的 --nvtx-capture=profile 从此处开始追踪信息，因此预热不会进入报告
    with nvtx_range("profile", device):
        for _ in range(step):
            with nvtx_range("training_step", device):
                zero_grad()
                with nvtx_range("forward", device):
                    if mode == "forward":
                        with torch.no_grad():
                            with autocast_context():
                                logits, duration = time_operation(lambda: model(x))
                    else:
                        with autocast_context():
                            logits, duration = time_operation(lambda: model(x))
                timings["forward"].append(duration)

                if mode != "forward":
                    with nvtx_range("loss", device):
                        with autocast_context():
                            loss, _ = time_operation(lambda: backward_value(logits))
                    with nvtx_range("backward", device):
                        _, duration = time_operation(lambda: backward(loss))
                    timings["backward"].append(duration)

                    if mode == "full":
                        with nvtx_range("optimizer_step", device):
                            _, duration = time_operation(optimizer_step)
                        timings["optimizer_step"].append(duration)

    if memory_profile:
        torch.cuda.memory._dump_snapshot("profiles/memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

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
        results = {}
        for compiled in (False, True):
            torch.manual_seed(0)
            label = "compiled" if compiled else "uncompiled"
            print(f"\n{model_type} ({label}, {device}, context_length={context_length}, mode={benchmark_mode})")
            gpt_lm = BasicsTransformerLM(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=config["d_model"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                d_ff=config["d_ff"],
            ).to(device)
            if compiled:
                gpt_lm = torch.compile(gpt_lm)
            opti = AdamW(gpt_lm.parameters())
            results[label] = model_benchmark(
                model=gpt_lm,
                step=10,
                device=device,
                warmup_steps=5,
                opti=opti,
                mode=benchmark_mode,
            )
            del opti, gpt_lm
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        for name in results["uncompiled"]:
            eager = results["uncompiled"][name]["mean"]
            compiled = results["compiled"][name]["mean"]
            print(f"{name}: {render_duration(eager)} -> {render_duration(compiled)} ({eager / compiled:.2f}x)")
        if benchmark_mode == "full":
            eager = sum(result["mean"] for result in results["uncompiled"].values())
            compiled = sum(result["mean"] for result in results["compiled"].values())
            print(f"forward+backward+optimizer_step: {render_duration(eager)} -> {render_duration(compiled)} ({eager / compiled:.2f}x)")
