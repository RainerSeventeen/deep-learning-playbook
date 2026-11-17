"""比较 activation checkpoint 对 LLM 反向时间和激活显存的影响。"""

from dataclasses import dataclass
from pathlib import Path
import statistics
import timeit

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import yaml

from cs336_basics.model import BasicsTransformerLM


model_type = "medium"
vocab_size = 10_000
batch_size = 4
sequence_length = 512
checkpoint_counts = (2, 3, 4, 5, 6, 8, 12, 24)
warmup_steps = 3
measure_steps = 10


def load_model_config(name: str) -> dict[str, int]:
    config_path = Path(__file__).with_name("model_config.yaml")
    with config_path.open(encoding="utf-8") as f:
        model_configs = yaml.safe_load(f)

    if name not in model_configs:
        choices = ", ".join(model_configs)
        raise ValueError(f"Unknown model type {name!r}; choices: {choices}")
    return model_configs[name]


def checkpoint_segment_sizes(num_blocks: int, num_checkpoints: int) -> list[int]:
    """将 block 均匀划分为指定数量的 checkpoint 分段。"""
    if num_checkpoints <= 0:
        raise ValueError("The checkpoint count must be positive")
    if num_blocks == 0:
        return []
    if num_checkpoints > num_blocks:
        raise ValueError("The checkpoint count cannot exceed the Transformer block count")

    blocks_per_segment, larger_segment_count = divmod(num_blocks, num_checkpoints)
    return [blocks_per_segment + 1] * larger_segment_count + [blocks_per_segment] * (
        num_checkpoints - larger_segment_count
    )


def forward(
    model: BasicsTransformerLM,
    token_ids: torch.Tensor,
    use_checkpoint: bool,
    num_checkpoints: int = 1,
) -> torch.Tensor:
    """将 Transformer block 划分为 k 个分段，每段使用一个 checkpoint。"""
    if not use_checkpoint:
        return model(token_ids)

    x = model.token_embeddings(token_ids)
    block_offset = 0
    for segment_size in checkpoint_segment_sizes(len(model.layers), num_checkpoints):
        layers = model.layers[block_offset : block_offset + segment_size]

        def run_segment(hidden_states: torch.Tensor, layers: nn.ModuleList = layers) -> torch.Tensor:
            for layer in layers:
                hidden_states = layer(hidden_states)
            return hidden_states

        x = checkpoint(run_segment, x, use_reentrant=False)
        block_offset += segment_size
    x = model.ln_final(x)
    return model.lm_head(x)


@dataclass
class SavedTensorStats:
    tensor_count: int = 0
    bytes: int = 0

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        # 叶子张量是模型参数或输入，已经由调用方持有；这里仅统计中间结果。
        if tensor.grad_fn is not None:
            self.tensor_count += 1
            self.bytes += tensor.numel() * tensor.element_size()
        return tensor

    @staticmethod
    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor


@dataclass(frozen=True)
class Result:
    name: str
    duration_seconds: float
    saved_tensor_count: float
    saved_tensor_bytes: float
    peak_activation_bytes: float | None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_step(
    model: BasicsTransformerLM,
    token_ids: torch.Tensor,
    use_checkpoint: bool,
    num_checkpoints: int = 1,
) -> tuple[float, SavedTensorStats, int | None]:
    device = token_ids.device
    model.zero_grad(set_to_none=True)
    synchronize(device)

    baseline_memory = None
    if device.type == "cuda":
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

    stats = SavedTensorStats()
    start = timeit.default_timer()
    with torch.autograd.graph.saved_tensors_hooks(stats.pack, stats.unpack):
        logits = forward(model, token_ids, use_checkpoint=use_checkpoint, num_checkpoints=num_checkpoints)
        loss = logits.float().square().mean()
        loss.backward()
    synchronize(device)

    peak_activation_bytes = None
    if baseline_memory is not None:
        peak_activation_bytes = torch.cuda.max_memory_allocated(device) - baseline_memory
    return timeit.default_timer() - start, stats, peak_activation_bytes


def measure(
    name: str,
    model: BasicsTransformerLM,
    token_ids: torch.Tensor,
    use_checkpoint: bool,
    num_checkpoints: int = 1,
    warmup: int = warmup_steps,
    steps: int = measure_steps,
) -> Result:
    for _ in range(warmup):
        run_step(model, token_ids, use_checkpoint, num_checkpoints)

    durations = []
    saved_tensor_counts = []
    saved_tensor_bytes = []
    peak_activation_bytes = []
    for _ in range(steps):
        duration, stats, peak_bytes = run_step(model, token_ids, use_checkpoint, num_checkpoints)
        durations.append(duration)
        saved_tensor_counts.append(stats.tensor_count)
        saved_tensor_bytes.append(stats.bytes)
        if peak_bytes is not None:
            peak_activation_bytes.append(peak_bytes)

    return Result(
        name=name,
        duration_seconds=statistics.mean(durations),
        saved_tensor_count=statistics.mean(saved_tensor_counts),
        saved_tensor_bytes=statistics.mean(saved_tensor_bytes),
        peak_activation_bytes=statistics.mean(peak_activation_bytes) if peak_activation_bytes else None,
    )


def format_bytes(size: float | None) -> str:
    if size is None:
        return "CUDA only"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def print_result(result: Result) -> None:
    print(f"{result.name}:")
    print(f"  Forward + backward: {result.duration_seconds * 1e3:.2f} ms")
    print(f"  Saved non-leaf tensors (outer hook): {result.saved_tensor_count:.0f}")
    print(f"  Saved non-leaf tensor bytes (outer hook): {format_bytes(result.saved_tensor_bytes)}")
    print(f"  Peak activation memory increase: {format_bytes(result.peak_activation_bytes)}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU to measure peak activation memory")

    device = torch.device("cuda")
    torch.manual_seed(0)
    config = load_model_config(model_type)
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=sequence_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).to(device)
    token_ids = torch.randint(vocab_size, (batch_size, sequence_length), device=device)

    print(
        f"Device: {torch.cuda.get_device_name(device)}, model={model_type}, "
        f"layers={config['num_layers']}, tokens=({batch_size}, {sequence_length})"
    )
    print_result(measure("No checkpoint", model, token_ids, use_checkpoint=False))
    for k in checkpoint_counts:
        segment_sizes = checkpoint_segment_sizes(config["num_layers"], k)
        print(f"Checkpoint configuration: k={k}, block groups={segment_sizes}")
        print_result(
            measure(
                f"Activation checkpoint (k={k})",
                model,
                token_ids,
                use_checkpoint=True,
                num_checkpoints=k,
            )
        )


if __name__ == "__main__":
    main()
