from dataclasses import dataclass

import torch
from torch import nn

from cs336_systems.benchmark import batch_size, context_length, model_benchmark, vocab_size


class ToyModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, hidden_features: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, hidden_features, bias=False)
        self.ln = nn.LayerNorm(hidden_features)
        self.output = nn.Linear(hidden_features, vocab_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.ln(x)
        return self.output(x)


@dataclass(frozen=True)
class PrecisionConfig:
    name: str
    autocast_dtype: torch.dtype | None


def make_benchmark_components(device: torch.device, config: PrecisionConfig):
    """为一种精度配置创建模型、优化器和梯度缩放器。"""
    torch.manual_seed(0)
    model = ToyModel(vocab_size, d_model=2048, hidden_features=4096).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler(device.type) if config.autocast_dtype == torch.float16 else None

    return model, optimizer, scaler


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("该基准需要 CUDA GPU")

    device = torch.device("cuda")
    configs = [
        PrecisionConfig("fp32", None),
        PrecisionConfig("autocast-fp16", torch.float16),
    ]
    if torch.cuda.is_bf16_supported():
        configs.append(PrecisionConfig("autocast-bf16", torch.bfloat16))

    print(f"device: {torch.cuda.get_device_name(device)}")
    print(
        "model: token embedding -> 2048 -> 4096 -> vocabulary projection, "
        f"batch_size={batch_size}, context_length={context_length}"
    )
    for config in configs:
        print(f"\n{config.name}")
        mode = "full"
        print(f"  mode={mode}")
        model, optimizer, scaler = make_benchmark_components(device, config)
        model_benchmark(
            model=model,
            step=50,
            device=device,
            warmup_steps=10,
            opti=optimizer,
            mode=mode,
            autocast_dtype=config.autocast_dtype,
            scaler=scaler,
        )


if __name__ == "__main__":
    main()
