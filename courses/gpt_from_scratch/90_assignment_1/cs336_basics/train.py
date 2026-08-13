import argparse
import logging
import os
import random
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import IO, Any, BinaryIO

import einops
import numpy.typing as npt
import numpy as np
import torch
import yaml

from .config import load_config
from .loss import cross_entropy
from .model import TransformerLM
from .optimizer import AdamW


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    给定一个 1D 的 numpy 输入, 随机抽取 batch 和 context 的数据

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). 
        第一个是 input, 第二个是相关的输出 (其实就是向后移一个 token)
    """
    dataset_size = len(dataset)
    # 随机抽取 bs , randint 是左闭右开
    starts = np.random.randint(
        0,
        dataset_size - context_length,
        size=batch_size
    )
    xs = []
    ys = []
    for i in range(batch_size):
        idx = starts[i]
        x = dataset[idx : idx + context_length]
        y = dataset[idx + 1 : idx + context_length + 1]
        xs.append(x)
        ys.append(y)
    
    # 沿着外层维度构造 array, 注意到这里内层长度必须一致
    xs, ys = np.stack(xs), np.stack(ys)
    xs = torch.tensor(xs, device=device)
    ys = torch.tensor(ys, device=device)
    return xs, ys


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    state = torch.load(src)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    iteration = state["iteration"]
    return iteration


def train_loop(
    config: Mapping[str, object],
    model: torch.nn.Module,
    opti: torch.optim.Optimizer,
    logger: logging.Logger | None = None,
) -> int:
    """
    核心 train 循环体
    curr_i 表示当前步数, 允许从中间继续训练
    """
    total_i = int(config["total_i"])
    curr_i = int(config.get("curr_i", 0))
    log_freq = int(config.get("log_freq", 1))
    if total_i < curr_i:
        raise ValueError("total_i must be greater than or equal to curr_i")
    if log_freq <= 0:
        raise ValueError("log_freq must be positive")

    data_path = config["data_path"]
    dataset = np.load(data_path)

    bs = int(config["batch_size"])
    ctx_len = int(config["context_len"])
    device = str(config.get("device", "cpu"))

    model.to(device)
    model.train()

    while curr_i < total_i:
        x, y = get_batch(dataset, bs, ctx_len, device)
        logits = model(x)
        # 计算 loss 全部展平到 1D 即可
        loss = cross_entropy(
            einops.rearrange(logits, "B T V -> (B T) V"),
            einops.rearrange(y, "B T -> (B T)"),
        )
        opti.zero_grad()
        loss.backward()
        opti.step()
        curr_i += 1

        if curr_i % log_freq == 0:
            message = f"Iter [{curr_i:4}] loss: {loss.item():.6f}"
            if logger is None:
                print(message)
            else:
                logger.info(message)

    return curr_i


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config: Mapping[str, object], device: str) -> TransformerLM:
    model = TransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        num_layers=int(config["num_layers"]),
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        theta=float(config.get("theta", 10_000.0)),
    )
    model.set_weights()
    return model.to(device)


def create_optimizer(
    config: Mapping[str, object], model: torch.nn.Module
) -> torch.optim.Optimizer:
    name = str(config.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"unsupported optimizer: {name}")

    betas = config.get("betas", (0.9, 0.99))
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("optimizer.betas must contain two values")

    return AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.1)),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(config.get("eps", 1e-8)),
    )


def create_logger(config: Mapping[str, object] | None = None) -> logging.Logger:
    config = config or {}
    logger = logging.getLogger("cs336.train")
    logger.setLevel(getattr(logging, str(config.get("level", "INFO")).upper()))
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if bool(config.get("console", True)):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    log_file = config.get("file")
    if log_file:
        log_path = Path(str(log_file))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def create_run_directory(output_dir: str | os.PathLike) -> Path:
    """Create a unique timestamp-named directory for one training run."""
    output_root = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}-{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def output_file_name(value: object | None, default: str) -> str:
    """Keep configured output files inside the current run directory."""
    if value is None:
        return default
    return Path(str(value)).name


def run_training(config: Mapping[str, Any]) -> int:
    """Create all training components and execute the training loop."""
    training_config = dict(config["training"])
    training_config.setdefault("context_len", config["model"]["context_length"])
    device = str(training_config.get("device", "cpu"))
    set_seed(int(config.get("seed", 42)))

    run_dir = create_run_directory(training_config.get("output_dir", "runs"))
    logging_config = dict(config.get("logging", {}))
    logging_config["file"] = run_dir / output_file_name(
        logging_config.get("file"), "train.log"
    )
    checkpoint_path = run_dir / output_file_name(
        training_config.get("checkpoint_path"), "checkpoint.pt"
    )

    resolved_config = dict(config)
    resolved_config["training"] = {
        **training_config,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
    }
    resolved_config["logging"] = {**logging_config, "file": str(logging_config["file"])}
    with (run_dir / "config.yml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolved_config, file, allow_unicode=True, sort_keys=False)

    logger = create_logger(logging_config)
    logger.info("run directory: %s", run_dir)
    model = create_model(config["model"], device)
    optimizer = create_optimizer(config["optimizer"], model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("device=%s parameters=%s", device, f"{parameter_count:,}")

    # 从 save 中恢复
    resume_from = training_config.get("resume_from")
    if resume_from:
        training_config["curr_i"] = load_checkpoint(resume_from, model, optimizer)
        logger.info("resumed from %s at iteration %d", resume_from, training_config["curr_i"])

    final_iteration = train_loop(training_config, model, optimizer, logger)

    save_checkpoint(model, optimizer, final_iteration, checkpoint_path)
    logger.info("saved checkpoint to %s", checkpoint_path)

    return final_iteration


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CS336 language model")
    parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    args = parser.parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    #  uv run -m cs336_basics.train --config ...
    main()
