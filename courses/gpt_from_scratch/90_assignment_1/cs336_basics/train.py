import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


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