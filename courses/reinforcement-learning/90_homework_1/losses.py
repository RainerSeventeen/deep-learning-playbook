"""Loss functions for imitation learning.

Each loss function takes the model/policy, a batch of states, and a batch of
expert actions, and returns a scalar loss. This signature keeps the training
loop in main.py generic across all methods.

Structure:
    TODO (students implement):
        - mse_loss (Problem 1): MSE regression loss for behavior cloning.
        - flow_matching_loss (Problem 3): MSE loss between predicted and
          target velocity. Compare with diffusion_loss for the pattern.
"""

import torch
import torch.nn as nn


def mse_loss(policy, s_batch: torch.Tensor,
             a_batch: torch.Tensor) -> torch.Tensor:
    """Compute the MSE regression loss for behavior cloning.

    Args:
        policy: BCPolicy network (callable: s_batch -> predicted actions).
        s_batch: states, shape (B, state_dim).
        a_batch: expert actions, shape (B, action_dim).

    Returns:
        Scalar MSE loss (mean over batch and action dimensions).
    """
    # 手写 MSE 不用 API
    predict_action = policy(s_batch)
    return ((predict_action - a_batch) ** 2).mean()


def flow_matching_loss(policy, s_batch: torch.Tensor,
                       a_batch: torch.Tensor) -> torch.Tensor:
    """Compute the flow matching loss (MSE on velocity prediction).

    The policy (FlowMatchingPolicy) carries its own schedule.

    Args:
        policy: FlowMatchingPolicy (model + schedule).
        s_batch: states, shape (B, state_dim).
        a_batch: expert actions, shape (B, action_dim).

    Returns:
        Scalar MSE loss (mean over batch and action dimensions).
    """
    schedule = policy.schedule
    model = policy.model

    # 随机抽一个时间点
    tau = torch.rand(a_batch.shape[0], device=a_batch.device)
    
    x_t, velocity = schedule.interpolate(a_batch, tau)
    velocity_pred = model(x_t, s_batch, tau)

    return ((velocity_pred - velocity) ** 2).mean()


