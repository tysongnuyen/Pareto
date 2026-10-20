"""Training-only counterfactual credit; no auction or reward rules live here.

COMA baseline: Foerster et al. (2018), Eq. 4.
Finite action expectation: Ciosek & Whiteson (2018), Expected Policy Gradients.
The clipped all-action objective is an engineering combination with PPO,
not an implementation of unmodified COMA or vanilla MAPPO.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


def centered_action_values(values: Tensor, probabilities: Tensor) -> Tensor:
    """Preserve exact zero credit when every action has the same outcome."""
    relative = values - values[0]
    return relative - (probabilities * relative).sum()


def expected_clipped_loss(
    probabilities: Tensor, old_probabilities: Tensor, advantages: Tensor,
    clip_epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Integrate over *all* valid own actions, holding other bidders fixed."""
    old = old_probabilities.detach()
    advantage = advantages.detach()
    # Algebraically multiply the ratio by old probability first. This avoids
    # 0*inf for saturated float32 policies, without changing the PPO objective
    # on its behavior support.
    clipped_mass = torch.maximum(old * (1.0 - clip_epsilon),
                                 torch.minimum(probabilities, old * (1.0 + clip_epsilon)))
    surrogate = torch.minimum(probabilities * advantage, clipped_mass * advantage)
    support = old > 0
    clipped = (probabilities < old * (1.0 - clip_epsilon)) | (probabilities > old * (1.0 + clip_epsilon))
    return -surrogate[support].sum(), (old * clipped).sum()


def categorical_entropy(probabilities: Tensor) -> Tensor:
    return -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()


def parameter_vector(module: nn.Module) -> Tensor:
    return torch.cat([p.detach().flatten() for p in module.parameters()]).clone()


def gradient_norm(module: nn.Module) -> float:
    squared = [p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(squared).sum().sqrt()) if squared else 0.0


def stack_states(states: list[dict[str, Tensor]], device: torch.device) -> dict[str, Tensor]:
    return {key: torch.stack([state[key] for state in states]).to(device) for key in states[0]}
