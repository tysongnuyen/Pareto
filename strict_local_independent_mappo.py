"""Two-timescale MAPPO with local bidders and a centralized critic.

This file implements the CTDE baseline agreed for the UAV problem:

* five independent placement actors;
* five independent local autoregressive allocation actors;
* one centralized placement critic;
* one centralized allocation state-value critic for counterfactual continuation;
* Pop-Art normalization for delayed placement returns, with raw-value output
  preservation; allocation retains its fixed per-UE reward scale;
* one environment placement step followed by one current-UE allocation step;
* ``none`` allocation actors choose ``participate -> PRB -> power`` without
  feasibility masks; legacy ``full``/``basic`` use Pareto-determined power;
* a fixed feasibility filter and deterministic minimum-cost auction execute;
* placement and allocation actors share the same served-minus-resource reward;
* allocation actors use local raw features and the shared PRB balance in
  ``none``; legacy modes also expose broadcast coordination summaries;
* exact simulator transitions plus an all-action COMA-style baseline give credit;
* critics use task-specific global state; allocation excludes placement
  candidates after the realized UAV positions are known; in ``none`` only its
  critic additionally sees minimum QoS-valid grid power for each PRB count.

Placement is a special high-level task.  Before a UAV has a position, there is
no meaningful coverage-local observation, so every placement actor receives the
common UE map and candidate set plus its own ID and resource limits.  During
allocation these common placement fields are zeroed by the environment.

The primary frozen-placement schedule first warms up allocation on one batch
without updating placement.  Later batches score sampled placements with the
allocator state available at batch entry, update placement from those aligned
zero-shot returns, and then adapt/restore the best allocation snapshot for the
next batch while joint training is active.  The final placement-only phase keeps
that restored allocator completely frozen.
``train_one_episode`` remains available for single-episode experiments.
"""

from __future__ import annotations

import copy
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F
from learning_credit import (
    centered_action_values, expected_clipped_loss, categorical_entropy,
    parameter_vector, gradient_norm, stack_states,
)


TensorDict = dict[str, Tensor]

#Bien doi data structure thanh tensor
def _nested_state_equal(left: Any, right: Any) -> bool:
    """Compare copied model/optimizer state without tensor truth ambiguity."""

    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_state_equal(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _copy_tensor(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Copy environment data so a later ``env.step`` cannot mutate a rollout."""

    return torch.as_tensor(
        np.array(value, copy=True),
        dtype=dtype,
        device=device,
    )

#Cai nay trong phien ban su dung pareto mask de danh dau nhung invalid actions bi vo hieu
def _masked_distribution(logits: Tensor, action_mask: Tensor) -> Categorical:
    """Create a categorical policy assigning exactly zero mass to invalid actions."""

    mask = action_mask.to(dtype=torch.bool, device=logits.device)
    if logits.shape != mask.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} does not match mask shape "
            f"{tuple(mask.shape)}"
        )
    if not bool(mask.any()):
        raise ValueError("At least one action must be valid")
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return Categorical(logits=masked_logits)

# Chon ra hanh vi
def _select_action(distribution: Categorical, deterministic: bool) -> Tensor:
    """Sample during training and take the categorical mode during evaluation."""
    #Chon thang co xac suat cao nhat
    if deterministic:
        return distribution.probs.argmax(dim=-1)
    # Neu khong thi chon ngau nhien
    return distribution.sample()

#Dam bao rang toan bo agent deu cho action ke ca la tu choi cung phai tra ve (0,0,0) hoac (0,x,y)) do phan tu thu 0 bieu thu la khong cap phat r
def _validated_joint_action(actions: np.ndarray, num_uavs: int) -> np.ndarray:
    """Require one integer action from every actor before an environment step."""

    joint_action = np.asarray(actions)
    if joint_action.shape != (num_uavs,):
        raise RuntimeError(
            f"Expected one action from each of {num_uavs} actors, "
            f"got shape {joint_action.shape}"
        )
    if not np.issubdtype(joint_action.dtype, np.integer):
        raise RuntimeError("Joint placement actions must have an integer dtype")
    return joint_action.astype(np.int64, copy=False)

#Tinh reward cho ca team
def _split_team_rewards(
    environment_reward: float,
    reward_breakdown: dict[str, Any],
) -> tuple[float, float]:
    """Return allocation and placement team rewards from one environment step.

    ``ver2.py`` stores costs as negative reward components. Placement and
    allocation both learn exactly the environment's served-minus-resource
    objective; simultaneous proposals add no separate reward term.
    """

    required_components = (
        "served_reward",
        "power_cost",
        "prb_cost",
    )
    missing = [
        component
        for component in required_components
        if component not in reward_breakdown
    ]
    if missing:
        raise RuntimeError(f"Reward breakdown is missing components: {missing}")

    shared_reward = sum(
        float(reward_breakdown[component])
        for component in required_components
    )
    if not np.isclose(
        shared_reward,
        float(environment_reward),
        rtol=1e-7,
        atol=1e-9,
    ):
        raise RuntimeError(
            "Environment reward does not match its served/resource breakdown"
        )
    #Tra ve cho COMA va moi truong do phai khai bao 2 bien o 2 dong
    return float(shared_reward), float(shared_reward)


class SetAttentionEncoder(nn.Module):
    """Encode an unordered set and return token embeddings and mean context."""
    #Tat ca dung chung attention encoder nay, co input dim
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Tanh(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.Tanh(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Accept (..., tokens, features), with an optional batch axis."""

        if features.ndim not in (2, 3) or features.shape[-2] == 0:
            raise ValueError("SetAttentionEncoder expects non-empty 2-D/3-D input")
        unbatched = features.ndim == 2
        tokens = self.input_projection(features)
        if unbatched:
            tokens = tokens.unsqueeze(0)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.feed_forward(tokens))
        if unbatched:
            tokens = tokens.squeeze(0)
        return tokens, tokens.mean(dim=-2)


class MaskedLocalSetEncoder(nn.Module):
    """Encode only the UE rows visible to one UAV."""
    #Cai nay cho Allocation local
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.encoder = SetAttentionEncoder(input_dim + 1, embed_dim, num_heads)

    def forward(self, features: Tensor, local_mask: Tensor) -> Tensor:
        mask = local_mask.to(dtype=torch.bool, device=features.device)
        if mask.shape != features.shape[:1]:
            raise ValueError("local_ue_mask does not match local_ue_features")
        if not bool(mask.any()):
            return torch.zeros(
                self.embed_dim,
                dtype=features.dtype,
                device=features.device,
            )
        order = torch.linspace(0, 1, features.shape[0], device=features.device).unsqueeze(-1)
        _, context = self.encoder(torch.cat((features, order), -1)[mask])
        return context


class PlacementActor(nn.Module):
    """An independent UAV policy scoring all placement candidates.

    Candidate queries attend to the common UE map.  Therefore each candidate
    obtains a different demand context instead of every candidate being scored
    from the same pooled UE vector.
    """

    def __init__(
        self,
        num_candidates: int,
        num_uavs: int,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.num_candidates = int(num_candidates)
        #ID UE, Vi tri UE, QoS
        self.ue_encoder = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )
        #X,y,h,theta, radius placement
        self.candidate_encoder = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )
        #Encode phan placement
        self.candidate_to_ue_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.candidate_norm = nn.LayerNorm(embed_dim)
        # Bind each committed coordinate to its owning UAV. Allocation actors
        # are independent, so the same set of positions with swapped owners is
        # not generally the same downstream state.
        self.previous_encoder = nn.Sequential(
            nn.Linear(5 + num_uavs, embed_dim),
            nn.Tanh(),
        )
        self.relative_encoder = nn.Sequential(nn.Linear(4, embed_dim), nn.Tanh())
        self.own_uav_encoder = nn.Sequential(
            nn.Linear(2 + num_uavs, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, observation: TensorDict) -> Tensor:
        ue_features = observation["ue_features"]
        ue_order = torch.linspace(
            0,
            1,
            ue_features.shape[0],
            dtype=ue_features.dtype,
            device=ue_features.device,
        ).unsqueeze(-1)
        ue_tokens = self.ue_encoder(
            torch.cat((ue_features, ue_order), dim=-1)
        ).unsqueeze(0)
        candidate_tokens = self.candidate_encoder(
            observation["candidate_features"]
        ).unsqueeze(0)
        candidate_context, _ = self.candidate_to_ue_attention(
            candidate_tokens,
            ue_tokens,
            ue_tokens,
            need_weights=False,
        )
        candidate_context = self.candidate_norm(
            candidate_tokens + candidate_context
        ).squeeze(0)
        # Explicit relative geometry and earlier commitments use existing map
        # and placement actions only; they do not change the candidate mask.
        candidates = observation["candidate_features"]
        ues = observation["ue_features"]
        relative_xy = ues[None, :, :2] - candidates[:, None, :2]
        distance_squared = relative_xy.square().sum(-1)
        # Multi-scale demand density; pool before the MLP to avoid retaining
        # candidates x UEs x embedding activations for every placement actor.
        relative = torch.stack((distance_squared.mean(1), *[
            (torch.exp(-distance_squared / scale) * ues[None, :, 2]).mean(1)
            for scale in (0.01, 0.05, 0.2)]), -1)
        candidate_context = candidate_context + self.relative_encoder(relative)
        previous = observation.get("previous_placements")
        if previous is not None and previous.shape[0]:
            candidate_context = candidate_context + self.previous_encoder(previous).mean(0)

        own_context = self.own_uav_encoder(
            torch.cat(
                (
                    observation["own_uav_limits"],
                    observation["agent_id_one_hot"],
                ),
                dim=-1,
            )
        )
        repeated_own_context = own_context.unsqueeze(0).expand(
            self.num_candidates,
            -1,
        )
        logits = self.score_head(
            torch.cat((candidate_context, repeated_own_context), dim=-1)
        ).squeeze(-1)
        if logits.shape != (self.num_candidates,):
            raise ValueError("PlacementActor produced an invalid logit shape")
        return logits


class GlobalUAVUERelationEncoder(nn.Module):
    """Encode global UAV--UE links without losing which UAV owns each link."""

    def __init__(
        self, embed_dim: int, num_heads: int, num_uavs: int = 5,
        *, link_feature_dim: int = 3,
    ) -> None:
        super().__init__()
        # UAV token (9 + ID), ordered UE token (10), and link/cost features.
        self.edge_encoder = nn.Sequential(
            nn.Linear(19 + num_uavs + link_feature_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )
        self.edge_score = nn.Linear(embed_dim, 1)
        self.uav_set_encoder = SetAttentionEncoder(
            embed_dim,
            embed_dim,
            num_heads,
        )

    def forward(
        self,
        uav_states: Tensor,
        ue_states: Tensor,
        link_features: Tensor,
    ) -> Tensor:
        num_uavs, num_ues = link_features.shape[-3:-1]
        uav_grid = uav_states.unsqueeze(-2).expand(*uav_states.shape[:-1], num_ues, -1)
        ue_grid = ue_states.unsqueeze(-3).expand(*ue_states.shape[:-2], num_uavs, num_ues, -1)
        edges = self.edge_encoder(
            torch.cat((uav_grid, ue_grid, link_features), dim=-1)
        )
        weights = F.softmax(self.edge_score(edges).squeeze(-1), dim=-1)
        per_uav_relations = torch.sum(weights.unsqueeze(-1) * edges, dim=-2)
        _, relation_context = self.uav_set_encoder(per_uav_relations)
        return relation_context


class PopArtLinear(nn.Linear):
    """Scalar Pop-Art output preserving raw values as target scale changes."""
    #Scale lai de tranh bien doi qua nhieu mot ky thuat fine tune lam on dinh model
    def __init__(
        self,
        input_dim: int,
        *,
        beta: float = 1e-4,
        min_std: float = 1e-4,
    ) -> None:
        if not 0.0 < beta <= 1.0:
            raise ValueError("Pop-Art beta must be in (0, 1]")
        if min_std <= 0.0:
            raise ValueError("Pop-Art min_std must be positive")
        super().__init__(input_dim, 1)
        self.beta = float(beta)
        self.min_std = float(min_std)
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("std", torch.ones(()))
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.long))

    def normalized_forward(self, features: Tensor) -> Tensor:
        """Return the normalized prediction learned by the linear layer."""

        return super().forward(features)

    def forward(self, features: Tensor) -> Tensor:
        """Return the value in the original, unnormalized target units."""

        return self.std * self.normalized_forward(features) + self.mean

    def normalize_targets(self, targets: Tensor) -> Tensor:
        return (targets - self.mean) / self.std

    @torch.no_grad()
    def update_statistics(self, targets: Tensor) -> None:
        """Update target moments and compensate ``weight``/``bias`` exactly."""

        flat_targets = targets.detach().to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        ).reshape(-1)
        if flat_targets.numel() == 0:
            raise ValueError("Pop-Art needs at least one target")
        if not bool(torch.isfinite(flat_targets).all()):
            raise ValueError("Pop-Art targets must be finite")

        batch_mean = flat_targets.mean()
        batch_variance = flat_targets.var(unbiased=False)
        old_mean = self.mean.clone()
        old_std = self.std.clone()
        batch_size = int(flat_targets.numel())
        new_count = int(self.sample_count.item()) + batch_size
        if self.beta == 1.0:
            batch_weight = 1.0
        else:
            # ``beta`` is a per-target EMA step.  Do not bias-correct this
            # weight: the initial unit variance is an intentional numerical
            # prior, especially when convergence gating leaves a one-sample
            # final placement batch.
            batch_weight = -math.expm1(
                batch_size * math.log1p(-self.beta)
            )

        new_mean = (1.0 - batch_weight) * old_mean + batch_weight * batch_mean
        new_variance = (
            (1.0 - batch_weight)
            * (old_std.square() + (old_mean - new_mean).square())
            + batch_weight
            * (batch_variance + (batch_mean - new_mean).square())
        )
        new_std = new_variance.clamp_min(self.min_std**2).sqrt()

        self.weight.mul_(old_std / new_std)
        self.bias.mul_(old_std).add_(old_mean - new_mean).div_(new_std)
        self.mean.copy_(new_mean)
        self.std.copy_(new_std)
        self.sample_count.fill_(new_count)


class CentralizedGlobalCritic(nn.Module):
    """Estimate one cooperative team value from the complete global state."""
    #Critic
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        include_candidate_features: bool = True,
        popart_beta: float | None = None,
        num_uavs: int = 5,
        allocation_cost_dim: int = 0,
    ) -> None:
        super().__init__()
        self.include_candidate_features = bool(include_candidate_features)
        self.allocation_cost_dim = int(allocation_cost_dim)
        # UE: xyz/demand (3), processed, served, PRB ratio, power ratio,
        # achieved-rate ratio, current mask = 9.
        self.num_uavs = num_uavs
        self.ue_encoder = SetAttentionEncoder(10, embed_dim, num_heads)
        # UAV: own dynamic state (7) and fixed normalized limits (2) = 9.
        self.uav_encoder = SetAttentionEncoder(9 + num_uavs, embed_dim, num_heads)
        self.candidate_encoder = (
            nn.Sequential(nn.Linear(5, embed_dim), nn.Tanh(), nn.Linear(embed_dim, embed_dim))
            if self.include_candidate_features
            else None
        )
        self.relation_encoder = GlobalUAVUERelationEncoder(
            embed_dim,
            num_heads,
            num_uavs,
            link_feature_dim=3 + self.allocation_cost_dim,
        )
        self.phase_embedding = nn.Embedding(3, embed_dim)
        context_count = 5 if self.include_candidate_features else 4
        self.context_dim = context_count * embed_dim
        output_layer: nn.Module = (
            PopArtLinear(embed_dim, beta=popart_beta)
            if popart_beta is not None
            else nn.Linear(embed_dim, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.context_dim, 2 * embed_dim),
            nn.Tanh(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Tanh(),
            output_layer,
        )

    def _context(self, state: TensorDict) -> Tensor:
        ue_states = state["ue_state_features"]
        # Row order is the processing schedule already present in the state.
        order = torch.linspace(0, 1, ue_states.shape[-2], device=ue_states.device)
        order = order.reshape(*([1] * (ue_states.ndim - 2)), -1, 1).expand(*ue_states.shape[:-1], 1)
        ue_states = torch.cat((ue_states, order), dim=-1)
        ids = torch.eye(self.num_uavs, device=ue_states.device)
        ids = ids.expand(*state["uav_features"].shape[:-2], -1, -1)
        uav_states = torch.cat(
            (state["uav_features"], state["uav_limits"], ids),
            dim=-1,
        )
        _, ue_context = self.ue_encoder(ue_states)
        _, uav_context = self.uav_encoder(uav_states)
        contexts = [ue_context, uav_context]
        if self.include_candidate_features:
            if self.candidate_encoder is None:
                raise RuntimeError("Placement candidate encoder is unavailable")
            candidate_context = self.candidate_encoder(
                state["candidate_features"]
            ).mean(dim=-2)
            contexts.append(candidate_context)
        link_features = state["link_features"]
        if self.allocation_cost_dim:
            costs = state["allocation_cost_features"]
            if costs.shape != (*link_features.shape[:-1], self.allocation_cost_dim):
                raise ValueError("Allocation critic cost feature shape does not match links")
            # Table is static and shared between CF branches. Only unprocessed
            # UEs contribute costs, using this branch's own processed flags.
            pending = (1.0 - ue_states[..., 3]).unsqueeze(-2).unsqueeze(-1)
            link_features = torch.cat((link_features, costs * pending), dim=-1)
        relation_context = self.relation_encoder(
            uav_states,
            ue_states,
            link_features,
        )
        phase_context = self.phase_embedding(state["phase"].long())
        contexts.extend((relation_context, phase_context))
        return torch.cat(contexts, dim=-1)

    def forward(self, state: TensorDict) -> Tensor:
        return self.value_head(self._context(state)).squeeze(-1)

    @property
    def popart_head(self) -> PopArtLinear:
        output_layer = self.value_head[-1]
        if not isinstance(output_layer, PopArtLinear):
            raise RuntimeError("This critic does not use Pop-Art")
        return output_layer

    def normalized_value(self, state: TensorDict) -> Tensor:
        features = self.value_head[:-1](self._context(state))
        return self.popart_head.normalized_forward(features).squeeze(-1)

    def update_popart(self, targets: Tensor) -> None:
        self.popart_head.update_statistics(targets)

    def normalize_popart_targets(self, targets: Tensor) -> Tensor:
        return self.popart_head.normalize_targets(targets)


class LocalAutoregressiveAllocationActor(nn.Module):
    """Local bidder with an optional learned power distribution given PRBs."""

    def __init__(
        self,
        num_uavs: int,
        max_prbs: int,
        coordination_feature_dim: int,
        embed_dim: int,
        num_heads: int,
        num_power_levels: int = 0,
    ) -> None:
        super().__init__()
        self.max_prbs = int(max_prbs)
        self.compact_local = num_power_levels > 0  # Learned power is the none policy.
        self.local_ue_encoder = MaskedLocalSetEncoder(
            9, embed_dim, num_heads
        )
        # None: remaining PRB/power (2), current QoS (1), link (3), actor ID.
        # Relative UE geometry stays in the set; fixed limits and duplicate PRB
        # broadcast are unused. Legacy full/basic keep their original inputs.
        self.local_vector_encoder = nn.Sequential(
            nn.Linear(
                (6 + num_uavs if self.compact_local
                 else 15 + num_uavs + coordination_feature_dim),
                embed_dim,
            ),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )
        self.participate_head = nn.Linear(embed_dim, 2)
        self.prb_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, self.max_prbs),
        )
        # Legacy policies keep their original parameter names and shapes.
        self.power_head = (
            nn.Sequential(
                nn.Linear(embed_dim + 1, embed_dim),
                nn.Tanh(),
                nn.Linear(embed_dim, num_power_levels),
            )
            if num_power_levels > 0 else None
        )

    def encode(self, observation: TensorDict) -> Tensor:
        local_context = self.local_ue_encoder(
            observation["local_ue_features"],
            observation["local_ue_mask"],
        )
        if self.compact_local:
            local_features = (
                observation["own_uav_features"][..., 5:7],
                observation["current_ue_features"][..., 2:3],
                observation["current_link_features"],
                observation["agent_id_one_hot"],
            )
        else:
            local_features = (
                observation["own_uav_features"],
                observation["own_uav_limits"],
                observation["current_ue_features"],
                observation["current_link_features"],
                observation["agent_id_one_hot"],
                observation["coordination_features"],
            )
        local_vector = torch.cat(local_features, dim=-1)
        vector_context = self.local_vector_encoder(local_vector)
        context = self.context_encoder(
            torch.cat(
                (local_context, vector_context),
                dim=-1,
            )
        )
        # Keep current QoS/link accessible to every decision head even when
        # mixing the UE-set context suppresses the local branch.
        return context + vector_context if self.compact_local else context

    def participate_logits(self, context: Tensor) -> Tensor:
        return self.participate_head(context)

    def prb_logits(self, context: Tensor) -> Tensor:
        return self.prb_head(context)

    def power_logits(self, context: Tensor, prb_index: Tensor) -> Tensor:
        """Condition power on the chosen PRB, without exposing simulator QoS."""

        if self.power_head is None:
            raise RuntimeError("Legacy allocation policies have no power head")
        prb_fraction = prb_index.to(context).unsqueeze(-1) / max(
            1, self.max_prbs - 1
        )
        return self.power_head(torch.cat((context, prb_fraction), dim=-1))


@dataclass
class AgentDecision:
    """One independent actor transition before a PPO update."""

    observation: TensorDict
    action: Tensor
    action_mask: Tensor
    old_log_prob: Tensor
    prb_mask: Tensor | None = None
    pair_mask: Tensor | None = None
    counterfactual_actions: Tensor | None = None
    counterfactual_probabilities: Tensor | None = None
    counterfactual_outcomes: Tensor | None = None


@dataclass
class PlacementEpisodeRecord:
    """Five placement decisions retained until the episode return is known."""

    initial_global_state: TensorDict
    decisions: list[AgentDecision]
    selection_order: tuple[int, ...]
    old_value: Tensor
    episode_return: float | None = None


@dataclass
class AllocationStepRecord:
    """Five local bidder decisions and one shared global transition."""

    global_state: TensorDict
    decisions: list[AgentDecision]
    joint_action: Tensor
    reward: float | None = None
    done: bool | None = None
    # Unique simulator outcomes, stored on CPU; identical losing bids share one.
    outcome_states: list[TensorDict] | None = None
    outcome_rewards: Tensor | None = None
    outcome_dones: Tensor | None = None
    actual_outcome: int = 0
    winner: int | None = None


class StrictLocalIndependentMAPPO(nn.Module):
    """CTDE with local actors, configurable broadcasts, and global V critics.

    The historical class name is retained for API compatibility. Allocation
    actors do not observe other UAVs' raw state or actions. In ``full`` and
    ``basic`` modes they receive centrally computed broadcast summaries.
    ``none`` exposes only the shared PRB balance in addition to local features.
    """

    REQUIRED_NUM_UAVS = 5
    CHECKPOINT_FRAMEWORK = "mappo_auction_prb_policy_expected_credit_v5"
    CHECKPOINT_SCHEMA_VERSION = 4
    OPTIMIZER_COUNTER_KEYS = (
        "placement_actor",
        "placement_critic",
        "allocation_actor",
        "allocation_critic",
    )
    ALLOCATION_OBSERVATION_SCOPE = (
        "local_raw_plus_broadcast_coordination_summary_v3"
    )

    def __init__(
        self,
        *,
        num_candidates: int,
        max_prbs: int,
        num_power_levels: int,
        coordination_feature_dim: int = 8,
        coordination_mode: str = "none",
        embed_dim: int = 64,
        num_heads: int = 4,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 1.0,
        clip_epsilon: float = 0.2,
        placement_entropy_coef: float = 5e-4,
        allocation_entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        reward_scale: float = 1.0,
        placement_popart_beta: float = 1e-4,
        max_grad_norm: float = 0.5,
        placement_batch_episodes: int = 10,
        placement_minibatch_size: int = 10,
        placement_ppo_epochs: int = 4,
        allocation_ppo_epochs: int = 4,
        allocation_minibatch_size: int = 500,
        target_kl: float = 0.02,
        diagnostics_callback: Callable[[str, dict[str, float]], None] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if max_prbs <= 0 or num_power_levels <= 0:
            raise ValueError("Allocation PRB and power dimensions must be positive")
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be positive")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in (actor_lr, critic_lr)
        ):
            raise ValueError("Actor and critic learning rates must be finite and positive")
        if coordination_mode not in {"full", "basic", "none"}:
            raise ValueError("coordination_mode must be full, basic, or none")
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if num_candidates < self.REQUIRED_NUM_UAVS:
            raise ValueError("Placement needs at least five unique candidates")
        if placement_batch_episodes <= 1:
            raise ValueError("placement_batch_episodes must be greater than one")
        if placement_minibatch_size <= 0:
            raise ValueError("placement_minibatch_size must be positive")
        if placement_minibatch_size > placement_batch_episodes:
            raise ValueError(
                "placement_minibatch_size cannot exceed placement_batch_episodes"
            )
        if placement_ppo_epochs <= 0 or allocation_ppo_epochs <= 0:
            raise ValueError("PPO epoch counts must be positive")
        if allocation_minibatch_size <= 0:
            raise ValueError("allocation_minibatch_size must be positive")
        if float(gamma) != 1.0:
            raise ValueError(
                "gamma must be exactly 1.0 so training and deterministic "
                "selection optimize the same undiscounted team return"
            )
        if not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in (placement_entropy_coef, allocation_entropy_coef)
        ):
            raise ValueError("Entropy coefficients must be non-negative")
        if not np.isfinite(reward_scale) or reward_scale <= 0.0:
            raise ValueError("reward_scale must be finite and positive")
        if not np.isfinite(placement_popart_beta) or not (
            0.0 < placement_popart_beta <= 1.0
        ):
            raise ValueError("placement_popart_beta must be in (0, 1]")

        self.num_uavs = self.REQUIRED_NUM_UAVS
        self.num_candidates = int(num_candidates)
        self.max_prbs = int(max_prbs)
        self.num_power_levels = int(num_power_levels)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.coordination_feature_dim = int(coordination_feature_dim)
        self.coordination_mode = coordination_mode
        self.learned_power = coordination_mode == "none"
        self.allocation_policy = (
            "participate_then_prb_then_conditional_power_coverage_only"
            if self.learned_power else
            "participate_then_prb_with_deterministic_pareto_power"
        )
        self.allocation_entropy_regularization = (
            "detached_branch_conditional_v1"
            if self.learned_power else "joint_entropy_v1"
        )
        if self.learned_power:
            self.CHECKPOINT_FRAMEWORK = "mappo_auction_prb_power_compact_actor_expected_credit_v9"
            self.CHECKPOINT_SCHEMA_VERSION = 8
            self.ALLOCATION_OBSERVATION_SCOPE = "local_ue_set_current_qos_link_shared_prb_own_power_v5"
        self.allocation_actor_feature_version = (
            "compact_qos_link_residual_v1" if self.learned_power else "local_vector_v1"
        )
        self.allocation_critic_feature_version = (
            "global_min_power_by_prb_qos_v1" if self.learned_power else "global_v1"
        )
        self.allocation_action_dims = (
            2,
            self.max_prbs,
            self.num_power_levels,
        )
        self.gamma = 1.0
        self.clip_epsilon = float(clip_epsilon)
        self.placement_entropy_coef = float(placement_entropy_coef)
        self.allocation_entropy_coef = float(allocation_entropy_coef)
        self.value_coef = float(value_coef)
        self.reward_scale = float(reward_scale)
        self.placement_popart_beta = float(placement_popart_beta)
        self.max_grad_norm = float(max_grad_norm)
        self.placement_batch_episodes = int(placement_batch_episodes)
        self.placement_minibatch_size = int(placement_minibatch_size)
        self.placement_ppo_epochs = int(placement_ppo_epochs)
        self.allocation_ppo_epochs = int(allocation_ppo_epochs)
        self.allocation_minibatch_size = int(allocation_minibatch_size)
        self.device = torch.device(device)
        if not np.isfinite(target_kl) or target_kl <= 0:
            raise ValueError("target_kl must be finite and positive")
        self.target_kl = float(target_kl)
        self.diagnostics_callback = diagnostics_callback
        self.loaded_trainer_state: dict[str, Any] | None = None

        # No parameters are shared: every ModuleList entry is a new network.
        self.placement_actors = nn.ModuleList(
            [
                PlacementActor(
                    self.num_candidates,
                    self.num_uavs,
                    embed_dim,
                    num_heads,
                )
                for _ in range(self.num_uavs)
            ]
        )
        self.allocation_actors = nn.ModuleList(
            [
                LocalAutoregressiveAllocationActor(
                    self.num_uavs,
                    self.max_prbs,
                    self.coordination_feature_dim,
                    embed_dim,
                    num_heads,
                    num_power_levels=(
                        self.num_power_levels if self.learned_power else 0
                    ),
                )
                for _ in range(self.num_uavs)
            ]
        )
        # Same architecture, completely different task-specific parameters.
        self.placement_critic = CentralizedGlobalCritic(
            embed_dim,
            num_heads,
            popart_beta=self.placement_popart_beta,
        )
        self.allocation_critic = CentralizedGlobalCritic(
            embed_dim,
            num_heads,
            include_candidate_features=False,
            allocation_cost_dim=2 * self.max_prbs if self.learned_power else 0,
        )
        self.to(self.device)
        # Resolve indexless "cuda" to the actual parameter device (e.g. cuda:0).
        self.device = next(self.parameters()).device

        self.placement_actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=actor_lr)
            for actor in self.placement_actors
        ]
        self.allocation_actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=actor_lr)
            for actor in self.allocation_actors
        ]
        self.placement_critic_optimizer = torch.optim.Adam(
            self.placement_critic.parameters(), lr=critic_lr
        )
        self.allocation_critic_optimizer = torch.optim.Adam(
            self.allocation_critic.parameters(), lr=critic_lr
        )

        self._placement_rollout: list[PlacementEpisodeRecord] = []
        self._allocation_rollout: list[AllocationStepRecord] = []
        # Lifetime work counters deliberately live outside allocation snapshots:
        # restoring the best inner model must not erase optimizer work performed.
        self.optimizer_step_totals = {
            key: 0 for key in self.OPTIMIZER_COUNTER_KEYS
        }

    @classmethod
    def from_env(
        cls,
        environment: Any,
        **kwargs: Any,
    ) -> "StrictLocalIndependentMAPPO":
        """Read candidate and per-UAV action dimensions from ``ver2.py``."""

        if int(environment.num_uavs) != cls.REQUIRED_NUM_UAVS:
            raise ValueError(
                f"Expected 5 UAVs, environment has {environment.num_uavs}"
            )
        kwargs.setdefault("reward_scale", max(1, int(environment.num_ues)))
        environment_mode = str(environment.coordination_mode)
        requested_mode = str(kwargs.get("coordination_mode", environment_mode))
        if requested_mode != environment_mode:
            raise ValueError(
                "Learner coordination_mode must match the environment: "
                f"learner={requested_mode!r}, environment={environment_mode!r}"
            )
        kwargs["coordination_mode"] = environment_mode
        if environment_mode == "none" and (
            not np.array_equal(environment.prb_levels, np.arange(1, 11))
            or len(environment.power_levels_w) != 10
            or not np.allclose(environment.power_levels_w, np.arange(1, 11) / 10)
        ):
            raise ValueError("none allocation requires PRBs 1..10 and power 0.1..1 W")
        return cls(
            num_candidates=int(environment.num_solutions),
            max_prbs=int(environment.max_prbs),
            num_power_levels=int(environment.num_power_levels),
            coordination_feature_dim=int(environment.COORDINATION_FEATURE_DIM),
            **kwargs,
        )

    def _global_state_to_tensors(
        self,
        state: dict[str, Any],
        *,
        include_candidate_features: bool = True,
        include_allocation_cost_features: bool = True,
    ) -> TensorDict:
        """Convert the privileged critic state, including every UAV and UE."""

        ue_state_features = torch.cat(
            (
                _copy_tensor(state["ue_features"], device=self.device),
                _copy_tensor(state["processed"], device=self.device).unsqueeze(-1),
                _copy_tensor(state["served"], device=self.device).unsqueeze(-1),
                _copy_tensor(
                    state["allocated_prbs_ratio"], device=self.device
                ).unsqueeze(-1),
                _copy_tensor(
                    state["allocated_power_ratio"], device=self.device
                ).unsqueeze(-1),
                _copy_tensor(
                    state["achieved_rate_ratio"], device=self.device
                ).unsqueeze(-1),
                _copy_tensor(
                    state["current_ue_mask"], device=self.device
                ).unsqueeze(-1),
            ),
            dim=-1,
        )
        tensors = {
            "phase": _copy_tensor(
                state["phase"],
                device=self.device,
                dtype=torch.long,
            ),
            "ue_state_features": ue_state_features,
            "uav_features": _copy_tensor(
                state["uav_features"], device=self.device
            ),
            "uav_limits": _copy_tensor(
                state["uav_limits"], device=self.device
            ),
            "link_features": _copy_tensor(
                state["link_features"], device=self.device
            ),
        }
        if include_candidate_features:
            tensors["candidate_features"] = _copy_tensor(
                state["candidate_features"], device=self.device
            )
        elif self.learned_power and include_allocation_cost_features:
            tensors["allocation_cost_features"] = _copy_tensor(
                state["allocation_cost_features"], device=self.device
            )
        return tensors

    def _placement_observation_to_tensors(
        self,
        observation: dict[str, Any],
    ) -> TensorDict:
        return {
            "ue_features": _copy_tensor(
                observation["ue_features"], device=self.device
            ),
            "candidate_features": _copy_tensor(
                observation["candidate_features"], device=self.device
            ),
            "own_uav_limits": _copy_tensor(
                observation["own_uav_limits"], device=self.device
            ),
            "agent_id_one_hot": _copy_tensor(
                observation["agent_id_one_hot"], device=self.device
            ),
        }

    def _allocation_observation_to_tensors(
        self, observation: dict[str, Any]
    ) -> TensorDict:
        """Copy only one UAV's decentralized allocation observation."""

        return {
            key: _copy_tensor(observation[key], device=self.device)
            for key in (
                "own_uav_features",
                "own_uav_limits",
                "local_ue_features",
                "local_ue_mask",
                "current_ue_features",
                "current_link_features",
                "agent_id_one_hot",
                "coordination_features",
            )
        }

    @torch.no_grad()
    def act_placement(
        self,
        global_state: dict[str, Any],
        actor_observations: Sequence[dict[str, Any]],
        *,
        deterministic: bool = False,
        store_for_update: bool = True,
        selection_order: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Collect five unique placement actions before the joint step."""

        if len(actor_observations) != self.num_uavs:
            raise ValueError("act_placement requires five observations")
        order = (
            tuple(range(self.num_uavs))
            if selection_order is None
            else tuple(int(index) for index in selection_order)
        )
        if len(order) != self.num_uavs or set(order) != set(
            range(self.num_uavs)
        ):
            raise ValueError(
                "selection_order must be a permutation of all UAV indices"
            )
        actions = np.zeros(self.num_uavs, dtype=np.int64)
        decisions_by_uav: dict[int, AgentDecision] = {}
        available_candidates = torch.ones(
            self.num_candidates,
            dtype=torch.bool,
            device=self.device,
        )
        previous_placements: list[Tensor] = []
        for uav_index in order:
            actor = self.placement_actors[uav_index]
            raw_observation = actor_observations[uav_index]
            observation = self._placement_observation_to_tensors(raw_observation)
            observation["previous_placements"] = (
                torch.stack(previous_placements) if previous_placements
                else torch.empty(
                    (0, 5 + self.num_uavs),
                    device=self.device,
                )
            )
            action_mask = _copy_tensor(
                raw_observation["action_mask"][: self.num_candidates],
                device=self.device,
                dtype=torch.bool,
            ) & available_candidates
            distribution = _masked_distribution(actor(observation), action_mask)
            action = _select_action(distribution, deterministic)
            actions[uav_index] = int(action.item())
            previous_placements.append(
                torch.cat(
                    (
                        observation["candidate_features"][action],
                        observation["agent_id_one_hot"],
                    )
                ).clone()
            )
            selected_xyz = observation["candidate_features"][action, :3]
            same_xyz = torch.isclose(
                observation["candidate_features"][:, :3],
                selected_xyz.unsqueeze(0),
                rtol=0.0,
                atol=1e-7,
            ).all(dim=-1)
            available_candidates &= ~same_xyz
            if store_for_update:
                decisions_by_uav[uav_index] = AgentDecision(
                    observation=observation,
                    action=action.detach(),
                    action_mask=action_mask.detach(),
                    old_log_prob=distribution.log_prob(action).detach(),
                    counterfactual_probabilities=distribution.probs.detach(),
                )

        if store_for_update:
            state = self._global_state_to_tensors(global_state)
            self._placement_rollout.append(
                PlacementEpisodeRecord(
                    initial_global_state=state,
                    decisions=[
                        decisions_by_uav[index]
                        for index in range(self.num_uavs)
                    ],
                    selection_order=order,
                    old_value=self.placement_critic(state).detach(),
                )
            )
        return actions

    def _pareto_prb_power_map(
        self,
        pair_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return valid PRBs and their deterministic Pareto power indices."""

        if pair_mask.shape != (self.max_prbs, self.num_power_levels):
            raise ValueError("Allocation pair mask has the wrong shape")
        if pair_mask.dtype != torch.bool:
            raise TypeError("Allocation pair mask must have boolean dtype")
        model_device = next(self.parameters()).device
        if pair_mask.device != model_device:
            raise ValueError("Allocation pair mask is on the wrong device")
        powers_per_prb = pair_mask.count_nonzero(dim=1)
        ambiguous_prbs = torch.nonzero(
            powers_per_prb > 1, as_tuple=False
        ).flatten()
        if ambiguous_prbs.numel():
            raise RuntimeError(
                "Pareto pair mask must contain exactly one power for each "
                "valid PRB; multiple powers found for PRBs "
                f"{ambiguous_prbs.cpu().tolist()}"
            )
        prb_mask = powers_per_prb == 1
        power_indices = pair_mask.to(dtype=torch.long).argmax(dim=1)
        return prb_mask, power_indices

    def _allocation_action_catalog(
        self,
        actor: LocalAutoregressiveAllocationActor,
        observation: TensorDict,
        participate_mask: Tensor,
        pair_mask: Tensor,
        *,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Enumerate the exact compound-action distribution for COMA/KL."""

        if self.learned_power:
            self._require_coverage_allocation(observation, participate_mask, pair_mask)
            if not bool(participate_mask[1]):
                return (torch.zeros((1, 3), dtype=torch.long, device=self.device),
                        observation["current_link_features"].new_ones(1))
            actor_context = actor.encode(observation) if context is None else context
            participate = _masked_distribution(
                actor.participate_logits(actor_context), participate_mask
            ).probs
            prb = actor.prb_logits(actor_context).softmax(dim=-1)
            prb_indices = torch.arange(self.max_prbs, device=self.device)
            power = actor.power_logits(
                actor_context.expand(self.max_prbs, -1), prb_indices
            ).softmax(dim=-1)
            pairs = torch.nonzero(pair_mask, as_tuple=False)
            actions = torch.cat((
                torch.zeros((1, 3), dtype=torch.long, device=self.device),
                torch.cat((torch.ones_like(pairs[:, :1]), pairs), dim=1),
            ))
            probabilities = torch.cat((
                participate[:1],
                participate[1] * prb[pairs[:, 0]] * power[pairs[:, 0], pairs[:, 1]],
            ))
            return actions, probabilities / probabilities.sum()

        prb_mask, power_indices = self._pareto_prb_power_map(pair_mask)
        actor_context = actor.encode(observation) if context is None else context
        participate_distribution = _masked_distribution(
            actor.participate_logits(actor_context), participate_mask
        )
        actions: list[Tensor] = []
        probabilities: list[Tensor] = []
        if bool(participate_mask[0]):
            actions.append(
                torch.zeros(3, dtype=torch.long, device=self.device)
            )
            probabilities.append(participate_distribution.probs[0])
        if bool(participate_mask[1]):
            prb_distribution = _masked_distribution(
                actor.prb_logits(actor_context), prb_mask
            )
            for prb_index in torch.nonzero(prb_mask, as_tuple=False).flatten():
                power_index = power_indices[prb_index]
                actions.append(
                    torch.stack(
                        (
                            torch.ones(
                                (), dtype=torch.long, device=self.device
                            ),
                            prb_index.long(),
                            power_index,
                        )
                    )
                )
                probabilities.append(
                    participate_distribution.probs[1]
                    * prb_distribution.probs[prb_index]
                )
        if not actions:
            raise RuntimeError("Allocation policy has no valid compound action")
        probability_tensor = torch.stack(probabilities)
        probability_tensor = probability_tensor / probability_tensor.sum()
        return torch.stack(actions), probability_tensor

    def _require_coverage_allocation(
        self, observation: TensorDict, participate_mask: Tensor, pair_mask: Tensor
    ) -> None:
        """Allow only coverage masking, using information visible to this actor."""

        if participate_mask.shape != (2,) or pair_mask.shape != (
            self.max_prbs, self.num_power_levels
        ):
            raise ValueError("Allocation masks have the wrong shape")
        if any(mask.dtype != torch.bool for mask in (participate_mask, pair_mask)):
            raise TypeError("Allocation masks must have boolean dtype")
        model_device = next(self.parameters()).device
        if any(mask.device != model_device for mask in (participate_mask, pair_mask)):
            raise ValueError("Allocation masks are on the wrong device")
        covered = observation["current_link_features"][1] >= 0.5
        if not bool(participate_mask[0] and (participate_mask[1] == covered)
                    and (pair_mask == covered).all()):
            raise RuntimeError(
                "none allocation must expose every action in coverage and only reject outside coverage"
            )

    @torch.no_grad()
    def act_allocation(
        self,
        global_state: dict[str, Any],
        actor_observations: Sequence[dict[str, Any]] | None = None,
        *,
        deterministic: bool = False,
        store_for_update: bool = True,
    ) -> tuple[np.ndarray, AllocationStepRecord | None]:
        """Sample participation, PRBs, and learned or legacy-determined power."""

        if actor_observations is None or len(actor_observations) != self.num_uavs:
            raise ValueError("act_allocation requires five local observations")
        state = self._global_state_to_tensors(
            global_state,
            include_candidate_features=False,
        ) if store_for_update else None
        actions = np.zeros((self.num_uavs, 3), dtype=np.int64)
        decisions: list[AgentDecision] = []
        for uav_index, (actor, raw_observation) in enumerate(
            zip(self.allocation_actors, actor_observations)
        ):
            observation = self._allocation_observation_to_tensors(
                raw_observation
            )
            participate_mask = _copy_tensor(
                raw_observation["allocation_participate_mask"],
                device=self.device,
                dtype=torch.bool,
            )
            prb_mask = _copy_tensor(
                raw_observation["allocation_prb_mask"],
                device=self.device,
                dtype=torch.bool,
            )
            power_mask = _copy_tensor(
                raw_observation["allocation_power_mask"],
                device=self.device,
                dtype=torch.bool,
            )
            pair_mask = _copy_tensor(
                raw_observation["allocation_pair_mask"],
                device=self.device,
                dtype=torch.bool,
            )
            if self.learned_power:
                self._require_coverage_allocation(observation, participate_mask, pair_mask)
                expected_prb_mask = pair_mask.any(dim=1)
                power_indices = None
            else:
                expected_prb_mask, power_indices = self._pareto_prb_power_map(pair_mask)
            if not torch.equal(prb_mask, expected_prb_mask):
                raise RuntimeError("PRB mask is inconsistent with the pair mask")
            if not torch.equal(power_mask, pair_mask.any(dim=0)):
                raise RuntimeError("Power mask is inconsistent with the pair mask")
            context = actor.encode(observation)
            participate_distribution = _masked_distribution(
                actor.participate_logits(context), participate_mask
            )
            participate = _select_action(
                participate_distribution, deterministic
            )
            action = torch.zeros(3, dtype=torch.long, device=self.device)
            action[0] = participate
            old_log_prob = participate_distribution.log_prob(participate)
            if int(participate.item()) == 1:
                prb_distribution = _masked_distribution(
                    actor.prb_logits(context), prb_mask
                )
                prb_index = _select_action(prb_distribution, deterministic)
                if self.learned_power:
                    power_distribution = _masked_distribution(
                        actor.power_logits(context, prb_index), pair_mask[prb_index]
                    )
                    power_index = _select_action(power_distribution, deterministic)
                    old_log_prob = old_log_prob + power_distribution.log_prob(power_index)
                else:
                    power_index = power_indices[prb_index]
                action[1] = prb_index
                action[2] = power_index
                old_log_prob = (
                    old_log_prob
                    + prb_distribution.log_prob(prb_index)
                )
            actions[uav_index] = action.cpu().numpy()
            if not store_for_update:
                continue
            catalog_actions, catalog_probabilities = (
                self._allocation_action_catalog(
                    actor,
                    observation,
                    participate_mask,
                    pair_mask,
                    context=context,
                )
            )
            actions[uav_index] = action.cpu().numpy()
            decisions.append(
                AgentDecision(
                    observation=observation,
                    action=action.detach(),
                    action_mask=participate_mask.detach(),
                    prb_mask=prb_mask.detach(),
                    pair_mask=pair_mask.detach(),
                    counterfactual_actions=catalog_actions.detach(),
                    counterfactual_probabilities=(
                        catalog_probabilities.detach()
                    ),
                    old_log_prob=old_log_prob.detach(),
                )
            )

        if not store_for_update:
            return actions, None
        joint_action = torch.as_tensor(
            actions, dtype=torch.long, device=self.device
        )
        record = AllocationStepRecord(
            global_state=state,
            decisions=decisions,
            joint_action=joint_action,
        )
        return actions, record

    def _allocation_entropy_components(
        self, actions, probabilities, *, detach_branch_weights=False
    ):
        """Decompose entropy; optionally stop child entropy moving parent heads."""
        participation = torch.stack((probabilities[actions[:, 0] == 0].sum(),
                                     probabilities[actions[:, 0] == 1].sum()))
        hp = categorical_entropy(participation)
        prb_joint = probabilities.new_zeros(self.max_prbs).scatter_add(
            0, actions[:, 1], probabilities * actions[:, 0])
        mass = participation[1]
        prb_weight = mass.detach() if detach_branch_weights else mass
        hr = prb_weight * categorical_entropy(prb_joint / mass.clamp_min(1e-38))
        power_entropy = probabilities.new_zeros(())
        if self.learned_power:
            participating = actions[:, 0] == 1
            joint = probabilities[participating]
            prb_indices = actions[participating, 1]
            conditional = joint / prb_joint[prb_indices].clamp_min(1e-38)
            conditional_entropy = probabilities.new_zeros(self.max_prbs).scatter_add(
                0, prb_indices, -conditional * conditional.clamp_min(1e-38).log())
            power_weights = prb_joint.detach() if detach_branch_weights else prb_joint
            power_entropy = (power_weights * conditional_entropy).sum()
        return hp, hr, power_entropy

    def _allocation_entropy_regularizer(self, actions, probabilities):
        """Explore each none head without rewarding its number of child actions.

        The entropy value remains the joint entropy, but detached branch masses
        prevent 100 resource pairs from pushing participation toward 100/101.
        Conditional distributions retain their own exploration gradients.
        """
        if not self.learned_power:
            return categorical_entropy(probabilities)
        return sum(self._allocation_entropy_components(
            actions, probabilities, detach_branch_weights=True))

    def _allocation_policy_terms(self, actor, decision):
        """Exact joint entropy decomposed along the autoregressive policy."""
        actions, probabilities = self._allocation_action_catalog(
            actor, decision.observation, decision.action_mask, decision.pair_mask)
        selected = (actions == decision.action).all(-1)
        if int(selected.sum()) != 1:
            raise RuntimeError("Stored action is outside its behavior catalog")
        log_prob = probabilities[selected][0].clamp_min(1e-38).log()
        hp, hr, power_entropy = self._allocation_entropy_components(actions, probabilities)
        total = categorical_entropy(probabilities)
        return log_prob, total, hp, hr, power_entropy

    @torch.no_grad()
    def prepare_counterfactuals(self, environment, record):
        """Use the unchanged simulator to enumerate one-agent deviations.

        Only the winning allocation affects future state in ver2.py. Reusing
        its outcome makes indistinguishable losing bids have identical Q.
        Rewards are the exact shared environment rewards, never shaped labels.
        """
        if record.reward is not None or record.outcome_states is not None:
            raise RuntimeError("Prepare counterfactuals once, before the real step")
        joint = record.joint_action.cpu().numpy()
        states, rewards, dones, outcomes = [], [], [], {}
        # Avoid a 5,000-float copy for every hypothetical outcome. The table is
        # identical within this placement; processed flags remain branch-local.
        cost_features = record.global_state.get("allocation_cost_features")
        if cost_features is not None:
            cost_features = cost_features.cpu()

        def outcome_index(action):
            winner = environment.winning_uav_for_actions(action)
            key = None if winner is None else (winner, *map(int, action[winner, 1:]))
            if key not in outcomes:
                state, reward, terminated, truncated, _ = environment.preview_allocation_step(
                    action, include_info=False)
                if truncated:
                    raise ValueError("Counterfactual training requires complete finite episodes")
                tensors = self._global_state_to_tensors(
                    state, include_candidate_features=False,
                    include_allocation_cost_features=False,
                )
                outcomes[key] = len(states)
                states.append({k: v.cpu() for k, v in tensors.items()})
                if cost_features is not None:
                    states[-1]["allocation_cost_features"] = cost_features
                rewards.append(float(reward) / self.reward_scale)
                dones.append(bool(terminated))
            return outcomes[key]

        with environment._cache_counterfactual_proposals():
            record.winner = environment.winning_uav_for_actions(joint)
            record.actual_outcome = outcome_index(joint)
            for uav, decision in enumerate(record.decisions):
                indices = []
                for action in decision.counterfactual_actions.cpu().numpy():
                    alternative = joint.copy()
                    alternative[uav] = action
                    indices.append(outcome_index(alternative))
                decision.counterfactual_outcomes = torch.tensor(indices, device=self.device)
        record.outcome_states = states
        record.outcome_rewards = torch.tensor(rewards, device=self.device)
        record.outcome_dones = torch.tensor(dones, device=self.device)

    def store_allocation_outcome(
        self,
        record: AllocationStepRecord,
        *,
        team_reward: float,
        done: bool,
    ) -> None:
        """Attach the shared team outcome to one joint bidder transition."""

        if record.reward is not None or record.done is not None:
            raise RuntimeError("This allocation transition already has an outcome")
        if record.outcome_rewards is None:
            raise RuntimeError("Counterfactual preparation is required before the real step")
        if not np.isclose(float(record.outcome_rewards[record.actual_outcome]) * self.reward_scale,
                          team_reward, rtol=1e-6, atol=1e-7):
            raise RuntimeError("Counterfactual preview changed the shared reward")
        if bool(record.outcome_dones[record.actual_outcome]) != bool(done):
            raise RuntimeError("Counterfactual preview termination mismatch")
        record.reward = float(team_reward)
        record.done = bool(done)
        self._allocation_rollout.append(record)

    def _compute_allocation_returns(self) -> Tensor:
        """Compute exact undiscounted returns across complete episode rollouts."""

        rewards = torch.as_tensor(
            [float(record.reward) for record in self._allocation_rollout],
            dtype=torch.float32,
            device=self.device,
        ) / self.reward_scale
        dones = torch.as_tensor(
            [bool(record.done) for record in self._allocation_rollout],
            dtype=torch.float32,
            device=self.device,
        )
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros((), device=self.device)
        for index in reversed(range(len(self._allocation_rollout))):
            nonterminal = 1.0 - dones[index]
            running_return = (
                rewards[index]
                + self.gamma * nonterminal * running_return
            )
            returns[index] = running_return
        return returns.detach()

    @torch.no_grad()
    def _counterfactual_values(self, record):
        if record.outcome_states is None:
            raise RuntimeError("Call prepare_counterfactuals before stepping the environment")
        values = []
        for start in range(0, len(record.outcome_states), 64):
            values.append(self.allocation_critic(stack_states(
                record.outcome_states[start:start + 64], self.device)))
        continuation = torch.cat(values)
        return record.outcome_rewards + self.gamma * (~record.outcome_dones) * continuation

    @torch.no_grad()
    def _mean_allocation_policy_kl(self, *, policy_catalogs=None):
        # Reuse this required forward pass for final entropy diagnostics.
        if policy_catalogs is not None:
            policy_catalogs.clear()
        by_agent = [[] for _ in range(self.num_uavs)]
        for record in self._allocation_rollout:
            for uav, (actor, decision) in enumerate(zip(self.allocation_actors, record.decisions)):
                old = decision.counterfactual_probabilities
                if len(old) <= 1:
                    if policy_catalogs is not None:
                        policy_catalogs.append((decision.counterfactual_actions, old))
                    continue
                actions, current = self._allocation_action_catalog(
                    actor, decision.observation, decision.action_mask, decision.pair_mask)
                if not torch.equal(actions, decision.counterfactual_actions):
                    raise RuntimeError("Allocation catalog changed during PPO")
                if policy_catalogs is not None:
                    policy_catalogs.append((decision.counterfactual_actions, current))
                kl = (old * (old.clamp_min(1e-38).log() - current.clamp_min(1e-38).log())).sum()
                by_agent[uav].append(float(kl.clamp_min(0)))
        self._last_allocation_kl_by_agent = [float(np.mean(x)) if x else 0.0 for x in by_agent]
        # Do not let forced-reject states dilute the stability measurement.
        active = [x for group in by_agent for x in group]
        return float(np.mean(active)) if active else 0.0

    def update_allocation_rollout(self, *, bootstrap_global_state, bootstrap_done):
        """Fit V on real returns; integrate PPO over simulator counterfactuals.

        Q_cf = r_cf / reward_scale + V(s'_cf), terminal V = 0. The learner
        enforces gamma=1 so this matches placement and evaluation returns.
        Other actors' bids, auction rules, and team reward remain fixed.
        V is learned only from actually visited states and full MC returns;
        counterfactual future estimates remain approximations, not oracle data.
        """
        del bootstrap_global_state
        records = self._allocation_rollout
        if not records or any(r.reward is None or r.done is None for r in records):
            raise RuntimeError("A complete allocation rollout is required")
        if not bootstrap_done or not records[-1].done:
            raise ValueError("Allocation updates require complete episodes")
        if any(r.outcome_states is None for r in records):
            raise RuntimeError("Counterfactual outcomes were not prepared")
        returns = self._compute_allocation_returns()
        n = len(records)
        minibatch = min(n, self.allocation_minibatch_size)
        before = [parameter_vector(actor) for actor in self.allocation_actors]
        for optimizer in self.allocation_actor_optimizers:
            optimizer.zero_grad(set_to_none=True)
        critic_losses, policy_losses, entropies, clip_fractions = [], [], [], []
        grad_logs = [[] for _ in range(self.num_uavs)]
        # Batch state attention. The next observed state is already in records;
        # full MC targets at that state train the continuation value as well.
        critic_steps = 0
        for _ in range(self.allocation_ppo_epochs):
            for indices in torch.randperm(n).split(min(minibatch, 64)):
                ids = indices.tolist()
                prediction = self.allocation_critic(stack_states([records[i].global_state for i in ids], self.device))
                loss = F.mse_loss(prediction, returns[indices.to(self.device)])
                self.allocation_critic_optimizer.zero_grad(set_to_none=True)
                (self.value_coef * loss).backward()
                nn.utils.clip_grad_norm_(self.allocation_critic.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                self.allocation_critic_optimizer.step()
                self.optimizer_step_totals["allocation_critic"] += 1
                critic_losses.append(float(loss.detach()))
                critic_steps += 1

        # Freeze all CF targets and old probabilities for every PPO epoch.
        catalogs, actual_advantages, rollout_q_values = [], [], []
        for record in records:
            q = self._counterfactual_values(record)
            rollout_q_values.append(q[record.actual_outcome])
            advantages = [centered_action_values(q[d.counterfactual_outcomes], d.counterfactual_probabilities)
                          for d in record.decisions]
            catalogs.append(advantages)
            actual_advantages.append(torch.stack([
                advantage[(decision.counterfactual_actions == decision.action).all(-1)][0]
                for advantage, decision in zip(advantages, record.decisions)]))
        raw = torch.stack(actual_advantages)
        # Per-agent RMS scaling keeps exact zeros/signs and avoids amplifying
        # numerical noise. Exclude singleton masks; never mean-center globally.
        scales = []
        for uav in range(self.num_uavs):
            second_moments = [(r.decisions[uav].counterfactual_probabilities * catalogs[i][uav].square()).sum()
                              for i, r in enumerate(records)
                              if len(r.decisions[uav].counterfactual_probabilities) > 1]
            rms = torch.stack(second_moments).mean().sqrt() if second_moments else returns.new_tensor(0.)
            scales.append(rms.clamp_min(1.0 / self.reward_scale))
        steps = 0
        sample_uses = 0
        stopped_actors = [False] * self.num_uavs
        actor_steps = [0] * self.num_uavs
        last_policy_catalogs = []
        for _ in range(self.allocation_ppo_epochs):
            for indices in torch.randperm(n).split(minibatch):
                updated = False
                for uav, (actor, optimizer) in enumerate(zip(self.allocation_actors, self.allocation_actor_optimizers)):
                    if stopped_actors[uav]:
                        continue
                    losses, entropy_terms, entropy_regularizers, clips = [], [], [], []
                    for index in indices.tolist():
                        decision = records[index].decisions[uav]
                        if len(decision.counterfactual_probabilities) <= 1:
                            continue
                        actions, probabilities = self._allocation_action_catalog(
                            actor, decision.observation, decision.action_mask, decision.pair_mask)
                        if not torch.equal(actions, decision.counterfactual_actions):
                            raise RuntimeError("Counterfactual catalog changed during PPO")
                        loss, clip = expected_clipped_loss(probabilities, decision.counterfactual_probabilities,
                                                           catalogs[index][uav] / scales[uav], self.clip_epsilon)
                        losses.append(loss)
                        entropy_terms.append(categorical_entropy(probabilities))
                        entropy_regularizers.append(
                            self._allocation_entropy_regularizer(actions, probabilities))
                        clips.append(clip)
                    if not losses:
                        continue
                    sample_uses += len(losses)
                    policy_loss = torch.stack(losses).mean()
                    entropy = torch.stack(entropy_terms).mean()
                    entropy_regularizer = torch.stack(entropy_regularizers).mean()
                    optimizer.zero_grad(set_to_none=True)
                    (policy_loss - self.allocation_entropy_coef * entropy_regularizer).backward()
                    norm = nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                    grad_logs[uav].append(float(norm))
                    if float(norm) > 0:
                        optimizer.step()
                        self.optimizer_step_totals["allocation_actor"] += 1
                        actor_steps[uav] += 1
                        updated = True
                    policy_losses.append(float(policy_loss.detach()))
                    entropies.append(float(entropy.detach()))
                    clip_fractions.append(float(torch.stack(clips).mean()))
                steps += int(updated)
                kl = self._mean_allocation_policy_kl(policy_catalogs=last_policy_catalogs)
                # Keep the crossing step; freeze only that actor until the next rollout.
                for uav, actor_kl in enumerate(self._last_allocation_kl_by_agent):
                    stopped_actors[uav] |= actor_kl > self.target_kl
                if all(stopped_actors):
                    break
            if all(stopped_actors):
                break
        mean = lambda xs: float(np.mean(xs)) if xs else 0.0
        # Both diagnostics use the fitted critic, in reward / reward_scale units.
        # V is evaluated on actual current states; Q includes the actual reward
        # and continuation value. This read-only pass builds no training graph.
        with torch.no_grad():
            values = torch.cat([
                self.allocation_critic(stack_states(
                    [r.global_state for r in records[start:start + 64]], self.device))
                for start in range(0, n, 64)
            ])
        q_values = torch.stack(rollout_q_values)
        metrics = {
            'allocation_actor_loss': mean(policy_losses), 'allocation_critic_loss': mean(critic_losses),
            'allocation_entropy': mean(entropies), 'allocation_clip_fraction': mean(clip_fractions),
            'allocation_value_mean': float(values.mean()), 'allocation_q_mean': float(q_values.mean()),
            'allocation_value_max_abs': float(values.abs().max()), 'allocation_q_max_abs': float(q_values.abs().max()),
            'allocation_return_mean': float(returns.mean()) * self.reward_scale,
            'allocation_value_target_mean': float(returns.mean()),
            'allocation_rollout_return': sum(float(r.reward) for r in records),
            'allocation_rollout_size': float(n), 'allocation_gradient_steps': float(steps),
            'allocation_critic_gradient_steps': float(critic_steps), 'allocation_policy_kl': kl,
            'allocation_kl_early_stop': float(any(stopped_actors)),
            'allocation_policy_sample_uses': float(sample_uses),
            'allocation_reject_fraction': mean([int(d.action[0]) == 0 for r in records for d in r.decisions]),
            'allocation_counterfactual_advantage_mean': float(raw.mean()),
            'allocation_counterfactual_nonzero_fraction': float((raw.abs() > 1e-8).float().mean()),
            'allocation_unique_outcomes': float(sum(len(r.outcome_states) for r in records)),
        }
        entropy_parts = []
        with torch.no_grad():
            for actions, probabilities in last_policy_catalogs:
                entropy_parts.append([float(v) for v in
                                      self._allocation_entropy_components(actions, probabilities)])
        for index, name in enumerate(('participation', 'prb', 'power')):
            metrics[f'allocation_{name}_entropy'] = mean([parts[index] for parts in entropy_parts])
        for uav, actor in enumerate(self.allocation_actors):
            prefix = f'allocation_uav_{uav}_'
            participation = mean([int(r.decisions[uav].action[0]) == 1 for r in records])
            metrics.update({
                prefix + 'selection_fraction': participation,  # Legacy CSV alias.
                prefix + 'participation_fraction': participation,
                prefix + 'win_fraction': mean([r.winner == uav for r in records]),
                prefix + 'forced_fraction': mean([len(r.decisions[uav].counterfactual_probabilities) == 1 for r in records]),
                prefix + 'counterfactual_advantage_mean': float(raw[:, uav].mean()),
                prefix + 'counterfactual_abs_advantage_mean': float(raw[:, uav].abs().mean()),
                prefix + 'counterfactual_nonzero_fraction': float((raw[:, uav].abs() > 1e-8).float().mean()),
                prefix + 'gradient_norm': mean(grad_logs[uav]),
                prefix + 'parameter_delta': float((parameter_vector(actor) - before[uav]).norm()),
                prefix + 'policy_kl': self._last_allocation_kl_by_agent[uav],
                prefix + 'kl_early_stop': float(stopped_actors[uav]),
                prefix + 'optimizer_steps': float(actor_steps[uav]),
            })
        if self.diagnostics_callback is not None:
            self.diagnostics_callback('allocation', metrics)
        records.clear()
        return metrics

    @property
    def allocation_rollout_size(self) -> int:
        """Number of completed allocation transitions waiting for PPO."""

        return len(self._allocation_rollout)

    @property
    def placement_rollout_size(self) -> int:
        """Number of complete placement episodes waiting for PPO."""

        return sum(
            record.episode_return is not None
            for record in self._placement_rollout
        )

    def store_placement_outcome(self, episode_return: float) -> None:
        """Attach a team return to the oldest unresolved placement."""

        if not self._placement_rollout:
            raise RuntimeError("No placement record is available")
        record = next(
            (
                candidate
                for candidate in self._placement_rollout
                if candidate.episode_return is None
            ),
            None,
        )
        if record is None:
            raise RuntimeError("Every placement record already has an outcome")
        record.episode_return = float(episode_return)

    def store_placement_outcomes(
        self, episode_returns: Sequence[float]
    ) -> None:
        """Attach one aligned global return to every pending placement."""

        pending = [
            record
            for record in self._placement_rollout
            if record.episode_return is None
        ]
        if len(pending) != len(episode_returns):
            raise ValueError(
                f"Expected {len(pending)} placement returns, got "
                f"{len(episode_returns)}"
            )
        for record, episode_return in zip(pending, episode_returns):
            record.episode_return = float(episode_return)

    def allocation_snapshot(self) -> dict[str, Any]:
        """Capture only allocation models/optimizers for inner best restore."""

        return copy.deepcopy(
            {
                "actors": self.allocation_actors.state_dict(),
                "critic": self.allocation_critic.state_dict(),
                "actor_optimizers": [
                    optimizer.state_dict()
                    for optimizer in self.allocation_actor_optimizers
                ],
                "critic_optimizer": self.allocation_critic_optimizer.state_dict(),
            }
        )

    def restore_allocation_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore the best inner allocation state without touching placement."""

        self.allocation_actors.load_state_dict(snapshot["actors"])
        self.allocation_critic.load_state_dict(snapshot["critic"])
        for optimizer, state in zip(
            self.allocation_actor_optimizers,
            snapshot["actor_optimizers"],
        ):
            optimizer.load_state_dict(state)
        self.allocation_critic_optimizer.load_state_dict(
            snapshot["critic_optimizer"]
        )

    @torch.no_grad()
    def _placement_policy_kl(self):
        by_agent = []
        for uav, actor in enumerate(self.placement_actors):
            divergences = []
            for record in self._placement_rollout:
                decision = record.decisions[uav]
                current = _masked_distribution(actor(decision.observation), decision.action_mask).probs
                old = decision.counterfactual_probabilities
                positive = old > 0
                divergences.append((old[positive] * (old[positive].log() - current[positive].clamp_min(1e-38).log())).sum())
            by_agent.append(float(torch.stack(divergences).mean().clamp_min(0)))
        return by_agent

    def update_placement_rollout(self):
        """PPO on the same delayed team returns, with per-actor diagnostics."""
        records = self._placement_rollout
        if len(records) < 2 or any(r.episode_return is None for r in records):
            raise RuntimeError("At least two complete placement episodes are required")
        raw_targets = torch.tensor([r.episode_return for r in records], device=self.device)
        old_values = torch.stack([r.old_value for r in records])
        raw_advantages = (raw_targets - old_values).detach()
        advantages = (raw_advantages - raw_advantages.mean()) / raw_advantages.std(unbiased=False).clamp_min(1e-6)
        self.placement_critic.update_popart(raw_targets)
        targets = self.placement_critic.normalize_popart_targets(raw_targets)
        before = [parameter_vector(actor) for actor in self.placement_actors]
        grad_logs = [[] for _ in range(self.num_uavs)]
        policy_losses, critic_losses, entropies, clips = [], [], [], []
        n = len(records)
        minibatch = min(n, self.placement_minibatch_size)
        steps, actor_steps, sample_uses = 0, 0, 0
        stopped = False
        for _ in range(self.placement_ppo_epochs):
            for indices in torch.randperm(n).split(minibatch):
                ids = indices.tolist()
                # Separate backward per actor bounds memory and avoids dividing
                # an independent actor's gradient by the number of UAVs.
                for uav, (actor, optimizer) in enumerate(zip(self.placement_actors, self.placement_actor_optimizers)):
                    losses, entropy_terms, clip_terms = [], [], []
                    for index in ids:
                        decision = records[index].decisions[uav]
                        distribution = _masked_distribution(actor(decision.observation), decision.action_mask)
                        ratio = (distribution.log_prob(decision.action) - decision.old_log_prob).exp()
                        advantage = advantages[index]
                        losses.append(-torch.minimum(ratio * advantage,
                            ratio.clamp(1-self.clip_epsilon, 1+self.clip_epsilon) * advantage))
                        entropy_terms.append(distribution.entropy())
                        clip_terms.append(((ratio-1).abs() > self.clip_epsilon).float())
                    policy_loss = torch.stack(losses).mean()
                    entropy = torch.stack(entropy_terms).mean()
                    optimizer.zero_grad(set_to_none=True)
                    (policy_loss - self.placement_entropy_coef * entropy).backward()
                    norm = nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                    grad_logs[uav].append(float(norm))
                    if float(norm) > 0:
                        optimizer.step()
                        self.optimizer_step_totals["placement_actor"] += 1
                        actor_steps += 1
                    policy_losses.append(float(policy_loss.detach()))
                    entropies.append(float(entropy.detach()))
                    clips.append(float(torch.stack(clip_terms).mean()))
                    sample_uses += len(ids)
                prediction = self.placement_critic.normalized_value(stack_states(
                    [records[i].initial_global_state for i in ids], self.device))
                critic_loss = F.mse_loss(prediction, targets[indices.to(self.device)])
                self.placement_critic_optimizer.zero_grad(set_to_none=True)
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.placement_critic.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                self.placement_critic_optimizer.step()
                self.optimizer_step_totals["placement_critic"] += 1
                critic_losses.append(float(critic_loss.detach()))
                steps += 1
                kls = self._placement_policy_kl()
                if max(kls) > self.target_kl:
                    stopped = True
                    break
            if stopped:
                break
        metrics = {
            'placement_updated': 1.0, 'placement_episodes_collected': float(n),
            'placement_actor_loss': float(np.mean(policy_losses)),
            'placement_critic_loss': float(np.mean(critic_losses)),
            'placement_entropy': float(np.mean(entropies)),
            'placement_clip_fraction': float(np.mean(clips)),
            'placement_advantage_mean': float(raw_advantages.mean()),
            'placement_return_mean': float(raw_targets.mean()),
            'placement_value_target_mean': float(targets.mean()),
            'placement_value_mean': float(old_values.mean()),
            'placement_popart_mean': float(self.placement_critic.popart_head.mean),
            'placement_popart_std': float(self.placement_critic.popart_head.std),
            'placement_gradient_steps': float(steps),
            'placement_actor_optimizer_steps': float(actor_steps),
            'placement_policy_sample_uses': float(sample_uses),
            'placement_kl_early_stop': float(stopped),
            'placement_policy_kl': float(np.mean(kls)),
        }
        for uav, actor in enumerate(self.placement_actors):
            metrics[f'placement_uav_{uav}_gradient_norm'] = float(np.mean(grad_logs[uav]))
            metrics[f'placement_uav_{uav}_parameter_delta'] = float((parameter_vector(actor) - before[uav]).norm())
            metrics[f'placement_uav_{uav}_policy_kl'] = kls[uav]
        if self.diagnostics_callback is not None:
            self.diagnostics_callback('placement', metrics)
        records.clear()
        return metrics

    @staticmethod
    def _validated_trainer_state(
        trainer_state: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(trainer_state, dict):
            raise TypeError("trainer_state must be a dictionary")
        if trainer_state.get("resume_boundary") != "outer_frozen_batch":
            raise ValueError(
                "Exact resume is supported only at an outer frozen-batch "
                "boundary; trainer_state['resume_boundary'] must be "
                "'outer_frozen_batch'"
            )
        return copy.deepcopy(trainer_state)

    @classmethod
    def _validated_optimizer_step_totals(
        cls,
        totals: Any,
    ) -> dict[str, int]:
        """Validate the lifetime counters stored outside optimizer snapshots."""

        if not isinstance(totals, dict) or set(totals) != set(
            cls.OPTIMIZER_COUNTER_KEYS
        ):
            raise ValueError(
                "Checkpoint optimizer counters have an invalid schema"
            )
        validated: dict[str, int] = {}
        for key in cls.OPTIMIZER_COUNTER_KEYS:
            value = totals[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise ValueError(
                    f"Checkpoint optimizer counter {key!r} must be a "
                    "non-negative integer"
                )
            validated[key] = int(value)
        return validated

    def optimizer_step_counts(self) -> dict[str, int]:
        """Return exact lifetime ``optimizer.step()`` call counts."""

        counts = dict(self.optimizer_step_totals)
        counts["total"] = sum(counts.values())
        return counts

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if cuda_count else [],
            "torch_cuda_device_count": cuda_count,
        }

    @staticmethod
    def _restore_rng_state(rng_state: dict[str, Any]) -> None:
        required = {
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
            "torch_cuda_device_count",
        }
        missing = required.difference(rng_state)
        if missing:
            raise ValueError(
                f"Checkpoint RNG state is missing fields: {sorted(missing)}"
            )
        saved_cuda_count = int(rng_state["torch_cuda_device_count"])
        current_cuda_count = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        if saved_cuda_count != current_cuda_count:
            raise RuntimeError(
                "Exact RNG restore requires the same CUDA device count: "
                f"checkpoint={saved_cuda_count}, runtime={current_cuda_count}"
            )
        cuda_states = list(rng_state["torch_cuda"])
        if len(cuda_states) != saved_cuda_count:
            raise ValueError("Checkpoint CUDA RNG state count is inconsistent")
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"].cpu())
        if cuda_states:
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])

    def save_checkpoint(
        self,
        path: str | Path,
        episode: int,
        *,
        trainer_state: dict[str, Any] | None = None,
    ) -> None:
        """Save models plus optional outer-boundary trainer/RNG resume state."""

        saved_trainer_state = None
        if trainer_state is not None:
            if self._placement_rollout or self._allocation_rollout:
                raise RuntimeError(
                    "Cannot create an exact-resume checkpoint with pending "
                    "placement or allocation rollouts"
                )
            saved_trainer_state = self._validated_trainer_state(trainer_state)

        checkpoint = {
            "checkpoint_schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "episode": int(episode),
            "target_kl": self.target_kl,
            "credit_estimator": "exact_reward_plus_learned_next_state_value",
            "num_uavs": self.num_uavs,
            "num_candidates": self.num_candidates,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "framework": self.CHECKPOINT_FRAMEWORK,
            "allocation_policy": self.allocation_policy,
            "allocation_entropy_regularization": self.allocation_entropy_regularization,
            "allocation_actor_feature_version": self.allocation_actor_feature_version,
            "allocation_critic_feature_version": self.allocation_critic_feature_version,
            "max_prbs": self.max_prbs,
            "num_power_levels": self.num_power_levels,
            "gamma": self.gamma,
            "reward_scale": self.reward_scale,
            "placement_popart_beta": self.placement_popart_beta,
            "allocation_observation_scope": (
                self.ALLOCATION_OBSERVATION_SCOPE
            ),
            "coordination_feature_dim": self.coordination_feature_dim,
            "coordination_mode": self.coordination_mode,
            "model_state_dict": self.state_dict(),
            "placement_actor_optimizers": [
                optimizer.state_dict()
                for optimizer in self.placement_actor_optimizers
            ],
            "allocation_actor_optimizers": [
                optimizer.state_dict()
                for optimizer in self.allocation_actor_optimizers
            ],
            "placement_critic_optimizer": (
                self.placement_critic_optimizer.state_dict()
            ),
            "allocation_critic_optimizer": (
                self.allocation_critic_optimizer.state_dict()
            ),
            "optimizer_step_totals": dict(self.optimizer_step_totals),
            "trainer_state": saved_trainer_state,
            "rng_state": self._capture_rng_state(),
            "resume_capability": (
                "exact_outer_frozen_batch"
                if saved_trainer_state is not None
                else "model_optimizer_rng_only"
            ),
        }
        path = Path(path)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b", dir=path.parent, prefix=f".{path.name}.",
                suffix=".tmp", delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                torch.save(checkpoint, temporary_file)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        restore_training_state: bool = False,
    ) -> int:
        """Restore model state and optionally exact outer-boundary RNG state.

        The return value remains the saved episode for existing callers.
        ``loaded_trainer_state`` exposes a defensive copy of trainer progress.
        """

        checkpoint = torch.load(
            Path(path),
            map_location=map_location or self.device,
            weights_only=False,
        )
        checkpoint_num_uavs = checkpoint.get("num_uavs")
        if checkpoint_num_uavs != self.num_uavs:
            raise ValueError(
                "Checkpoint topology does not match this five-UAV learner: "
                f"checkpoint num_uavs={checkpoint_num_uavs!r}"
            )
        expected_framework = self.CHECKPOINT_FRAMEWORK
        if checkpoint.get("framework") != expected_framework:
            raise ValueError(
                "Checkpoint allocation architecture is incompatible: "
                f"expected {expected_framework!r}, got "
                f"{checkpoint.get('framework')!r}. Start a new run or migrate "
                "the checkpoint explicitly."
            )
        if (
            checkpoint.get("checkpoint_schema_version")
            != self.CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                "Checkpoint schema does not match this learner: expected "
                f"{self.CHECKPOINT_SCHEMA_VERSION!r}, got "
                f"{checkpoint.get('checkpoint_schema_version')!r}"
            )
        if (
            checkpoint.get("max_prbs") != self.max_prbs
            or checkpoint.get("num_power_levels") != self.num_power_levels
        ):
            raise ValueError("Checkpoint allocation action dimensions do not match")
        if checkpoint.get("allocation_policy") != self.allocation_policy:
            raise ValueError("Checkpoint allocation policy semantics do not match")
        if checkpoint.get("allocation_actor_feature_version", "local_vector_v1") != self.allocation_actor_feature_version:
            raise ValueError(
                "Checkpoint allocation actor features do not match; "
                "start a new run for the compact QoS/link residual actor"
            )
        if checkpoint.get("allocation_critic_feature_version", "global_v1") != self.allocation_critic_feature_version:
            raise ValueError(
                "Checkpoint allocation critic features do not match; "
                "start a new run for the critic cost table"
            )
        if restore_training_state and checkpoint.get(
            "allocation_entropy_regularization", "joint_entropy_v1"
        ) != self.allocation_entropy_regularization:
            raise ValueError(
                "Checkpoint allocation entropy regularization does not match "
                "exact resume; start a new run for the changed training objective"
            )
        architecture = {
            "num_candidates": self.num_candidates,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        }
        mismatches = {
            key: (checkpoint.get(key), expected)
            for key, expected in architecture.items()
            if checkpoint.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "Checkpoint model architecture does not match this learner: "
                f"{mismatches}"
            )
        if checkpoint.get("gamma") != self.gamma:
            raise ValueError(
                "Checkpoint gamma does not match the required undiscounted "
                f"objective: checkpoint={checkpoint.get('gamma')!r}, "
                f"learner={self.gamma!r}"
            )
        checkpoint_reward_scale = checkpoint.get("reward_scale")
        if checkpoint_reward_scale is not None and not np.isclose(
            float(checkpoint_reward_scale), self.reward_scale
        ):
            raise ValueError(
                "Checkpoint reward scale does not match this learner: "
                f"checkpoint reward_scale={checkpoint_reward_scale!r}, "
                f"learner reward_scale={self.reward_scale!r}"
            )
        checkpoint_scope = checkpoint.get("allocation_observation_scope")
        if checkpoint_scope != self.ALLOCATION_OBSERVATION_SCOPE:
            raise ValueError(
                "Checkpoint allocation observation scope does not match this "
                f"learner: checkpoint={checkpoint_scope!r}, "
                f"expected={self.ALLOCATION_OBSERVATION_SCOPE!r}"
            )
        if checkpoint.get("coordination_feature_dim") != (
            self.coordination_feature_dim
        ):
            raise ValueError("Checkpoint coordination feature size does not match")
        if checkpoint.get("coordination_mode") != self.coordination_mode:
            raise ValueError(
                "Checkpoint coordination mode does not match this learner: "
                f"checkpoint={checkpoint.get('coordination_mode')!r}, "
                f"learner={self.coordination_mode!r}"
            )
        checkpoint_popart_beta = checkpoint.get("placement_popart_beta")
        if checkpoint_popart_beta is None:
            raise ValueError(
                "Checkpoint predates placement Pop-Art; start a new training "
                "run or migrate the checkpoint explicitly"
            )
        if not np.isclose(
            float(checkpoint_popart_beta),
            self.placement_popart_beta,
        ):
            raise ValueError(
                "Checkpoint placement Pop-Art beta does not match this "
                f"learner: checkpoint={checkpoint_popart_beta!r}, "
                f"learner={self.placement_popart_beta!r}"
            )
        stored_trainer_state = checkpoint.get("trainer_state")
        if stored_trainer_state is not None:
            stored_trainer_state = self._validated_trainer_state(
                stored_trainer_state
            )
        stored_optimizer_step_totals = self._validated_optimizer_step_totals(
            checkpoint.get("optimizer_step_totals")
        )
        if restore_training_state:
            if self._placement_rollout or self._allocation_rollout:
                raise RuntimeError(
                    "Exact resume requires a learner with empty rollouts"
                )
            if (
                stored_trainer_state is None
                or checkpoint.get("resume_capability")
                != "exact_outer_frozen_batch"
            ):
                raise ValueError(
                    "Checkpoint has no exact outer frozen-batch trainer state"
                )
            if not isinstance(checkpoint.get("rng_state"), dict):
                raise ValueError("Checkpoint has no restorable RNG state")
        self.load_state_dict(checkpoint["model_state_dict"])
        for optimizer, state in zip(
            self.placement_actor_optimizers,
            checkpoint["placement_actor_optimizers"],
        ):
            optimizer.load_state_dict(state)
        for optimizer, state in zip(
            self.allocation_actor_optimizers,
            checkpoint["allocation_actor_optimizers"],
        ):
            optimizer.load_state_dict(state)
        self.placement_critic_optimizer.load_state_dict(
            checkpoint["placement_critic_optimizer"]
        )
        self.allocation_critic_optimizer.load_state_dict(
            checkpoint["allocation_critic_optimizer"]
        )
        self.optimizer_step_totals = stored_optimizer_step_totals
        self.loaded_trainer_state = copy.deepcopy(stored_trainer_state)
        if restore_training_state:
            self._restore_rng_state(checkpoint["rng_state"])
        return int(checkpoint["episode"])


def _collect_allocation_episode(
    environment: Any,
    learner: StrictLocalIndependentMAPPO,
    global_state: dict[str, Any],
    *,
    initial_placement_reward: float = 0.0,
    allocation_rollout_horizon: int | None = None,
    defer_allocation_update: bool = False,
    deterministic_actions: bool = False,
) -> dict[str, Any]:
    """Train allocation from an already-applied, fixed placement."""

    placement_episode_return = float(initial_placement_reward)
    allocation_episode_return = 0.0
    done = environment.phase == environment.TERMINAL_PHASE
    allocation_updates: list[dict[str, float]] = []
    info = environment._get_info()

    while not done:
        allocation_action, step_record = learner.act_allocation(
            global_state,
            environment.get_actor_observations(),
            deterministic=deterministic_actions,
            store_for_update=True,
        )
        if step_record is None:
            raise RuntimeError("Training allocation did not produce a record")

        learner.prepare_counterfactuals(environment, step_record)
        next_global_state, environment_reward, terminated, truncated, info = (
            environment.step(allocation_action)
        )
        done = bool(terminated or truncated)
        allocation_team_reward, placement_team_reward = _split_team_rewards(
            environment_reward,
            info["reward_breakdown"],
        )
        learner.store_allocation_outcome(
            step_record,
            team_reward=allocation_team_reward,
            done=done,
        )
        allocation_episode_return += allocation_team_reward
        placement_episode_return += placement_team_reward
        global_state = next_global_state

        horizon_reached = (
            allocation_rollout_horizon is not None
            and learner.allocation_rollout_size >= allocation_rollout_horizon
        )
        if done and (horizon_reached or not defer_allocation_update):
            allocation_updates.append(
                learner.update_allocation_rollout(
                    bootstrap_global_state=None if done else global_state,
                    bootstrap_done=done,
                )
            )

    return {
        "allocation_return": float(allocation_episode_return),
        "placement_return": float(placement_episode_return),
        "served_count": int(info["served_count"]),
        "served_ratio": float(info["served_ratio"]),
        "allocation_updates": allocation_updates,
        "final_info": info,
    }


def train_one_episode(
    environment: Any,
    learner: StrictLocalIndependentMAPPO,
    *,
    reset_seed: int | None = None,
    ue_seed: int | None = None,
    allocation_rollout_horizon: int | None = None,
    defer_allocation_update: bool = False,
    deterministic_actions: bool = False,
) -> dict[str, Any]:
    """Collect one episode with a shared placement/allocation objective."""

    if allocation_rollout_horizon is not None and allocation_rollout_horizon <= 0:
        raise ValueError("allocation_rollout_horizon must be positive or None")
    if learner.allocation_rollout_size and not defer_allocation_update:
        raise RuntimeError("Start an episode with an empty allocation rollout")
    if reset_seed is not None and ue_seed is not None:
        raise ValueError("Pass either reset_seed or ue_seed, not both")
    if ue_seed is not None:
        global_state, info = environment.reset(
            options={"ue_seed": int(ue_seed)}
        )
    elif reset_seed is None:
        global_state, info = environment.reset()
    else:
        global_state, info = environment.reset(seed=reset_seed)

    placement_actions = learner.act_placement(
        global_state,
        environment.get_actor_observations(),
        deterministic=deterministic_actions,
        store_for_update=True,
    )
    joint_placement_actions = _validated_joint_action(
        placement_actions, learner.num_uavs
    )
    global_state, immediate_placement_reward, terminated, truncated, info = (
        environment.step(joint_placement_actions)
    )
    allocation_metrics = _collect_allocation_episode(
        environment,
        learner,
        global_state,
        initial_placement_reward=float(immediate_placement_reward),
        allocation_rollout_horizon=allocation_rollout_horizon,
        defer_allocation_update=defer_allocation_update,
        deterministic_actions=deterministic_actions,
    )
    allocation_episode_return = float(allocation_metrics["allocation_return"])
    placement_episode_return = float(allocation_metrics["placement_return"])
    allocation_updates = allocation_metrics["allocation_updates"]
    info = allocation_metrics["final_info"]

    learner.store_placement_outcome(placement_episode_return)
    if learner.placement_rollout_size >= learner.placement_batch_episodes:
        placement_metrics = learner.update_placement_rollout()
    else:
        placement_metrics = {
            "placement_updated": 0.0,
            "placement_episodes_collected": float(
                learner.placement_rollout_size
            ),
        }
    return {
        "episode_return": float(placement_episode_return),
        "allocation_return": float(allocation_episode_return),
        "placement_return": float(placement_episode_return),
        "served_count": int(info["served_count"]),
        "served_ratio": float(info["served_ratio"]),
        "placement_metrics": placement_metrics,
        "allocation_updates": allocation_updates,
        "final_info": info,
    }


@torch.no_grad()
def evaluate_fixed_placement_episode(
    environment: Any,
    learner: StrictLocalIndependentMAPPO,
    *,
    ue_seed: int,
    placement_actions: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Evaluate one fixed map/placement with deterministic local bidders."""

    global_state, _ = environment.reset(options={"ue_seed": int(ue_seed)})
    placement = _validated_joint_action(
        np.asarray(placement_actions), learner.num_uavs
    )
    global_state, placement_reward, terminated, truncated, info = (
        environment.step(placement)
    )
    allocation_return = 0.0
    placement_return = float(placement_reward)
    done = bool(terminated or truncated)
    while not done:
        allocation_action, _ = learner.act_allocation(
            global_state,
            environment.get_actor_observations(),
            deterministic=True,
            store_for_update=False,
        )
        global_state, reward, terminated, truncated, info = environment.step(
            allocation_action
        )
        allocation_reward, placement_step_reward = _split_team_rewards(
            reward, info["reward_breakdown"]
        )
        allocation_return += allocation_reward
        placement_return += placement_step_reward
        done = bool(terminated or truncated)
    return {
        "episode_return": float(placement_return),
        "allocation_return": float(allocation_return),
        "placement_return": float(placement_return),
        "served_count": int(info["served_count"]),
        "served_ratio": float(info["served_ratio"]),
        "placement_actions": placement.copy(),
        "final_info": info,
    }


def _fixed_batch_score(
    _environment: Any,
    evaluations: Sequence[dict[str, Any]],
) -> tuple[float]:
    """Return the sole selection objective: mean scalar team return."""

    team_returns = np.asarray(
        [result["placement_return"] for result in evaluations],
        dtype=np.float64,
    )
    if not team_returns.size or not np.isfinite(team_returns).all():
        raise ValueError("Fixed-batch team returns must be non-empty and finite")
    return (round(float(team_returns.mean()), 12),)


def _fixed_batch_diagnostics(
    environment: Any,
    evaluations: Sequence[dict[str, Any]],
) -> dict[str, float | int]:
    """Report service/resource diagnostics without changing model selection."""

    served = np.asarray(
        [result["served_count"] for result in evaluations], dtype=np.float64
    )
    resource_costs = np.asarray(
        [
            environment.power_weight
            * result["final_info"]["total_used_power_w"]
            / environment.power_penalty_denominator_w
            + environment.prb_weight
            * result["final_info"]["total_used_prbs"]
            / environment.total_max_prbs
            for result in evaluations
        ],
        dtype=np.float64,
    )
    return {
        "mean_team_return": _fixed_batch_score(
            environment, evaluations
        )[0],
        "mean_served_count": float(served.mean()),
        "min_served_count": int(served.min()),
        "mean_resource_cost": float(resource_costs.mean()),
    }


def train_frozen_placement_batch(
    environment: Any,
    learner: StrictLocalIndependentMAPPO,
    *,
    ue_seeds: Sequence[int],
    placement_orders: Sequence[Sequence[int]] | None = None,
    max_inner_updates: int = 50,
    inner_patience: int = 10,
    allocation_entropy_schedule: Callable[[int], float] | None = None,
    deterministic_actions: bool = False,
    update_placement: bool = True,
    update_allocation: bool = True,
) -> dict[str, Any]:
    """Score placement zero-shot, then optionally optimize allocation.

    Every inner update collects one complete 50-UE trajectory from every
    frozen map/placement pair. Stop after ``inner_patience`` consecutive
    evaluations without improvement, or at ``max_inner_updates``, and
    retain the final actors, critic and optimizers, even if deterministic
    evaluation deteriorates. Best scores only control the patience counter. Placement
    samples may accumulate across calls before their PPO batch is full.
    When ``update_placement`` is false the
    sampled placements warm up only the allocator and create no placement
    rollout.  Otherwise placement PPO uses the initial evaluations collected
    before any allocation update in this batch.  When ``update_allocation`` is
    false, those same evaluations update placement while every allocation
    model parameter and optimizer-step counter remains frozen.
    """

    seeds = tuple(int(seed) for seed in ue_seeds)
    if not seeds or len(seeds) > learner.placement_batch_episodes:
        raise ValueError(
            "The frozen batch must fit within placement_batch_episodes"
        )
    if max_inner_updates <= 0:
        raise ValueError("max_inner_updates must be positive")
    if not isinstance(inner_patience, (int, np.integer)) or isinstance(inner_patience, bool) or inner_patience <= 0:
        raise ValueError("inner_patience must be a positive integer")
    if not update_placement and not update_allocation:
        raise ValueError("At least one of placement or allocation must be active")
    if learner.allocation_rollout_size:
        raise RuntimeError("Start a frozen batch with an empty allocation rollout")
    if learner.placement_rollout_size != len(learner._placement_rollout):
        raise RuntimeError("Previous placement samples have no outcomes")
    if update_placement and learner.placement_rollout_size + len(seeds) > learner.placement_batch_episodes:
        raise ValueError("Frozen batch would overflow the placement PPO batch")
    if placement_orders is None:
        orders = tuple(tuple(range(learner.num_uavs)) for _ in seeds)
    else:
        orders = tuple(
            tuple(int(index) for index in order)
            for order in placement_orders
        )
    if len(orders) != len(seeds):
        raise ValueError("Each frozen placement requires one selection order")
    expected_uavs = set(range(learner.num_uavs))
    if any(
        len(order) != learner.num_uavs or set(order) != expected_uavs
        for order in orders
    ):
        raise ValueError(
            "Every placement selection order must permute all UAV indices"
        )
    if len(set(zip(seeds, orders))) != len(seeds):
        raise ValueError("Each repeated UE map must use a distinct placement order")

    frozen_allocation_state = (
        learner.allocation_snapshot()
        if not update_allocation
        else None
    )
    frozen_allocation_steps = (
        learner.optimizer_step_counts() if not update_allocation else {}
    )
    frozen_allocation_entropy = float(learner.allocation_entropy_coef)

    placements: list[np.ndarray] = []
    for ue_seed, selection_order in zip(seeds, orders):
        global_state, _ = environment.reset(options={"ue_seed": ue_seed})
        placement = learner.act_placement(
            global_state,
            environment.get_actor_observations(),
            deterministic=deterministic_actions,
            store_for_update=update_placement,
            selection_order=selection_order,
        )
        placements.append(
            _validated_joint_action(placement, learner.num_uavs).copy()
        )

    results_by_problem: list[list[dict[str, Any]]] = [
        [] for _ in seeds
    ]
    entropy_by_problem: list[list[float]] = [[] for _ in seeds]
    allocation_updates: list[dict[str, float]] = []
    initial_evaluations = [
        evaluate_fixed_placement_episode(
            environment,
            learner,
            ue_seed=ue_seed,
            placement_actions=placement,
        )
        for ue_seed, placement in zip(seeds, placements)
    ]
    if update_placement:
        placement_returns = [
            float(result["placement_return"])
            for result in initial_evaluations
        ]
        learner.store_placement_outcomes(placement_returns)
        placement_metrics = (
            learner.update_placement_rollout()
            if learner.placement_rollout_size == learner.placement_batch_episodes
            else {"placement_updated": 0.0,
                  "placement_episodes_collected": float(learner.placement_rollout_size)}
        )
        placement_metrics["placement_target_accepted"] = 1.0
        placement_target_source = "initial_zero_shot"
    else:
        if learner._placement_rollout:
            raise RuntimeError("Allocation warm-up created a placement rollout")
        placement_returns = []
        placement_metrics = {
            "placement_updated": 0.0,
            "placement_target_accepted": 0.0,
            "placement_episodes_collected": 0.0,
        }
        placement_target_source = "discarded_allocator_warmup"
    best_score: tuple[float] | None = _fixed_batch_score(
        environment, initial_evaluations
    )
    initial_diagnostics = _fixed_batch_diagnostics(
        environment, initial_evaluations
    )
    evaluation_history: list[dict[str, Any]] = [
        {
            "inner_update": 0,
            **initial_diagnostics,
            "policy_kl": 0.0,
            "improved": True,
            "checks_without_improvement": 0,
        }
    ]
    if not update_allocation:
        current_allocation_state = learner.allocation_snapshot()
        current_steps = learner.optimizer_step_counts()
        allocation_unchanged = (
            frozen_allocation_state is not None
            and _nested_state_equal(
                current_allocation_state, frozen_allocation_state
            )
            and current_steps["allocation_actor"]
            == frozen_allocation_steps["allocation_actor"]
            and current_steps["allocation_critic"]
            == frozen_allocation_steps["allocation_critic"]
            and float(learner.allocation_entropy_coef)
            == frozen_allocation_entropy
        )
        if not allocation_unchanged:
            raise RuntimeError(
                "Placement-only training modified frozen allocation state"
            )
        return {
            "placement_metrics": placement_metrics,
            "placement_target_source": "frozen_allocator_zero_shot",
            "placement_target_returns": placement_returns,
            "placement_actions": [
                placement.copy() for placement in placements
            ],
            "placement_orders": list(orders),
            "ue_seeds": list(seeds),
            "initial_evaluations": initial_evaluations,
            "final_evaluations": copy.deepcopy(initial_evaluations),
            "best_evaluations": copy.deepcopy(initial_evaluations),
            "best_score": best_score,
            "best_inner_update": 0,
            "allocation_updates": [],
            "allocation_episode_results_by_problem": [
                [] for _ in seeds
            ],
            "allocation_entropy_coefficients_by_problem": [
                [] for _ in seeds
            ],
            "allocation_inner_updates": 0,
            "allocation_inner_episodes": 0,
            "allocation_training_active": False,
            "allocation_state_unchanged": True,
            "allocation_converged": False,
            "allocation_stopped_by_safety_cap": False,
            "allocation_stop_reason": "frozen",
            "allocation_convergence_change": 0.0,
            "allocation_evaluation_history": evaluation_history,
            "deterministic_evaluation_passes": 1,
        }

    best_evaluations: list[dict[str, Any]] | None = copy.deepcopy(
        initial_evaluations
    )
    best_inner_update = 0
    checks_without_improvement = 0
    stopped_by_patience = False
    for inner_update in range(1, max_inner_updates + 1):
        episode_count = inner_update * len(seeds)
        if allocation_entropy_schedule is not None:
            entropy_coefficient = float(
                allocation_entropy_schedule(episode_count)
            )
            if not math.isfinite(entropy_coefficient) or entropy_coefficient < 0:
                raise ValueError(
                    "allocation entropy schedule must return a finite, "
                    "non-negative value"
                )
            learner.allocation_entropy_coef = entropy_coefficient

        update_results: list[dict[str, Any]] = []
        for problem_index, (ue_seed, placement) in enumerate(
            zip(seeds, placements)
        ):
            global_state, _ = environment.reset(
                options={"ue_seed": ue_seed}
            )
            global_state, placement_reward, terminated, truncated, _ = (
                environment.step(placement)
            )
            if terminated or truncated:
                raise RuntimeError("Placement unexpectedly ended an episode")
            result = _collect_allocation_episode(
                environment,
                learner,
                global_state,
                initial_placement_reward=float(placement_reward),
                defer_allocation_update=True,
                deterministic_actions=deterministic_actions,
            )
            result["inner_update"] = inner_update
            results_by_problem[problem_index].append(result)
            entropy_by_problem[problem_index].append(
                float(learner.allocation_entropy_coef)
            )
            update_results.append(result)

        expected_rollout_size = len(seeds) * int(environment.num_ues)
        if learner.allocation_rollout_size != expected_rollout_size:
            raise RuntimeError(
                "Each allocation update requires one complete episode per "
                f"problem ({expected_rollout_size} transitions)"
            )
        update_metrics = learner.update_allocation_rollout(
            bootstrap_global_state=None,
            bootstrap_done=True,
        )
        allocation_updates.append(update_metrics)
        for result in update_results:
            result["allocation_updates"] = [update_metrics]

        deterministic_evaluations = [
            evaluate_fixed_placement_episode(
                environment,
                learner,
                ue_seed=ue_seed,
                placement_actions=placement,
            )
            for ue_seed, placement in zip(seeds, placements)
        ]
        score = _fixed_batch_score(environment, deterministic_evaluations)
        diagnostics = _fixed_batch_diagnostics(
            environment, deterministic_evaluations
        )
        improved = best_score is None or score[0] > best_score[0] + 1e-12
        matched_best = best_score is not None and score == best_score
        if improved or matched_best:
            # Keep diagnostic best scores without rolling back training state.
            best_score = score
            best_evaluations = copy.deepcopy(deterministic_evaluations)
            best_inner_update = inner_update
        if improved:
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        policy_kl = float(update_metrics["allocation_policy_kl"])
        evaluation_history.append(
            {
                "inner_update": inner_update,
                **diagnostics,
                "policy_kl": policy_kl,
                "improved": improved,
                "matched_best": matched_best,
                "checks_without_improvement": checks_without_improvement,
            }
        )
        if checks_without_improvement >= inner_patience:
            stopped_by_patience = True
            break
    final_evaluations = deterministic_evaluations

    return {
        "placement_metrics": placement_metrics,
        "placement_target_source": placement_target_source,
        "placement_target_returns": placement_returns,
        "placement_actions": [placement.copy() for placement in placements],
        "placement_orders": list(orders),
        "ue_seeds": list(seeds),
        "initial_evaluations": initial_evaluations,
        "final_evaluations": final_evaluations,
        "best_evaluations": best_evaluations,
        "best_score": best_score,
        "best_inner_update": best_inner_update,
        "allocation_updates": allocation_updates,
        "allocation_episode_results_by_problem": results_by_problem,
        "allocation_entropy_coefficients_by_problem": entropy_by_problem,
        "allocation_inner_updates": inner_update,
        "allocation_inner_episodes": inner_update * len(seeds),
        "allocation_training_active": True,
        "allocation_state_unchanged": False,
        "allocation_converged": False,
        "allocation_stopped_by_safety_cap": not stopped_by_patience,
        "allocation_stop_reason": "no_improvement" if stopped_by_patience else "max_inner_updates",
        "allocation_convergence_change": float(
            evaluation_history[-1]["policy_kl"]
        ),
        "allocation_evaluation_history": evaluation_history,
        "deterministic_evaluation_passes": inner_update + 1,
    }


def train_frozen_placement_cycle(
    environment: Any,
    learner: StrictLocalIndependentMAPPO,
    *,
    ue_seed: int,
    convergence_tolerance: float = 0.03,
    convergence_window: int = 10,
    convergence_patience: int = 4,
    allocation_batch_episodes: int = 10,
    allocation_entropy_schedule: Callable[[int], float] | None = None,
    deterministic_actions: bool = False,
) -> dict[str, Any]:
    """Optimize allocation under one fixed map and placement until stable.

    Convergence compares consecutive, non-overlapping blocks of allocation
    returns. The placement target is the mean return of the final block.
    """

    if not 0.0 < convergence_tolerance < 1.0:
        raise ValueError("convergence_tolerance must be in (0, 1)")
    if convergence_window <= 0 or convergence_patience <= 0:
        raise ValueError("convergence window and patience must be positive")
    if allocation_batch_episodes <= 0:
        raise ValueError("allocation_batch_episodes must be positive")
    if convergence_window % allocation_batch_episodes:
        raise ValueError(
            "convergence_window must be divisible by "
            "allocation_batch_episodes"
        )
    if learner.allocation_rollout_size:
        raise RuntimeError("Start a frozen-placement cycle with an empty rollout")

    global_state, _ = environment.reset(options={"ue_seed": int(ue_seed)})
    placement_actions = learner.act_placement(
        global_state,
        environment.get_actor_observations(),
        deterministic=deterministic_actions,
        store_for_update=True,
    )
    placement_actions = _validated_joint_action(
        placement_actions,
        learner.num_uavs,
    )

    inner_results: list[dict[str, Any]] = []
    window_means: list[float] = []
    stable_checks = 0
    relative_change = float("nan")
    converged = False
    entropy_coefficients: list[float] = []

    inner_episode = 0
    while not converged:
        inner_episode += 1
        if allocation_entropy_schedule is not None:
            entropy_coefficient = float(
                allocation_entropy_schedule(inner_episode)
            )
            if not math.isfinite(entropy_coefficient) or entropy_coefficient < 0:
                raise ValueError(
                    "allocation entropy schedule must return a finite, "
                    "non-negative value"
                )
            learner.allocation_entropy_coef = entropy_coefficient
        entropy_coefficients.append(float(learner.allocation_entropy_coef))
        if inner_episode > 1:
            global_state, _ = environment.reset(
                options={"ue_seed": int(ue_seed)}
            )
        global_state, placement_reward, terminated, truncated, _ = (
            environment.step(placement_actions)
        )
        if terminated or truncated:
            raise RuntimeError("Placement unexpectedly terminated the episode")

        result = _collect_allocation_episode(
            environment,
            learner,
            global_state,
            initial_placement_reward=float(placement_reward),
            defer_allocation_update=True,
            deterministic_actions=deterministic_actions,
        )
        inner_results.append(result)

        if inner_episode % allocation_batch_episodes == 0:
            result["allocation_updates"].append(
                learner.update_allocation_rollout(
                    bootstrap_global_state=None,
                    bootstrap_done=True,
                )
            )

        if inner_episode % convergence_window == 0:
            current_mean = float(
                np.mean(
                    [
                        item["allocation_return"]
                        for item in inner_results[-convergence_window:]
                    ]
                )
            )
            if not math.isfinite(current_mean):
                raise FloatingPointError(
                    "Allocation convergence return became non-finite"
                )
            window_means.append(current_mean)
            if len(window_means) < 2:
                continue
            previous_mean = window_means[-2]
            relative_change = abs(current_mean - previous_mean) / max(
                abs(previous_mean),
                1e-8,
            )
            stable_checks = (
                stable_checks + 1
                if relative_change < convergence_tolerance
                else 0
            )
            if stable_checks >= convergence_patience:
                converged = True

    if learner.allocation_rollout_size:
        inner_results[-1]["allocation_updates"].append(
            learner.update_allocation_rollout(
                bootstrap_global_state=None,
                bootstrap_done=True,
            )
        )

    score_results = inner_results[-convergence_window:]
    placement_score = float(
        np.mean([item["placement_return"] for item in score_results])
    )
    learner.store_placement_outcome(placement_score)
    if learner.placement_rollout_size >= learner.placement_batch_episodes:
        placement_metrics = learner.update_placement_rollout()
    else:
        placement_metrics = {
            "placement_updated": 0.0,
            "placement_episodes_collected": float(
                learner.placement_rollout_size
            ),
        }
    placement_metrics["placement_target_accepted"] = 1.0

    allocation_updates = [
        update
        for result in inner_results
        for update in result["allocation_updates"]
    ]
    final_result = inner_results[-1]
    return {
        "episode_return": placement_score,
        "allocation_return": placement_score,
        "placement_return": placement_score,
        "served_count": int(final_result["served_count"]),
        "served_ratio": float(final_result["served_ratio"]),
        "placement_metrics": placement_metrics,
        "allocation_updates": allocation_updates,
        "final_info": final_result["final_info"],
        "placement_actions": placement_actions.copy(),
        "placement_score_infos": [
            result["final_info"] for result in score_results
        ],
        "allocation_episode_results": inner_results,
        "allocation_inner_episodes": len(inner_results),
        "allocation_converged": converged,
        "allocation_convergence_change": relative_change,
        "allocation_convergence_window_means": window_means,
        "allocation_entropy_coefficients": entropy_coefficients,
    }


@torch.no_grad()
def evaluate_one_episode(
    environment: Any,
    learner: StrictLocalIndependentMAPPO,
    *,
    reset_seed: int | None = None,
    ue_seed: int | None = None,
    placement_order: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate placement and all five local bidders deterministically."""

    if reset_seed is not None and ue_seed is not None:
        raise ValueError("Pass either reset_seed or ue_seed, not both")
    if ue_seed is not None:
        global_state, info = environment.reset(
            options={"ue_seed": int(ue_seed)}
        )
    elif reset_seed is None:
        global_state, info = environment.reset()
    else:
        global_state, info = environment.reset(seed=reset_seed)

    placement_actions = learner.act_placement(
        global_state,
        environment.get_actor_observations(),
        deterministic=True,
        store_for_update=False,
        selection_order=placement_order,
    )
    joint_placement_actions = _validated_joint_action(
        placement_actions, learner.num_uavs
    )
    global_state, immediate_placement_reward, terminated, truncated, info = (
        environment.step(joint_placement_actions)
    )
    placement_episode_return = float(immediate_placement_reward)
    allocation_episode_return = 0.0
    done = bool(terminated or truncated)

    while not done:
        allocation_action, _ = learner.act_allocation(
            global_state,
            environment.get_actor_observations(),
            deterministic=True,
            store_for_update=False,
        )
        global_state, environment_reward, terminated, truncated, info = (
            environment.step(allocation_action)
        )
        allocation_team_reward, placement_team_reward = _split_team_rewards(
            environment_reward,
            info["reward_breakdown"],
        )
        allocation_episode_return += allocation_team_reward
        placement_episode_return += placement_team_reward
        done = bool(terminated or truncated)

    return {
        "episode_return": float(placement_episode_return),
        "allocation_return": float(allocation_episode_return),
        "placement_return": float(placement_episode_return),
        "served_count": int(info["served_count"]),
        "served_ratio": float(info["served_ratio"]),
        "placement_actions": placement_actions.copy(),
        "final_info": info,
    }
