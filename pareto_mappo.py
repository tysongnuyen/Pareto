"""Shared-budget Pareto allocation with sampled PPO–GAE and compact local actors."""
import torch
from torch import nn
from ver2 import gym_env
from local_mappo_baseline import LocalMAPPOBaseline
from strict_local_independent_mappo import (
    CentralizedGlobalCritic, LocalAutoregressiveAllocationActor,
)


class SharedParetoEnv(gym_env):
    def __init__(self, **kwargs):
        if kwargs.get("coordination_mode", "none") != "none":
            raise ValueError("Shared Pareto requires coordination_mode=none")
        super().__init__(
            **kwargs, allocation_mask_mode="pareto", allocation_cost_features=False
        )


class ParetoAllocationActor(LocalAutoregressiveAllocationActor):
    """Compact local residual actor; power is selected by the singleton mask."""

    def __init__(self, *args, num_power_levels, **kwargs):
        super().__init__(*args, num_power_levels=num_power_levels, **kwargs)
        self.num_power_levels = num_power_levels
        # Retain the common action/log-probability API without a trainable head.
        self.power_head = None

    def power_logits(self, context, prb_index):
        return context.new_zeros((*context.shape[:-1], self.num_power_levels))


class ParetoMAPPO(LocalMAPPOBaseline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.CHECKPOINT_FRAMEWORK = "local_pareto_shared_no_cost_ppo_gae_v2"
        self.allocation_policy = "participate_then_prb_with_deterministic_pareto_power"
        self.allocation_actor_feature_version = "compact_qos_link_residual_pareto_v1"
        self.allocation_critic_feature_version = "global_attention_no_cost_v1"
        # Keep all retained initial weights and the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            actors = nn.ModuleList([
                ParetoAllocationActor(
                    self.num_uavs, self.max_prbs, self.coordination_feature_dim,
                    self.embed_dim, self.num_heads,
                    num_power_levels=self.num_power_levels,
                ) for _ in range(self.num_uavs)
            ])
            actors.load_state_dict({
                key: value for key, value in self.allocation_actors.state_dict().items()
                if ".power_head." not in key
            })
            critic = CentralizedGlobalCritic(
                self.embed_dim, self.num_heads, include_candidate_features=False,
                allocation_cost_dim=0,
            )
            state = self.allocation_critic.state_dict()
            key = "relation_encoder.edge_encoder.0.weight"
            width = critic.relation_encoder.edge_encoder[0].in_features
            state[key] = state[key][:, :width]
            critic.load_state_dict(state)
        actor_lr = self.allocation_actor_optimizers[0].param_groups[0]["lr"]
        critic_lr = self.allocation_critic_optimizer.param_groups[0]["lr"]
        self.allocation_actors = actors.to(self.device)
        self.allocation_critic = critic.to(self.device)
        self.allocation_actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.allocation_actors
        ]
        self.allocation_critic_optimizer = torch.optim.Adam(
            self.allocation_critic.parameters(), lr=critic_lr
        )

    def _global_state_to_tensors(self, state, *, include_candidate_features=True,
                                include_allocation_cost_features=True):
        return super()._global_state_to_tensors(
            state, include_candidate_features=include_candidate_features,
            include_allocation_cost_features=False,
        )

    @classmethod
    def from_env(cls, environment, **kwargs):
        if not environment.shared_prb_pool or environment.allocation_mask_mode != "pareto":
            raise ValueError("ParetoMAPPO requires the shared-budget Pareto environment")
        return super().from_env(environment, **kwargs)

    def _require_coverage_allocation(self, observation, participate_mask, pair_mask):
        """Validate the shared Pareto contract at the inherited collector boundary."""
        if participate_mask.shape != (2,) or pair_mask.shape != (self.max_prbs, self.num_power_levels):
            raise ValueError("Allocation masks have the wrong shape")
        if any(m.dtype != torch.bool for m in (participate_mask, pair_mask)):
            raise TypeError("Allocation masks must be boolean")
        if any(m.device != next(self.parameters()).device for m in (participate_mask, pair_mask)):
            raise ValueError("Allocation masks are on the wrong device")
        if not bool(participate_mask[0]) or bool(participate_mask[1]) != bool(pair_mask.any()):
            raise ValueError("Reject/participation masks disagree with Pareto pairs")
        if bool(pair_mask.any()) and not bool(observation["current_link_features"][1] >= .5):
            raise ValueError("Out-of-coverage UAV cannot bid")
        self._pareto_prb_power_map(pair_mask)
