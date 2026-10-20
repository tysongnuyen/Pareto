from __future__ import annotations
import copy
from contextlib import contextmanager
from typing import Any, Sequence
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import env
seed = list(range(14, 24))


def allocation_cost_key(cost: float) -> float:
    """Quantize ranking only; reward and resource accounting retain precision."""

    # ponytail: 1e-12 rank resolution; increase if smaller costs are meaningful.
    return round(float(cost), 12)


def resolve_reward_weights(
    coordination_mode: str,
    power_weight: float | None,
    prb_weight: float | None,
) -> tuple[float, float]:
    """Resolve CLI/env defaults identically, including resume fingerprints."""

    if coordination_mode not in ("full", "basic", "none"):
        raise ValueError("coordination_mode must be one of: full, basic, none")
    # In none, one 0.1 W step and one PRB both cost 1/135.
    default_power_weight = 4.0 / 27.0
    default_prb_weight = 1.0 if coordination_mode == "none" else 4.0 / 27.0
    weights = (
        float(default_power_weight if power_weight is None else power_weight),
        float(default_prb_weight if prb_weight is None else prb_weight),
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in weights):
        raise ValueError("Reward and cost weights must be finite and non-negative")
    return weights


def _validated_multidiscrete_action(
    action: Any,
    action_space: spaces.MultiDiscrete,
    label: str,
) -> np.ndarray:
    """Validate the caller's dtype and values before producing int64 indices."""

    try:
        values = np.asarray(action)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} action cannot be converted to an array") from exc
    if not action_space.contains(values):
        raise ValueError(f"{label} action {values} is outside {action_space}")
    return values.astype(np.int64, copy=False)


class _DistinctPlacementSpace(spaces.MultiDiscrete):
    """Multi-discrete candidate IDs with distinct UAV coordinates."""

    def __init__(
        self,
        candidate_xyz: Sequence[tuple[float, float, float]],
        num_uavs: int,
    ) -> None:
        super().__init__(
            np.full(num_uavs, len(candidate_xyz), dtype=np.int64)
        )
        coordinate_lookup: dict[tuple[float, float, float], int] = {}
        coordinate_ids: list[int] = []
        candidates_by_coordinate: list[list[int]] = []
        for candidate_id, coordinate in enumerate(candidate_xyz):
            coordinate_id = coordinate_lookup.get(coordinate)
            if coordinate_id is None:
                coordinate_id = len(candidates_by_coordinate)
                coordinate_lookup[coordinate] = coordinate_id
                candidates_by_coordinate.append([])
            coordinate_ids.append(coordinate_id)
            candidates_by_coordinate[coordinate_id].append(candidate_id)
        if len(candidates_by_coordinate) < num_uavs:
            raise ValueError("Not enough distinct placement coordinates for all UAVs")
        self._coordinate_ids = np.asarray(coordinate_ids, dtype=np.int64)
        self._candidates_by_coordinate = tuple(
            np.asarray(candidate_ids, dtype=np.int64)
            for candidate_ids in candidates_by_coordinate
        )

    def contains(self, x: Any) -> bool:
        if not super().contains(x):
            return False
        candidate_ids = np.asarray(x, dtype=np.int64)
        return bool(
            np.unique(self._coordinate_ids[candidate_ids]).size
            == candidate_ids.size
        )

    def sample(
        self,
        mask: tuple[np.ndarray, ...] | None = None,
        probability: tuple[np.ndarray, ...] | None = None,
    ) -> np.ndarray:
        # Let Gymnasium validate optional masks/probabilities and preserve its
        # usual sample whenever it already satisfies the coupled constraint.
        sampled = super().sample(mask=mask, probability=probability)
        if self.contains(sampled):
            return sampled

        weights = np.ones((self.shape[0], len(self._coordinate_ids)))
        if mask is not None:
            for uav_index, row_mask in enumerate(mask):
                weights[uav_index] = np.asarray(row_mask, dtype=np.float64)
                if not weights[uav_index].any():
                    weights[uav_index, int(self.start[uav_index])] = 1.0
        elif probability is not None:
            for uav_index, row_probability in enumerate(probability):
                weights[uav_index] = np.asarray(
                    row_probability, dtype=np.float64
                )

        group_weights = np.stack(
            [
                np.bincount(
                    self._coordinate_ids,
                    weights=row,
                    minlength=len(self._candidates_by_coordinate),
                )
                for row in weights
            ]
        )
        order = np.argsort(
            np.count_nonzero(group_weights, axis=1), kind="stable"
        )
        assignments = np.full(self.shape[0], -1, dtype=np.int64)

        def assign(position: int, used: set[int]) -> bool:
            if position == len(order):
                return True
            uav_index = int(order[position])
            available = np.flatnonzero(group_weights[uav_index] > 0)
            available = available[
                [int(value) not in used for value in available]
            ]
            if not len(available):
                return False
            priorities = -np.log(
                np.maximum(self.np_random.random(len(available)), 1e-300)
            ) / group_weights[uav_index, available]
            for coordinate_id in available[np.argsort(priorities)]:
                coordinate_id = int(coordinate_id)
                assignments[uav_index] = coordinate_id
                if assign(position + 1, used | {coordinate_id}):
                    return True
            assignments[uav_index] = -1
            return False

        if not assign(0, set()):
            raise ValueError(
                "Placement mask/probability has no action with distinct "
                "(x, y, z) coordinates"
            )
        for uav_index, coordinate_id in enumerate(assignments):
            candidates = self._candidates_by_coordinate[int(coordinate_id)]
            candidate_weights = weights[uav_index, candidates]
            candidates = candidates[candidate_weights > 0]
            candidate_weights = candidate_weights[candidate_weights > 0]
            sampled[uav_index] = self.np_random.choice(
                candidates, p=candidate_weights / candidate_weights.sum()
            )
        return sampled.astype(self.dtype, copy=False)


class gym_env(gym.Env):
    #metadata = {"render_modes": []}

    PLACEMENT_PHASE = 0
    ALLOCATION_PHASE = 1
    TERMINAL_PHASE = 2
    COORDINATION_FEATURE_DIM = 8
    EPISODES_PER_SEED = 200
    MIN_GAIN_DB = -200.0
    MAX_GAIN_DB = 100.0
    def __init__(self,served_reward: float = 1.0,
            power_weight: float | None = None,
            prb_weight: float | None = None,
            shuffle_ue_order: bool = True,
            seed: Sequence[int] = seed,
            coordination_mode: str = "none",
            allocation_mask_mode: str | None = None,
            allocation_cost_features: bool = True)->None:
        super().__init__()
        self.power_weight, self.prb_weight = resolve_reward_weights(
            coordination_mode, power_weight, prb_weight
        )
        self.coordination_mode = coordination_mode
        self.shared_prb_pool = coordination_mode == "none"
        self.allocation_mask_mode = (
            ("coverage" if self.shared_prb_pool else "pareto")
            if allocation_mask_mode is None else allocation_mask_mode
        )
        if self.allocation_mask_mode not in {"coverage", "pareto"}:
            raise ValueError("allocation_mask_mode must be coverage or pareto")
        if not self.shared_prb_pool and self.allocation_mask_mode != "pareto":
            raise ValueError("Legacy resource modes require Pareto masks")
        self.include_allocation_cost_features = bool(allocation_cost_features)
        self.seed = tuple(int(value) for value in seed)
        if not self.seed:
            raise ValueError("seed must not be empty")
        self.default_seed = self.seed[0]
        self.served_reward = float(served_reward)
        weights = (self.served_reward, self.power_weight, self.prb_weight)
        if not all(np.isfinite(value) and value >= 0.0 for value in weights):
            raise ValueError("Reward and cost weights must be finite and non-negative")
        # Weights scale actual normalized consumption; their sum is not a
        # per-action cost and need not be below served_reward in any mode.
        self.shuffle_ue_order = bool(shuffle_ue_order)

        self.solutions = list(env.make_solution())
        self.len_solutions = len(self.solutions)
        self.uavs = list(env.make_uav())
        self.num_uavs = len(self.uavs)
        self.ues = list(env.make_ue())
        self.num_ues = len(self.ues)
        self.num_solutions = self.len_solutions
        self.max_prbs_per_uav = np.array([int(uav.max_prbs) for uav in self.uavs], dtype=np.int32)
        self.max_power_w_per_uav = np.array([float(uav.max_power_w) for uav in self.uavs], dtype = np.float64)

        self.total_max_prbs = int(env.total_system_prbs)
        self.total_max_power_w = float(self.max_power_w_per_uav.sum())
        if self.shared_prb_pool:
            # Per-UAV bounds describe access to the same pool, never quotas.
            self.max_prbs_per_uav.fill(self.total_max_prbs)
            self.max_prbs_per_action = int(env.allocation_max_prbs)
            self.prb_levels = np.arange(1, self.max_prbs_per_action + 1, dtype=np.int32)
            self.power_levels_w = env.make_power_levels_w(
                max_power_w=env.allocation_max_power_w, include_zero=False
            )
        else:
            self.max_prbs_per_action = int(self.max_prbs_per_uav.max())
            self.prb_levels = np.arange(1, self.max_prbs_per_action + 1, dtype=np.int32)
            self.power_levels_w = env.make_power_levels_w()
        self.max_prbs = len(self.prb_levels)
        self.power_penalty_denominator_w = (
            float(env.uav_max_power_w) if self.shared_prb_pool else self.total_max_power_w
        )
        self.allocation_action_semantics = (
            "participate_prb_index_power_index_shared_coverage_v2"
            if self.shared_prb_pool else "participate_prb_index_power_index_pareto_v1"
        )
        if self.shared_prb_pool and self.allocation_mask_mode == "pareto":
            self.allocation_action_semantics = "participate_prb_index_power_index_shared_pareto_v1"
        self.num_power_levels = len(self.power_levels_w)
        # Placement and allocation deliberately have different controllers.
        # Five placement actors emit candidate IDs. Five allocation actors use
        # local observations plus the configured team broadcast and each emit
        # [participate, PRB index, power index]. The bid is derived from
        # resource use by the fixed auction, not learned.
        self.max_action_choices = self.num_solutions
        self.placement_action_space = _DistinctPlacementSpace(
            [
                (position.x_u, position.y_u, position.h_u)
                for position in self.solutions
            ],
            self.num_uavs,
        )
        self.allocation_action_space = spaces.MultiDiscrete(
            np.tile(
                np.asarray([2, self.max_prbs, self.num_power_levels]),
                (self.num_uavs, 1),
            ).astype(np.int64)
        )
        self.action_space = spaces.Dict(
            {
                "placement": self.placement_action_space,
                "allocation": self.allocation_action_space,
            }
        )

        self._prepare_static_features()
        self._build_observation_spaces()

        self.ues: list[env.UE] = []
        self.episode_ue_seed: int | None = None
        self._seed_index = 0
        self._episodes_on_seed = 0
        self._episode_uses_scheduled_seed = True
        self.phase = self.PLACEMENT_PHASE
        self.current_ue = 0
        self._counterfactual_proposals = None
        self._counterfactual_gains = None
        self.placement_indices = np.full(self.num_uavs, -1, dtype=np.int32)
        self.selected_positions: list[env.solution] = []
        self.shared_prbs_remaining = self.total_max_prbs
        self._remaining_prbs = self.max_prbs_per_uav.copy()
        self.remaining_power_w = self.max_power_w_per_uav.copy()
        self.processed = np.zeros(self.num_ues, dtype=np.int8)
        self.served = np.zeros(self.num_ues, dtype=np.int8)
        self.serving_uav = np.full(self.num_ues, -1, dtype=np.int32)
        self.allocated_prbs_ratio = np.zeros(self.num_ues, dtype=np.float32)
        self.allocated_power_ratio = np.zeros(self.num_ues, dtype=np.float32)
        self.achieved_rate_ratio = np.zeros(self.num_ues, dtype=np.float32)
        self.link_features = np.zeros(
            (self.num_uavs, self.num_ues, 3), dtype=np.float32
        )
        self._allocation_cost_features = np.zeros(
            (self.num_uavs, self.num_ues, 2 * self.max_prbs), dtype=np.float32
        )
        self._allocation_cost_features.setflags(write=False)
        self.pareto_pair_masks = np.zeros(
            (
                self.num_uavs,
                self.num_ues,
                self.max_prbs,
                self.num_power_levels,
            ),
            dtype=np.int8,
        )
        self.allocations: list[dict[str, Any] | None] = [None] * self.num_ues
        self.last_evaluation: dict[str, Any] | None = None
        self.last_reward_breakdown = self._empty_reward_breakdown()
        self._rng_initialized = False

    @property
    def remaining_prbs(self) -> np.ndarray:
        """Expose each UAV's access balance; shared entries must not be summed."""

        if not self.shared_prb_pool:
            return self._remaining_prbs
        balances = np.full(self.num_uavs, self.shared_prbs_remaining, dtype=np.int32)
        balances.setflags(write=False)
        return balances

    @remaining_prbs.setter
    def remaining_prbs(self, values: np.ndarray) -> None:
        if self.shared_prb_pool:
            raise AttributeError("Set shared_prbs_remaining for the shared PRB pool")
        self._remaining_prbs = values

    def _prepare_static_features(self)->None:
        self.max_radius = float(env.coverage_radius(max(env.phys.H_U), max(env.phys.Theta_U)))
        self.max_height = float(max(env.phys.H_U))
        self.max_theta = float(max(env.phys.Theta_U))
        self.max_required_mbps = float(max(env.required_mbs))
        self.area_diagonal_m = float(np.hypot(env.width_m, env.length_m))
        self.candidate_positions = np.asarray(
            [
                [
                    position.x_u,
                    position.y_u,
                    position.h_u,
                    position.theta_u,
                    position.r_u,
                ]
                for position in self.solutions
            ],
            dtype=np.float32,
        )
        scales = np.asarray(
            [
                env.width_m,
                env.length_m,
                self.max_height,
                self.max_theta,
                self.max_radius,
            ],
            dtype=np.float32,
        )             
        self.candidate_features = np.divide(
            self.candidate_positions,
            scales,
            out=np.zeros_like(self.candidate_positions),
            where=scales > 0,
        ).clip(0.0, 1.0)
        # These arrays are immutable across every episode. Returning read-only
        # views avoids copying ~50k floats on each of 680k allocation steps.
        self.candidate_positions.setflags(write=False)
        self.candidate_features.setflags(write=False)

        self.uav_limits = np.column_stack(
            (
                self.max_prbs_per_uav / float(
                    self.total_max_prbs if self.shared_prb_pool else self.max_prbs
                ),
                self.max_power_w_per_uav
                / float(self.max_power_w_per_uav.max()),
            )
        ).astype(np.float32)
        self.agent_ids = np.eye(self.num_uavs, dtype=np.int8)


    def _build_observation_spaces(self) -> None:
        ue_low = np.asarray(
            [0.0, 0.0, min(env.required_mbs)], dtype=np.float32
        )
        ue_high = np.asarray(
            [env.width_m, env.length_m, self.max_required_mbps],
            dtype=np.float32,
        )
        position_high = np.asarray(
            [
                env.width_m,
                env.length_m,
                self.max_height,
                self.max_theta,
                self.max_radius,
            ],
            dtype=np.float32,
        )
        #Cai nay la cho global tuc la cac ham co dau global thi se su dung observation nay
        self.observation_space = spaces.Dict(
            {
                "phase": spaces.Discrete(3),
                "current_ue": spaces.Discrete(self.num_ues + 1),
                "current_ue_mask": spaces.MultiBinary(self.num_ues),
                "global_ues": spaces.Box(
                    low=np.tile(ue_low, (self.num_ues, 1)),
                    high=np.tile(ue_high, (self.num_ues, 1)),
                    dtype=np.float32,
                ),
                "ue_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_ues, 3),
                    dtype=np.float32,
                ),
                "candidate_positions": spaces.Box(
                    low=0.0,
                    high=np.tile(position_high, (self.num_solutions, 1)),
                    dtype=np.float32,
                ),
                "candidate_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_solutions, 5),
                    dtype=np.float32,
                ),
                "placement_indices": spaces.Box(
                    low=-1,
                    high=self.num_solutions - 1,
                    shape=(self.num_uavs,),
                    dtype=np.int32,
                ),
                "uav_positions": spaces.Box(
                    low=0.0,
                    high=np.tile(position_high, (self.num_uavs, 1)),
                    dtype=np.float32,
                ),
                "uav_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_uavs, 7),
                    dtype=np.float32,
                ),
                "uav_limits": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_uavs, 2),
                    dtype=np.float32,
                ),
                "remaining_prbs": spaces.Box(
                    low=0,
                    high=self.max_prbs_per_uav,
                    dtype=np.int32,
                ),
                "remaining_power_w": spaces.Box(
                    low=0.0,
                    high=self.max_power_w_per_uav.astype(np.float32),
                    dtype=np.float32,
                ),
                # Xu ly bao nhieu cai nay de in ra
                "processed": spaces.MultiBinary(self.num_ues),
                # Phuc vu bao nhieu cai nay de in ra
                "served": spaces.MultiBinary(self.num_ues),
                #Cai nay de ghi log
                "serving_uav": spaces.Box(
                    low=-1,
                    high=self.num_uavs - 1,
                    shape=(self.num_ues,),
                    dtype=np.int32,
                ),
                # Dung de ghi log
                "allocated_prbs_ratio": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_ues,),
                    dtype=np.float32,
                ),
                "allocated_power_ratio": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_ues,),
                    dtype=np.float32,
                ),
                "achieved_rate_ratio": spaces.Box(
                    low=0.0,
                    high=2.0,
                    shape=(self.num_ues,),
                    dtype=np.float32,
                ),
                "link_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_uavs, self.num_ues, 3),
                    dtype=np.float32,
                ),
                "allocation_cost_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_uavs, self.num_ues, 2 * self.max_prbs),
                    dtype=np.float32,
                ),
                #Danh dau Agent nao phuc vu UAV nao
                "agent_ids": spaces.MultiBinary((self.num_uavs, self.num_uavs)),
                #Cac action nao duoc chon may cai nay cho critic
                #Allocation thi chi duoc biet tap hanh dong chung va chung cung dung de lay doi chieu ket qua thoi vi du nhu chon cap phat 5prb thi o thu 5 se duoc danh dau la tru
                "action_mask": spaces.MultiBinary(
                    (self.num_uavs, self.max_action_choices)
                ),
                "allocation_uav_mask": spaces.MultiBinary(self.num_uavs + 1),
                "allocation_participate_mask": spaces.MultiBinary(
                    (self.num_uavs, 2)
                ),
                "allocation_prb_mask": spaces.MultiBinary(
                    (self.num_uavs, self.max_prbs)
                ),
                "allocation_power_mask": spaces.MultiBinary(
                    (self.num_uavs, self.num_power_levels)
                ),
                "allocation_pair_mask": spaces.MultiBinary(
                    (
                        self.num_uavs,
                        self.max_prbs,
                        self.num_power_levels,
                    )
                ),
            }
        )
        #Obeservation space cua actor
        self.single_actor_observation_space = spaces.Dict(
            {
                "phase": spaces.Discrete(3),
                "agent_id": spaces.Discrete(self.num_uavs), # Xac dinh xem UAV nao 1, hay 2 hay 3,...
                "agent_id_one_hot": spaces.MultiBinary(self.num_uavs), # Dinh dang kieu [0,0,1,0,0] De tranh agent vo tinh hoc,tim ra rang ID cua UAV hay so thu tu co tac dong len kq
                "ue_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_ues, 3),
                    dtype=np.float32,
                ),
                "candidate_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_solutions, 5),
                    dtype=np.float32,
                ),
                "own_uav_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(7,),
                    dtype=np.float32,
                ),
                "own_uav_limits": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
                # Allocation actors are allowed to see only UEs covered by
                # their own UAV. Rows outside that local set are zero-padded
                # and excluded by local_ue_mask.
                # [relative x, relative y, demand, normalized distance,
                #  covered, normalized gain, processed, served, current].
                "local_ue_features": spaces.Box(
                    low=np.tile(
                        np.asarray(
                            [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            dtype=np.float32,
                        ),
                        (self.num_ues, 1),
                    ),
                    high=np.ones((self.num_ues, 9), dtype=np.float32),
                    dtype=np.float32,
                ),
                #Nhung UE nao co trong vung phu cua UAV do
                "local_ue_mask": spaces.MultiBinary(self.num_ues),
                #Xet UE hien tai dang duoc env chon va cac thon tin cua no
                "current_ue_mask": spaces.MultiBinary(self.num_ues),
                "current_ue_features": spaces.Box(
                    # [relative x, relative y, normalized demand].
                    low=np.asarray([-1.0, -1.0, 0.0], dtype=np.float32),
                    high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "current_link_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(3,),
                    dtype=np.float32,
                ),
                #Cac feature lien quan toi khoang cach nhung ma hinh nhu la hien tai em loai bo roi
                #Do ban dau xay dung cho nhieu che do khac nen no van con
                "coordination_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.COORDINATION_FEATURE_DIM,),
                    dtype=np.float32,
                ),
                #Cai nay nhu doan duoi cai global
                #Do ban dau em suy nghi toi viec cho biet het thong
                "action_mask": spaces.MultiBinary(self.max_action_choices),
                "allocation_participate_mask": spaces.MultiBinary(2),
                "allocation_prb_mask": spaces.MultiBinary(self.max_prbs),
                "allocation_power_mask": spaces.MultiBinary(
                    self.num_power_levels
                ),
                "allocation_pair_mask": spaces.MultiBinary(
                    (self.max_prbs, self.num_power_levels)
                ),
            }
        )

    def reset(self,seed: int | None = None,options: dict[str,Any] |  None = None)-> tuple[dict[str,np.ndarray | np.int64], dict[str,Any]]:
        if self._counterfactual_proposals is not None:
            raise RuntimeError("Cannot reset inside the counterfactual proposal cache")
        if seed is not None:
            super().reset(seed=seed)
            self._rng_initialized = True
        elif not self._rng_initialized:
            super().reset(seed=self.default_seed)
            self._rng_initialized = True
        else:
            super().reset(seed=None)

        if options and "ue_seed" in options:
            ue_seed = int(options["ue_seed"])
            self._episode_uses_scheduled_seed = False
        else:
            ue_seed = self.seed[self._seed_index]
            self._episode_uses_scheduled_seed = True
        self.episode_ue_seed = ue_seed

        self.ues = list(env.make_ue(seed=ue_seed))
        if len(self.ues) != self.num_ues:
            raise ValueError(
                f"env.make_ue() returned {len(self.ues)} UEs, expected "
                f"{self.num_ues}"
            )
        #Dam bao rang UE duoc lua chon ngau nhien
        if self.shuffle_ue_order:
            # Each physical layout always has one reproducible UE sequence.
            order_rng = np.random.default_rng(ue_seed)
            order = order_rng.permutation(self.num_ues)
            self.ues = [self.ues[int(index)] for index in order]

        self.phase = self.PLACEMENT_PHASE
        self.current_ue = 0
        self.placement_indices.fill(-1)
        self.selected_positions = []
        self.shared_prbs_remaining = self.total_max_prbs
        self._remaining_prbs = self.max_prbs_per_uav.copy()
        self.remaining_power_w = self.max_power_w_per_uav.copy()
        self.processed.fill(0)
        self.served.fill(0)
        self.serving_uav.fill(-1)
        self.allocated_prbs_ratio.fill(0.0)
        self.allocated_power_ratio.fill(0.0)
        self.achieved_rate_ratio.fill(0.0)
        self.link_features.fill(0.0)
        # Replace rather than mutate: earlier observations may retain the table.
        self._allocation_cost_features = np.zeros_like(self._allocation_cost_features)
        self._allocation_cost_features.setflags(write=False)
        self.pareto_pair_masks.fill(0)
        self.allocations = [None] * self.num_ues
        self.last_evaluation = None
        self.last_reward_breakdown = self._empty_reward_breakdown()
        return self._get_obs(), self._get_info()

    def _get_ue_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return raw UE data and a numerically stable normalized copy."""

        global_ues = np.asarray(
            [[ue.x_m, ue.y_m, ue.require_mbps] for ue in self.ues],
            dtype=np.float32,
        )
        scales = np.asarray(
            [env.width_m, env.length_m, self.max_required_mbps],
            dtype=np.float32,
        )
        ue_features = np.divide(
            global_ues,
            scales,
            out=np.zeros_like(global_ues),
            where=scales > 0,
        ).clip(0.0, 1.0)
        return global_ues, ue_features

    def _get_uav_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return raw UAV placements and normalized geometry/resource state."""

        positions = np.zeros((self.num_uavs, 5), dtype=np.float32)
        if self.selected_positions:
            positions[:] = [
                [
                    position.x_u,
                    position.y_u,
                    position.h_u,
                    position.theta_u,
                    position.r_u,
                ]
                for position in self.selected_positions
            ]

        geometry_scales = np.asarray(
            [
                env.width_m,
                env.length_m,
                self.max_height,
                self.max_theta,
                self.max_radius,
            ],
            dtype=np.float32,
        )
        geometry = np.divide(
            positions,
            geometry_scales,
            out=np.zeros_like(positions),
            where=geometry_scales > 0,
        ).clip(0.0, 1.0)
        remaining_prb_ratio = (
            self.remaining_prbs / self.max_prbs_per_uav
        ).astype(np.float32)
        remaining_power_ratio = (
            self.remaining_power_w / self.max_power_w_per_uav
        ).astype(np.float32)
        features = np.column_stack(
            (geometry, remaining_prb_ratio, remaining_power_ratio)
        ).astype(np.float32)
        return positions, features

    def _get_current_ue_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_ues, dtype=np.int8)
        if self.phase == self.ALLOCATION_PHASE:
            mask[self.current_ue] = 1
        return mask

    def _get_local_ue_view(
        self,
        uav_index: int,
        ue_features: np.ndarray,
        uav_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the locally observable UE set for one allocation actor.

        The array keeps a fixed ``num_ues`` size for Gymnasium compatibility,
        but every UE outside this UAV's coverage is represented by an all-zero
        row and a zero entry in ``local_mask``.  Therefore an allocation actor
        cannot read another UAV's individual coverage, link quality, or
        resource state. Depending on ``coordination_mode``, a separate
        centrally computed broadcast supplies aggregate team context.
        """
        #Luu tru thong tin local nhu la vi tri UE va vi tri tuong doi bo di cung duoc, QoS cua UE, khoang cach cua UE toi UAV local do
        #Channel gain cua UAV va UE do, UE do da duoc phuc vu chua, UE co phai UE dang xet hay khong?
        local_features = np.zeros((self.num_ues, 9), dtype=np.float32)
        #UE do co duoc phu hay khong
        local_mask = np.zeros(self.num_ues, dtype=np.int8)
        if self.phase != self.ALLOCATION_PHASE:
            return local_features, local_mask

        covered = self.link_features[uav_index, :, 1] >= 0.5
        local_mask[covered] = 1
        if not np.any(covered):
            return local_features, local_mask

        # Both UE and UAV x/y positions use the same width/length normalizers,
        # so their difference is a stable signed relative coordinate in [-1,1].
        relative_xy = ue_features[:, :2] - uav_features[uav_index, :2]
        local_features[covered, 0:2] = relative_xy[covered]
        local_features[covered, 2] = ue_features[covered, 2]
        local_features[covered, 3] = self.link_features[uav_index, covered, 0]
        local_features[covered, 4] = 1.0
        local_features[covered, 5] = self.link_features[uav_index, covered, 2]
        local_features[covered, 6] = self.processed[covered]
        local_features[covered, 7] = self.served[covered]
        if covered[self.current_ue]:
            local_features[self.current_ue, 8] = 1.0
        return local_features, local_mask

    def _get_obs(self) -> dict[str, np.ndarray | np.int64]:
        """Build the global Markov state consumed by the centralized critic."""

        global_ues, ue_features = self._get_ue_arrays()
        uav_positions, uav_features = self._get_uav_arrays()
        current_ue = (
            self.current_ue
            if self.phase == self.ALLOCATION_PHASE
            else self.num_ues
        )
        allocation_masks = self.get_allocation_action_masks()
        #De ghi log
        return {
            "phase": np.int64(self.phase),
            "current_ue": np.int64(current_ue),
            "current_ue_mask": self._get_current_ue_mask(),
            "global_ues": global_ues,
            "ue_features": ue_features,
            "candidate_positions": self.candidate_positions,
            "candidate_features": self.candidate_features,
            "placement_indices": self.placement_indices.copy(),
            "uav_positions": uav_positions,
            "uav_features": uav_features,
            "uav_limits": self.uav_limits.copy(),
            "remaining_prbs": self.remaining_prbs.copy(),
            "remaining_power_w": self.remaining_power_w.astype(np.float32),
            "processed": self.processed.copy(),
            "served": self.served.copy(),
            "serving_uav": self.serving_uav.copy(),
            "allocated_prbs_ratio": self.allocated_prbs_ratio.copy(),
            "allocated_power_ratio": self.allocated_power_ratio.copy(),
            "achieved_rate_ratio": self.achieved_rate_ratio.copy(),
            "link_features": self.link_features.copy(),
            # Immutable placement data; the critic masks processed UEs per state.
            "allocation_cost_features": self._allocation_cost_features,
            "agent_ids": self.agent_ids.copy(),
            "action_mask": self._get_action_mask(),
            "allocation_uav_mask": allocation_masks["uav"],
            "allocation_participate_mask": allocation_masks["participate"],
            "allocation_prb_mask": allocation_masks["prb"],
            "allocation_power_mask": allocation_masks["power"],
            "allocation_pair_mask": allocation_masks["pair"],
        }

    def _get_coordination_features(
        self,
        uav_index: int,
        participate_mask: np.ndarray,
    ) -> np.ndarray:
        """Build the centrally computed per-UAV team broadcast.

        Features are: eligible-UAV fraction, this UAV's normalized link rank,
        best eligible link gain, team remaining-PRB ratio, and team
        remaining-power ratio. The last three describe this UAV's future
        exclusive workload: remaining exclusive-UE fraction and the minimum
        PRB/power reserve required by their cheapest QoS-valid Pareto actions.
        ``basic`` exposes only the first five signals, ``full`` exposes all
        eight. In ``none``, only slot 3 reports the shared PRB balance; the
        other seven coordination signals are disabled. The fixed eight-element
        shape is retained in every mode for model compatibility. The simulator
        recomputes this broadcast before each UE decision and models its
        delivery as instantaneous and cost-free; it does not claim a physical
        control-channel implementation.
        """

        features = np.zeros(self.COORDINATION_FEATURE_DIM, dtype=np.float32)
        if self.phase != self.ALLOCATION_PHASE:
            return features
        if self.coordination_mode == "none":
            features[3] = float(self.shared_prbs_remaining / self.total_max_prbs)
            return features

        eligible = participate_mask[:, 1].astype(bool)
        eligible_ids = np.flatnonzero(eligible)
        eligible_count = len(eligible_ids)
        features[0] = eligible_count / self.num_uavs
        if eligible_count:
            link_gains = self.link_features[:, self.current_ue, 2]
            ranked_ids = sorted(
                eligible_ids.tolist(),
                key=lambda index: (-float(link_gains[index]), int(index)),
            )
            if eligible[uav_index]:
                rank = ranked_ids.index(uav_index)
                features[1] = (
                    1.0
                    if eligible_count == 1
                    else 1.0 - rank / (eligible_count - 1)
                )
            features[2] = float(link_gains[ranked_ids[0]])
        features[3] = float(self.remaining_prbs.sum() / self.total_max_prbs)
        features[4] = float(
            self.remaining_power_w.sum() / self.total_max_power_w
        )
        if self.coordination_mode == "basic":
            return np.clip(features, 0.0, 1.0)
        future_indices = np.arange(self.current_ue + 1, self.num_ues)
        if future_indices.size:
            coverage_count = (
                self.link_features[:, future_indices, 1] >= 0.5
            ).sum(axis=0)
            exclusive_indices = future_indices[
                np.logical_and(
                    coverage_count == 1,
                    self.link_features[uav_index, future_indices, 1] >= 0.5,
                )
            ]
            features[5] = len(exclusive_indices) / self.num_ues
            reserve_prbs = 0.0
            reserve_power_w = 0.0
            for ue_index in exclusive_indices:
                prb_indices, power_indices = np.nonzero(
                    self.pareto_pair_masks[uav_index, ue_index]
                )
                if not len(prb_indices):
                    continue
                costs = (
                    self.prb_weight
                    * (prb_indices + 1)
                    / self.total_max_prbs
                    + self.power_weight
                    * self.power_levels_w[power_indices]
                    / self.total_max_power_w
                )
                cheapest = int(np.argmin(costs))
                reserve_prbs += int(prb_indices[cheapest]) + 1
                reserve_power_w += float(
                    self.power_levels_w[power_indices[cheapest]]
                )
            features[6] = reserve_prbs / self.max_prbs_per_uav[uav_index]
            features[7] = (
                reserve_power_w / self.max_power_w_per_uav[uav_index]
            )
        return np.clip(features, 0.0, 1.0)

    def get_actor_observations(
        self,
    ) -> list[dict[str, np.ndarray | np.int64]]:

        """Build the actor observations for the current phase.

        Placement is a special high-level task: before a position exists, an
        actor receives the shared UE map and candidate set. During allocation
        those common arrays are zeroed. Each actor then receives its local UAV
        view plus a centrally computed team broadcast selected by
        ``coordination_mode``; the fixed auction still performs joint
        arbitration. The global state returned by ``reset``/``step`` remains
        unchanged for the centralized critic.
        """

        _, ue_features = self._get_ue_arrays()
        _, uav_features = self._get_uav_arrays()
        current_mask = self._get_current_ue_mask()
        action_mask = self._get_action_mask()
        allocation_masks = self.get_allocation_action_masks()

        if self.phase == self.PLACEMENT_PHASE:
            placement_ue_features = ue_features.copy()
            placement_candidate_features = self.candidate_features.copy()
        else:
            # Placement-only common inputs are inactive during allocation.
            placement_ue_features = np.zeros_like(ue_features)
            placement_candidate_features = np.zeros_like(
                self.candidate_features
            )

        observations: list[dict[str, np.ndarray | np.int64]] = []
        for uav_index in range(self.num_uavs):
            if self.phase == self.ALLOCATION_PHASE:
                current_link = self.link_features[uav_index, self.current_ue]
                current_ue_features = np.asarray(
                    [
                        ue_features[self.current_ue, 0]
                        - uav_features[uav_index, 0],
                        ue_features[self.current_ue, 1]
                        - uav_features[uav_index, 1],
                        ue_features[self.current_ue, 2],
                    ],
                    dtype=np.float32,
                )
                #Mode none thi chi tra ve toan bo thong tin local
                if self.coordination_mode == "none" and current_link[1] < 0.5:
                    current_link = np.zeros(3, dtype=np.float32)
                    current_ue_features.fill(0.0)
            else:
                current_link = np.zeros(3, dtype=np.float32)
                current_ue_features = np.zeros(3, dtype=np.float32)
            local_ue_features, local_ue_mask = self._get_local_ue_view(
                uav_index,
                ue_features,
                uav_features,
            )
            observations.append(
                {
                    "phase": np.int64(self.phase),
                    "agent_id": np.int64(uav_index),
                    "agent_id_one_hot": self.agent_ids[uav_index].copy(),
                    "ue_features": placement_ue_features.copy(),
                    "candidate_features": (
                        placement_candidate_features.copy()
                    ),
                    "own_uav_features": uav_features[uav_index].copy(),
                    "own_uav_limits": self.uav_limits[uav_index].copy(),
                    "local_ue_features": local_ue_features,
                    "local_ue_mask": local_ue_mask,
                    "current_ue_mask": current_mask.copy(),
                    "current_ue_features": current_ue_features.copy(),
                    "current_link_features": current_link.copy(),
                    "coordination_features": self._get_coordination_features(
                        uav_index,
                        allocation_masks["participate"],
                    ),
                    "action_mask": action_mask[uav_index].copy(),
                    "allocation_participate_mask": allocation_masks[
                        "participate"
                    ][uav_index].copy(),
                    "allocation_prb_mask": allocation_masks["prb"][
                        uav_index
                    ].copy(),
                    "allocation_power_mask": allocation_masks["power"][
                        uav_index
                    ].copy(),
                    "allocation_pair_mask": allocation_masks["pair"][
                        uav_index
                    ].copy(),
                }
            )
        return observations

    def _get_info(self) -> dict[str, Any]:
        served_count = int(self.served.sum())
        if self.shared_prb_pool:
            # Derive attribution from committed allocations, not duplicated
            # access balances. Only the scalar shared pool is mutable budget.
            used_prbs = np.zeros(self.num_uavs, dtype=np.int32)
            for allocation in self.allocations:
                if allocation is not None and allocation.get("accepted"):
                    used_prbs[int(allocation["uav_id"])] += int(allocation["num_prbs"])
        else:
            used_prbs = self.max_prbs_per_uav - self.remaining_prbs
        used_power_w = self.max_power_w_per_uav - self.remaining_power_w
        return {
            "phase": ("placement", "allocation", "terminal")[self.phase],
            "coordination_mode": self.coordination_mode,
            "allocation_action_semantics": self.allocation_action_semantics,
            "shared_prb_pool": self.shared_prb_pool,
            "shared_prbs_remaining": (
                int(self.shared_prbs_remaining) if self.shared_prb_pool else None
            ),
            "episode_ue_seed": self.episode_ue_seed,
            "current_ue_index": (
                self.current_ue if self.phase == self.ALLOCATION_PHASE else None
            ),
            "current_ue_id": (
                self.ues[self.current_ue].id
                if self.phase == self.ALLOCATION_PHASE
                else None
            ),
            "served_count": served_count,
            "served_ratio": served_count / self.num_ues,
            "used_prbs_per_uav": used_prbs.tolist(),
            "total_used_prbs": (
                int(self.total_max_prbs - self.shared_prbs_remaining)
                if self.shared_prb_pool else int(used_prbs.sum())
            ),
            "used_power_w_per_uav": used_power_w.tolist(),
            "total_used_power_w": float(used_power_w.sum()),
            "placement_indices": self.placement_indices.tolist(),
            "allocations": copy.deepcopy(self.allocations),
            "last_evaluation": copy.deepcopy(self.last_evaluation),
            "reward_breakdown": self.last_reward_breakdown.copy(),
        }

    def _build_link_features(self) -> None:
        self.link_features.fill(0.0)
        costs = np.zeros_like(self._allocation_cost_features)
        for uav_index, position in enumerate(self.selected_positions):
            for ue_index, ue in enumerate(self.ues):
                distance_m = float(
                    env.dis_2d(position.x_u, position.y_u, ue.x_m, ue.y_m)
                )
                covered = float(distance_m <= position.r_u)
                channel_gain = max(float(env.gain(position, ue)), np.finfo(float).tiny)
                gain_db = 10.0 * np.log10(channel_gain)
                gain_db = float(
                    np.clip(gain_db, self.MIN_GAIN_DB, self.MAX_GAIN_DB)
                )
                gain_normalized = (gain_db - self.MIN_GAIN_DB) / (
                    self.MAX_GAIN_DB - self.MIN_GAIN_DB
                )
                self.link_features[uav_index, ue_index] = (
                    np.clip(distance_m / self.area_diagonal_m, 0.0, 1.0),
                    covered,
                    gain_normalized,
                )
                if self.include_allocation_cost_features and self.shared_prb_pool and covered:
                    # Critic-only: minimum grid power meeting QoS for each PRB.
                    # Ignore dynamic budgets; do not prune any actor actions.
                    for prb_index, num_prbs in enumerate(self.prb_levels):
                        for power_w in self.power_levels_w:
                            rate, _ = env.data_rate_mbps(
                                float(power_w), int(num_prbs), channel_gain
                            )
                            if rate >= float(ue.require_mbps):
                                costs[uav_index, ue_index, prb_index] = (
                                    power_w / self.power_levels_w[-1]
                                )
                                costs[uav_index, ue_index, self.max_prbs + prb_index] = 1.0
                                break
        costs.setflags(write=False)
        self._allocation_cost_features = costs

    def _build_pareto_pair_masks(self) -> None:
        """Precompute QoS-valid, resource-nondominated PRB/power pairs.

        For a fixed PRB count, every feasible power above the minimum is
        dominated. Across PRB counts, a pair remains on the Pareto frontier
        only when it needs strictly less power than every lower-PRB pair.
        Dynamic budget masking can safely remove frontier pairs later because
        a pair's dominator never consumes more of either resource.
        """

        self.pareto_pair_masks.fill(0)
        if self.allocation_mask_mode != "pareto":
            return
        num_prbs = self.prb_levels.astype(np.float64)[:, None]
        power_w = self.power_levels_w[None, :]
        noise_w = (
            env.phys.k_B
            * env.phys.T
            * env.phys.Bandwidth_BRP
            * env.phys.noise_figure_linear
            * num_prbs
        )
        for uav_index, position in enumerate(self.selected_positions):
            for ue_index, ue in enumerate(self.ues):
                if self.link_features[uav_index, ue_index, 1] < 0.5:
                    continue
                channel_gain = float(env.gain(position, ue))
                sinr = power_w * channel_gain / noise_w
                rates_mbps = (
                    num_prbs
                    * env.phys.Bandwidth_BRP
                    * np.log2(1.0 + sinr)
                    / 1e6
                )
                feasible = rates_mbps >= float(ue.require_mbps)
                best_power_index = self.num_power_levels
                for prb_index in range(self.max_prbs):
                    feasible_power = np.flatnonzero(feasible[prb_index])
                    if not len(feasible_power):
                        continue
                    power_index = int(feasible_power[0])
                    if power_index < best_power_index:
                        self.pareto_pair_masks[
                            uav_index, ue_index, prb_index, power_index
                        ] = 1
                        best_power_index = power_index

    def _get_action_mask(self) -> np.ndarray:
        """Return the phase mask used only by the five placement actors."""

        mask = np.zeros(
            (self.num_uavs, self.max_action_choices), dtype=np.int8
        )
        if self.phase == self.PLACEMENT_PHASE:
            mask[:, : self.num_solutions] = 1
        return mask

    def get_allocation_action_masks(self) -> dict[str, np.ndarray]:
        """Return coverage or Pareto masks independently of shared PRB resources.

        Coverage mode permits any pair inside coverage; env checks feasibility.
        Pareto mode works with shared budgets too, independently of broadcasts.

        The Pareto mask removes actions that cannot pass coverage, remaining
        resource, and QoS constraints, then removes resource-dominated pairs.
        It does not rank UAVs and does not use the critic; the fixed auction
        still chooses the minimum resource bid among submitted proposals.
        """

        uav_mask = np.zeros(self.num_uavs + 1, dtype=np.int8)
        participate_mask = np.zeros((self.num_uavs, 2), dtype=np.int8)
        prb_mask = np.zeros((self.num_uavs, self.max_prbs), dtype=np.int8)
        power_mask = np.zeros(
            (self.num_uavs, self.num_power_levels), dtype=np.int8
        )
        pair_mask = np.zeros(
            (self.num_uavs, self.max_prbs, self.num_power_levels),
            dtype=np.int8,
        )
        if self.phase != self.ALLOCATION_PHASE:
            return {
                "uav": uav_mask,
                "participate": participate_mask,
                "prb": prb_mask,
                "power": power_mask,
                "pair": pair_mask,
            }

        # Reject is always available. The other heads are ignored on reject.
        uav_mask[0] = 1
        participate_mask[:, 0] = 1
        if self.allocation_mask_mode == "coverage":
            covered = self.link_features[:, self.current_ue, 1] >= 0.5
            uav_mask[1:] = covered
            participate_mask[:, 1] = covered
            prb_mask[covered] = 1
            power_mask[covered] = 1
            pair_mask[covered] = 1
            return {
                "uav": uav_mask,
                "participate": participate_mask,
                "prb": prb_mask,
                "power": power_mask,
                "pair": pair_mask,
            }
        for uav_index in range(self.num_uavs):
            remaining_prbs = int(self.remaining_prbs[uav_index])
            remaining_power_w = float(self.remaining_power_w[uav_index])
            valid_pairs = self.pareto_pair_masks[
                uav_index, self.current_ue
            ].copy()
            valid_pairs[remaining_prbs:] = 0
            valid_pairs[:, self.power_levels_w > remaining_power_w + 1e-12] = 0
            if not valid_pairs.any():
                continue
            pair_mask[uav_index] = valid_pairs
            uav_mask[uav_index + 1] = 1
            participate_mask[uav_index, 1] = 1
            prb_mask[uav_index] = valid_pairs.any(axis=1).astype(np.int8)
            power_mask[uav_index] = valid_pairs.any(axis=0).astype(np.int8)
        return {
            "uav": uav_mask,
            "participate": participate_mask,
            "prb": prb_mask,
            "power": power_mask,
            "pair": pair_mask,
        }

    def _evaluate_allocation_proposal(
        self,
        uav_index: int,
        power_index: int,
        num_prbs: int,
        power_w: float,
    ) -> dict[str, Any]:
        """Evaluate the unique UAV proposal without consuming resources."""
        #Cai nay de cho COMA va ghi log thoi
        cache = self._counterfactual_proposals
        key = (uav_index, power_index, num_prbs, power_w)
        if cache is not None and key in cache:
            return cache[key]
        ue = self.ues[self.current_ue]
        position = self.selected_positions[uav_index]
        gains = self._counterfactual_gains
        if gains is not None and uav_index in gains:
            channel_gain = gains[uav_index]
        else:
            channel_gain = float(env.gain(position, ue))
            if gains is not None:
                gains[uav_index] = channel_gain
        rate_mbps, sinr = env.data_rate_mbps(
            power_w, num_prbs, channel_gain
        )
        rate_mbps = float(rate_mbps)
        sinr = float(sinr)
        covered = bool(
            self.link_features[uav_index, self.current_ue, 1] >= 0.5
        )

        reason = "accepted"
        if num_prbs <= 0 or power_w <= 0.0:
            reason = "zero_resource"
        elif not covered:
            reason = "out_of_coverage"
        elif num_prbs > self.remaining_prbs[uav_index]:
            reason = "prb_budget_exceeded"
        elif power_w > self.remaining_power_w[uav_index] + 1e-12:
            reason = "power_budget_exceeded"
        elif rate_mbps < ue.require_mbps:
            reason = "qos_not_met"

        # None uses the winner's 2 W budget, not the team's 10 W budget.
        power_cost = self.power_weight * power_w / self.power_penalty_denominator_w
        prb_cost = self.prb_weight * num_prbs / self.total_max_prbs
        proposal = {
            "ue_id": ue.id,
            "uav_id": uav_index,
            "position_index": int(self.placement_indices[uav_index]),
            "power_index": power_index,
            "power_w": power_w,
            "num_prbs": num_prbs,
            "channel_gain": channel_gain,
            "sinr": sinr,
            "rate_mbps": rate_mbps,
            "required_mbps": float(ue.require_mbps),
            "covered": covered,
            "accepted": reason == "accepted",
            "reason": reason,
            "bid": float(power_cost + prb_cost),
            "resource_cost": float(power_cost + prb_cost),
            "power_cost": float(power_cost),
            "prb_cost": float(prb_cost),
        }
        if cache is not None:
            cache[key] = proposal
        return proposal

    @contextmanager
    def _cache_counterfactual_proposals(self):
        """Reuse proposals only across previews of one unchanged UE/budget state."""

        if self.phase != self.ALLOCATION_PHASE:
            raise RuntimeError("Counterfactual caching requires allocation phase")
        if self._counterfactual_proposals is not None:
            raise RuntimeError("Counterfactual proposal cache is already active")
        self._counterfactual_proposals = {}
        self._counterfactual_gains = {}
        try:
            yield
        finally:
            self._counterfactual_proposals = None
            self._counterfactual_gains = None

    @staticmethod
    def _proposal_rank(proposal: dict[str, Any]) -> tuple[float, float, int]:
        """Implement the deterministic reverse auction: lowest bid wins."""

        return (
            allocation_cost_key(proposal["bid"]),
            -float(proposal["rate_mbps"]),
            int(proposal["uav_id"]),
        )

    def _resolve_joint_allocation(
        self, actions: np.ndarray
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
        """Apply the fixed feasibility filter and select the minimum-cost bid.

        Each row is ``[participate, prb_index, power_index]``.  The filter is
        physical only: coverage, remaining budgets, and QoS.  The critic and
        any estimate of future return are deliberately absent from execution.
        """

        evaluated: list[dict[str, Any]] = []
        for uav_index, row in enumerate(actions):
            participate, prb_index, power_index = map(int, row)
            if participate == 0:
                continue
            evaluated.append(
                self._evaluate_allocation_proposal(
                    uav_index,
                    power_index,
                    int(self.prb_levels[prb_index]),
                    float(self.power_levels_w[power_index]),
                )
            )
        feasible = [proposal for proposal in evaluated if proposal["accepted"]]
        winner = min(feasible, key=self._proposal_rank) if feasible else None
        return evaluated, feasible, winner

    def winning_uav_for_actions(
        self, action: np.ndarray | Sequence[Sequence[int]]
    ) -> int | None:
        """Return the fixed auction winner without changing environment state."""

        actions = _validated_multidiscrete_action(
            action,
            self.allocation_action_space,
            "Allocation",
        )
        _, _, winner = self._resolve_joint_allocation(actions)
        return None if winner is None else int(winner["uav_id"])

    def _evaluate_joint_allocation(
        self, actions: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        """Resolve five local proposals and commit only the auction winner."""

        ue = self.ues[self.current_ue]
        breakdown = self._empty_reward_breakdown()
        evaluated, feasible, winner = self._resolve_joint_allocation(actions)
        if winner is None:
            evaluation = {
                "ue_id": ue.id,
                "accepted": False,
                "reason": (
                    "not_allocated" if not evaluated else "no_feasible_proposal"
                ),
                "num_proposals": len(evaluated),
                "proposing_uav_ids": [
                    int(proposal["uav_id"]) for proposal in evaluated
                ],
                "proposal_evaluations": evaluated,
            }
            self.last_evaluation = evaluation
            self.allocations[self.current_ue] = evaluation.copy()
            return 0.0, breakdown

        uav_index = int(winner["uav_id"])
        num_prbs = int(winner["num_prbs"])
        power_w = float(winner["power_w"])
        rate_mbps = float(winner["rate_mbps"])
        evaluation = winner.copy()
        evaluation.update(
            {
                "accepted": True,
                "reason": (
                    "accepted" if len(evaluated) == 1
                    else "accepted_after_arbitration"
                ),
                "num_proposals": len(evaluated),
                "proposing_uav_ids": [
                    int(proposal["uav_id"]) for proposal in evaluated
                ],
                "feasible_uav_ids": [
                    int(proposal["uav_id"]) for proposal in feasible
                ],
                "winning_uav_id": uav_index,
                "proposal_evaluations": evaluated,
            }
        )
        self.last_evaluation = evaluation
        self.allocations[self.current_ue] = evaluation.copy()

        if self.shared_prb_pool:
            self.shared_prbs_remaining -= num_prbs
        else:
            self.remaining_prbs[uav_index] -= num_prbs
        self.remaining_power_w[uav_index] = max(
            0.0, self.remaining_power_w[uav_index] - power_w
        )
        self.served[self.current_ue] = 1
        self.serving_uav[self.current_ue] = uav_index
        self.allocated_prbs_ratio[self.current_ue] = (
            num_prbs / self.max_prbs_per_uav[uav_index]
        )
        self.allocated_power_ratio[self.current_ue] = (
            power_w / self.max_power_w_per_uav[uav_index]
        )
        self.achieved_rate_ratio[self.current_ue] = np.clip(
            rate_mbps / float(ue.require_mbps), 0.0, 2.0
        )
        power_cost = float(winner["power_cost"])
        prb_cost = float(winner["prb_cost"])
        breakdown["served_reward"] = self.served_reward
        breakdown["power_cost"] = -power_cost
        breakdown["prb_cost"] = -prb_cost
        net_reward = self.served_reward - power_cost - prb_cost
        breakdown["net_reward"] = net_reward
        return net_reward, breakdown

    def _apply_placement(self, actions: np.ndarray) -> None:
        selected_positions = [
            self.solutions[int(index)] for index in actions
        ]
        selected_xyz = [
            (position.x_u, position.y_u, position.h_u)
            for position in selected_positions
        ]
        if len(set(selected_xyz)) != self.num_uavs:
            raise ValueError(
                "Placement actions must select distinct (x, y, z) "
                "coordinates for every UAV"
            )

        self.placement_indices = actions.astype(np.int32, copy=True)
        self.selected_positions = selected_positions
        self._build_link_features()
        if self.allocation_mask_mode == "pareto":
            self._build_pareto_pair_masks()
        self.phase = self.ALLOCATION_PHASE

        duplicate_count = self.num_uavs - len(set(selected_xyz))
        self.last_evaluation = {
            "accepted": True,
            "reason": "positions_fixed",
            "duplicate_count": duplicate_count,
        }

    @staticmethod
    def _empty_reward_breakdown() -> dict[str, float]:
        return {
            "served_reward": 0.0,
            "power_cost": 0.0,
            "prb_cost": 0.0,
            "net_reward": 0.0,
        }

    def preview_allocation_step(
        self, action: np.ndarray | Sequence[Sequence[int]], *, include_info: bool = True
    ) -> tuple[
        dict[str, np.ndarray | np.int64],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Evaluate then restore; training may omit unused info construction."""

        if self.phase != self.ALLOCATION_PHASE:
            raise RuntimeError("Counterfactual preview requires allocation phase")
        snapshot = {
            "phase": self.phase,
            "current_ue": self.current_ue,
            "remaining_prbs": self._remaining_prbs.copy(),
            "shared_prbs_remaining": self.shared_prbs_remaining,
            "remaining_power_w": self.remaining_power_w.copy(),
            "processed": self.processed.copy(),
            "served": self.served.copy(),
            "serving_uav": self.serving_uav.copy(),
            "allocated_prbs_ratio": self.allocated_prbs_ratio.copy(),
            "allocated_power_ratio": self.allocated_power_ratio.copy(),
            "achieved_rate_ratio": self.achieved_rate_ratio.copy(),
            "allocations": self.allocations.copy(),
            "last_evaluation": self.last_evaluation,
            "last_reward_breakdown": self.last_reward_breakdown.copy(),
            "seed_index": self._seed_index,
            "episodes_on_seed": self._episodes_on_seed,
        }
        try:
            return self._step(action, include_info=include_info)
        finally:
            self.phase = int(snapshot["phase"])
            self.current_ue = int(snapshot["current_ue"])
            self._remaining_prbs = snapshot["remaining_prbs"]
            self.shared_prbs_remaining = int(snapshot["shared_prbs_remaining"])
            self.remaining_power_w = snapshot["remaining_power_w"]
            self.processed = snapshot["processed"]
            self.served = snapshot["served"]
            self.serving_uav = snapshot["serving_uav"]
            self.allocated_prbs_ratio = snapshot["allocated_prbs_ratio"]
            self.allocated_power_ratio = snapshot["allocated_power_ratio"]
            self.achieved_rate_ratio = snapshot["achieved_rate_ratio"]
            self.allocations = snapshot["allocations"]
            self.last_evaluation = snapshot["last_evaluation"]
            self.last_reward_breakdown = snapshot["last_reward_breakdown"]
            self._seed_index = int(snapshot["seed_index"])
            self._episodes_on_seed = int(snapshot["episodes_on_seed"])

    def step(self, action):
        if self._counterfactual_proposals is not None:
            raise RuntimeError("Cannot commit a step inside the counterfactual proposal cache")
        return self._step(action)

    def _step(
        self,
        action: (
            dict[str, np.ndarray]
            | np.ndarray
            | Sequence[int]
            | Sequence[Sequence[int]]
        ),
        *,
        include_info: bool = True,
    ) -> tuple[
        dict[str, np.ndarray | np.int64],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if self.phase == self.TERMINAL_PHASE:
            raise RuntimeError("Episode is finished; call reset() before step()")

        if isinstance(action, dict):
            if not self.action_space.contains(action):
                raise ValueError(
                    f"Action {action} is outside {self.action_space}"
                )
            phase_key = (
                "placement"
                if self.phase == self.PLACEMENT_PHASE
                else "allocation"
            )
            action = action[phase_key]

        if self.phase == self.PLACEMENT_PHASE:
            actions = _validated_multidiscrete_action(
                action,
                self.placement_action_space,
                "Placement",
            )
            active_mask = self._get_action_mask()
            invalid = [
                uav_index
                for uav_index, code in enumerate(actions)
                if not active_mask[uav_index, int(code)]
            ]
            if invalid:
                raise ValueError(
                    "Placement action contains phase-invalid candidate IDs "
                    f"for UAVs {invalid}. Apply observation['action_mask'] "
                    "before sampling."
                )

            self._apply_placement(actions)
            breakdown = self._empty_reward_breakdown()
            self.last_reward_breakdown = breakdown
            return (
                self._get_obs(),
                0.0,
                False,
                False,
                self._get_info(),
            )

        actions = _validated_multidiscrete_action(
            action,
            self.allocation_action_space,
            "Allocation",
        )
        step_reward, breakdown = self._evaluate_joint_allocation(actions)
        self.processed[self.current_ue] = 1
        self.current_ue += 1
        terminated = self.current_ue == self.num_ues

        if terminated:
            self.phase = self.TERMINAL_PHASE
            if self._episode_uses_scheduled_seed:
                self._episodes_on_seed += 1
                if self._episodes_on_seed == self.EPISODES_PER_SEED:
                    self._episodes_on_seed = 0
                    self._seed_index = (self._seed_index + 1) % len(self.seed)

        self.last_reward_breakdown = breakdown
        return (
            self._get_obs(),
            float(step_reward),
            terminated,
            False,
            self._get_info() if include_info else {},
        )

UAVEnv = gym_env

def _self_check() -> None:
    """Exercise the Gym action contract, broadcasts, and QoS/Pareto masks."""

    try:
        gym_env(coordination_mode="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid coordination mode was accepted")

    environment = gym_env(coordination_mode="full")
    observation, info = environment.reset(seed=env.seed)
    assert info["episode_ue_seed"] == seed[0]
    assert info["coordination_mode"] == "full"
    assert environment.observation_space.contains(observation)
    assert all(
        environment.single_actor_observation_space.contains(actor_observation)
        for actor_observation in environment.get_actor_observations()
    )

    for _ in range(32):
        sampled_action = environment.action_space.sample()
        assert environment.action_space.contains(sampled_action)
        sampled_positions = [
            environment.solutions[int(index)]
            for index in sampled_action["placement"]
        ]
        assert len(
            {
                (position.x_u, position.y_u, position.h_u)
                for position in sampled_positions
            }
        ) == environment.num_uavs

    dict_environment = gym_env()
    dict_environment.reset(seed=env.seed)
    dict_environment.step(dict_environment.action_space.sample())
    assert dict_environment.phase == dict_environment.ALLOCATION_PHASE
    dict_environment.step(dict_environment.action_space.sample())

    first_ue = environment.ues[0]
    ranked_candidates = sorted(
        range(environment.num_solutions),
        key=lambda index: env.gain(environment.solutions[index], first_ue),
        reverse=True,
    )
    if environment.num_solutions >= environment.num_uavs:
        placement_action = np.asarray(
            ranked_candidates[: environment.num_uavs], dtype=np.int64
        )
    else:
        placement_action = np.resize(
            np.asarray(ranked_candidates, dtype=np.int64),
            environment.num_uavs,
        )

    observation, _, terminated, _, _ = environment.step(placement_action)
    assert not terminated
    assert observation["phase"] == environment.ALLOCATION_PHASE
    assert environment.observation_space.contains(observation)

    full_coordination = np.stack(
        [
            actor_observation["coordination_features"]
            for actor_observation in environment.get_actor_observations()
        ]
    )
    environment.coordination_mode = "basic"
    basic_coordination = np.stack(
        [
            actor_observation["coordination_features"]
            for actor_observation in environment.get_actor_observations()
        ]
    )
    np.testing.assert_allclose(
        basic_coordination[:, :5], full_coordination[:, :5]
    )
    assert not basic_coordination[:, 5:].any()
    environment.coordination_mode = "full"

    local_environment = gym_env()
    local_environment.reset(seed=env.seed)
    local_environment.step(placement_action)
    assert local_environment.coordination_mode == "none"
    assert local_environment.shared_prbs_remaining == 135
    assert local_environment.max_prbs == 10
    assert local_environment.num_power_levels == 10
    assert not local_environment.pareto_pair_masks.any()
    for actor_observation in local_environment.get_actor_observations():
        assert local_environment.single_actor_observation_space.contains(actor_observation)
        expected_coordination = np.zeros(8, dtype=np.float32)
        expected_coordination[3] = 1.0
        np.testing.assert_array_equal(
            actor_observation["coordination_features"], expected_coordination
        )
        covered = actor_observation["current_link_features"][1] >= 0.5
        assert (actor_observation["allocation_pair_mask"] == covered).all()

    local_action = np.zeros((local_environment.num_uavs, 3), dtype=np.int64)
    local_action[0] = [1, 9, 9]
    preview, preview_reward, _, _, preview_info = local_environment.preview_allocation_step(local_action)
    assert preview_info["last_evaluation"]["accepted"]
    assert np.isclose(preview_reward, 1.0 - (4.0 / 27.0) * 1.0 / 2.0 - 10.0 / 135.0)
    assert preview_info["total_used_prbs"] == 10
    assert local_environment.shared_prbs_remaining == 135
    assert not local_environment.served.any()
    committed, committed_reward, _, _, _ = local_environment.step(local_action)
    assert local_environment.observation_space.contains(committed)
    assert local_environment.shared_prbs_remaining == 125
    assert committed_reward == preview_reward
    for key in preview:
        np.testing.assert_array_equal(preview[key], committed[key])

    masks = environment.get_allocation_action_masks()
    assert masks["uav"].shape == (environment.num_uavs + 1,)
    assert masks["prb"].shape == (environment.num_uavs, environment.max_prbs)
    assert masks["power"].shape == (
        environment.num_uavs,
        environment.num_power_levels,
    )
    assert masks["pair"].shape == (
        environment.num_uavs,
        environment.max_prbs,
        environment.num_power_levels,
    )
    assert masks["uav"][0] == 1

    # A UAV may participate exactly when at least one QoS-valid Pareto pair
    # fits its remaining resources.
    for uav_index in range(environment.num_uavs):
        pairs = np.argwhere(masks["pair"][uav_index])
        if not len(pairs):
            assert masks["uav"][uav_index + 1] == 0
            assert not masks["prb"][uav_index].any()
            assert not masks["power"][uav_index].any()
            continue
        assert masks["uav"][uav_index + 1] == 1
        assert masks["participate"][uav_index, 1] == 1
        for prb_index, power_index in pairs:
            rate_mbps, _ = env.data_rate_mbps(
                environment.power_levels_w[power_index],
                int(prb_index) + 1,
                env.gain(environment.selected_positions[uav_index], first_ue),
            )
            assert rate_mbps >= first_ue.require_mbps

    # Find one QoS-feasible proposal for the fixed filter and auction.
    allocation_action = np.zeros((environment.num_uavs, 3), dtype=np.int64)
    feasible_uav: int | None = None
    for uav_index in range(environment.num_uavs):
        if not masks["uav"][uav_index + 1]:
            continue
        pairs = np.argwhere(masks["pair"][uav_index])
        if len(pairs):
            prb_index, power_index = pairs[0]
            allocation_action[uav_index] = (
                1,
                int(prb_index),
                int(power_index),
            )
            feasible_uav = uav_index
        if feasible_uav is not None:
            break

    observation, _, _, _, info = environment.step(allocation_action)
    if feasible_uav is None:
        assert info["last_evaluation"]["reason"] == "not_allocated"
    else:
        assert info["last_evaluation"]["accepted"]
        assert info["last_evaluation"]["winning_uav_id"] == feasible_uav
        assert observation["served"][0] == 1
    assert environment.observation_space.contains(observation)

    environment._episodes_on_seed = environment.EPISODES_PER_SEED - 1
    environment.current_ue = environment.num_ues - 1
    _, _, terminated, _, _ = environment.step(
        np.zeros((environment.num_uavs, 3), dtype=np.int64)
    )
    assert terminated
    _, info = environment.reset()
    assert info["episode_ue_seed"] == seed[1]


if __name__ == "__main__":
    _self_check()
    print("gym_env self-check: OK")
